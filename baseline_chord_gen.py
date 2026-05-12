"""
baseline_chord_gen.py
---------------------
Baseline chord progression generator using Qwen3.5-2B (no fine-tuning).
Validates output against music theory rules, and renders a MIDI file if valid.

Usage (Colab):
    !pip install transformers accelerate music21 midiutil

    from baseline_chord_gen import generate

    result = generate(
        key="C",
        mode="major",
        style="jazz",
        num_bars=8,
        output_midi_path="output.mid"
    )
    print(result)
"""

import json
import re
import statistics
from dataclasses import dataclass
from typing import Optional

# ── music libs ───────────────────────────────────────────────────────────────
from music21 import harmony, key as m21key, scale as m21scale
from midiutil import MIDIFile

# ── model libs ───────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-2B"

STYLE_DESCRIPTIONS = {
    "jazz":   "Use 7th chords, 9th chords, and ii-V-I progressions. Aim for harmonic complexity.",
    "pop":    "Use mostly I, IV, V, vi chords. Keep it simple and singable.",
    "blues":  "Use dominant 7th chords on I, IV, and V. Follow 12-bar blues conventions.",
    "folk":   "Use simple triads — I, IV, V, and maybe ii or vi. Keep it diatonic.",
    "bossa":  "Similar to jazz but favour maj7, min7, dom7 chords with smooth voice leading.",
}

# Chords music21 struggles to parse — map them to equivalents
CHORD_ALIASES = {
    "maj": "maj",
    "min": "m",
    "m":   "m",
}

MIDI_TEMPO    = 120   # BPM
BEATS_PER_BAR = 4
VELOCITY      = 75


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    prompt: str
    raw_output: str
    chords: Optional[list[str]]
    valid: bool
    validation_errors: list[str]
    reward: float
    midi_path: Optional[str]
    mp3_path: Optional[str] = None
    report_path: Optional[str] = None
    reward_breakdown: Optional[dict] = None

    def play(self):
        """Display an inline audio player in Jupyter/Colab."""
        if self.mp3_path is None:
            print("No MP3 available.")
            return
        try:
            from IPython.display import Audio, display
            display(Audio(self.mp3_path))
        except ImportError:
            print(f"IPython not available. MP3 saved to: {self.mp3_path}")

    def format_report(self) -> str:
        lines = [
            "── Chord Progression Report ─────────────────────────",
            f"Chords  : {self.chords}",
            f"Valid   : {self.valid}",
            f"Reward  : {self.reward}",
        ]
        if self.validation_errors:
            lines.append(f"Errors  : {self.validation_errors}")
        if self.midi_path:
            lines.append(f"MIDI    : {self.midi_path}")
        if self.mp3_path:
            lines.append(f"MP3     : {self.mp3_path}")
        lines += [
            "── Prompt ───────────────────────────────────────────",
            self.prompt,
            "── Raw Output ───────────────────────────────────────",
            self.raw_output,
            "─────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


@dataclass
class BatchResult:
    results: list[GenerationResult]
    n: int
    pass_rate: float
    mean_reward: float
    std_reward: float
    median_reward: float
    min_reward: float
    max_reward: float
    mean_breakdown: dict  # {"key": mean, "style": mean, "voice": mean}

    def format_report(self) -> str:
        lines = [
            "══ Batch Report ════════════════════════════════════",
            f"Runs       : {self.n}",
            f"Pass rate  : {self.pass_rate:.0%}  ({sum(r.valid for r in self.results)}/{self.n})",
            f"Reward     : mean={self.mean_reward:.3f}  std={self.std_reward:.3f}  "
            f"median={self.median_reward:.3f}  min={self.min_reward:.3f}  max={self.max_reward:.3f}",
            f"Breakdown  : key={self.mean_breakdown['key']:.3f}  "
            f"style={self.mean_breakdown['style']:.3f}  "
            f"voice={self.mean_breakdown['voice']:.3f}",
            "── Per-run ──────────────────────────────────────────",
        ]
        for i, r in enumerate(self.results):
            mark = "✓" if r.valid else "✗"
            chords_str = " ".join(r.chords) if r.chords else "<unparsed>"
            lines.append(f"  [{i+1:>2}] {mark} reward={r.reward:.3f}  {chords_str}")
        lines.append("════════════════════════════════════════════════════")
        return "\n".join(lines)

    def best(self) -> GenerationResult:
        """Return the result with the highest reward."""
        return max(self.results, key=lambda r: r.reward)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(key: str, mode: str, style: str, num_bars: int) -> str:
    style_hint = STYLE_DESCRIPTIONS.get(style, "")
    return (
        f"Generate a {num_bars}-bar chord progression in {key} {mode}. "
        f"Style: {style}. {style_hint}\n\n"
        f"Rules:\n"
        f"- Output ONLY a JSON array of {num_bars} chord symbol strings, nothing else.\n"
        f"- Example format: [\"Cmaj7\", \"Am7\", \"Dm7\", \"G7\"]\n"
        f"- Use standard chord symbols (e.g. Cmaj7, Dm7, G7, Am, Fmaj7).\n"
        f"- Do not include bar numbers, explanations, or any other text.\n"
        f"/no_think"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_chord_list(raw: str) -> Optional[list[str]]:
    """Extract a JSON array of chord strings from raw model output."""
    # Strip thinking tags if they sneak through
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try to find a JSON array anywhere in the output
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        chords = json.loads(match.group())
        if isinstance(chords, list) and all(isinstance(c, str) for c in chords):
            return [c.strip() for c in chords]
    except json.JSONDecodeError:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Validation (reward functions)
# ─────────────────────────────────────────────────────────────────────────────

def get_scale_pitch_names(key_str: str, mode: str) -> set[str]:
    """Return the set of pitch name strings (e.g. {'C','D','E',...}) for a key."""
    tonic = key_str.replace("b", "-")  # music21 uses '-' for flats
    if mode == "major":
        sc = m21scale.MajorScale(tonic)
    elif mode == "minor":
        sc = m21scale.MinorScale(tonic)
    else:
        sc = m21scale.MajorScale(tonic)
    return {p.name for p in sc.getPitches(f"{tonic}4", f"{tonic}5")}


def try_parse_chord(chord_str: str) -> Optional[harmony.ChordSymbol]:
    """Attempt to parse a chord symbol; return None on failure."""
    try:
        c = harmony.ChordSymbol(chord_str)
        # Trigger full resolution so errors surface now
        _ = c.pitches
        return c
    except Exception:
        return None


def validate(
    chords: list[str],
    key_str: str,
    mode: str,
    num_bars: int,
    style: str,
) -> tuple[list[str], float, dict]:
    """
    Run all reward checks. Returns (errors, reward_score ∈ [0,1], breakdown).

    Reward breakdown:
        0.00  → unparseable / wrong length (hard gate)
        +0.50  → key conformance score (fraction of chords with diatonic root)
        +0.30  → style score
        +0.20  → voice-leading smoothness
    """
    errors: list[str] = []
    breakdown = {"key": 0.0, "style": 0.0, "voice": 0.0}

    # ── Hard gate 1: length ──────────────────────────────────────────────────
    if len(chords) != num_bars:
        errors.append(f"Length mismatch: expected {num_bars}, got {len(chords)}")
        return errors, 0.0, breakdown

    # ── Hard gate 2: parseability ────────────────────────────────────────────
    parsed = []
    for c in chords:
        obj = try_parse_chord(c)
        if obj is None:
            errors.append(f"Unparseable chord: '{c}'")
        parsed.append(obj)

    if any(p is None for p in parsed):
        return errors, 0.1, breakdown  # partial credit — at least right length

    # ── Soft reward 1: key conformance (weight 0.5) ──────────────────────────
    scale_pitches = get_scale_pitch_names(key_str, mode)
    diatonic_count = sum(
        1 for c in parsed if c.root().name in scale_pitches
    )
    r_key = diatonic_count / len(parsed)

    non_diatonic = [
        chords[i] for i, c in enumerate(parsed)
        if c.root().name not in scale_pitches
    ]
    if non_diatonic:
        errors.append(
            f"Non-diatonic roots ({len(non_diatonic)}/{num_bars}): {non_diatonic}"
        )

    # ── Soft reward 2: style score (weight 0.3) ──────────────────────────────
    r_style = _style_score(parsed, chords, style)

    # ── Soft reward 3: voice leading (weight 0.2) ────────────────────────────
    r_voice = _voice_leading_score(parsed)

    reward = (r_key * 0.5) + (r_style * 0.3) + (r_voice * 0.2)
    breakdown = {
        "key": round(r_key, 3),
        "style": round(r_style, 3),
        "voice": round(r_voice, 3),
    }
    return errors, round(reward, 3), breakdown


def _style_score(parsed: list, chord_strs: list[str], style: str) -> float:
    """Heuristic style match score [0,1]."""
    if style == "jazz" or style == "bossa":
        # Reward 7th+ chords
        extended = sum(
            1 for c in parsed
            if any(q in c.commonName for q in ["seventh", "ninth", "eleventh"])
        )
        return min(extended / max(len(parsed) * 0.6, 1), 1.0)

    elif style == "blues":
        # Reward dominant 7ths
        dom7 = sum(
            1 for c in parsed
            if "dominant" in c.commonName and "seventh" in c.commonName
        )
        return min(dom7 / max(len(parsed) * 0.5, 1), 1.0)

    elif style in ("pop", "folk"):
        # Reward simple triads / no extensions
        simple = sum(
            1 for c in parsed
            if len(c.pitches) <= 4
        )
        return simple / len(parsed)

    return 0.5  # unknown style — neutral


def _voice_leading_score(parsed: list) -> float:
    """Score smoothness of root motion [0,1]. Smaller intervals = higher score."""
    if len(parsed) < 2:
        return 1.0
    intervals = []
    for a, b in zip(parsed, parsed[1:]):
        semitones = abs(a.root().midi - b.root().midi) % 12
        semitones = min(semitones, 12 - semitones)  # fold to [0,6]
        intervals.append(semitones)
    # 0 semitones = 1.0, 6 semitones = 0.0
    avg = sum(intervals) / len(intervals)
    return round(1.0 - (avg / 6.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# MIDI rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_midi(chords: list[str], output_path: str, tempo: int = MIDI_TEMPO):
    """Write a MIDI file with block chords, one chord per bar."""
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)

    beat = 0
    for chord_str in chords:
        c = harmony.ChordSymbol(chord_str)
        pitches = [p.midi for p in c.pitches]
        for p in pitches:
            midi.addNote(0, 0, p, beat, BEATS_PER_BAR - 0.1, VELOCITY)
        beat += BEATS_PER_BAR

    with open(output_path, "wb") as f:
        midi.writeFile(f)


def render_mp3(midi_path: str, output_path: str):
    """Synthesize MIDI to MP3.

    Requires (Colab): !apt-get install -y fluidsynth && pip install midi2audio pydub
    Requires (Mac):   brew install fluidsynth ffmpeg && pip install midi2audio pydub
    """
    import os, tempfile
    from midi2audio import FluidSynth
    from pydub import AudioSegment

    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    try:
        FluidSynth().midi_to_audio(midi_path, wav_path)
        AudioSegment.from_wav(wav_path).export(output_path, format="mp3")
    finally:
        os.unlink(wav_path)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None

def load_model(model_id: str = MODEL_ID):
    global _model, _tokenizer
    if _model is None:
        print(f"Loading {model_id} ...")
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("Model loaded.")
    return _model, _tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate(
    key: str = "C",
    mode: str = "major",
    style: str = "jazz",
    num_bars: int = 8,
    output_midi_path: Optional[str] = "output.mid",
    output_mp3_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
    model_id: str = MODEL_ID,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    quiet: bool = False,
) -> GenerationResult:
    """
    Generate and validate a chord progression using Qwen3.5-2B.

    Returns a GenerationResult with validation details and reward score.
    If valid (reward > 0.5 and correct length), writes MIDI and optionally MP3/report.

    Set quiet=True to suppress printing and auto-playback (useful for batch runs).
    Pass output_midi_path=None to skip writing MIDI even when valid.
    """
    model, tokenizer = load_model(model_id)

    # Build prompt
    prompt = build_prompt(key, mode, style, num_bars)
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # disable <think> mode
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )

    # Decode only the new tokens
    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Parse
    chords = parse_chord_list(raw_output)

    if chords is None:
        return GenerationResult(
            prompt=prompt,
            raw_output=raw_output,
            chords=None,
            valid=False,
            validation_errors=["Could not parse a chord list from output"],
            reward=0.0,
            midi_path=None,
        )

    # Validate
    errors, reward, breakdown = validate(chords, key, mode, num_bars, style)
    valid = reward > 0.5 and len(chords) == num_bars

    # Render MIDI only if valid
    midi_path = None
    mp3_path = None
    if valid and output_midi_path:
        render_midi(chords, output_midi_path)
        midi_path = output_midi_path
        if not quiet:
            print(f"✓ Valid progression! MIDI saved to: {output_midi_path}")
        if output_mp3_path:
            render_mp3(output_midi_path, output_mp3_path)
            mp3_path = output_mp3_path
            if not quiet:
                print(f"✓ MP3 saved to: {output_mp3_path}")
    elif not quiet:
        if not valid:
            print(f"✗ Invalid progression (reward={reward}). No MIDI generated.")

    result = GenerationResult(
        prompt=prompt,
        raw_output=raw_output,
        chords=chords,
        valid=valid,
        validation_errors=errors,
        reward=reward,
        midi_path=midi_path,
        mp3_path=mp3_path,
        reward_breakdown=breakdown,
    )

    if not quiet and result.mp3_path:
        result.play()

    report = result.format_report()
    if not quiet:
        print(report)
    if output_report_path:
        with open(output_report_path, "w") as f:
            f.write(report)
        result.report_path = output_report_path

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(
    n: int,
    key: str = "C",
    mode: str = "major",
    style: str = "jazz",
    num_bars: int = 8,
    model_id: str = MODEL_ID,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    save_best_midi: Optional[str] = None,
    save_best_mp3: Optional[str] = None,
    verbose: bool = True,
) -> BatchResult:
    """
    Generate `n` chord progressions with identical settings and aggregate scores.

    By default, no per-run artifacts are written. Pass `save_best_midi` /
    `save_best_mp3` to render only the highest-reward run after the loop.
    """
    results: list[GenerationResult] = []
    for i in range(n):
        if verbose:
            print(f"[{i+1}/{n}] generating ...", end=" ", flush=True)
        r = generate(
            key=key,
            mode=mode,
            style=style,
            num_bars=num_bars,
            output_midi_path=None,
            output_mp3_path=None,
            output_report_path=None,
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            quiet=True,
        )
        results.append(r)
        if verbose:
            mark = "✓" if r.valid else "✗"
            print(f"{mark} reward={r.reward:.3f}  {r.chords}")

    rewards = [r.reward for r in results]
    pass_rate = sum(r.valid for r in results) / n
    mean_reward = statistics.fmean(rewards)
    std_reward = statistics.pstdev(rewards) if n > 1 else 0.0
    median_reward = statistics.median(rewards)

    breakdowns = [r.reward_breakdown for r in results if r.reward_breakdown]
    if breakdowns:
        mean_breakdown = {
            k: round(statistics.fmean(b[k] for b in breakdowns), 3)
            for k in ("key", "style", "voice")
        }
    else:
        mean_breakdown = {"key": 0.0, "style": 0.0, "voice": 0.0}

    batch = BatchResult(
        results=results,
        n=n,
        pass_rate=pass_rate,
        mean_reward=round(mean_reward, 3),
        std_reward=round(std_reward, 3),
        median_reward=round(median_reward, 3),
        min_reward=min(rewards),
        max_reward=max(rewards),
        mean_breakdown=mean_breakdown,
    )

    if verbose:
        print(batch.format_report())

    # Optionally render the best run
    if save_best_midi:
        best = batch.best()
        if best.chords and best.valid:
            render_midi(best.chords, save_best_midi)
            best.midi_path = save_best_midi
            if verbose:
                print(f"\n★ Best run (reward={best.reward}) → {save_best_midi}")
            if save_best_mp3:
                render_mp3(save_best_midi, save_best_mp3)
                best.mp3_path = save_best_mp3
                if verbose:
                    print(f"★ Best MP3 → {save_best_mp3}")
                    best.play()
        elif verbose:
            print("\n(no valid run to save)")

    return batch


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline chord progression generator")
    parser.add_argument("--key",      default="C",        help="Root key (e.g. C, F, Bb)")
    parser.add_argument("--mode",     default="major",    choices=["major", "minor"])
    parser.add_argument("--style",    default="jazz",     choices=list(STYLE_DESCRIPTIONS))
    parser.add_argument("--bars",     default=8,          type=int)
    parser.add_argument("--output",   default="output.mid")
    parser.add_argument("--mp3",      default=None,         help="Path for MP3 output (requires fluidsynth, pydub)")
    parser.add_argument("--report",   default=None,         help="Path to save text report")
    parser.add_argument("--model",    default=MODEL_ID)
    parser.add_argument("--temp",     default=0.7,        type=float)
    args = parser.parse_args()

    result = generate(
        key=args.key,
        mode=args.mode,
        style=args.style,
        num_bars=args.bars,
        output_midi_path=args.output,
        output_mp3_path=args.mp3,
        output_report_path=args.report,
        model_id=args.model,
        temperature=args.temp,
    )

    print("\n── Result ──────────────────────────────")
    print(f"Chords:  {result.chords}")
    print(f"Valid:   {result.valid}")
    print(f"Reward:  {result.reward}")
    if result.validation_errors:
        print(f"Errors:  {result.validation_errors}")
    if result.midi_path:
        print(f"MIDI:    {result.midi_path}")
