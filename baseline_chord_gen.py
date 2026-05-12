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

import itertools
import statistics
from dataclasses import dataclass
from typing import Optional

# ── music libs ───────────────────────────────────────────────────────────────
from music21 import harmony
from midiutil import MIDIFile

# ── model libs ───────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ── reward primitives (model-free, also used by RL trainer) ──────────────────
from chord_rewards import (
    STYLE_DESCRIPTIONS,
    build_prompt,
    parse_chord_list,
    validate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-2B"

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


@dataclass
class GridCell:
    """One (key, mode, style, num_bars) point in a grid sweep."""
    key: str
    mode: str
    style: str
    num_bars: int
    batch: BatchResult


@dataclass
class GridResult:
    cells: list[GridCell]
    n_cells: int
    samples_per_cell: int
    overall_pass_rate: float
    overall_mean_reward: float
    overall_mean_breakdown: dict

    def as_rows(self) -> list[dict]:
        """One row per cell, flat dict — easy to drop into pandas."""
        return [
            {
                "key":         c.key,
                "mode":        c.mode,
                "style":       c.style,
                "num_bars":    c.num_bars,
                "n":           c.batch.n,
                "pass_rate":   c.batch.pass_rate,
                "mean_reward": c.batch.mean_reward,
                "std_reward":  c.batch.std_reward,
                "min_reward":  c.batch.min_reward,
                "max_reward":  c.batch.max_reward,
                **{f"mean_{k}": v for k, v in c.batch.mean_breakdown.items()},
            }
            for c in self.cells
        ]

    def group_by(self, axis: str) -> dict[str, dict]:
        """Aggregate cells by a single axis (e.g. 'style' or 'key')."""
        groups: dict[str, list[BatchResult]] = {}
        for c in self.cells:
            groups.setdefault(getattr(c, axis), []).append(c.batch)
        out = {}
        for k, batches in groups.items():
            all_results = [r for b in batches for r in b.results]
            rewards = [r.reward for r in all_results]
            out[k] = {
                "n":           len(all_results),
                "pass_rate":   round(sum(r.valid for r in all_results) / len(all_results), 3),
                "mean_reward": round(statistics.fmean(rewards), 3),
                "std_reward":  round(statistics.pstdev(rewards) if len(rewards) > 1 else 0.0, 3),
            }
        return out

    def format_report(self) -> str:
        lines = [
            "═══ Grid Report ════════════════════════════════════════════════════",
            f"Cells: {self.n_cells}   samples/cell: {self.samples_per_cell}   "
            f"total runs: {self.n_cells * self.samples_per_cell}",
            f"Overall pass rate : {self.overall_pass_rate:.0%}",
            f"Overall mean rwd  : {self.overall_mean_reward:.3f}",
            f"Overall breakdown : key={self.overall_mean_breakdown['key']:.3f}  "
            f"style={self.overall_mean_breakdown['style']:.3f}  "
            f"voice={self.overall_mean_breakdown['voice']:.3f}",
            "── Per-cell ────────────────────────────────────────────────────────",
            f"  {'key':<4} {'mode':<6} {'style':<6} {'bars':>4}  "
            f"{'pass':>5}  {'mean':>6}  {'std':>6}  {'key':>5} {'sty':>5} {'voi':>5}",
        ]
        for c in self.cells:
            b = c.batch
            lines.append(
                f"  {c.key:<4} {c.mode:<6} {c.style:<6} {c.num_bars:>4}  "
                f"{b.pass_rate:>4.0%}   {b.mean_reward:>6.3f}  {b.std_reward:>6.3f}  "
                f"{b.mean_breakdown['key']:>5.3f} "
                f"{b.mean_breakdown['style']:>5.3f} "
                f"{b.mean_breakdown['voice']:>5.3f}"
            )
        lines.append("════════════════════════════════════════════════════════════════════")
        return "\n".join(lines)

    def best(self) -> GenerationResult:
        """Highest-reward run across the entire grid."""
        return max(
            (r for c in self.cells for r in c.batch.results),
            key=lambda r: r.reward,
        )


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
            dtype=torch.bfloat16,
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
# Grid sweep
# ─────────────────────────────────────────────────────────────────────────────

def generate_grid(
    keys: list[str] = ("C",),
    modes: list[str] = ("major",),
    styles: list[str] = ("jazz",),
    num_bars: list[int] = (8,),
    samples_per_cell: int = 5,
    model_id: str = MODEL_ID,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    verbose: bool = True,
) -> GridResult:
    """
    Sweep over the Cartesian product of (keys × modes × styles × num_bars),
    drawing `samples_per_cell` samples per combination. Returns a GridResult
    with per-cell BatchResults and overall aggregates.

    No per-run artifacts are written — this is for evaluation, not listening.
    Use batch.best() / grid.best() and render that one if you want audio.
    """
    cells: list[GridCell] = []
    combos = list(itertools.product(keys, modes, styles, num_bars))

    for i, (k, m, s, nb) in enumerate(combos, 1):
        if verbose:
            print(f"\n══ Cell {i}/{len(combos)}: key={k} mode={m} style={s} bars={nb} ══")
        batch = generate_batch(
            n=samples_per_cell,
            key=k, mode=m, style=s, num_bars=nb,
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            verbose=verbose,
        )
        cells.append(GridCell(key=k, mode=m, style=s, num_bars=nb, batch=batch))

    # Overall aggregates across every run in every cell
    all_results = [r for c in cells for r in c.batch.results]
    rewards = [r.reward for r in all_results]
    overall_pass_rate = round(sum(r.valid for r in all_results) / len(all_results), 3)
    overall_mean_reward = round(statistics.fmean(rewards), 3)

    breakdowns = [r.reward_breakdown for r in all_results if r.reward_breakdown]
    if breakdowns:
        overall_mean_breakdown = {
            k: round(statistics.fmean(b[k] for b in breakdowns), 3)
            for k in ("key", "style", "voice")
        }
    else:
        overall_mean_breakdown = {"key": 0.0, "style": 0.0, "voice": 0.0}

    grid = GridResult(
        cells=cells,
        n_cells=len(cells),
        samples_per_cell=samples_per_cell,
        overall_pass_rate=overall_pass_rate,
        overall_mean_reward=overall_mean_reward,
        overall_mean_breakdown=overall_mean_breakdown,
    )

    if verbose:
        print()
        print(grid.format_report())

    return grid


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
