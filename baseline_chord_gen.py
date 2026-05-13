"""
baseline_chord_gen.py
---------------------
Sectional chord progression generator using Qwen/Qwen3.5-0.5B.
Validates output against structural compliance reward (validate_sectional).

Usage (Colab):
    from baseline_chord_gen import generate_sectional

    result = generate_sectional(
        style="jazz",
        sections=["intro", "verse", "chorus", "verse", "chorus", "outro"],
        output_mp3_path="output.mp3",
    )
    result.play()
"""

from dataclasses import dataclass
from typing import Optional

# ── music libs ───────────────────────────────────────────────────────────────
from music21 import harmony
from midiutil import MIDIFile

# ── reward primitives (model-free, also used by RL trainer) ──────────────────
from chord_rewards import (
    build_prompt_sectional,
    parse_sectional_progression,
    validate_sectional,
)

# ── notation adapter ──────────────────────────────────────────────────────────
from data.notation import to_music21


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3.5-0.5B"

MIDI_TEMPO    = 120   # BPM
BEATS_PER_BAR = 4
VELOCITY      = 75


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResultSectional:
    prompt: str
    raw_output: str
    requested_sections: list[str]
    parsed_sections: list | None       # list of (name, idx, [chords]) tuples, or None if parse failed
    valid: bool
    reward: float
    errors: list[str]
    breakdown: dict
    midi_path: Optional[str] = None
    mp3_path: Optional[str] = None
    report_path: Optional[str] = None

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
            "── Sectional Chord Progression Report ────────────────",
            f"Sections: {self.requested_sections}",
            f"Valid    : {self.valid}",
            f"Reward   : {self.reward:.3f}",
        ]
        if self.errors:
            lines.append(f"Errors   : {self.errors}")
        if self.midi_path:
            lines.append(f"MIDI     : {self.midi_path}")
        if self.mp3_path:
            lines.append(f"MP3      : {self.mp3_path}")
        lines.append("── Parsed Sections ──────────────────────────────────")
        if self.parsed_sections:
            for name, idx, chords in self.parsed_sections:
                lines.append(f"  {name}_{idx}: {chords}")
        else:
            lines.append("  <unparsed>")
        lines += [
            "── Prompt ───────────────────────────────────────────",
            self.prompt,
            "── Raw Output ───────────────────────────────────────",
            self.raw_output,
            "─────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MIDI / MP3 rendering
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
# Model loading (cached across calls)
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None


def load_model(model_id: str = MODEL_ID):
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
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
# Sectional helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_sections(parsed_sections: list) -> list[str]:
    """Flatten [(name, idx, chords), ...] into a single ordered chord list."""
    return [c for (_, _, chords) in parsed_sections for c in chords]


def _render_sectional_audio(
    parsed_sections: list,
    output_midi_path: str,
    output_mp3_path: Optional[str],
    quiet: bool,
) -> tuple[Optional[str], Optional[str]]:
    """Flatten sections, convert notation, write MIDI + optionally MP3."""
    flat_chords = _flatten_sections(parsed_sections)
    music21_chords = [to_music21(chord) for chord in flat_chords]
    render_midi(music21_chords, output_midi_path)
    if not quiet:
        print(f"✓ Valid progression! MIDI saved to: {output_midi_path}")

    mp3_path = None
    if output_mp3_path:
        render_mp3(output_midi_path, output_mp3_path)
        mp3_path = output_mp3_path
        if not quiet:
            print(f"✓ MP3 saved to: {output_mp3_path}")
    return output_midi_path, mp3_path


# ─────────────────────────────────────────────────────────────────────────────
# Main generation function
# ─────────────────────────────────────────────────────────────────────────────

def generate_sectional(
    style: str,
    sections: list[str],
    output_midi_path: Optional[str] = None,
    output_mp3_path: Optional[str] = None,
    output_report_path: Optional[str] = None,
    model_id: str = MODEL_ID,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    quiet: bool = False,
) -> GenerationResultSectional:
    """
    Generate and validate a sectional chord progression using Qwen/Qwen3.5-0.5B.

    Returns a GenerationResultSectional with validation details and reward score.
    If valid (reward > 0.5 and no errors), writes MIDI and optionally MP3/report.

    Set quiet=True to suppress printing and auto-playback (useful for batch runs).
    Pass output_midi_path=None to skip writing MIDI even when valid.
    """
    model, tokenizer = load_model(model_id)

    prompt = build_prompt_sectional(style, sections)
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # disable <think> mode
    )
    import torch
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )

    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    errors, reward, breakdown = validate_sectional(raw_output, sections)
    parsed_sections = parse_sectional_progression(raw_output)
    valid = reward > 0.5 and len(errors) == 0

    midi_path = None
    mp3_path = None
    if valid and parsed_sections and output_midi_path:
        midi_path, mp3_path = _render_sectional_audio(
            parsed_sections["sections"], output_midi_path, output_mp3_path, quiet
        )
    elif not quiet and not valid:
        print(f"✗ Invalid progression (reward={reward:.3f}). No MIDI generated.")

    result = GenerationResultSectional(
        prompt=prompt,
        raw_output=raw_output,
        requested_sections=sections,
        parsed_sections=parsed_sections["sections"] if parsed_sections else None,
        valid=valid,
        reward=reward,
        errors=errors,
        breakdown=breakdown,
        midi_path=midi_path,
        mp3_path=mp3_path,
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
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sectional chord progression generator")
    parser.add_argument("--style", default="jazz",
                        choices=["jazz", "pop", "rock", "country", "soul", "blues"])
    parser.add_argument("--sections", nargs="+",
                        default=["intro", "verse", "chorus", "verse", "chorus", "outro"],
                        metavar="SECTION")
    parser.add_argument("--output-midi", default=None)
    parser.add_argument("--output-mp3",  default=None)
    parser.add_argument("--report",      default=None)
    parser.add_argument("--model",       default=MODEL_ID)
    parser.add_argument("--temp",        default=0.7, type=float)
    args = parser.parse_args()

    result = generate_sectional(
        style=args.style,
        sections=args.sections,
        output_midi_path=args.output_midi,
        output_mp3_path=args.output_mp3,
        output_report_path=args.report,
        model_id=args.model,
        temperature=args.temp,
    )

    print(f"\nReward: {result.reward:.3f}  Valid: {result.valid}")
    if result.errors:
        print(f"Errors: {result.errors}")
