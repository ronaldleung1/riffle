"""
chord_rewards.py
----------------
Pure (model-free) primitives for chord-progression generation:
prompt building, output parsing, and verifiable rewards.

This module has no transformers/torch dependency — only `music21` —
so it can be imported cheaply from inference, batch eval, and RL
training loops alike.

Public API:
    STYLE_DESCRIPTIONS, CHORD_ALIASES
    build_prompt(key, mode, style, num_bars) -> str
    parse_chord_list(raw) -> list[str] | None
    validate(chords, key, mode, num_bars, style) -> (errors, reward, breakdown)
"""

import json
import re
from typing import Optional

from music21 import harmony, scale as m21scale


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

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
# Validation primitives
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


def _style_score(parsed: list, style: str) -> float:
    """Heuristic style match score [0,1]."""
    if style in ("jazz", "bossa"):
        # Reward 7th+ chords
        extended = sum(
            1 for c in parsed
            if any(q in c.commonName for q in ["seventh", "ninth", "eleventh"])
        )
        return min(extended / max(len(parsed) * 0.6, 1), 1.0)

    if style == "blues":
        # Reward dominant 7ths
        dom7 = sum(
            1 for c in parsed
            if "dominant" in c.commonName and "seventh" in c.commonName
        )
        return min(dom7 / max(len(parsed) * 0.5, 1), 1.0)

    if style in ("pop", "folk"):
        # Reward simple triads / no extensions
        simple = sum(1 for c in parsed if len(c.pitches) <= 4)
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
    avg = sum(intervals) / len(intervals)
    return round(1.0 - (avg / 6.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Verifiable reward
# ─────────────────────────────────────────────────────────────────────────────

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
    diatonic_count = sum(1 for c in parsed if c.root().name in scale_pitches)
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
    r_style = _style_score(parsed, style)

    # ── Soft reward 3: voice leading (weight 0.2) ────────────────────────────
    r_voice = _voice_leading_score(parsed)

    reward = (r_key * 0.5) + (r_style * 0.3) + (r_voice * 0.2)
    breakdown = {
        "key": round(r_key, 3),
        "style": round(r_style, 3),
        "voice": round(r_voice, 3),
    }
    return errors, round(reward, 3), breakdown
