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

    Simple mode (Phase 1 baseline):
        build_prompt(key, mode, style, num_bars) -> str
        parse_chord_list(raw) -> list[str] | None
        validate(chords, key, mode, num_bars, style) -> (errors, reward, breakdown)

    Sectional mode (RLVR, Phase 2):
        build_prompt_sectional(style, sections) -> str
        parse_sectional_progression(raw) -> dict | None
        validate_sectional(raw, requested_sections) -> (errors, reward, breakdown)
"""

import json
import re
from typing import Optional

from music21 import harmony, scale as m21scale

from data.notation import to_chordonomicon


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

STYLE_DESCRIPTIONS = {
    "jazz":       "Use 7th chords, 9th chords, and ii-V-I progressions. Aim for harmonic complexity.",
    "pop":        "Use mostly I, IV, V, vi chords. Keep it simple and singable.",
    "blues":      "Use dominant 7th chords on I, IV, and V. Follow 12-bar blues conventions.",
    "folk":       "Use simple triads — I, IV, V, and maybe ii or vi. Keep it diatonic.",
    "bossa":      "Similar to jazz but favour maj7, min7, dom7 chords with smooth voice leading.",
    "rock":       "Riff-driven, power-chord-friendly, often I-bVII-IV motion.",
    "country":    "Mostly diatonic triads, I-IV-V dominant, occasional sus4.",
    "soul":       "Extended chords, secondary dominants, smooth voice leading.",
    "electronic": "Loopable, repetitive, often modal or pedal-tone-based.",
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


# --- Sectional Mode (RLVR) ---

def build_prompt_sectional(style: str, sections: list[str]) -> str:
    """Build a prompt for sectional chord-progression generation.

    Args:
        style: Genre/style name (e.g. 'rock', 'pop').
        sections: Ordered list of section names without index suffixes
                  (e.g. ['intro', 'verse', 'chorus', 'verse', 'chorus', 'outro']).

    Returns:
        A prompt string with XML-tag format instructions.
    """
    style_hint = STYLE_DESCRIPTIONS.get(style, "")
    section_list_csv = ", ".join(sections)
    return (
        f"Generate a song-form chord progression.\n"
        f"Style: {style}. {style_hint}\n"
        f"Structure: {section_list_csv}.\n"
        f"\n"
        f"Rules:\n"
        f"- Output the progression using XML-like section tags.\n"
        f"- Format: <intro_1> C G Am F <verse_1> C F G C ...\n"
        f"- Each section must contain 2 to 16 chords.\n"
        f"- Use chord symbols where 's' is sharp and 'b' is flat (e.g., Fs7, Bb, Cmin7/Fs).\n"
        f"- Sections must appear in the order listed above.\n"
        f"- Do not include any explanation, just the tagged chord sequence.\n"
        f"/no_think"
    )


def parse_sectional_progression(raw: str) -> dict | None:
    """Parse sectional chord output into a structured dict.

    Strips ``<think>...</think>`` blocks, then finds ``<name_index>`` tags
    followed by chord tokens. Each chord token is normalised to Chordonomicon
    notation via :func:`data.notation.to_chordonomicon`.

    Returns:
        ``{"sections": [(name, index, [chords]), ...]}`` in parse order,
        or ``None`` if no valid sections are found.
    """
    # Strip think blocks (non-greedy, DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Find sections: <name_index> followed by chord tokens
    pattern = re.compile(r"<(\w+?)_(\d+)>\s*([^<]+)", re.DOTALL)
    sections = []
    for match in pattern.finditer(cleaned):
        name = match.group(1).lower()
        index = int(match.group(2))
        raw_chords = match.group(3).split()
        chords = [to_chordonomicon(token) for token in raw_chords if token.strip()]
        sections.append((name, index, chords))

    # Return None if nothing was found or all chord lists are empty
    if not sections or all(len(chords) == 0 for _, _, chords in sections):
        return None

    return {"sections": sections}


def _lcs(a: list, b: list) -> int:
    """Return the length of the longest common subsequence of two lists.

    Both arguments should be lists of strings (e.g. section names).
    """
    m, n = len(a), len(b)
    # dp[i][j] = LCS length for a[:i], b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _section_max_jaccard(parsed_sections: list) -> float:
    """Compute the maximum pairwise Jaccard similarity between distinct section types.

    Each distinct section *name* (e.g. 'verse', 'chorus') is collapsed into the
    union of its chord sets across all instances.  The Jaccard index is then
    computed for every pair of distinct names, and the maximum is returned.

    Returns:
        Max Jaccard in [0.0, 1.0].  Returns 0.0 if fewer than 2 distinct names.
    """
    # Merge chord sets per section name
    chord_sets: dict[str, set] = {}
    for name, _, chords in parsed_sections:
        chord_sets.setdefault(name, set()).update(chords)

    names = list(chord_sets.keys())
    if len(names) < 2:
        return 0.0

    max_jac = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = chord_sets[names[i]]
            b = chord_sets[names[j]]
            union = a | b
            if not union:
                jac = 0.0
            else:
                jac = len(a & b) / len(union)
            if jac > max_jac:
                max_jac = jac

    return max_jac


def validate_sectional(
    raw: str,
    requested_sections: list[str],
) -> tuple[list[str], float, dict]:
    """Validate sectional chord output and compute an RLVR reward.

    Args:
        raw: Raw model output (may contain ``<think>`` blocks).
        requested_sections: Ordered list of section names as requested in the
                            prompt (e.g. ``['intro', 'verse', 'chorus']``).

    Returns:
        A tuple ``(errors, reward, breakdown)`` where:

        * ``errors`` is a list of human-readable error strings (may be empty).
        * ``reward`` is a float in ``[0.0, 1.0]``.
        * ``breakdown`` is a dict with keys ``presence``, ``order``,
          ``compliance``, ``max_jaccard``, and ``gamed``.
    """
    # 1. Parse
    parsed = parse_sectional_progression(raw)
    if parsed is None:
        return (
            ["unparseable: no valid section tags"],
            0.0,
            {
                "presence": 0.0,
                "order": 0.0,
                "compliance": 0.0,
                "max_jaccard": 0.0,
                "gamed": False,
            },
        )

    # 2. Presence and order rewards
    requested_names_seq = [s.lower() for s in requested_sections]
    output_names_seq = [name for (name, _, _) in parsed["sections"]]

    requested_set = set(requested_names_seq)
    output_set = set(output_names_seq)

    r_presence = len(requested_set & output_set) / len(requested_set)
    r_order = _lcs(requested_names_seq, output_names_seq) / len(requested_names_seq)
    r_compliance = 0.5 * r_presence + 0.5 * r_order

    # 3. Gaming detection via cross-section Jaccard
    max_jac = _section_max_jaccard(parsed["sections"])
    gamed = max_jac > 0.9
    gaming_penalty = 1.0 if gamed else 0.0

    # 4. Final reward
    reward = r_compliance * (1 - gaming_penalty)

    # 5. Error list
    errors: list[str] = []
    if not requested_set.issubset(output_set):
        missing = requested_set - output_set
        errors.append(f"missing sections: {sorted(missing)}")
    if r_order < 1.0:
        errors.append("section order does not match request")
    if gamed:
        errors.append(f"sections too similar (max Jaccard = {max_jac:.2f})")

    return (
        errors,
        reward,
        {
            "presence": r_presence,
            "order": r_order,
            "compliance": r_compliance,
            "max_jaccard": max_jac,
            "gamed": gamed,
        },
    )
