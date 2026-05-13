"""Regex adapters between Chordonomicon and music21 chord-symbol notations.

Chordonomicon notation (canonical): uses 's' for sharp and 'b' for flat in root positions.
  Examples: Fs7, Bb, Cmin7/Fs, Asm7b5

music21 notation: uses '#' for sharp and '-' for flat.
  Examples: F#7, B-, Cm7/F#, A#m7b5
"""

import re


def to_music21(chord: str) -> str:
    """Convert a Chordonomicon-notation chord symbol to music21 notation.

    Conversion rules (applied in order):
    1. Root sharp: [A-G]s (at start or after /) becomes [A-G]#
    2. Root flat: [A-G]b (at start or after /) becomes [A-G]-
       (preserves 'b' in modifiers like b9, b5, b13)
    3. min -> m (case-sensitive)
    """
    # Rule 1: Root sharp - replace [A-G]s with [A-G]# at root or after /
    # Match: (start or /) + note + 's' NOT followed by 'us' (to avoid matching 'sus')
    def replace_root_sharp(match):
        prefix = match.group(1) or ''  # empty string at start, or '/'
        note = match.group(2)  # A-G
        return prefix + note + '#'

    chord = re.sub(r'(^|/)([A-G])s(?!us)', replace_root_sharp, chord)

    # Rule 2: Root flat - replace [A-G]b with [A-G]- at root or after /
    # The regex ([A-G])b naturally only matches when 'b' is directly after a note letter,
    # so modifiers like b9, b5, b13 are not matched (they have non-note letters before the 'b')
    def replace_root_flat(match):
        prefix = match.group(1) or ''  # empty string at start, or '/'
        note = match.group(2)  # A-G
        return prefix + note + '-'

    chord = re.sub(r'(^|/)([A-G])b', replace_root_flat, chord)

    # Rule 3: min -> m
    chord = re.sub(r'min', 'm', chord)

    return chord


def to_chordonomicon(chord: str) -> str:
    """Convert a music21-notation chord symbol to Chordonomicon notation.

    Conversion rules (applied in order):
    1. Root sharp: [A-G]# becomes [A-G]s (at root or after /)
    2. Root flat: [A-G]- becomes [A-G]b (at root or after /)
    3. 'm' is left as-is (already in Chordonomicon format)
    """
    # Rule 1: Root sharp - replace [A-G]# with [A-G]s at root or after /
    def replace_root_sharp(match):
        prefix = match.group(1) or ''  # empty string at start, or '/'
        note = match.group(2)  # A-G
        return prefix + note + 's'

    chord = re.sub(r'(^|/)([A-G])#', replace_root_sharp, chord)

    # Rule 2: Root flat - replace [A-G]- with [A-G]b at root or after /
    def replace_root_flat(match):
        prefix = match.group(1) or ''  # empty string at start, or '/'
        note = match.group(2)  # A-G
        return prefix + note + 'b'

    chord = re.sub(r'(^|/)([A-G])-', replace_root_flat, chord)

    return chord
