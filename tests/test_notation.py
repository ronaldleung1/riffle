"""Tests for notation.py adapter between Chordonomicon and music21 notations."""

import pytest
from data.notation import to_music21, to_chordonomicon


class TestToMusic21:
    """Test conversion from Chordonomicon to music21 notation."""

    # Test natural chords (no accidentals)
    def test_natural_root(self):
        assert to_music21("C") == "C"

    def test_natural_minor(self):
        assert to_music21("Cm") == "Cm"

    def test_natural_maj7(self):
        assert to_music21("Cmaj7") == "Cmaj7"

    def test_cmin7_becomes_cm7(self):
        """Cmin7 loses 'min' form and becomes Cm7."""
        assert to_music21("Cmin7") == "Cm7"

    # Test sharp chords (root position)
    def test_sharp_root(self):
        assert to_music21("Fs") == "F#"

    def test_sharp_dominant7(self):
        assert to_music21("Fs7") == "F#7"

    def test_sharp_minor7(self):
        assert to_music21("Fsm7") == "F#m7"

    # Test flat chords (root position)
    def test_flat_root(self):
        assert to_music21("Bb") == "B-"

    def test_flat_dominant7(self):
        assert to_music21("Bb7") == "B-7"

    def test_flat_minor7(self):
        assert to_music21("Bbm7") == "B-m7"

    # Test root flat with modifiers (b modifier preserved)
    def test_flat_root_with_b9_modifier(self):
        """Root flat converted, b9 modifier preserved."""
        assert to_music21("Bb7b9") == "B-7b9"

    def test_flat_root_with_b5_modifier(self):
        """Root flat converted, b5 modifier preserved."""
        assert to_music21("Bbm7b5") == "B-m7b5"

    # Test chords without root accidentals but with modifiers
    def test_natural_root_with_b5_modifier(self):
        assert to_music21("Cm7b5") == "Cm7b5"

    def test_natural_root_with_b9_modifier(self):
        assert to_music21("C7b9") == "C7b9"

    # Test sharp with modifiers
    def test_sharp_root_with_b5_modifier(self):
        assert to_music21("Asm7b5") == "A#m7b5"

    def test_sharp_root_with_b9_modifier(self):
        assert to_music21("Fs7b9") == "F#7b9"

    # Test chords with slash bass
    def test_bass_note_natural(self):
        assert to_music21("Bb/D") == "B-/D"

    def test_bass_note_sharp(self):
        assert to_music21("Cm7/Fs") == "Cm7/F#"

    def test_bass_note_flat(self):
        assert to_music21("Bb/Fs") == "B-/F#"

    def test_complex_with_bass_flat_and_modifier(self):
        assert to_music21("Cmin7/Fs") == "Cm7/F#"

    # Test chords with sus
    def test_sus4_natural(self):
        assert to_music21("G7sus4") == "G7sus4"

    def test_sus2_natural(self):
        assert to_music21("Csus2") == "Csus2"


class TestToChordonomicon:
    """Test conversion from music21 to Chordonomicon notation."""

    # Test natural chords (no accidentals)
    def test_natural_root(self):
        assert to_chordonomicon("C") == "C"

    def test_natural_minor(self):
        assert to_chordonomicon("Cm") == "Cm"

    def test_natural_maj7(self):
        assert to_chordonomicon("Cmaj7") == "Cmaj7"

    def test_natural_min7(self):
        assert to_chordonomicon("Cm7") == "Cm7"

    # Test sharp chords (root position)
    def test_sharp_root(self):
        assert to_chordonomicon("F#") == "Fs"

    def test_sharp_dominant7(self):
        assert to_chordonomicon("F#7") == "Fs7"

    def test_sharp_minor7(self):
        assert to_chordonomicon("F#m7") == "Fsm7"

    # Test flat chords (root position)
    def test_flat_root(self):
        assert to_chordonomicon("B-") == "Bb"

    def test_flat_dominant7(self):
        assert to_chordonomicon("B-7") == "Bb7"

    def test_flat_minor7(self):
        assert to_chordonomicon("B-m7") == "Bbm7"

    # Test root flat with modifiers (b modifier preserved)
    def test_flat_root_with_b9_modifier(self):
        """Root flat converted, b9 modifier preserved."""
        assert to_chordonomicon("B-7b9") == "Bb7b9"

    def test_flat_root_with_b5_modifier(self):
        """Root flat converted, b5 modifier preserved."""
        assert to_chordonomicon("B-m7b5") == "Bbm7b5"

    # Test chords without root accidentals but with modifiers
    def test_natural_root_with_b5_modifier(self):
        assert to_chordonomicon("Cm7b5") == "Cm7b5"

    def test_natural_root_with_b9_modifier(self):
        assert to_chordonomicon("C7b9") == "C7b9"

    # Test sharp with modifiers
    def test_sharp_root_with_b5_modifier(self):
        assert to_chordonomicon("A#m7b5") == "Asm7b5"

    def test_sharp_root_with_b9_modifier(self):
        assert to_chordonomicon("F#7b9") == "Fs7b9"

    # Test chords with slash bass
    def test_bass_note_natural(self):
        assert to_chordonomicon("B-/D") == "Bb/D"

    def test_bass_note_sharp(self):
        assert to_chordonomicon("Cm7/F#") == "Cm7/Fs"

    def test_bass_note_flat(self):
        assert to_chordonomicon("B-/F#") == "Bb/Fs"

    def test_complex_with_bass_sharp(self):
        assert to_chordonomicon("Cm7/F#") == "Cm7/Fs"

    # Test chords with sus
    def test_sus4_natural(self):
        assert to_chordonomicon("G7sus4") == "G7sus4"

    def test_sus2_natural(self):
        assert to_chordonomicon("Csus2") == "Csus2"


class TestRoundTrip:
    """Test round-trip conversions (Chordonomicon -> music21 -> Chordonomicon)."""

    # Fixture of Chordonomicon chords that should round-trip
    round_trip_chords = [
        "C",
        "Cm",
        "Cmaj7",
        "Fs",
        "Fs7",
        "Fsm7",
        "Bb",
        "Bb7",
        "Bbm7",
        "Bb7b9",
        "Cm7b5",
        "Asm7b5",
        "Cm7/Fs",
        "Bb/D",
        "Bb/Fs",
        "G7sus4",
    ]

    @pytest.mark.parametrize("chord", round_trip_chords)
    def test_round_trip(self, chord):
        """Test that Chordonomicon -> music21 -> Chordonomicon preserves the chord."""
        music21_form = to_music21(chord)
        back_to_chordonomicon = to_chordonomicon(music21_form)
        assert back_to_chordonomicon == chord, (
            f"Round-trip failed for {chord}: "
            f"{chord} -> {music21_form} -> {back_to_chordonomicon}"
        )


class TestOneWayConversions:
    """Test one-way conversions that don't round-trip due to notation loss."""

    def test_cmin7_to_cm7_one_way(self):
        """Cmin7 converts to Cm7; reverse is also Cm7."""
        assert to_music21("Cmin7") == "Cm7"
        assert to_chordonomicon("Cm7") == "Cm7"
        # No round-trip: Cmin7 -> Cm7 -> Cm7 (loses 'min' form)

    def test_music21_to_chordonomicon_fsharp_example(self):
        """F#m7b5/A (music21) -> Fsm7b5/A (Chordonomicon) -> F#m7b5/A (back to music21)."""
        music21_input = "F#m7b5/A"
        chordonomicon_form = to_chordonomicon(music21_input)
        assert chordonomicon_form == "Fsm7b5/A"

        back_to_music21 = to_music21(chordonomicon_form)
        assert back_to_music21 == music21_input
