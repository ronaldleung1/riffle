"""Tests for the Sectional Mode (RLVR) extensions in chord_rewards.py."""

import pytest

from chord_rewards import (
    _lcs,
    _section_max_jaccard,
    parse_sectional_progression,
    validate_sectional,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_raw(*section_defs):
    """Build a minimal raw model output string from (name, index, chords) tuples.

    Example: _make_raw(("verse", 1, ["C", "G"]), ("chorus", 1, ["F", "G"]))
    """
    parts = []
    for name, idx, chords in section_defs:
        parts.append(f"<{name}_{idx}> {' '.join(chords)}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 1. Perfect output — all requested sections present, distinct content
# ---------------------------------------------------------------------------

class TestPerfectOutput:
    def test_reward_is_one(self):
        # Use four clearly distinct chord sets so Jaccard stays well below 0.9
        raw = _make_raw(
            ("intro",  1, ["C",  "G",  "Am", "F"]),
            ("verse",  1, ["D",  "Bm", "Em", "A"]),
            ("chorus", 1, ["Eb", "Bb", "Gm", "Cm"]),
            ("outro",  1, ["Fs", "Bs", "Ds", "Gs"]),
        )
        requested = ["intro", "verse", "chorus", "outro"]
        errors, reward, breakdown = validate_sectional(raw, requested)

        assert reward == pytest.approx(1.0)
        assert errors == []
        assert breakdown["presence"] == pytest.approx(1.0)
        assert breakdown["order"] == pytest.approx(1.0)
        assert breakdown["gamed"] is False


# ---------------------------------------------------------------------------
# 2. Missing one section — bridge absent
# ---------------------------------------------------------------------------

class TestMissingSection:
    def test_missing_bridge_lowers_presence(self):
        # Output has intro, verse, chorus — no bridge
        raw = _make_raw(
            ("intro",  1, ["C", "G"]),
            ("verse",  1, ["Am", "F"]),
            ("chorus", 1, ["F", "G"]),
        )
        requested = ["intro", "verse", "chorus", "bridge"]
        errors, reward, breakdown = validate_sectional(raw, requested)

        assert breakdown["presence"] < 1.0
        assert any("missing sections" in e for e in errors)
        assert "bridge" in errors[0]

    def test_reward_is_less_than_one(self):
        raw = _make_raw(
            ("intro",  1, ["C", "G"]),
            ("verse",  1, ["Am", "F"]),
            ("chorus", 1, ["F", "G"]),
        )
        requested = ["intro", "verse", "chorus", "bridge"]
        _, reward, _ = validate_sectional(raw, requested)
        assert reward < 1.0


# ---------------------------------------------------------------------------
# 3. Wrong order — sections present but reordered
# ---------------------------------------------------------------------------

class TestWrongOrder:
    def test_order_error_raised(self):
        # Request: intro, verse, chorus — output gives: intro, chorus, verse
        raw = _make_raw(
            ("intro",  1, ["C", "G"]),
            ("chorus", 1, ["F", "G"]),
            ("verse",  1, ["Am", "F"]),
        )
        requested = ["intro", "verse", "chorus"]
        errors, reward, breakdown = validate_sectional(raw, requested)

        assert breakdown["order"] < 1.0
        assert any("section order" in e for e in errors)

    def test_presence_still_full_when_all_present(self):
        raw = _make_raw(
            ("intro",  1, ["C", "G"]),
            ("chorus", 1, ["F", "G"]),
            ("verse",  1, ["Am", "F"]),
        )
        requested = ["intro", "verse", "chorus"]
        _, _, breakdown = validate_sectional(raw, requested)
        assert breakdown["presence"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Gamed — every section has identical chords → max_jaccard > 0.9, reward 0
# ---------------------------------------------------------------------------

class TestGamedOutput:
    SAME_CHORDS = ["C", "G", "Am", "F"]

    def test_reward_is_zero(self):
        raw = _make_raw(
            ("intro",  1, self.SAME_CHORDS),
            ("verse",  1, self.SAME_CHORDS),
            ("chorus", 1, self.SAME_CHORDS),
        )
        requested = ["intro", "verse", "chorus"]
        _, reward, breakdown = validate_sectional(raw, requested)

        assert reward == pytest.approx(0.0)
        assert breakdown["gamed"] is True
        assert breakdown["max_jaccard"] == pytest.approx(1.0)

    def test_gamed_error_message(self):
        raw = _make_raw(
            ("intro",  1, self.SAME_CHORDS),
            ("verse",  1, self.SAME_CHORDS),
            ("chorus", 1, self.SAME_CHORDS),
        )
        requested = ["intro", "verse", "chorus"]
        errors, _, _ = validate_sectional(raw, requested)
        assert any("too similar" in e for e in errors)


# ---------------------------------------------------------------------------
# 5. Unparseable — no section tags at all
# ---------------------------------------------------------------------------

class TestUnparseable:
    def test_no_tags_returns_zero_reward(self):
        raw = "C G Am F G C F G"
        errors, reward, breakdown = validate_sectional(raw, ["verse", "chorus"])

        assert reward == pytest.approx(0.0)
        assert any("unparseable" in e for e in errors)

    def test_breakdown_all_zero(self):
        raw = "no tags here at all"
        _, _, breakdown = validate_sectional(raw, ["verse"])

        assert breakdown["presence"] == pytest.approx(0.0)
        assert breakdown["order"] == pytest.approx(0.0)
        assert breakdown["gamed"] is False


# ---------------------------------------------------------------------------
# 6. <think> stripping — reasoning block wraps valid tags
# ---------------------------------------------------------------------------

class TestThinkStripping:
    def test_think_block_stripped_before_parsing(self):
        # verse and chorus use completely different chord sets to avoid gaming penalty
        raw = (
            "<think>I will write the progression now. "
            "Let me plan: intro=C, verse=Am...</think>"
            "<verse_1> C G Am F "
            "<chorus_1> D Bm Em A"
        )
        requested = ["verse", "chorus"]
        errors, reward, breakdown = validate_sectional(raw, requested)

        assert reward == pytest.approx(1.0)
        assert errors == []

    def test_parsed_sections_correct_after_stripping(self):
        raw = "<think>some reasoning</think><verse_1> C G <chorus_1> F Am"
        result = parse_sectional_progression(raw)
        assert result is not None
        names = [name for name, _, _ in result["sections"]]
        assert names == ["verse", "chorus"]


# ---------------------------------------------------------------------------
# 7. Notation normalisation — F#m7 stored as Fsm7
# ---------------------------------------------------------------------------

class TestNotationNormalisation:
    def test_sharp_normalised_to_chordonomicon(self):
        # Model emits F#m7; should be stored as Fsm7
        raw = "<verse_1> F#m7 C G Am"
        result = parse_sectional_progression(raw)
        assert result is not None
        _, _, chords = result["sections"][0]
        assert "Fsm7" in chords
        assert "F#m7" not in chords

    def test_flat_normalised_to_chordonomicon(self):
        # Model emits B-7 (music21 flat notation); should become Bb7
        raw = "<verse_1> B-7 C G Am"
        result = parse_sectional_progression(raw)
        assert result is not None
        _, _, chords = result["sections"][0]
        assert "Bb7" in chords
        assert "B-7" not in chords


# ---------------------------------------------------------------------------
# 8. Parser returns None for empty section lists
# ---------------------------------------------------------------------------

class TestEmptySectionList:
    def test_tag_with_no_chords_returns_none(self):
        # Tag present but nothing follows (or only whitespace before next tag)
        raw = "<verse_1> <chorus_1>"
        result = parse_sectional_progression(raw)
        assert result is None

    def test_completely_empty_raw_returns_none(self):
        result = parse_sectional_progression("")
        assert result is None

    def test_only_think_block_returns_none(self):
        raw = "<think>just thinking, no actual output</think>"
        result = parse_sectional_progression(raw)
        assert result is None


# ---------------------------------------------------------------------------
# Unit test: _lcs
# ---------------------------------------------------------------------------

class TestLCS:
    def test_basic_subsequence(self):
        assert _lcs(["a", "b", "c"], ["a", "c"]) == 2

    def test_identical_lists(self):
        assert _lcs(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_no_common(self):
        assert _lcs(["a", "b"], ["c", "d"]) == 0

    def test_empty_lists(self):
        assert _lcs([], []) == 0

    def test_one_empty(self):
        assert _lcs(["a", "b"], []) == 0

    def test_section_names(self):
        requested = ["intro", "verse", "chorus", "bridge", "outro"]
        output = ["intro", "verse", "bridge", "outro"]
        # LCS: intro, verse, bridge, outro = 4
        assert _lcs(requested, output) == 4


# ---------------------------------------------------------------------------
# Unit test: _section_max_jaccard
# ---------------------------------------------------------------------------

class TestSectionMaxJaccard:
    def test_identical_sections_returns_one(self):
        # verse and chorus have exact same chord set
        sections = [
            ("verse",  1, ["C", "G", "Am", "F"]),
            ("chorus", 1, ["C", "G", "Am", "F"]),
        ]
        assert _section_max_jaccard(sections) == pytest.approx(1.0)

    def test_disjoint_sections_returns_zero(self):
        sections = [
            ("verse",  1, ["C", "G"]),
            ("chorus", 1, ["Am", "F"]),
        ]
        assert _section_max_jaccard(sections) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # A={C,G,Am}, B={C,G,F} → intersection={C,G}, union={C,G,Am,F} → 2/4=0.5
        sections = [
            ("verse",  1, ["C", "G", "Am"]),
            ("chorus", 1, ["C", "G", "F"]),
        ]
        jac = _section_max_jaccard(sections)
        assert jac == pytest.approx(0.5)

    def test_single_section_name_returns_zero(self):
        # Only one distinct name — no pairs to compare
        sections = [
            ("verse", 1, ["C", "G"]),
            ("verse", 2, ["Am", "F"]),
        ]
        assert _section_max_jaccard(sections) == pytest.approx(0.0)

    def test_merges_same_name_instances(self):
        # verse_1={C,G}, verse_2={C,F} → merged verse={C,G,F}
        # chorus_1={C,G,F}
        # Jaccard = |{C,G,F} ∩ {C,G,F}| / |{C,G,F} ∪ {C,G,F}| = 3/3 = 1.0
        sections = [
            ("verse",  1, ["C", "G"]),
            ("verse",  2, ["C", "F"]),
            ("chorus", 1, ["C", "G", "F"]),
        ]
        jac = _section_max_jaccard(sections)
        assert jac == pytest.approx(1.0)
