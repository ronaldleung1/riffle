"""Unit tests for data/build_dataset.py — pipeline stages with mock rows.

Does NOT load the actual Chordonomicon dataset (264MB). Tests each stage
against synthetic input.
"""

import json

from data.build_dataset import (
    MAX_CHORDS_PER_SECTION,
    _row_to_example,
    emit_jsonl_row,
    is_parseable,
    map_genre,
    parse_song_sections,
    passes_filters,
    truncate_sections,
)


# ─────────────────────────────────────────────────────────────────────────────
# parse_song_sections
# ─────────────────────────────────────────────────────────────────────────────

class TestParseSongSections:
    def test_basic_parse(self):
        s = "<intro_1> C G Am F <verse_1> Am Dm G C"
        result = parse_song_sections(s)
        assert result == [
            ("intro", 1, ["C", "G", "Am", "F"]),
            ("verse", 1, ["Am", "Dm", "G", "C"]),
        ]

    def test_multiple_instances(self):
        s = "<verse_1> C G <verse_2> Am F <chorus_1> F C"
        result = parse_song_sections(s)
        assert [name for name, _, _ in result] == ["verse", "verse", "chorus"]
        assert [idx for _, idx, _ in result] == [1, 2, 1]

    def test_empty_string_returns_none(self):
        assert parse_song_sections("") is None

    def test_no_tags_returns_none(self):
        assert parse_song_sections("just some text C G Am F") is None

    def test_lowercases_section_names(self):
        s = "<Verse_1> C G"
        result = parse_song_sections(s)
        assert result == [("verse", 1, ["C", "G"])]


# ─────────────────────────────────────────────────────────────────────────────
# is_parseable
# ─────────────────────────────────────────────────────────────────────────────

class TestIsParseable:
    def test_valid_chord(self):
        assert is_parseable("Cmaj7") is True

    def test_chordonomicon_sharp(self):
        # to_music21 converts Fs7 → F#7 before parsing
        assert is_parseable("Fs7") is True

    def test_chordonomicon_flat(self):
        assert is_parseable("Bb") is True

    def test_gibberish_unparseable(self):
        assert is_parseable("XYZ123") is False


# ─────────────────────────────────────────────────────────────────────────────
# passes_filters
# ─────────────────────────────────────────────────────────────────────────────

class TestPassesFilters:
    def test_no_sections_rejected(self):
        ok, reason = passes_filters([])
        assert ok is False
        assert reason == "no_sections"

    def test_too_short(self):
        sections = [("intro", 1, ["C", "G", "Am"])]  # 3 chords < 8
        ok, reason = passes_filters(sections)
        assert ok is False
        assert reason == "too_short"

    def test_too_long(self):
        sections = [("verse", 1, ["C"] * 250)]  # 250 chords > 200
        ok, reason = passes_filters(sections)
        assert ok is False
        assert reason == "too_long"

    def test_low_diversity(self):
        # 10 chords, but all the same — < 3 distinct
        sections = [("verse", 1, ["C"] * 10)]
        ok, reason = passes_filters(sections)
        assert ok is False
        assert reason == "low_diversity"

    def test_unparseable_chord(self):
        sections = [("verse", 1, ["C", "G", "Am", "F", "C", "G", "Am", "XYZ123"])]
        ok, reason = passes_filters(sections)
        assert ok is False
        assert reason == "unparseable_chord"

    def test_valid_song(self):
        sections = [
            ("intro", 1, ["C", "G", "Am", "F"]),
            ("verse", 1, ["C", "G", "Am", "F"]),
        ]  # 8 chords, 4 distinct, all parseable
        ok, reason = passes_filters(sections)
        assert ok is True
        assert reason is None


# ─────────────────────────────────────────────────────────────────────────────
# map_genre
# ─────────────────────────────────────────────────────────────────────────────

class TestMapGenre:
    def test_main_genre_jazz(self):
        assert map_genre({"main_genre": "jazz"}) == "jazz"

    def test_main_genre_pop(self):
        assert map_genre({"main_genre": "pop"}) == "pop"

    def test_main_genre_rock(self):
        assert map_genre({"main_genre": "rock"}) == "rock"

    def test_blues_via_genres_tag(self):
        assert map_genre({"main_genre": "alternative", "genres": "delta blues, slide"}) == "blues"

    def test_blues_via_genres_list(self):
        assert map_genre({"main_genre": "rock", "genres": ["blues rock", "indie"]}) == "rock"
        # main_genre wins over blues detection

    def test_unsupported_genre_dropped(self):
        assert map_genre({"main_genre": "metal", "genres": ""}) is None

    def test_missing_genre(self):
        assert map_genre({}) is None

    def test_case_insensitive(self):
        assert map_genre({"main_genre": "JAZZ"}) == "jazz"


# ─────────────────────────────────────────────────────────────────────────────
# truncate_sections
# ─────────────────────────────────────────────────────────────────────────────

class TestTruncateSections:
    def test_caps_at_max(self):
        long_section = ("verse", 1, ["C"] * 30)
        result = truncate_sections([long_section])
        assert len(result[0][2]) == MAX_CHORDS_PER_SECTION

    def test_short_section_unchanged(self):
        short_section = ("verse", 1, ["C", "G"])
        result = truncate_sections([short_section])
        assert result[0][2] == ["C", "G"]

    def test_preserves_name_and_idx(self):
        result = truncate_sections([("chorus", 3, ["C"] * 20)])
        assert result[0][0] == "chorus"
        assert result[0][1] == 3


# ─────────────────────────────────────────────────────────────────────────────
# emit_jsonl_row
# ─────────────────────────────────────────────────────────────────────────────

class TestEmitJsonlRow:
    def test_row_keys(self):
        row = emit_jsonl_row("song_42", "jazz", [("intro", 1, ["C", "G"]), ("verse", 1, ["Am", "F"])])
        assert set(row.keys()) == {"id", "style", "sections", "prompt", "completion"}

    def test_id_and_style(self):
        row = emit_jsonl_row("song_42", "jazz", [("intro", 1, ["C"])])
        assert row["id"] == "song_42"
        assert row["style"] == "jazz"

    def test_completion_format(self):
        row = emit_jsonl_row("x", "pop", [("intro", 1, ["C", "G"]), ("verse", 2, ["Am"])])
        assert row["completion"] == "<intro_1> C G <verse_2> Am"

    def test_sections_list(self):
        row = emit_jsonl_row("x", "jazz", [("intro", 1, ["C"]), ("verse", 1, ["G"])])
        assert row["sections"] == ["intro", "verse"]

    def test_prompt_contains_style_and_structure(self):
        row = emit_jsonl_row("x", "jazz", [("intro", 1, ["C"]), ("verse", 1, ["G"])])
        assert "jazz" in row["prompt"]
        assert "intro, verse" in row["prompt"]


# ─────────────────────────────────────────────────────────────────────────────
# _row_to_example (end-to-end on a synthetic row)
# ─────────────────────────────────────────────────────────────────────────────

class TestRowToExample:
    def test_happy_path(self):
        row = {
            "id": "song_1",
            "main_genre": "jazz",
            "chords": "<intro_1> Cmaj7 Am7 Dm7 G7 <verse_1> Cmaj7 Am7 Dm7 G7",
        }
        example, reason = _row_to_example(row)
        assert reason is None
        assert example is not None
        assert example["style"] == "jazz"
        assert example["id"] == "song_1"

    def test_drops_unsupported_genre(self):
        row = {"main_genre": "metal", "chords": "<verse_1> C G Am F"}
        example, reason = _row_to_example(row)
        assert example is None
        assert reason == "wrong_genre"

    def test_drops_too_short(self):
        row = {"main_genre": "jazz", "chords": "<verse_1> C G"}
        example, reason = _row_to_example(row)
        assert example is None
        assert reason == "too_short"

    def test_drops_no_sections(self):
        row = {"main_genre": "jazz", "chords": "C G Am F C G Am F C G"}
        example, reason = _row_to_example(row)
        assert example is None
        assert reason == "no_sections"

    def test_emits_valid_jsonl(self):
        # Round-trip the output through json to confirm it's serializable.
        row = {
            "id": "song_1",
            "main_genre": "jazz",
            "chords": "<intro_1> Cmaj7 Am7 Dm7 G7 <verse_1> Cmaj7 Am7 Dm7 G7",
        }
        example, _ = _row_to_example(row)
        serialized = json.dumps(example)
        roundtrip = json.loads(serialized)
        assert roundtrip == example
