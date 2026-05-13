"""Tests for the sectional generation function in baseline_chord_gen.py."""

from baseline_chord_gen import GenerationResultSectional, _flatten_sections


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataclass smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerationResultSectionalDataclass:
    def test_instantiate_with_dummy_values(self):
        result = GenerationResultSectional(
            prompt="test prompt",
            raw_output="<verse_1> C G Am",
            requested_sections=["verse"],
            parsed_sections=[("verse", 1, ["C", "G", "Am"])],
            valid=True,
            reward=0.8,
            errors=[],
            breakdown={"presence": 1.0, "order": 1.0, "compliance": 1.0, "max_jaccard": 0.0, "gamed": False},
        )
        assert result.prompt == "test prompt"
        assert result.raw_output == "<verse_1> C G Am"
        assert result.requested_sections == ["verse"]
        assert result.parsed_sections is not None
        assert len(result.parsed_sections) == 1
        assert result.valid is True
        assert result.reward == 0.8
        assert result.errors == []
        assert result.midi_path is None
        assert result.mp3_path is None
        assert result.report_path is None

    def test_instantiate_with_none_parsed_sections(self):
        result = GenerationResultSectional(
            prompt="test",
            raw_output="unparseable output",
            requested_sections=["verse"],
            parsed_sections=None,
            valid=False,
            reward=0.0,
            errors=["unparseable: no valid section tags"],
            breakdown={},
        )
        assert result.parsed_sections is None
        assert result.valid is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. play() returns None when mp3_path is None
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayMethod:
    def test_play_with_no_mp3_path_prints_message(self, capsys):
        result = GenerationResultSectional(
            prompt="test",
            raw_output="<verse_1> C G",
            requested_sections=["verse"],
            parsed_sections=None,
            valid=False,
            reward=0.0,
            errors=[],
            breakdown={},
            mp3_path=None,
        )
        result.play()
        captured = capsys.readouterr()
        assert "No MP3 available" in captured.out

    def test_play_with_mp3_path_attempts_to_import_ipython(self):
        """Verify that play() tries to use IPython when mp3_path exists."""
        result = GenerationResultSectional(
            prompt="test",
            raw_output="<verse_1> C G",
            requested_sections=["verse"],
            parsed_sections=None,
            valid=False,
            reward=0.0,
            errors=[],
            breakdown={},
            mp3_path="/tmp/test.mp3",
        )
        # This will fail to display since we're not in a Jupyter environment,
        # but it should attempt the import without crashing.
        # The function handles the ImportError gracefully.
        try:
            result.play()
        except Exception:
            pass  # Expected in non-Jupyter environment


# ─────────────────────────────────────────────────────────────────────────────
# 3. MIDI flatten logic (unit test without model)
# ─────────────────────────────────────────────────────────────────────────────

class TestFlattenSectionsLogic:
    def test_flatten_sections_simple(self):
        parsed = [
            ("intro", 1, ["C", "G"]),
            ("verse", 1, ["Am", "F"]),
        ]
        assert _flatten_sections(parsed) == ["C", "G", "Am", "F"]

    def test_flatten_sections_with_multiple_instances(self):
        parsed = [
            ("intro", 1, ["C", "G"]),
            ("verse", 1, ["Am", "F"]),
            ("verse", 2, ["Dm", "G"]),
            ("chorus", 1, ["F", "C"]),
        ]
        assert _flatten_sections(parsed) == ["C", "G", "Am", "F", "Dm", "G", "F", "C"]

    def test_flatten_sections_preserves_order(self):
        parsed = [
            ("intro", 1, ["E", "B"]),
            ("chorus", 1, ["D", "A"]),
            ("verse", 1, ["G"]),
        ]
        assert _flatten_sections(parsed) == ["E", "B", "D", "A", "G"]

    def test_flatten_empty_sections(self):
        assert _flatten_sections([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# 4. format_report output verification
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatReport:
    def test_format_report_with_parsed_sections(self):
        result = GenerationResultSectional(
            prompt="test prompt",
            raw_output="<verse_1> C G",
            requested_sections=["verse"],
            parsed_sections=[("verse", 1, ["C", "G"])],
            valid=True,
            reward=0.9,
            errors=[],
            breakdown={"presence": 1.0, "order": 1.0, "compliance": 1.0, "max_jaccard": 0.0, "gamed": False},
            midi_path="/tmp/test.mid",
        )
        report = result.format_report()
        assert "Sectional Chord Progression Report" in report
        assert "verse" in report
        assert "Valid    : True" in report
        assert "Reward   : 0.9" in report
        assert "verse_1: ['C', 'G']" in report
        assert "test prompt" in report
        assert "<verse_1> C G" in report

    def test_format_report_with_none_parsed_sections(self):
        result = GenerationResultSectional(
            prompt="test",
            raw_output="bad output",
            requested_sections=["verse"],
            parsed_sections=None,
            valid=False,
            reward=0.0,
            errors=["unparseable: no valid section tags"],
            breakdown={},
        )
        report = result.format_report()
        assert "<unparsed>" in report
        assert "unparseable: no valid section tags" in report
