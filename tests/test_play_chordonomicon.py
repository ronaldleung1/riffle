"""Smoke tests for play_chordonomicon module."""

import inspect
import pytest


def test_import():
    """Test that play_random can be imported."""
    from scripts.play_chordonomicon import play_random
    assert play_random is not None


def test_signature():
    """Test that play_random has the correct signature."""
    from scripts.play_chordonomicon import play_random

    sig = inspect.signature(play_random)
    params = set(sig.parameters.keys())

    assert "style" in params
    assert "section" in params
    assert "seed" in params
