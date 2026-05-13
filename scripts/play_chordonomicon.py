"""Importable module to play random Chordonomicon progressions.

Usage in Jupyter/Colab:
    from scripts.play_chordonomicon import play_random
    play_random(style="jazz")
    play_random(style="rock", section="chorus")
"""

import os
import random
import tempfile
from datasets import load_dataset
from chord_rewards import parse_sectional_progression
from data.notation import to_music21
from baseline_chord_gen import render_midi, render_mp3


_DATASET = None


def _load_dataset():
    """Lazily load the dataset on first call."""
    global _DATASET
    if _DATASET is None:
        _DATASET = load_dataset("ailsntua/Chordonomicon", split="train")
    return _DATASET


def play_random(
    style: str | None = None,
    section: str | None = None,
    seed: int | None = None,
):
    """
    Play a random Chordonomicon progression inline in Jupyter/Colab.

    Args:
        style: filter to rows where main_genre == this value. None = no filter.
        section: if given, play only the chords from sections matching this name
                 (e.g., section="chorus" plays just chorus sections). None = play full song.
        seed: random seed for reproducible row selection.

    Returns:
        IPython.display.Audio object for inline playback (or None if dataset row
        couldn't be parsed). Also prints a summary line: song_id, genre, decade.
    """
    dataset = _load_dataset()

    # Set seed if provided
    if seed is not None:
        random.seed(seed)

    # Filter by style if provided
    if style is not None:
        filtered = [row for row in dataset if row["main_genre"] == style]
        if not filtered:
            print(f"No songs found with main_genre='{style}'")
            return None
        row = random.choice(filtered)
    else:
        row = random.choice(list(dataset))

    # Parse the chords field
    parsed = parse_sectional_progression(row["chords"])
    if parsed is None:
        print(f"Could not parse chords for song {row['id']}")
        return None

    # Extract chord list
    if section is not None:
        # Collect chords from matching sections (case-insensitive)
        section_lower = section.lower()
        chords = []
        for sec_name, sec_idx, sec_chords in parsed["sections"]:
            if sec_name == section_lower:
                chords.extend(sec_chords)

        if not chords:
            print(f"No sections matching '{section}' found in song {row['id']}")
            return None
    else:
        # Flatten all sections
        chords = []
        for sec_name, sec_idx, sec_chords in parsed["sections"]:
            chords.extend(sec_chords)

    # Convert to music21 notation
    chords_m21 = [to_music21(chord) for chord in chords]

    # Render to MIDI (mkstemp returns a file descriptor we must close immediately)
    mid_fd, mid_path = tempfile.mkstemp(suffix=".mid")
    os.close(mid_fd)
    render_midi(chords_m21, mid_path)

    # Render to MP3
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(mp3_fd)
    render_mp3(mid_path, mp3_path)

    # Print summary
    section_names = [sec_name for sec_name, _, _ in parsed["sections"]]
    print(f"Song {row['id']} | genre={row['main_genre']} | decade={row['decade']} | sections={section_names}")

    # Return Audio object
    try:
        from IPython.display import Audio
        return Audio(mp3_path)
    except ImportError:
        print(f"IPython not available. MP3 saved to: {mp3_path}")
        return None
