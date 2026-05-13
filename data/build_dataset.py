"""
data/build_dataset.py — Chordonomicon → JSONL training-data pipeline.

Pipeline (matches DATA_PLAN.md §2):
  1. Load ailsntua/Chordonomicon via HuggingFace `datasets`.
  2. Parse inline section tags (intro/verse/chorus/...) from each row's `chords`.
  3. Filter: unparseable chords, length out of [8, 200], zero sections, < 3 distinct chords.
  4. Map main_genre → style (jazz/pop/rock/country/soul/blues; drop the rest).
  5. Truncate each section to MAX_CHORDS_PER_SECTION = 16 chords.
  6. Emit one JSONL row per song; random 90/5/5 train/val/test split.

CLI:
    python data/build_dataset.py                  # full build into data/chordonomicon_{train,val,test}.jsonl
    python data/build_dataset.py --sample 1000    # sanity check on 1000 random rows (no JSONL written)
    python data/build_dataset.py --out-dir path   # override output directory
    python data/build_dataset.py --seed 42        # override RNG seed
    python data/build_dataset.py --max-per-style N  # cap per-style sample count
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

# When invoked as `python data/build_dataset.py`, Python sets `data/` (not the
# repo root) as sys.path[0], which breaks `from chord_rewards import ...`.
# Prepend the repo root so the script works regardless of CWD or invocation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chord_rewards import build_prompt_sectional, try_parse_chord
from data.notation import to_music21


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MAX_CHORDS_PER_SECTION = 16
MIN_TOTAL_CHORDS = 8
MAX_TOTAL_CHORDS = 200
MIN_DISTINCT_CHORDS = 3

GENRE_TO_STYLE = {
    "jazz": "jazz",
    "pop": "pop",
    "rock": "rock",
    "country": "country",
    "soul": "soul",
}
# `blues` is detected via the `genres` tag substring instead of main_genre.

SECTION_RE = re.compile(r"<(\w+?)_(\d+)>\s*([^<]+)")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: parse + filter
# ─────────────────────────────────────────────────────────────────────────────

def parse_song_sections(chord_str: str) -> Optional[list]:
    """Parse a Chordonomicon `chords` field. Returns [(name, idx, [chords]), ...] or None."""
    if not chord_str:
        return None
    sections = []
    for match in SECTION_RE.finditer(chord_str):
        name = match.group(1).lower()
        idx = int(match.group(2))
        chord_tokens = [tok for tok in match.group(3).split() if tok]
        sections.append((name, idx, chord_tokens))
    return sections or None


def is_parseable(chord: str) -> bool:
    """Check whether a chord can be understood by music21 (after notation adapter)."""
    return try_parse_chord(to_music21(chord)) is not None


def passes_filters(parsed_sections: list) -> tuple[bool, Optional[str]]:
    """Apply quality filters. Returns (kept, reject_reason)."""
    if not parsed_sections:
        return False, "no_sections"

    all_chords = [c for _, _, chords in parsed_sections for c in chords]
    if len(all_chords) < MIN_TOTAL_CHORDS:
        return False, "too_short"
    if len(all_chords) > MAX_TOTAL_CHORDS:
        return False, "too_long"
    if len(set(all_chords)) < MIN_DISTINCT_CHORDS:
        return False, "low_diversity"
    if any(not is_parseable(c) for c in all_chords):
        return False, "unparseable_chord"
    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: genre → style mapping
# ─────────────────────────────────────────────────────────────────────────────

def map_genre(row: dict) -> Optional[str]:
    """Map a Chordonomicon row to a project style label, or None to drop."""
    main_genre = (row.get("main_genre") or "").lower()
    if main_genre in GENRE_TO_STYLE:
        return GENRE_TO_STYLE[main_genre]
    # `blues` detection via genres tag list / string.
    genres = row.get("genres") or ""
    if isinstance(genres, list):
        genres = " ".join(genres)
    if "blues" in str(genres).lower():
        return "blues"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: truncate + emit JSONL
# ─────────────────────────────────────────────────────────────────────────────

def truncate_sections(parsed_sections: list, max_chords: int = MAX_CHORDS_PER_SECTION) -> list:
    """Cap each section's chord list to max_chords."""
    return [(name, idx, chords[:max_chords]) for name, idx, chords in parsed_sections]


def emit_jsonl_row(row_id, style: str, truncated_sections: list) -> dict:
    """Build one JSONL training row."""
    section_names = [name for name, _, _ in truncated_sections]
    completion_parts = []
    for name, idx, chords in truncated_sections:
        completion_parts.append(f"<{name}_{idx}> {' '.join(chords)}")
    completion = " ".join(completion_parts)
    return {
        "id": row_id,
        "style": style,
        "sections": section_names,
        "prompt": build_prompt_sectional(style, section_names),
        "completion": completion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main build / sanity pipelines
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_example(row: dict) -> tuple[Optional[dict], Optional[str]]:
    """Process one Chordonomicon row. Returns (example_or_none, reject_reason_or_none)."""
    style = map_genre(row)
    if style is None:
        return None, "wrong_genre"

    parsed = parse_song_sections(row.get("chords", ""))
    if parsed is None:
        return None, "no_sections"

    ok, reason = passes_filters(parsed)
    if not ok:
        return None, reason

    truncated = truncate_sections(parsed)
    example = emit_jsonl_row(row.get("id", ""), style, truncated)
    return example, None


def _iter_rows(dataset, sample_size: Optional[int], seed: int) -> Iterable[dict]:
    """Iterate over dataset rows; if sample_size given, sample uniformly without replacement."""
    n = len(dataset)
    if sample_size is None or sample_size >= n:
        return iter(dataset)
    rng = random.Random(seed)
    indices = rng.sample(range(n), sample_size)
    return (dataset[i] for i in indices)


def run_sanity_check(sample_size: int, seed: int) -> None:
    """Run the full pipeline on a sample, print stats (no JSONL written)."""
    from datasets import load_dataset
    print(f"Loading Chordonomicon (sample of {sample_size}, seed={seed})...")
    dataset = load_dataset("ailsntua/Chordonomicon", split="train")
    print(f"Dataset loaded: {len(dataset)} total rows. Sampling {sample_size}...")

    kept = 0
    reject_reasons = Counter()
    style_counts = Counter()

    for row in _iter_rows(dataset, sample_size, seed):
        example, reason = _row_to_example(row)
        if example is None:
            reject_reasons[reason] += 1
        else:
            kept += 1
            style_counts[example["style"]] += 1

    print()
    print("=== Sanity Check Results ===")
    print(f"Sampled:  {sample_size}")
    print(f"Kept:     {kept} ({100 * kept / sample_size:.1f}%)")
    print(f"Rejected: {sample_size - kept}")
    print()
    print("Rejection reasons:")
    for reason, count in reject_reasons.most_common():
        print(f"  {reason:20s} {count}")
    print()
    print("Style distribution (kept):")
    for style, count in style_counts.most_common():
        share = 100 * count / kept if kept else 0
        print(f"  {style:10s} {count:5d}  ({share:.1f}%)")


def build_splits(out_dir: Path, max_per_style: int, seed: int) -> None:
    """Run the full pipeline and write data/chordonomicon_{train,val,test}.jsonl."""
    from datasets import load_dataset
    print("Loading Chordonomicon (full)...")
    dataset = load_dataset("ailsntua/Chordonomicon", split="train")
    print(f"Dataset loaded: {len(dataset)} rows.")

    by_style: dict[str, list[dict]] = {}
    reject_reasons = Counter()

    for row in dataset:
        example, reason = _row_to_example(row)
        if example is None:
            reject_reasons[reason] += 1
            continue
        by_style.setdefault(example["style"], []).append(example)

    # Cap per-style and shuffle.
    rng = random.Random(seed)
    pooled: list[dict] = []
    for style, examples in by_style.items():
        rng.shuffle(examples)
        capped = examples[:max_per_style]
        print(f"  {style:10s} {len(examples)} kept → {len(capped)} after cap")
        pooled.extend(capped)
    rng.shuffle(pooled)

    n = len(pooled)
    n_val = max(1, n // 20)   # 5%
    n_test = max(1, n // 20)  # 5%
    n_train = n - n_val - n_test

    splits = {
        "train": pooled[:n_train],
        "val": pooled[n_train:n_train + n_val],
        "test": pooled[n_train + n_val:],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = out_dir / f"chordonomicon_{split}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(rows):6d} rows → {path}")

    print()
    print(f"Total kept: {n} / {len(dataset)} ({100*n/len(dataset):.1f}%)")
    print("Rejection reasons:")
    for reason, count in reject_reasons.most_common():
        print(f"  {reason:20s} {count}")


def main():
    parser = argparse.ArgumentParser(description="Build Chordonomicon → JSONL training data.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Run sanity check on N random rows (no JSONL output).")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Output directory for JSONL files (default: ./data).")
    parser.add_argument("--max-per-style", type=int, default=8000,
                        help="Cap kept examples per style label (default: 8000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for sampling and shuffling (default: 42).")
    args = parser.parse_args()

    if args.sample is not None:
        run_sanity_check(args.sample, args.seed)
    else:
        build_splits(Path(args.out_dir), args.max_per_style, args.seed)


if __name__ == "__main__":
    main()
