"""CLI script to explore the Chordonomicon dataset."""

import random
import statistics
from collections import Counter
from datasets import load_dataset
from chord_rewards import parse_sectional_progression


def main():
    """Explore and print dataset statistics."""
    print("Loading Chordonomicon dataset...")
    dataset = load_dataset("ailsntua/Chordonomicon", split="train")

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset basics
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Dataset Overview ===")
    print(f"Dataset: ailsntua/Chordonomicon")
    print(f"Total rows: {len(dataset)}")
    print(f"Schema: {dict(dataset.features)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Genre distribution
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Genre Distribution (Top 15) ===")
    genres = [row["main_genre"] for row in dataset]
    genre_counts = Counter(genres)
    total = len(genres)

    for genre, count in genre_counts.most_common(15):
        pct = 100.0 * count / total
        print(f"{genre:20s} {count:6d}  {pct:5.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # Section-tag distribution
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Section-Tag Count Distribution ===")
    section_counts = []
    for row in dataset:
        parsed = parse_sectional_progression(row["chords"])
        if parsed is not None:
            section_counts.append(len(parsed["sections"]))

    if section_counts:
        p10 = statistics.quantiles(section_counts, n=10)[0]
        p25 = statistics.quantiles(section_counts, n=4)[0]
        p50 = statistics.quantiles(section_counts, n=2)[0]
        p75 = statistics.quantiles(section_counts, n=4)[2]
        p90 = statistics.quantiles(section_counts, n=10)[8]
        p99 = statistics.quantiles(section_counts, n=100)[98]

        print(f"Median (p50): {p50:.1f}")
        print(f"Mean:         {statistics.mean(section_counts):.1f}")
        print(f"p10:          {p10:.1f}")
        print(f"p25:          {p25:.1f}")
        print(f"p50:          {p50:.1f}")
        print(f"p75:          {p75:.1f}")
        print(f"p90:          {p90:.1f}")
        print(f"p99:          {p99:.1f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Top 50 chord symbols
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== Top 50 Chord Symbols ===")
    all_chords = Counter()
    for row in dataset:
        chord_tokens = row["chords"].split()
        for token in chord_tokens:
            if not token.startswith("<"):
                all_chords[token] += 1

    for i, (chord, count) in enumerate(all_chords.most_common(50), 1):
        print(f"{i:2d}. {chord:10s} {count:6d}")

    # ─────────────────────────────────────────────────────────────────────────
    # 3 random rows
    # ─────────────────────────────────────────────────────────────────────────
    print("\n=== 3 Random Rows ===")
    random.seed(42)
    indices = random.sample(range(len(dataset)), 3)

    for idx, row_idx in enumerate(indices, 1):
        row = dataset[row_idx]
        chords_str = row["chords"]
        if len(chords_str) > 300:
            chords_str = chords_str[:300] + "..."

        print(f"\n--- Row {idx} (id={row['id']}) ---")
        print(f"main_genre: {row['main_genre']}")
        print(f"genres:     {row['genres']}")
        print(f"decade:     {row['decade']}")
        print(f"chords:     {chords_str}")


if __name__ == "__main__":
    main()
