"""Measure cross-section-name Jaccard overlap in local Chordonomicon JSONL splits.

Mirrors ``parse_sectional_progression`` + ``_section_max_jaccard`` in
``chord_rewards.py`` without importing that module (avoids ``music21``).
For each distinct section *name*, merge chord symbols across instances;
max pairwise Jaccard between names; ``gamed`` if max > threshold (default 0.9).

Run from repo root::

    python scripts/analyze_section_jaccard_dataset.py
    python scripts/analyze_section_jaccard_dataset.py --threshold 0.9 data/chordonomicon_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.notation import to_chordonomicon


def _parse_completion(raw: str) -> dict | None:
    """Same structure as ``chord_rewards.parse_sectional_progression`` (no music21)."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    pattern = re.compile(r"<(\w+?)_(\d+)>\s*([^<]+)", re.DOTALL)
    sections = []
    for match in pattern.finditer(cleaned):
        name = match.group(1).lower()
        index = int(match.group(2))
        raw_chords = match.group(3).split()
        chords = [
            to_chordonomicon(token)
            for token in raw_chords
            if token.strip() and token != "/"
        ]
        sections.append((name, index, chords))
    if not sections or all(len(chords) == 0 for _, _, chords in sections):
        return None
    return {"sections": sections}


def _section_max_jaccard(parsed_sections: list) -> float:
    """Mirror ``chord_rewards._section_max_jaccard`` (pairwise Jaccard on merged name sets)."""
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
            jac = 0.0 if not union else len(a & b) / len(union)
            if jac > max_jac:
                max_jac = jac
    return max_jac

_DEFAULT_FILES = [
    _REPO_ROOT / "data" / "chordonomicon_train.jsonl",
    _REPO_ROOT / "data" / "chordonomicon_val.jsonl",
    _REPO_ROOT / "data" / "chordonomicon_test.jsonl",
]


def _bucket(x: float) -> str:
    if x <= 0.5:
        return "0.00–0.50"
    if x <= 0.7:
        return "0.50–0.70"
    if x <= 0.9:
        return "0.70–0.90"
    return "0.90–1.00"


def _row_stats(row: dict) -> tuple[bool, float | None, str]:
    """Returns (unparseable, max_jaccard_or_None, style).

    ``max_jaccard`` is None when unparseable or fewer than two distinct section names.
    """
    style = row.get("style", "unknown")
    parsed = _parse_completion(row.get("completion", ""))
    if parsed is None:
        return True, None, style
    names = {s[0] for s in parsed["sections"]}
    if len(names) < 2:
        return False, None, style
    return False, _section_max_jaccard(parsed["sections"]), style


def _build_report(
    path: Path,
    total: int,
    unparseable: int,
    eligible: int,
    gamed: int,
    max_jacs: list[float],
    by_style_eligible: dict[str, int],
    by_style_gamed: dict[str, int],
) -> dict:
    buckets = {"0.00–0.50": 0, "0.50–0.70": 0, "0.70–0.90": 0, "0.90–1.00": 0}
    for mj in max_jacs:
        buckets[_bucket(mj)] += 1

    out: dict = {
        "path": str(path),
        "total_rows": total,
        "unparseable": unparseable,
        "eligible_rows": eligible,
        "gamed_count": gamed,
        "gamed_rate_of_eligible": (gamed / eligible) if eligible else 0.0,
        "gamed_rate_of_total": (gamed / total) if total else 0.0,
        "buckets": buckets,
    }
    if max_jacs:
        out["max_jaccard_mean"] = statistics.mean(max_jacs)
        out["max_jaccard_median"] = statistics.median(max_jacs)
        qs = statistics.quantiles(max_jacs, n=100)
        out["max_jaccard_p90"] = qs[89]
        out["max_jaccard_p99"] = qs[98]
    else:
        out["max_jaccard_mean"] = None
        out["max_jaccard_median"] = None
        out["max_jaccard_p90"] = None
        out["max_jaccard_p99"] = None

    style_rates = {}
    for st, el in sorted(by_style_eligible.items()):
        gd = by_style_gamed.get(st, 0)
        style_rates[st] = {"eligible": el, "gamed": gd, "rate": gd / el if el else 0.0}
    out["by_style"] = style_rates
    return out


def analyze_path(path: Path, threshold: float) -> dict:
    total = 0
    unparseable = 0
    # Rows where at least two distinct section names exist (same as reward logic for max_jac).
    eligible = 0
    gamed = 0
    max_jacs: list[float] = []
    by_style_gamed: dict[str, int] = {}
    by_style_eligible: dict[str, int] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            bad, mj, style = _row_stats(json.loads(line))
            if bad:
                unparseable += 1
                continue
            if mj is None:
                continue

            eligible += 1
            by_style_eligible[style] = by_style_eligible.get(style, 0) + 1

            max_jacs.append(mj)
            if mj > threshold:
                gamed += 1
                by_style_gamed[style] = by_style_gamed.get(style, 0) + 1

    return _build_report(
        path, total, unparseable, eligible, gamed, max_jacs, by_style_eligible, by_style_gamed
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jsonl",
        nargs="*",
        type=Path,
        default=_DEFAULT_FILES,
        help="JSONL files (default: train/val/test under data/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Same as validate_sectional gaming threshold (default: 0.9)",
    )
    args = parser.parse_args()

    grand_total = 0
    grand_eligible = 0
    grand_gamed = 0

    for p in args.jsonl:
        if not p.is_file():
            print(f"SKIP (missing): {p}", file=sys.stderr)
            continue
        r = analyze_path(p.resolve(), args.threshold)
        grand_total += r["total_rows"]
        grand_eligible += r["eligible_rows"]
        grand_gamed += r["gamed_count"]

        print(f"\n=== {r['path']} ===")
        print(f"  rows:              {r['total_rows']}")
        print(f"  unparseable:       {r['unparseable']}")
        print(f"  eligible (≥2 names): {r['eligible_rows']}")
        print(f"  gamed (max Jaccard > {args.threshold}): {r['gamed_count']}")
        print(f"  gamed / eligible:  {100 * r['gamed_rate_of_eligible']:.2f}%")
        print(f"  gamed / all rows:  {100 * r['gamed_rate_of_total']:.2f}%")
        if r["max_jaccard_mean"] is not None:
            print(
                f"  max_jaccard: mean={r['max_jaccard_mean']:.3f} "
                f"median={r['max_jaccard_median']:.3f} "
                f"p90={r['max_jaccard_p90']:.3f} p99={r['max_jaccard_p99']:.3f}"
            )
        print(f"  distribution (eligible only): {r['buckets']}")
        print("  by style (gamed / eligible):")
        for st, info in r["by_style"].items():
            print(f"    {st:12s} {info['gamed']:5d} / {info['eligible']:5d}  ({100 * info['rate']:.1f}%)")

    if grand_eligible:
        print("\n=== COMBINED (files read) ===")
        print(f"  total rows:        {grand_total}")
        print(f"  eligible:          {grand_eligible}")
        print(f"  gamed:             {grand_gamed}")
        print(
            f"  gamed / eligible:  {100 * grand_gamed / grand_eligible:.2f}% "
            f"(threshold {args.threshold})"
        )


if __name__ == "__main__":
    main()
