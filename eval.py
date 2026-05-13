"""
eval.py — Three-way comparison: base / SFT / SFT+GRPO on sectional prompts.

Runs generate_sectional across a small held-out grid of (style × structure)
cells, scores each output with validate_sectional, and writes:
  - eval_results.csv          (one row per generation, raw data)
  - eval_summary.csv          (per-checkpoint mean reward / pass rate / etc.)
  - eval_summary.md           (headline markdown table for the report)

Usage:
    python eval.py \
        --base-model Qwen/Qwen3.5-0.8B \
        --sft riffle-sft \
        --grpo riffle-grpo \
        --samples 5 \
        --out-dir eval_out
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


# Held-out eval grid (kept small for time budget).
EVAL_STYLES = ["jazz", "pop"]
EVAL_STRUCTURES = [
    ["intro", "verse", "chorus", "verse", "chorus", "outro"],
    ["verse", "chorus", "verse", "chorus", "bridge", "chorus"],
    ["intro", "verse", "verse", "chorus", "verse", "chorus", "outro"],
    ["intro", "verse", "chorus", "bridge", "chorus", "outro"],
    ["verse", "chorus", "verse", "chorus"],
]


def _evaluate_checkpoint(label: str, model_id: str, samples: int, seed: int) -> list[dict]:
    """Run all (style × structure × samples) cells against one checkpoint."""
    # Lazy import — keep CLI snappy.
    from baseline_chord_gen import generate_sectional

    rows = []
    for style in EVAL_STYLES:
        for structure in EVAL_STRUCTURES:
            for i in range(samples):
                print(f"[{label}] {style} / {'-'.join(structure)} / sample {i+1}")
                result = generate_sectional(
                    style=style,
                    sections=structure,
                    output_midi_path=None,
                    output_mp3_path=None,
                    output_report_path=None,
                    model_id=model_id,
                    temperature=0.7,
                    max_new_tokens=512,
                    quiet=True,
                )
                rows.append({
                    "checkpoint": label,
                    "style": style,
                    "structure": "-".join(structure),
                    "sample": i,
                    "reward": result.reward,
                    "valid": result.valid,
                    "presence": result.breakdown.get("presence", 0.0),
                    "order": result.breakdown.get("order", 0.0),
                    "gamed": result.breakdown.get("gamed", False),
                    "errors": "|".join(result.errors),
                })
    return rows


def _summarize(rows: list[dict]) -> dict:
    """Per-checkpoint aggregate stats."""
    by_ckpt: dict[str, list[dict]] = {}
    for r in rows:
        by_ckpt.setdefault(r["checkpoint"], []).append(r)

    summary = {}
    for ckpt, rs in by_ckpt.items():
        rewards = [r["reward"] for r in rs]
        summary[ckpt] = {
            "n": len(rs),
            "mean_reward": statistics.mean(rewards),
            "median_reward": statistics.median(rewards),
            "pass_rate": sum(1 for r in rs if r["valid"]) / len(rs),
            "mean_presence": statistics.mean(r["presence"] for r in rs),
            "mean_order": statistics.mean(r["order"] for r in rs),
            "gaming_rate": sum(1 for r in rs if r["gamed"]) / len(rs),
        }
    return summary


def _write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(summary: dict, path: Path):
    headers = ["checkpoint", "n", "mean_reward", "median_reward", "pass_rate",
               "mean_presence", "mean_order", "gaming_rate"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for ckpt in ["base", "sft", "sft+grpo"]:
        if ckpt not in summary:
            continue
        s = summary[ckpt]
        lines.append("| " + " | ".join([
            ckpt,
            f"{s['n']}",
            f"{s['mean_reward']:.3f}",
            f"{s['median_reward']:.3f}",
            f"{s['pass_rate']:.1%}",
            f"{s['mean_presence']:.3f}",
            f"{s['mean_order']:.3f}",
            f"{s['gaming_rate']:.1%}",
        ]) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="unsloth/Qwen3-1.7B",
                        help="Base model ID for the zero-shot condition")
    parser.add_argument("--sft", default=None, help="SFT checkpoint dir (optional)")
    parser.add_argument("--grpo", default=None, help="SFT+GRPO checkpoint dir (optional)")
    parser.add_argument("--samples", type=int, default=5,
                        help="Samples per (style, structure) cell")
    parser.add_argument("--out-dir", default="eval_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run each available checkpoint. baseline_chord_gen caches the model globally,
    # so we need to reset it between runs.
    import baseline_chord_gen

    conditions = [("base", args.base_model)]
    if args.sft:
        conditions.append(("sft", args.sft))
    if args.grpo:
        conditions.append(("sft+grpo", args.grpo))

    all_rows = []
    for label, model_id in conditions:
        baseline_chord_gen._model = None
        baseline_chord_gen._tokenizer = None
        rows = _evaluate_checkpoint(label, model_id, args.samples, args.seed)
        all_rows.extend(rows)

    summary = _summarize(all_rows)

    _write_csv(all_rows, out_dir / "eval_results.csv")
    print(f"  wrote {out_dir / 'eval_results.csv'}")

    summary_rows = [{"checkpoint": k, **v} for k, v in summary.items()]
    _write_csv(summary_rows, out_dir / "eval_summary.csv")
    print(f"  wrote {out_dir / 'eval_summary.csv'}")

    _write_summary_md(summary, out_dir / "eval_summary.md")
    print(f"  wrote {out_dir / 'eval_summary.md'}")

    print()
    print("=== Summary ===")
    print((out_dir / "eval_summary.md").read_text())


if __name__ == "__main__":
    main()
