"""
sft_train.py — Supervised fine-tune Qwen3.5-0.5B on Chordonomicon sectional data.

Usage (Colab or local):
    python sft_train.py \
        --train data/chordonomicon_train.jsonl \
        --val   data/chordonomicon_val.jsonl \
        --out   riffle-sft \
        --model Qwen/Qwen3.5-0.5B \
        --epochs 1 \
        --batch-size 8

Trains on completion tokens only (prompt tokens are loss-masked) via the chat
template. Full fine-tune (no LoRA) — the model is small enough that this fits
comfortably on a single Colab GPU and gives cleaner gradients than LoRA.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_chat_format(row: dict) -> dict:
    """Convert a JSONL training row to the chat-message format SFTTrainer expects."""
    return {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train JSONL")
    parser.add_argument("--val", default=None, help="Path to val JSONL (optional)")
    parser.add_argument("--out", default="riffle-sft", help="Output checkpoint dir")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.5B", help="Base model ID")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Cap training set size (smoke test: 1000)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Hard cap on training steps (smoke test: 100)")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Lazy imports — keep CLI snappy and let `--help` work without GPU libs.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"Loading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading train data: {args.train}")
    train_rows = _load_jsonl(args.train)
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    print(f"  train rows: {len(train_rows)}")
    train_ds = Dataset.from_list([_to_chat_format(r) for r in train_rows])

    eval_ds = None
    if args.val:
        val_rows = _load_jsonl(args.val)
        print(f"  val rows:   {len(val_rows)}")
        eval_ds = Dataset.from_list([_to_chat_format(r) for r in val_rows])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        max_length=args.max_len,
        completion_only_loss=True,   # mask prompt tokens from loss
        report_to=[],                # no wandb/tensorboard unless explicitly added
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting SFT...")
    trainer.train()

    print(f"Saving final checkpoint to {out_dir}")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("Done.")


if __name__ == "__main__":
    main()
