"""
sft_train.py — Supervised fine-tune Qwen3-1.7B on Chordonomicon sectional data.

Uses Unsloth + QLoRA for faster training with less VRAM.
Saves a merged 16-bit model so eval.py can load it with standard HuggingFace.

Usage (Colab):
    python sft_train.py \
        --train data/chordonomicon_train.jsonl \
        --val   data/chordonomicon_val.jsonl \
        --out   /content/drive/MyDrive/riffle_checkpoints/sft

Smoke test (verify pipeline before full run):
    python sft_train.py --train ... --val ... --out ... \
        --max-train-samples 500 --max-steps 50
"""

import argparse
import json
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
            {"role": "user",      "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train JSONL")
    parser.add_argument("--val",   default=None,  help="Path to val JSONL (optional)")
    parser.add_argument("--out",   default="riffle-sft", help="Output checkpoint dir")
    parser.add_argument("--model", default="unsloth/Qwen3-1.7B")
    parser.add_argument("--lora-rank",  type=int,   default=16)
    parser.add_argument("--epochs",     type=float, default=1.0)
    parser.add_argument("--batch-size", type=int,   default=1)
    parser.add_argument("--grad-accum", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--max-len",    type=int,   default=768)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Load base model in 4-bit for QLoRA; use --no-load-in-4bit if VRAM allows.")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 instead of fp16. Leave off for Colab T4.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Cap training set size (smoke test: 500)")
    parser.add_argument("--max-steps",  type=int, default=None,
                        help="Hard step cap (smoke test: 50)")
    parser.add_argument("--save-steps",    type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    if not args.bf16 and torch.cuda.is_bf16_supported():
        args.bf16 = True

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_len,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False,    # vLLM not needed for SFT
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2,   # *2 recommended by Unsloth
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    print(f"Loading train data: {args.train}")
    train_rows = _load_jsonl(args.train)
    if args.max_train_samples:
        train_rows = train_rows[:args.max_train_samples]
    print(f"  train rows: {len(train_rows)}")
    def _apply_template(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}

    train_ds = Dataset.from_list([_to_chat_format(r) for r in train_rows])
    train_ds = train_ds.map(_apply_template, remove_columns=["messages"])

    eval_ds = None
    if args.val:
        val_rows = _load_jsonl(args.val)
        print(f"  val rows:   {len(val_rows)}")
        eval_ds = Dataset.from_list([_to_chat_format(r) for r in val_rows])
        eval_ds = eval_ds.map(_apply_template, remove_columns=["messages"])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        fp16=not args.bf16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.save_steps if eval_ds else None,
        dataset_text_field="text",
        max_length=args.max_len,
        report_to=[],
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

    # Merge LoRA into base weights and save as standard HF checkpoint so
    # eval.py can load it without needing Unsloth.
    print(f"Saving merged 16-bit model to {out_dir}")
    model.save_pretrained_merged(str(out_dir), tokenizer, save_method="merged_16bit")
    print("Done.")


if __name__ == "__main__":
    main()
