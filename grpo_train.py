"""
grpo_train.py — GRPO fine-tune the SFT checkpoint using validate_sectional as reward.

Reward function (verifiable, no learned reward model):
  reward = r_compliance * (1 - gaming_penalty)
    r_compliance = 0.5 * r_presence + 0.5 * r_order   (sections present + correctly ordered)
    gaming_penalty = 1.0 if max pairwise Jaccard between distinct section types > 0.9 else 0.0

Reads the SFT-prepped train JSONL but only uses each row's `prompt` field — the
completion is discarded since GRPO learns from on-policy rollouts.

Usage:
    python grpo_train.py \
        --train data/chordonomicon_train.jsonl \
        --sft-checkpoint riffle-sft \
        --out riffle-grpo \
        --num-generations 4 \
        --max-steps 500
"""

import argparse
import json
from pathlib import Path
from typing import List

from datasets import Dataset

from chord_rewards import validate_sectional


def _load_prompts(path: str) -> list[dict]:
    """Load JSONL and keep only {prompt, sections} fields needed for GRPO."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"prompt": obj["prompt"], "sections": obj["sections"]})
    return rows


def make_reward_fn():
    """
    Build a reward function with the GRPOTrainer signature:
        reward(prompts, completions, **kwargs) -> list[float]

    kwargs includes any non-`prompt` dataset columns — we use `sections` to know
    what structure was requested for each row.
    """
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # `sections` is a list of lists, aligned with prompts/completions.
        section_lists = kwargs.get("sections", [None] * len(prompts))
        rewards = []
        for completion, requested in zip(completions, section_lists):
            if requested is None:
                rewards.append(0.0)
                continue
            _errors, reward, _breakdown = validate_sectional(completion, list(requested))
            rewards.append(float(reward))
        return rewards
    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="JSONL with prompt + sections fields")
    parser.add_argument("--sft-checkpoint", required=True, help="Path to SFT checkpoint dir")
    parser.add_argument("--out", default="riffle-grpo")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Rollouts per prompt (GRPO group size)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device prompts per step (effective = batch * num_generations)")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="GRPO uses much smaller LR than SFT")
    parser.add_argument("--kl-beta", type=float, default=0.04,
                        help="KL penalty coefficient against SFT reference")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Higher than SFT to encourage exploration during rollouts")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Cap training prompts (smoke test: 100)")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Lazy imports
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    print(f"Loading SFT checkpoint: {args.sft_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading train prompts: {args.train}")
    rows = _load_prompts(args.train)
    if args.max_train_samples:
        rows = rows[: args.max_train_samples]
    print(f"  prompts: {len(rows)}")

    # GRPOTrainer expects a `prompt` column and any extra columns flow to reward_fn kwargs.
    train_ds = Dataset.from_list(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.kl_beta,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_reward_fn(),
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    print("Starting GRPO...")
    trainer.train()

    print(f"Saving final checkpoint to {out_dir}")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("Done.")


if __name__ == "__main__":
    main()
