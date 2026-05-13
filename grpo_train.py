"""
grpo_train.py — GRPO fine-tune the SFT checkpoint using validate_sectional as reward.

Uses Unsloth + vLLM for fast on-policy rollout generation.
Saves a merged 16-bit model so eval.py can load it with standard HuggingFace.

Reward function (verifiable, no learned reward model):
  reward = r_compliance * (1 - gaming_penalty)
    r_compliance = 0.5 * r_presence + 0.5 * r_order
    gaming_penalty = 1.0 if max pairwise Jaccard > 0.9 else 0.0

Usage:
    python grpo_train.py \
        --train data/chordonomicon_train.jsonl \
        --sft-checkpoint /content/drive/MyDrive/riffle_checkpoints/sft \
        --out /content/drive/MyDrive/riffle_checkpoints/grpo
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
    Reward function with the GRPOTrainer signature:
        reward(prompts, completions, **kwargs) -> list[float]

    Unsloth's GRPO passes completions as list[list[dict]] (chat-message format).
    We unwrap the assistant content before calling validate_sectional.
    kwargs includes non-`prompt` dataset columns — we use `sections`.
    """
    def reward_fn(prompts: List, completions: List, **kwargs) -> List[float]:
        section_lists = kwargs.get("sections", [None] * len(prompts))
        rewards = []
        for completion, requested in zip(completions, section_lists):
            # Unsloth passes each completion as [{"role": "assistant", "content": "..."}]
            if isinstance(completion, list):
                text = completion[0]["content"]
            else:
                text = completion
            if requested is None:
                rewards.append(0.0)
                continue
            _errors, reward, _breakdown = validate_sectional(text, list(requested))
            rewards.append(float(reward))
        return rewards
    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",          required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--out",            default="riffle-grpo")
    parser.add_argument("--lora-rank",      type=int,   default=32)
    parser.add_argument("--num-generations",type=int,   default=4,
                        help="Rollouts per prompt (GRPO group size)")
    parser.add_argument("--batch-size",     type=int,   default=1,
                        help="Per-device prompts per step")
    parser.add_argument("--grad-accum",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=5e-6)
    parser.add_argument("--kl-beta",        type=float, default=0.04)
    parser.add_argument("--max-steps",      type=int,   default=500)
    parser.add_argument("--max-prompt-length",     type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Cap training prompts (smoke test: 100)")
    parser.add_argument("--save-steps",    type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6,
                        help="vLLM GPU fraction; reduce if OOM during rollout")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from vllm import SamplingParams
    from trl import GRPOConfig, GRPOTrainer

    max_seq_length = args.max_prompt_length + args.max_completion_length

    print(f"Loading SFT checkpoint: {args.sft_checkpoint}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.sft_checkpoint,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,                             # enable vLLM for GRPO rollouts
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    print(f"Loading train prompts: {args.train}")
    rows = _load_prompts(args.train)
    if args.max_train_samples:
        rows = rows[:args.max_train_samples]
    print(f"  prompts: {len(rows)}")

    # GRPOTrainer expects prompt as chat messages; extra columns flow to reward_fn kwargs.
    train_ds = Dataset.from_list([
        {
            "prompt":   [{"role": "user", "content": r["prompt"]}],
            "sections": r["sections"],
        }
        for r in rows
    ])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vllm_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        top_k=-1,
        seed=args.seed,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )

    grpo_config = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        output_dir=str(out_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.001,
        optim="adamw_8bit",
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

    # Merge LoRA and save as standard HF checkpoint for eval.py compatibility.
    print(f"Saving merged 16-bit model to {out_dir}")
    model.save_pretrained_merged(str(out_dir), tokenizer, save_method="merged_16bit")
    print("Done.")


if __name__ == "__main__":
    main()
