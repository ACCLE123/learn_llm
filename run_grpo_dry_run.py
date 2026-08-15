#!/usr/bin/env python3
"""Run a small GRPO + LoRA smoke test without reading the held-out test set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward import exact_answer_reward, extract_tagged_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-data", type=Path, default=Path("data/arithmetic/train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/arithmetic/validation.jsonl"))
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--validation-limit", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--save-steps",
        type=int,
        help="Checkpoint interval. Defaults to saving only at the final step.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grpo_dry_run"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path, limit: int) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit < 1 or not rows:
        raise ValueError("Dataset limits must be positive and datasets must not be empty.")
    # The trainer receives prompt and answer only. It never sees the test split.
    return [{"prompt": row["prompt"], "answer": row["answer"]} for row in rows[:limit]]


def grpo_exact_answer_reward(
    completions: list[str], answer: list[str], **_: Any
) -> list[float]:
    """TRL callback wrapper around the independently tested reward function."""
    return [exact_answer_reward(completion, reference) for completion, reference in zip(completions, answer)]


def evaluate_greedy(
    model: torch.nn.Module, tokenizer: Any, rows: list[dict[str, str]]
) -> dict[str, float | int]:
    """Evaluate a checkpoint on validation data with the fixed baseline protocol."""
    model.eval()
    correct = 0
    valid_format = 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for row in rows:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(rendered, return_tensors="pt").to(device)
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            prediction = extract_tagged_answer(completion)
            valid_format += prediction is not None
            correct += prediction == row["answer"]
    total = len(rows)
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total,
        "format_valid": valid_format,
        "format_valid_rate": valid_format / total,
    }


def main() -> None:
    args = parse_args()
    if args.train_limit < args.num_generations:
        raise ValueError("--train-limit must be at least --num-generations.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(args.train_data, args.train_limit)
    validation_rows = load_rows(args.validation_data, args.validation_limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.config.use_cache = False  # Required with gradient checkpointing.

    pre_train_validation = evaluate_greedy(model, tokenizer, validation_rows)
    print("Validation before training:", json.dumps(pre_train_validation))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    save_steps = args.save_steps if args.save_steps is not None else args.max_steps
    config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=6,
        report_to="none",
        num_generations=args.num_generations,
        max_completion_length=32,
        temperature=1.0,
        beta=0.04,
        gradient_checkpointing=True,
        use_cache=False,
        seed=args.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_exact_answer_reward,
        args=config,
        train_dataset=Dataset.from_list(train_rows),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))

    post_train_validation = evaluate_greedy(trainer.model, tokenizer, validation_rows)
    reward_logs = [
        entry for entry in trainer.state.log_history if "reward" in entry or "rewards/grpo_exact_answer_reward" in entry
    ]
    summary = {
        "model": args.model,
        "train_data": str(args.train_data),
        "validation_data": str(args.validation_data),
        "test_data_read": False,
        "train_examples_available": len(train_rows),
        "validation_examples": len(validation_rows),
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "save_steps": save_steps,
        "lora": {"r": 8, "alpha": 16, "target_modules": lora_config.target_modules},
        "pre_train_validation": pre_train_validation,
        "post_train_validation": post_train_validation,
        "train_metrics": train_result.metrics,
        "reward_logs": reward_logs,
        "final_adapter": str(args.output_dir / "final_adapter"),
    }
    (args.output_dir / "dry_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("Dry run completed:", json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
