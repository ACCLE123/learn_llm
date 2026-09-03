#!/usr/bin/env python3
"""Continue the selected SFT adapter with GRPO; never reads test data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from run_grpo_dry_run import evaluate_greedy, grpo_exact_answer_reward, load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--init-adapter", type=Path, default=Path("outputs/sft_formal_v1/checkpoint-1250"))
    parser.add_argument("--train-data", type=Path, default=Path("data/arithmetic/train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/arithmetic/validation.jsonl"))
    parser.add_argument("--train-limit", type=int, default=5_000)
    parser.add_argument("--validation-limit", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft_then_grpo"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.init_adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"SFT adapter not found: {args.init_adapter}")
    if args.max_steps < 1 or args.num_generations < 2:
        raise ValueError("--max-steps must be positive and --num-generations must be at least 2.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(args.train_data, args.train_limit)
    validation_rows = load_rows(args.validation_data, args.validation_limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()
    model = PeftModel.from_pretrained(base_model, args.init_adapter, is_trainable=True)
    model.print_trainable_parameters()

    pre_train_validation = evaluate_greedy(model, tokenizer, validation_rows)
    print("Validation before SFT→GRPO:", json.dumps(pre_train_validation))
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
        save_steps=args.save_steps,
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
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    trainer.model.config.use_cache = True
    post_train_validation = evaluate_greedy(trainer.model, tokenizer, validation_rows)
    reward_logs = [entry for entry in trainer.state.log_history if "reward" in entry]
    summary = {
        "experiment": "SFT checkpoint-1250 followed by GRPO step ablation",
        "model": args.model,
        "initial_adapter": str(args.init_adapter),
        "train_data": str(args.train_data),
        "validation_data": str(args.validation_data),
        "test_data_read": False,
        "train_examples_available": len(train_rows),
        "validation_examples": len(validation_rows),
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "pre_train_validation": pre_train_validation,
        "post_train_validation": post_train_validation,
        "train_metrics": train_result.metrics,
        "reward_logs": reward_logs,
        "final_adapter": str(args.output_dir / "final_adapter"),
    }
    (args.output_dir / "sft_then_grpo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("SFT→GRPO complete:", json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
