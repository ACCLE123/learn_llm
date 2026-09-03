#!/usr/bin/env python3
"""Train a LoRA SFT baseline on arithmetic answers without reading test data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from reward import extract_tagged_answer


class ArithmeticSFTDataset(Dataset[dict[str, list[int]]]):
    """Tokenized chat examples whose loss applies only to the assistant answer."""

    def __init__(self, rows: list[dict[str, str]], tokenizer: Any, max_length: int) -> None:
        self.examples: list[dict[str, list[int]]] = []
        for row in rows:
            user_messages = [{"role": "user", "content": row["prompt"]}]
            full_messages = user_messages + [
                {"role": "assistant", "content": f"<answer>{row['answer']}</answer>"}
            ]
            prompt_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("Chat-template prefix mismatch; refusing to train with invalid labels.")
            if len(full_ids) <= len(prompt_ids):
                raise ValueError("Answer was truncated; increase --max-length.")
            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-data", type=Path, default=Path("data/arithmetic/train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/arithmetic/validation.jsonl"))
    parser.add_argument("--train-limit", type=int, default=5_000)
    parser.add_argument("--validation-limit", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=1_250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft_formal_v1"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path, limit: int) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = rows[:limit]
    if not rows:
        raise ValueError(f"No examples loaded from {path}")
    return [{"prompt": row["prompt"], "answer": row["answer"]} for row in rows]


def evaluate_greedy(model: torch.nn.Module, tokenizer: Any, rows: list[dict[str, str]], batch_size: int = 8) -> dict[str, float | int]:
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    format_valid = 0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            prompt_width = inputs["input_ids"].shape[1]
            completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
            for completion, row in zip(completions, batch, strict=True):
                prediction = extract_tagged_answer(completion.strip())
                format_valid += prediction is not None
                correct += prediction == row["answer"]
    total = len(rows)
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total,
        "format_valid": format_valid,
        "format_valid_rate": format_valid / total,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(args.train_data, args.train_limit)
    validation_rows = load_rows(args.validation_data, args.validation_limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train_dataset = ArithmeticSFTDataset(train_rows, tokenizer, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    pre_train_validation = evaluate_greedy(model, tokenizer, validation_rows)
    print("Validation before SFT:", json.dumps(pre_train_validation))
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=6,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    trainer.model.config.use_cache = True
    post_train_validation = evaluate_greedy(trainer.model, tokenizer, validation_rows)

    summary = {
        "experiment": "SFT LoRA baseline",
        "model": args.model,
        "train_data": str(args.train_data),
        "validation_data": str(args.validation_data),
        "test_data_read": False,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "lora": {"r": 8, "alpha": 16},
        "pre_train_validation": pre_train_validation,
        "post_train_validation": post_train_validation,
        "train_metrics": train_result.metrics,
        "final_adapter": str(args.output_dir / "final_adapter"),
    }
    (args.output_dir / "sft_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("SFT complete:", json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
