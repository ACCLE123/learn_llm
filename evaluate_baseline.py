#!/usr/bin/env python3
"""Evaluate Qwen2.5-0.5B-Instruct on the arithmetic validation set.

This script performs inference only. It does not update model weights.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ANSWER_PATTERN = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/arithmetic/validation.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline_results.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/baseline_summary.json"))
    parser.add_argument("--limit", type=int, help="Evaluate only the first N validation examples.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_examples(path: Path, limit: int | None) -> list[dict[str, Any]]:
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def extract_answer(model_output: str) -> str | None:
    """Return the first tagged integer, or None when output violates the contract."""
    match = ANSWER_PATTERN.search(model_output)
    return match.group(1) if match else None


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    examples = load_examples(args.data, args.limit)

    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    formatted = 0
    with args.output.open("w", encoding="utf-8") as output_file, torch.inference_mode():
        for index, example in enumerate(examples, start=1):
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_tokens = generated[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            prediction = extract_answer(response)
            is_formatted = prediction is not None
            is_correct = prediction == example["answer"]
            formatted += is_formatted
            correct += is_correct
            result = {
                "id": example["id"],
                "prompt": example["prompt"],
                "reference_answer": example["answer"],
                "model_response": response,
                "parsed_answer": prediction,
                "format_valid": is_formatted,
                "correct": is_correct,
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(examples):
                print(f"Evaluated {index}/{len(examples)}")

    summary = {
        "model": args.model,
        "data": str(args.data),
        "device": device,
        "examples": len(examples),
        "correct": correct,
        "accuracy": correct / len(examples),
        "format_valid": formatted,
        "format_valid_rate": formatted / len(examples),
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
