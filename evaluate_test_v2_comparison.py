#!/usr/bin/env python3
"""One-time test_v2 comparison of base Qwen, GRPO-250, and SFT-1250."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_grpo_checkpoints import load_rows
from reward import extract_tagged_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", type=Path, default=Path("data/arithmetic/test_v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/test_v2_comparison.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def evaluate(model: torch.nn.Module, tokenizer: Any, rows: list[dict[str, str]], batch_size: int) -> dict[str, float | int]:
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
            width = inputs["input_ids"].shape[1]
            completions = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
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
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if not args.data.exists():
        raise FileNotFoundError(f"Create test_v2 first: {args.data}")
    rows = load_rows(args.data, None)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"Loading base model on {device} ...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    base_model.config.use_cache = True

    results: dict[str, Any] = {"baseline": evaluate(base_model, tokenizer, rows, args.batch_size)}
    adapters = {
        "grpo_checkpoint_250": Path("outputs/grpo_formal_v1/checkpoint-250"),
        "sft_checkpoint_1250": Path("outputs/sft_formal_v1/checkpoint-1250"),
    }
    model: PeftModel | None = None
    for name, path in adapters.items():
        if not (path / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"Adapter not found: {path}")
        if model is None:
            model = PeftModel.from_pretrained(base_model, path, adapter_name=name)
        else:
            model.load_adapter(path, adapter_name=name)
        model.set_adapter(name)
        results[name] = {"adapter": str(path), **evaluate(model, tokenizer, rows, args.batch_size)}
        print(name, json.dumps(results[name], ensure_ascii=False))

    summary = {
        "protocol": "one-time fresh test_v2 comparison",
        "data": str(args.data),
        "model": args.model,
        "generation": {"do_sample": False, "max_new_tokens": 32, "batch_size": args.batch_size},
        "results": results,
        "accuracy_gains_over_baseline": {
            name: metrics["accuracy"] - results["baseline"]["accuracy"]
            for name, metrics in results.items()
            if name != "baseline"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
