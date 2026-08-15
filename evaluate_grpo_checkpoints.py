#!/usr/bin/env python3
"""Rank saved LoRA GRPO checkpoints on the validation set only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import extract_tagged_answer


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/grpo_formal_v1"))
    parser.add_argument("--data", type=Path, default=Path("data/arithmetic/validation.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/grpo_formal_v1/checkpoint_validation.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples (for smoke tests).")
    return parser.parse_args()


def checkpoint_paths(directory: Path) -> list[tuple[int, Path]]:
    paths: list[tuple[int, Path]] = []
    for path in directory.glob("checkpoint-*"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match and (path / "adapter_model.safetensors").exists():
            paths.append((int(match.group(1)), path))
    if not paths:
        raise FileNotFoundError(f"No LoRA checkpoints found in {directory}")
    return sorted(paths)


def load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("Validation data is empty.")
    return [{"id": row["id"], "prompt": row["prompt"], "answer": row["answer"]} for row in rows]


def evaluate_adapter(
    model: PeftModel,
    adapter_name: str,
    tokenizer: Any,
    rows: list[dict[str, str]],
    batch_size: int,
) -> dict[str, float | int]:
    model.set_adapter(adapter_name)
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    format_valid = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
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
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    rows = load_rows(args.data, args.limit)
    checkpoints = checkpoint_paths(args.checkpoint_dir)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    print(f"Loading base model on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    base_model.config.use_cache = True

    results: list[dict[str, Any]] = []
    model: PeftModel | None = None
    for step, path in checkpoints:
        adapter_name = f"step_{step}"
        if model is None:
            model = PeftModel.from_pretrained(base_model, path, adapter_name=adapter_name)
        else:
            model.load_adapter(path, adapter_name=adapter_name)
        metrics = evaluate_adapter(model, adapter_name, tokenizer, rows, args.batch_size)
        result = {"checkpoint_step": step, "checkpoint": str(path), **metrics}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    ranked = sorted(results, key=lambda item: (item["accuracy"], item["format_valid_rate"]), reverse=True)
    summary = {
        "model": args.model,
        "data": str(args.data),
        "test_data_read": False,
        "generation": {"do_sample": False, "max_new_tokens": 32, "batch_size": args.batch_size},
        "ranked_checkpoints": ranked,
        "selected_checkpoint": ranked[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Selected checkpoint:", json.dumps(ranked[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
