#!/usr/bin/env python3
"""Run the single final test evaluation for a selected LoRA adapter."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from evaluate_grpo_checkpoints import evaluate_adapter, load_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Use only for a non-final smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # This entry point is intentionally fixed to the held-out test split and
    # selected adapter to make the final protocol explicit and auditable.
    data = Path("data/arithmetic/test.jsonl")
    output = Path("outputs/grpo_formal_v1/final_test_evaluation.json")
    checkpoint = Path("outputs/grpo_formal_v1/checkpoint-250")
    if not (checkpoint / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Selected adapter not found: {checkpoint}")

    # Import here so `--help` remains fast and all model work is explicit.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(data, args.limit)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    print(f"Loading selected adapter on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    base_model.config.use_cache = True
    model = PeftModel.from_pretrained(base_model, checkpoint, adapter_name="selected")
    metrics = evaluate_adapter(model, "selected", tokenizer, rows, args.batch_size)

    baseline_path = Path("outputs/test_baseline_summary.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    summary = {
        "protocol": "single final held-out test evaluation",
        "model": args.model,
        "selected_checkpoint": str(checkpoint),
        "data": str(data),
        "generation": {"do_sample": False, "max_new_tokens": 32, "batch_size": args.batch_size},
        "grpo": metrics,
        "baseline": {
            "accuracy": baseline["accuracy"],
            "format_valid_rate": baseline["format_valid_rate"],
        },
        "absolute_accuracy_gain": metrics["accuracy"] - baseline["accuracy"],
        "absolute_format_valid_rate_gain": metrics["format_valid_rate"] - baseline["format_valid_rate"],
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
