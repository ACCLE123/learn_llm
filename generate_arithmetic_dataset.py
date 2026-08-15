#!/usr/bin/env python3
"""Generate reproducible arithmetic datasets for GRPO experiments.

Each JSONL row includes a prompt for the model, a hidden reference answer for
the reward function, and metadata for analysis.  The model should receive only
the `prompt` field during GRPO sampling.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


OPERATION_SYMBOLS = {
    "add": "+",
    "subtract": "-",
    "multiply": "×",
    "divide": "÷",
}

PROMPT_TEMPLATE = """请计算下列算式。只输出 `<answer>整数</answer>`，不要解释。
问题：{left} {symbol} {right}"""


def positive_integer_with_digits(rng: random.Random, digits: int) -> int:
    """Return a positive integer with exactly `digits` decimal digits."""
    return rng.randint(10 ** (digits - 1), 10**digits - 1)


def make_example(
    rng: random.Random, operation: str, min_digits: int, max_digits: int
) -> dict[str, Any]:
    """Make one arithmetic problem with an integral, non-negative answer."""
    left_digits = rng.randint(min_digits, max_digits)
    right_digits = rng.randint(min_digits, max_digits)
    left = positive_integer_with_digits(rng, left_digits)
    right = positive_integer_with_digits(rng, right_digits)

    if operation == "add":
        answer = left + right
    elif operation == "subtract":
        # Avoid negative results while retaining a broad range of magnitudes.
        left, right = max(left, right), min(left, right)
        answer = left - right
    elif operation == "multiply":
        answer = left * right
    elif operation == "divide":
        # Construct dividend from a quotient and divisor, guaranteeing an exact
        # integer result and avoiding the ambiguous rounding convention.
        quotient = left
        divisor = right
        left, right, answer = quotient * divisor, divisor, quotient
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    return {
        "prompt": PROMPT_TEMPLATE.format(
            left=left, symbol=OPERATION_SYMBOLS[operation], right=right
        ),
        "answer": str(answer),
        "metadata": {
            "operation": operation,
            "left_operand": left,
            "right_operand": right,
            "left_digits": len(str(left)),
            "right_digits": len(str(right)),
        },
    }


def example_key(example: dict[str, Any]) -> tuple[str, int, int]:
    metadata = example["metadata"]
    left = metadata["left_operand"]
    right = metadata["right_operand"]
    # Addition and multiplication are commutative. Canonicalising their keys
    # prevents e.g. `5 × 350` entering train while `350 × 5` enters validation.
    if metadata["operation"] in {"add", "multiply"}:
        left, right = sorted((left, right))
    return (
        metadata["operation"],
        left,
        right,
    )


def generate_split(
    rng: random.Random,
    split: str,
    count: int,
    operations: list[str],
    min_digits: int,
    max_digits: int,
    excluded_keys: set[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen_keys = set(excluded_keys)
    attempts = 0
    max_attempts = count * 100

    while len(examples) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Could not generate enough unique examples. Increase the digit range "
                "or reduce the requested dataset size."
            )
        example = make_example(rng, rng.choice(operations), min_digits, max_digits)
        key = example_key(example)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        example["id"] = f"{split}-{len(examples):06d}"
        example["split"] = split
        examples.append(example)
    return examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/arithmetic"))
    parser.add_argument("--train-size", type=int, default=5_000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--min-digits", type=int, default=1)
    parser.add_argument("--max-digits", type=int, default=3)
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=sorted(OPERATION_SYMBOLS),
        default=["multiply"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_size < 1 or args.validation_size < 1:
        raise ValueError("--train-size and --validation-size must both be positive.")
    if args.min_digits < 1 or args.max_digits < args.min_digits:
        raise ValueError("Digit bounds must satisfy 1 <= min-digits <= max-digits.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_examples = generate_split(
        rng,
        "train",
        args.train_size,
        args.operations,
        args.min_digits,
        args.max_digits,
        excluded_keys=set(),
    )
    validation_examples = generate_split(
        rng,
        "validation",
        args.validation_size,
        args.operations,
        args.min_digits,
        args.max_digits,
        excluded_keys={example_key(example) for example in train_examples},
    )

    write_jsonl(args.output_dir / "train.jsonl", train_examples)
    write_jsonl(args.output_dir / "validation.jsonl", validation_examples)
    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    (args.output_dir / "generation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(train_examples)} training examples to {args.output_dir / 'train.jsonl'}")
    print(
        f"Wrote {len(validation_examples)} validation examples to "
        f"{args.output_dir / 'validation.jsonl'}"
    )


if __name__ == "__main__":
    main()
