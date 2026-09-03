#!/usr/bin/env python3
"""Create a fresh arithmetic test split disjoint from all existing splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from generate_arithmetic_dataset import example_key, generate_split, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/arithmetic"))
    parser.add_argument("--output", type=Path, default=Path("data/arithmetic/test_v2.jsonl"))
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size < 1:
        raise ValueError("--size must be positive.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing test split: {args.output}")

    excluded_keys = set()
    for split in ("train", "validation", "test"):
        path = args.source_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing existing split: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            excluded_keys.add(example_key(json.loads(line)))

    examples = generate_split(
        rng=random.Random(args.seed),
        split="test_v2",
        count=args.size,
        operations=["multiply"],
        min_digits=1,
        max_digits=3,
        excluded_keys=excluded_keys,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, examples)
    manifest = {
        "split": "test_v2",
        "output": str(args.output),
        "size": args.size,
        "seed": args.seed,
        "operations": ["multiply"],
        "digit_range": [1, 3],
        "disjoint_from": ["train", "validation", "test"],
    }
    manifest_path = args.output.with_name("test_v2_generation_config.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
