# GRPO arithmetic-data experiment

Generate a reproducible, automatically verifiable dataset for a first GRPO
experiment with Qwen2.5-0.5B.

```bash
python3 generate_arithmetic_dataset.py
```

This writes `data/arithmetic/train.jsonl` (5,000 examples) and
`data/arithmetic/validation.jsonl` (500 examples).  The splits are disjoint by
operation and operands. Default questions are multiplication with one- to
three-digit operands.

Every row has the following shape:

```json
{
  "id": "train-000000",
  "split": "train",
  "prompt": "请计算下列算式。只输出 `<answer>整数</answer>`，不要解释。\n问题：37 × 48",
  "answer": "1776",
  "metadata": {"operation": "multiply", "left_operand": 37, "right_operand": 48}
}
```

During GRPO generation, supply only `prompt` to the model. The reward function
can extract the value between `<answer>` tags and compare it exactly with
`answer`.

For a mixed-operation dataset:

```bash
python3 generate_arithmetic_dataset.py \
  --operations add subtract multiply divide \
  --min-digits 1 --max-digits 2 \
  --train-size 10000 --validation-size 1000 \
  --output-dir data/arithmetic_mixed
```
