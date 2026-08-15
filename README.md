# GRPO arithmetic-data experiment

Generate a reproducible, automatically verifiable dataset for a first GRPO
experiment with Qwen2.5-0.5B.

```bash
python3 generate_arithmetic_dataset.py
```

This writes `data/arithmetic/train.jsonl` (5,000 examples),
`data/arithmetic/validation.jsonl` (500 examples), and
`data/arithmetic/test.jsonl` (500 examples). The splits are disjoint by
operation and operands (including commutative equivalents). Default questions
are multiplication with one- to three-digit operands.

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

## Baseline evaluation

The baseline is inference-only: it measures the original model before any GRPO
training. Install the dependencies, then start with 20 questions to verify the
model download and output format:

```bash
python3 -m pip install -r requirements.txt
python3 evaluate_baseline.py --limit 20
```

Run the full 500-question validation set once the smoke test passes:

```bash
python3 evaluate_baseline.py
```

For the fixed, held-out test baseline, pass the test file explicitly:

```bash
python3 evaluate_baseline.py \
  --data data/arithmetic/test.jsonl \
  --output outputs/test_baseline_results.jsonl \
  --summary outputs/test_baseline_summary.json
```

The script selects CUDA, Apple Metal (MPS), or CPU automatically. It saves one
prediction per line in `outputs/baseline_results.jsonl` and aggregate accuracy
plus format-validity rate in `outputs/baseline_summary.json`.

## GRPO reward contract

`reward.py` defines `exact_answer_reward(completion, reference_answer)`. It
returns `1.0` only when the completion contains exactly one integer
`<answer>...</answer>` tag equal to the reference answer; otherwise it returns
`0.0`. Test this contract before connecting it to a trainer:

```bash
python3 -m unittest tests/test_reward.py -v
```

## GRPO dry run

The dry run reads only `train.jsonl` and `validation.jsonl`; it never opens the
held-out test split. It uses LoRA, 32 available training rows, four sampled
completions per prompt, and eight update steps.

```bash
conda run -n llm python run_grpo_dry_run.py
```

Outputs are written to `outputs/grpo_dry_run/`, including a LoRA adapter and a
JSON summary with training reward logs plus validation metrics before and after
training.

## Selecting a GRPO checkpoint

Evaluate saved LoRA checkpoints on validation only, then select the highest
accuracy checkpoint before running the held-out test evaluation:

```bash
conda run --no-capture-output -n llm python -u evaluate_grpo_checkpoints.py
```
