# GRPO formal experiment v1

The arithmetic dataset is frozen. This run reads only `train.jsonl` and
`validation.jsonl`; it does not open `test.jsonl`.

| Setting | Value |
|---|---:|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Train examples | 5,000 |
| Validation examples | 500 |
| Updates | 1,250 (one epoch at batch size 4) |
| Candidates per prompt | 4 |
| LoRA rank / alpha | 8 / 16 |
| Learning rate | `2e-5` |
| Warmup | 50 updates |
| KL coefficient (`beta`) | `0.04` |
| Completion limit | 32 tokens |
| Checkpoints | every 250 updates plus final adapter |

Run command:

```bash
conda run -n llm python run_grpo_dry_run.py \
  --train-limit 5000 --validation-limit 500 \
  --max-steps 1250 --num-generations 4 \
  --learning-rate 2e-5 --warmup-steps 50 --save-steps 250 \
  --output-dir outputs/grpo_formal_v1
```

After training, use validation results to select the checkpoint. Evaluate the
held-out test set exactly once for the selected final model and compare it to
the recorded 19.0% baseline.
