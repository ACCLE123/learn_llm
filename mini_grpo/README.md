# Minimal GRPO from scratch

This directory contains a deliberately small, readable GRPO implementation.
It is a learning implementation, not a replacement for `trl.GRPOTrainer`.

It implements the complete single-turn RLVR loop:

```text
prompt
  -> sample G completions with the LoRA policy
  -> deterministic reward per completion
  -> normalize rewards inside each prompt group
  -> recompute token log-probabilities
  -> clipped GRPO objective + reference-policy KL
  -> backward() and optimizer.step()
```

## What is deliberately included

- Qwen2.5-0.5B-Instruct with trainable LoRA adapters only;
- one short response as one rollout;
- exact-answer reward for the arithmetic training split;
- `G` completions per prompt and group-relative advantages;
- token-level masked policy loss, PPO-style clipping, and KL to the frozen base policy;
- a small JSONL training log and a saved final adapter.

## What is deliberately omitted

- distributed training, vLLM, rollout buffering, gradient accumulation;
- multiple reward functions, async rewards, checkpoint resume;
- value model / critic, GAE, and multi-step agent environments;
- validation and test evaluation. The trainer reads **only** `train.jsonl`.

The omissions make the correspondence to the math visible. For production
experiments, continue to use TRL.

## First verify the pure math

From the repository root:

```bash
conda run --no-capture-output -n llm python -m unittest mini_grpo/test_math.py -v
```

## Run a one-update smoke test

```bash
conda run --no-capture-output -n llm python -u mini_grpo/train_minimal_grpo.py \
  --train-limit 4 \
  --steps 1 \
  --batch-size 1 \
  --num-generations 4 \
  --output-dir mini_grpo/outputs/smoke
```

The script samples four answers for one training prompt, assigns four rewards,
calculates four advantages, averages their token losses, and updates LoRA once.
It never reads `validation.jsonl`, `test.jsonl`, or `test_v2.jsonl`.

## Source-map to TRL

| This teaching implementation | Corresponding TRL responsibility |
|---|---|
| `sample_rollouts` | `_generate_and_score_completions` |
| `exact_reward` | custom reward function invoked by `_calculate_rewards` |
| `group_advantages` | `rewards.view(-1, num_generations)` normalization |
| `completion_logps` | per-token log-probability computation |
| `grpo_loss` | `GRPOTrainer._compute_loss` |
| `optimizer.step()` in `train` | outer `Transformers Trainer` training loop |

The loss used here is, per sampled completion \(i\):

\[
L_i = \frac{1}{T_i}\sum_t\left[-\min(\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i)
+ \beta\,KL_{i,t}\right].
\]

The group advantage \(A_i\) is one scalar per completion and is broadcast over
that completion's valid generated tokens.
