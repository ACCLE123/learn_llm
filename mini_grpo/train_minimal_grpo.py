#!/usr/bin/env python3
"""A readable, single-turn GRPO + LoRA training loop.

This script intentionally implements only the algorithmic core needed to
understand GRPO. It is not a general-purpose or distributed trainer.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=REPOSITORY_ROOT / "data/arithmetic/train.jsonl",
    )
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "mini_grpo/outputs/run",
    )
    return parser.parse_args()


def device_for_training() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_train_rows(path: Path, limit: int) -> list[dict[str, str]]:
    """Load only the training split; answer is used exclusively by the verifier."""
    if limit < 1:
        raise ValueError("--train-limit must be positive.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return [{"prompt": row["prompt"], "answer": str(row["answer"])} for row in rows[:limit]]


def extract_tagged_answer(completion: str) -> str | None:
    """Keep the verifier local, so this directory is self-contained."""
    import re

    matches = re.findall(r"<answer>\s*(-?\d+)\s*</answer>", completion)
    return matches[0] if len(matches) == 1 else None


def exact_reward(completions: list[str], answer: str) -> Tensor:
    """One final, response-level reward for each sampled completion."""
    rewards = [float(extract_tagged_answer(text) == answer) for text in completions]
    return torch.tensor(rewards, dtype=torch.float32)


def group_advantages(rewards: Tensor, eps: float = 1e-4) -> Tensor:
    """Compute one standardized relative advantage per rollout in a prompt group."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("A GRPO group must contain at least two scalar rewards.")
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def completion_mask(completion_ids: Tensor, eos_token_id: int) -> Tensor:
    """Mask valid completion tokens, including the first EOS and excluding padding after it."""
    eos = completion_ids.eq(eos_token_id)
    eos_before = eos.cumsum(dim=-1) - eos.to(dtype=torch.long)
    return eos_before.eq(0).to(dtype=torch.float32)


def completion_logps(
    model: torch.nn.Module,
    prompt_ids: Tensor,
    completion_ids: Tensor,
) -> Tensor:
    """Return log pi(token_t | prompt, earlier completion tokens) for each completion token.

    All rows in this call share the same prompt, which keeps prompt padding out
    of this teaching implementation.
    """
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    logits = model(input_ids=input_ids, use_cache=False).logits
    prompt_length = prompt_ids.shape[1]
    token_logits = logits[:, prompt_length - 1 : -1, :]
    return token_logits.log_softmax(dim=-1).gather(
        dim=-1,
        index=completion_ids.unsqueeze(-1),
    ).squeeze(-1)


def sample_rollouts(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device, 
) -> tuple[Tensor, Tensor, list[str]]:
    """Sample G full responses from one prompt using the current policy."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_inputs = tokenizer(rendered, return_tensors="pt").to(device)
    prompt_ids = prompt_inputs["input_ids"] # 1 * L
    prompt_attention_mask = prompt_inputs["attention_mask"]
    repeated_prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0) # G * L
    repeated_attention_mask = prompt_attention_mask.repeat_interleave(num_generations, dim=0)

    model.eval()
    # Use no_grad, rather than inference_mode: the completion IDs are later
    # used as targets in a graph that computes trainable policy log-probabilities.
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=repeated_prompt_ids,
            attention_mask=repeated_attention_mask,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ) # G * (L + T)
    completion_ids = generated_ids[:, prompt_ids.shape[1] :] # G * T
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True) 
    return repeated_prompt_ids, completion_ids, completions


def reference_logps(
    model: torch.nn.Module,
    prompt_ids: Tensor,
    completion_ids: Tensor,
) -> Tensor:
    """Score completions under the frozen base policy, with LoRA disabled."""
    model.eval()
    with torch.inference_mode(), model.disable_adapter():
        return completion_logps(model, prompt_ids, completion_ids)


def grpo_loss(
    policy_logps: Tensor, # G * T
    old_policy_logps: Tensor, # G * T
    ref_logps: Tensor, # G * T
    advantages: Tensor, # G
    mask: Tensor, # G * T
    beta: float,
    clip_epsilon: float,
) -> tuple[Tensor, dict[str, float]]:
    """Compute clipped GRPO loss for one prompt group.

    `advantages` has shape [G]. Unsqueezing it broadcasts the rollout-level
    signal across every valid completion token, shape [G, T].
    """
    if advantages.ndim != 1:
        raise ValueError("Expected one scalar advantage per rollout.")

    advantage_per_token = advantages.unsqueeze(1) # G * 1
    ratio = torch.exp(policy_logps - old_policy_logps) # r (G * T)
    unclipped_objective = ratio * advantage_per_token # A * r (G * T)
    clipped_objective = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage_per_token #  A * clip(r) (G * T)
    policy_loss_per_token = -torch.minimum(unclipped_objective, clipped_objective)

    # A differentiable, non-negative estimate of KL(policy || reference).
    per_token_kl = torch.exp(ref_logps - policy_logps) - (ref_logps - policy_logps) - 1 # G * T
    per_token_loss = policy_loss_per_token + beta * per_token_kl #  G * T
    per_rollout_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1) #  G * T
    loss = per_rollout_loss.mean() # 1 * 1

    with torch.no_grad():
        valid_tokens = mask.sum().clamp_min(1)
        mean_kl = ((per_token_kl * mask).sum() / valid_tokens).item()
        clip_fraction = (((ratio - 1).abs() > clip_epsilon).to(mask.dtype) * mask).sum() / valid_tokens
    return loss, {"kl": mean_kl, "clip_fraction": clip_fraction.item()} # scalar


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        # Keeping dropout at zero makes rollout-time and loss-time policy
        # probabilities directly comparable in this small implementation.
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def train(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.num_generations < 2:
        raise ValueError("GRPO requires --num-generations >= 2.")
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = device_for_training()
    rows = load_train_rows(args.train_data, args.train_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    model.config.use_cache = False
    model = get_peft_model(model, lora_config())
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    print(f"Trainable LoRA parameters: {trainable_parameters:,} / {total_parameters:,}")

    logs: list[dict[str, Any]] = []
    for step in range(args.steps):
        started = time.perf_counter()
        batch = [rows[(step * args.batch_size + offset) % len(rows)] for offset in range(args.batch_size)]
        group_losses: list[Tensor] = []
        group_logs: list[dict[str, Any]] = []

        for row in batch:
            prompt_ids, completion_ids, completions = sample_rollouts(
                model=model,
                tokenizer=tokenizer,
                prompt=row["prompt"],
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
            )
            rewards = exact_reward(completions, row["answer"]).to(device)
            advantages = group_advantages(rewards)
            mask = completion_mask(completion_ids, tokenizer.eos_token_id)

            # These are rollout-time log probabilities. They are detached, so
            # PPO/GRPO's ratio compares the trainable current policy to a fixed
            # snapshot of the policy that generated this group.
            model.train()
            policy_logps = completion_logps(model, prompt_ids, completion_ids)
            old_policy_logps = policy_logps.detach()
            ref_logps = reference_logps(model, prompt_ids, completion_ids)
            loss, loss_metrics = grpo_loss(
                policy_logps=policy_logps,
                old_policy_logps=old_policy_logps,
                ref_logps=ref_logps,
                advantages=advantages,
                mask=mask,
                beta=args.beta,
                clip_epsilon=args.clip_epsilon,
            )
            group_losses.append(loss)
            group_logs.append(
                {
                    "rewards": rewards.detach().cpu().tolist(),
                    "advantages": advantages.detach().cpu().tolist(),
                    "completions": completions,
                    **loss_metrics,
                }
            )
        # group_losse
        # All prompt-group losses form one scalar objective and one update.
        batch_loss = torch.stack(group_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        record = {
            "step": step + 1,
            "loss": batch_loss.detach().item(),
            "grad_norm": float(grad_norm),
            "seconds": time.perf_counter() - started,
            "groups": group_logs,
        }
        logs.append(record)
        print(json.dumps(record, ensure_ascii=False))

    model.save_pretrained(args.output_dir / "final_adapter")
    tokenizer.save_pretrained(args.output_dir / "final_adapter")
    (args.output_dir / "training_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return logs


def main() -> None:
    args = parse_args()
    logs = train(args)
    print(f"Completed {len(logs)} updates. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
