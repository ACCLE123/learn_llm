#!/usr/bin/env python3
"""Evaluate the local Qwen agent on a fixed, small τ² Retail subset.

Retail needs τ²'s LLM-powered user simulator. Pass it explicitly with
``--user-llm``. Tasks with natural-language assertions also need a judge;
pass ``--judge-llm`` to avoid τ²'s default OpenAI judge.
The tasks run sequentially because one local Qwen instance is reused.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tau2.domains.retail.environment import get_environment
from tau2.evaluator.evaluator import EvaluationType
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.run import get_tasks
from tau2.runner.simulation import run_simulation
from tau2.user.user_simulator import UserSimulator

from tau2_retail_agent.backend import TransformersQwenBackend
from tau2_retail_agent.failure_analysis import build_failure_report
from tau2_retail_agent.tau2_adapter import Tau2QwenAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-1.7B", help="Local Qwen checkpoint.")
    parser.add_argument("--user-llm", help="LiteLLM model for τ²'s user simulator.")
    parser.add_argument(
        "--judge-llm",
        help=(
            "LiteLLM model for τ² natural-language-assertion evaluation. "
            "For example: deepseek/deepseek-chat. Without this flag, τ² uses its "
            "default OpenAI judge for tasks that need one."
        ),
    )
    parser.add_argument("--split", default="train", choices=("train", "test", "base"))
    parser.add_argument("--limit", type=int, default=5, help="Number of selected tasks.")
    parser.add_argument(
        "--selection",
        choices=("leading", "diverse"),
        default="leading",
        help="Use leading task IDs or diversify by evaluation-action coverage.",
    )
    parser.add_argument(
        "--task-ids",
        help="Comma-separated official task IDs. Overrides --selection and --limit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-decisions", type=int, default=12)
    parser.add_argument(
        "--candidate-tools",
        type=int,
        default=4,
        help="Maximum schema-retrieved tools shown to Qwen for one decision.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-generation-seconds", type=float, default=90.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--disable-structured-planning",
        action="store_true",
        help="Run the PAOV-v1 ablation without the model-generated private plan.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/retail_baseline"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print task selection without loading Qwen or calling an API.",
    )
    return parser.parse_args()


def safe_user_tools(environment: Any, task: Any) -> list[Any] | None:
    try:
        return environment.get_user_tools(include=task.user_tools) or None
    except ValueError:
        return None


def select_tasks(args: argparse.Namespace) -> list[Any]:
    """Select reproducible leading tasks or a generic action-diverse subset."""

    explicit_ids = [task_id.strip() for task_id in (args.task_ids or "").split(",") if task_id.strip()]
    if explicit_ids:
        return get_tasks("retail", task_split_name=args.split, task_ids=explicit_ids)
    tasks = get_tasks("retail", task_split_name=args.split)
    if args.selection == "leading":
        return tasks[: args.limit]

    remaining = list(tasks)
    selected: list[Any] = []
    seen_actions: set[str] = set()
    while remaining and len(selected) < args.limit:
        def novelty(task: Any) -> int:
            criteria = getattr(task, "evaluation_criteria", None)
            actions = getattr(criteria, "actions", []) if criteria else []
            return len({action.name for action in actions} - seen_actions)

        best_index, best_task = max(enumerate(remaining), key=lambda pair: (novelty(pair[1]), -pair[0]))
        selected.append(best_task)
        criteria = getattr(best_task, "evaluation_criteria", None)
        seen_actions.update(action.name for action in (getattr(criteria, "actions", []) or []))
        remaining.pop(best_index)
    return selected


def tool_error_count(simulation: Any) -> int:
    return sum(
        bool(getattr(message, "error", False))
        for message in (simulation.messages or [])
        if message.role == "tool"
    )


def configure_nl_judge(model: str | None) -> None:
    """Override only the evaluator's judge model, without editing τ² vendor code."""
    if not model:
        return
    import tau2.evaluator.evaluator_nl_assertions as nl_assertions_evaluator

    nl_assertions_evaluator.DEFAULT_LLM_NL_ASSERTIONS = model


def task_record(task: Any, simulation: Any, agent_state: Any) -> dict[str, Any]:
    reward_info = simulation.reward_info
    return {
        "task_id": task.id,
        "reward": reward_info.reward if reward_info else None,
        "reward_breakdown": reward_info.reward_breakdown if reward_info else None,
        "termination_reason": simulation.termination_reason.value,
        "duration_seconds": simulation.duration,
        "tool_result_errors": tool_error_count(simulation),
        "raw_model_generations": agent_state.raw_generations,
        "workflow": {
            "plans": [asdict(plan) for plan in agent_state.plans],
            "structured_plans": [asdict(plan) for plan in agent_state.structured_plans],
            "observations": [asdict(observation) for observation in agent_state.observations],
        },
        "simulation": simulation.model_dump(mode="json"),
    }


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    tasks = select_tasks(args)
    selection = {
        "split": args.split,
        "seed": args.seed,
        "task_ids": [task.id for task in tasks],
        "judge_llm": args.judge_llm,
        "agent_architecture": (
            "paov_v1_evidence_recovery"
            if args.disable_structured_planning
            else "paov_v2_structured_planning"
        ),
        "candidate_tools": args.candidate_tools,
        "selection": "explicit" if args.task_ids else args.selection,
    }

    if args.dry_run:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return
    if not args.user_llm:
        raise SystemExit(
            "Retail requires τ²'s LLM user simulator. Re-run with --user-llm <provider/model>, "
            "or use --dry-run to inspect the task selection without external calls."
        )

    configure_nl_judge(args.judge_llm)

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Loading {args.model} once for {len(tasks)} Retail tasks ...", flush=True)
    backend = TransformersQwenBackend.from_pretrained(
        args.model,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        max_generation_seconds=args.max_generation_seconds,
    )
    print(f"Model device: {backend.device}; thinking: {args.enable_thinking}", flush=True)
    records: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] task={task.id}", flush=True)
        environment = get_environment()
        agent = Tau2QwenAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            backend=backend,
            max_decisions=args.max_decisions,
            candidate_tool_limit=args.candidate_tools,
            enable_structured_planning=not args.disable_structured_planning,
        )
        user = UserSimulator(
            llm=args.user_llm,
            instructions=str(task.user_scenario),
            tools=safe_user_tools(environment, task),
        )
        orchestrator = Orchestrator(
            domain="retail",
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=args.max_steps,
            seed=args.seed,
            validate_communication=True,
            timeout=args.timeout_seconds,
        )
        simulation = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)
        record = task_record(task, simulation, orchestrator.agent_state)
        records.append(record)
        (run_dir / f"task_{task.id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    rewards = [record["reward"] for record in records if record["reward"] is not None]
    failure_report = build_failure_report(records)
    summary = {
        **selection,
        "model": args.model,
        "user_llm": args.user_llm,
        "tasks_completed": len(records),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "success_rate": sum(reward == 1.0 for reward in rewards) / len(rewards) if rewards else None,
        "behavioral_success_rate": (
            failure_report["behavioral_successes"] / len(records) if records else None
        ),
        "total_tool_result_errors": sum(record["tool_result_errors"] for record in records),
        "run_dir": str(run_dir),
    }
    (run_dir / "failure_report.json").write_text(
        json.dumps(failure_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
