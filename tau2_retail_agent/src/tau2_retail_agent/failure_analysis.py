"""Cross-task failure labels for τ² trajectory artifacts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def is_behavioral_success(record: dict[str, Any]) -> bool:
    """Return a conservative success signal independent of reward alone.

    A benchmark reward can be awarded when a simulator stops for reasons that
    do not demonstrate a clean agent execution. This metric therefore requires
    a perfect reward, no environment tool errors, and a non-budget termination.
    """

    return (
        record.get("reward") == 1.0
        and not record.get("tool_result_errors", 0)
        and record.get("termination_reason") not in {"agent_stop", "max_steps"}
    )


def classify_record(record: dict[str, Any]) -> list[str]:
    """Return non-exclusive, benchmark-agnostic labels for one trajectory."""

    labels: list[str] = []
    simulation = record.get("simulation") or {}
    messages = simulation.get("messages") or []
    reward_breakdown = record.get("reward_breakdown") or {}
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    tool_calls = [
        call
        for message in assistant_messages
        for call in (message.get("tool_calls") or [])
    ]
    final_content = str(assistant_messages[-1].get("content") or "") if assistant_messages else ""

    if record.get("reward") == 1.0:
        labels.append("benchmark_reward_success")
        labels.append(
            "behavioral_success" if is_behavioral_success(record) else "reward_success_with_execution_issues"
        )
    if "###STOP###" in final_content or record.get("termination_reason") == "agent_stop":
        labels.append("agent_stop_or_generation_timeout")
    if record.get("termination_reason") == "max_steps":
        labels.append("max_steps_exhausted")
    if not tool_calls:
        labels.append("no_tool_action")
    if record.get("tool_result_errors", 0):
        labels.append("tool_execution_error")
    if reward_breakdown.get("DB") == 0.0:
        labels.append("environment_goal_not_met")
    if reward_breakdown.get("NL_ASSERTION") == 0.0:
        labels.append("communication_requirement_not_met")
    if not labels:
        labels.append("uncategorized_failure")
    return labels


def build_failure_report(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate labels over heterogeneous tasks without task-specific rules."""

    task_reports = []
    counter: Counter[str] = Counter()
    benchmark_reward_successes = 0
    behavioral_successes = 0
    for record in records:
        labels = classify_record(record)
        behavioral_success = is_behavioral_success(record)
        benchmark_reward_successes += record.get("reward") == 1.0
        behavioral_successes += behavioral_success
        counter.update(labels)
        task_reports.append(
            {
                "task_id": record.get("task_id"),
                "reward": record.get("reward"),
                "behavioral_success": behavioral_success,
                "termination_reason": record.get("termination_reason"),
                "labels": labels,
            }
        )
    return {
        "tasks_analyzed": len(task_reports),
        "benchmark_reward_successes": benchmark_reward_successes,
        "behavioral_successes": behavioral_successes,
        "label_counts": dict(sorted(counter.items())),
        "tasks": task_reports,
    }
