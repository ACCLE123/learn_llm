"""Cross-task failure labels for τ² trajectory artifacts."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


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
        return ["success"]
    if "###STOP###" in final_content or record.get("termination_reason") == "agent_stop":
        labels.append("agent_stop_or_generation_timeout")
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
    for record in records:
        labels = classify_record(record)
        counter.update(labels)
        task_reports.append(
            {
                "task_id": record.get("task_id"),
                "reward": record.get("reward"),
                "termination_reason": record.get("termination_reason"),
                "labels": labels,
            }
        )
    return {
        "tasks_analyzed": len(task_reports),
        "label_counts": dict(sorted(counter.items())),
        "tasks": task_reports,
    }
