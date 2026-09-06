"""Tests for cross-task trajectory failure labels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tau2_retail_agent.failure_analysis import build_failure_report, classify_record


class FailureAnalysisTests(unittest.TestCase):
    def test_labels_timeout_without_tools(self) -> None:
        record = {
            "task_id": "a",
            "reward": 0.0,
            "termination_reason": "agent_stop",
            "tool_result_errors": 0,
            "reward_breakdown": {"DB": 0.0},
            "simulation": {"messages": [{"role": "assistant", "content": "###STOP###"}]},
        }
        self.assertEqual(
            classify_record(record),
            ["agent_stop_or_generation_timeout", "no_tool_action", "environment_goal_not_met"],
        )

    def test_aggregates_heterogeneous_failures(self) -> None:
        report = build_failure_report(
            [
                {
                    "task_id": "a",
                    "reward": 1.0,
                    "termination_reason": "user_stop",
                    "simulation": {"messages": []},
                },
                {
                    "task_id": "b",
                    "reward": 0.0,
                    "termination_reason": "user_stop",
                    "tool_result_errors": 1,
                    "reward_breakdown": {"DB": 0.0, "NL_ASSERTION": 0.0},
                    "simulation": {
                        "messages": [
                            {"role": "assistant", "content": "", "tool_calls": [{"name": "x"}]}
                        ]
                    },
                },
            ]
        )
        self.assertEqual(report["tasks_analyzed"], 2)
        self.assertEqual(report["label_counts"]["benchmark_reward_success"], 1)
        self.assertEqual(report["label_counts"]["behavioral_success"], 1)
        self.assertEqual(report["label_counts"]["tool_execution_error"], 1)
        self.assertEqual(report["label_counts"]["communication_requirement_not_met"], 1)

    def test_distinguishes_reward_from_clean_execution(self) -> None:
        report = build_failure_report(
            [
                {
                    "task_id": "a",
                    "reward": 1.0,
                    "termination_reason": "user_stop",
                    "tool_result_errors": 2,
                    "simulation": {"messages": []},
                }
            ]
        )

        self.assertEqual(report["benchmark_reward_successes"], 1)
        self.assertEqual(report["behavioral_successes"], 0)
        self.assertEqual(report["label_counts"]["reward_success_with_execution_issues"], 1)


if __name__ == "__main__":
    unittest.main()
