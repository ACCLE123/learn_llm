"""Dependency-free unit tests for the initial agent loop."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tau2_retail_agent.agent import AgentCore, TurnLimitReached
from tau2_retail_agent.backend import ToolCallParseError, parse_qwen_tool_calls
from tau2_retail_agent.models import Generation, Message, ToolCall, ToolSpec


class ScriptedBackend:
    def __init__(self, generations: list[Generation]) -> None:
        self.generations = generations
        self.requests: list[tuple[object, object]] = []

    def generate(self, messages: object, tools: object) -> Generation:
        self.requests.append((messages, tools))
        return self.generations.pop(0)


def order_lookup() -> ToolSpec:
    return ToolSpec(
        name="get_order",
        description="Look up an order by ID.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    )


class AgentCoreTests(unittest.TestCase):
    def test_records_policy_history_and_valid_tool_call(self) -> None:
        backend = ScriptedBackend(
            [Generation(tool_calls=(ToolCall(name="get_order", arguments={"order_id": "o-1"}),))]
        )
        agent = AgentCore(backend, [order_lookup()], "Verify the order before changes.")
        state = agent.get_init_state()

        turn = agent.generate_next_message(Message(role="user", content="Where is order o-1?"), state)

        self.assertEqual(turn.assistant_message.tool_calls[0].name, "get_order")
        self.assertEqual(state.decisions, 1)
        self.assertEqual(state.messages[0].role, "user")
        self.assertEqual(state.messages[1].role, "assistant")
        prompt, tools = backend.requests[0]
        self.assertIn("Verify the order", prompt[1].content)
        self.assertEqual(tools[0].name, "get_order")

    def test_rejects_unknown_tool_before_environment_execution(self) -> None:
        backend = ScriptedBackend(
            [Generation(tool_calls=(ToolCall(name="delete_everything", arguments={}),))]
        )
        agent = AgentCore(backend, [order_lookup()], "Never delete data.")
        state = agent.get_init_state()

        turn = agent.generate_next_message(Message(role="user", content="Help me."), state)

        self.assertEqual(turn.assistant_message.tool_calls, ())
        self.assertEqual(turn.rejected_calls[0].reason, "Unknown tool name.")
        self.assertIn("could not form", turn.assistant_message.content)

    def test_turn_limit_stops_unbounded_loop(self) -> None:
        backend = ScriptedBackend([Generation(content="I need more information.")])
        agent = AgentCore(backend, [order_lookup()], "Ask for details.", max_decisions=1)
        state = agent.get_init_state()
        agent.generate_next_message(Message(role="user", content="Hello"), state)

        with self.assertRaises(TurnLimitReached):
            agent.generate_next_message(Message(role="user", content="Any update?"), state)
        self.assertTrue(state.terminated)


class QwenParsingTests(unittest.TestCase):
    def test_parses_tool_call_and_preserves_text(self) -> None:
        generation = parse_qwen_tool_calls(
            "Let me check. <tool_call>{\"name\": \"get_order\", \"arguments\": {\"order_id\": \"o-1\"}}</tool_call>"
        )
        self.assertEqual(generation.content, "Let me check.")
        self.assertEqual(generation.tool_calls[0].arguments, {"order_id": "o-1"})

    def test_refuses_malformed_tool_json(self) -> None:
        with self.assertRaises(ToolCallParseError):
            parse_qwen_tool_calls("<tool_call>{not-json}</tool_call>")


if __name__ == "__main__":
    unittest.main()
