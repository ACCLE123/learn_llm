"""τ² integration tests; skipped outside the dedicated tau2-agent environment."""

from __future__ import annotations

import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TAU2_AVAILABLE = find_spec("tau2") is not None

if TAU2_AVAILABLE:
    from tau2.data_model.message import ToolMessage, UserMessage
    from tau2.environment.tool import as_tool

    from tau2_retail_agent.models import Generation, ToolCall
    from tau2_retail_agent.tau2_adapter import Tau2QwenAgent


class ScriptedBackend:
    def __init__(self, generations):
        self.generations = list(generations)

    def generate(self, messages, tools):
        if any(
            message.role == "system" and "private planner" in message.content.lower()
            for message in messages
        ):
            return Generation(content='{"mode":"read","missing_facts":["order"]}')
        return self.generations.pop(0)


@unittest.skipUnless(TAU2_AVAILABLE, "requires the tau2-agent Conda environment")
class Tau2AdapterTests(unittest.TestCase):
    def test_converts_a_valid_tool_call_to_tau2_message(self) -> None:
        def get_order(order_id: str) -> str:
            """Look up a Retail order."""

            return order_id

        backend = ScriptedBackend(
            [
                Generation(
                    tool_calls=(ToolCall(name="get_order", arguments={"order_id": "o-1"}),)
                ),
                Generation(content="The order is shipped."),
            ]
        )
        agent = Tau2QwenAgent(
            tools=[as_tool(get_order)], domain_policy="Use tools for order facts.", backend=backend
        )
        state = agent.get_init_state()

        response, state = agent.generate_next_message(UserMessage.text("Find order o-1."), state)

        self.assertEqual(response.content, "")
        self.assertEqual(response.tool_calls[0].name, "get_order")
        self.assertEqual(response.tool_calls[0].id, "call_1_1")

        response, state = agent.generate_next_message(
            ToolMessage(id="call_1_1", role="tool", content="Order is shipped."), state
        )
        self.assertEqual(response.content, "The order is shipped.")
        self.assertEqual(state.decisions, 2)


if __name__ == "__main__":
    unittest.main()
