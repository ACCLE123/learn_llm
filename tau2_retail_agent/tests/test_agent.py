"""Dependency-free unit tests for the initial agent loop."""

from __future__ import annotations

import json
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


def tool(name: str, *, kind: str = "generic", description: str | None = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description or name,
        parameters={"type": "object", "properties": {}},
        kind=kind,  # type: ignore[arg-type]
    )


class AgentCoreTests(unittest.TestCase):
    def test_retrieves_a_small_schema_relevant_subset(self) -> None:
        backend = ScriptedBackend([Generation(content="I will look it up.")])
        agent = AgentCore(
            backend,
            [
                tool("get_order", kind="read", description="Find order details by order id."),
                tool("cancel_order", kind="write", description="Cancel an order."),
                tool("list_catalog", kind="read", description="List product catalogue."),
                tool("get_user", kind="read", description="Look up a customer profile."),
                tool("return_items", kind="write", description="Return delivered items."),
            ],
            "Use tools for facts.",
            candidate_tool_limit=2,
        )
        state = agent.get_init_state()

        agent.generate_next_message(Message(role="user", content="Please find my order."), state)

        _, visible_tools = backend.requests[0]
        self.assertEqual(len(visible_tools), 2)
        self.assertEqual(visible_tools[0].name, "get_order")
        self.assertEqual(state.plans[-1].phase, "discover")

    def test_blocks_write_before_confirmation(self) -> None:
        backend = ScriptedBackend(
            [Generation(tool_calls=(ToolCall(name="change_record", arguments={}),))]
        )
        agent = AgentCore(backend, [tool("change_record", kind="write")], "Ask first.")
        state = agent.get_init_state()

        turn = agent.generate_next_message(Message(role="user", content="Change my record."), state)

        self.assertEqual(turn.assistant_message.tool_calls, ())
        self.assertIn("confirmation", turn.rejected_calls[0].reason)

    def test_allows_confirmed_write_then_enters_verify_phase(self) -> None:
        backend = ScriptedBackend(
            [
                Generation(content="Please confirm that you want to proceed."),
                Generation(tool_calls=(ToolCall(name="change_record", arguments={}),)),
                Generation(tool_calls=(ToolCall(name="get_record", arguments={}),)),
            ]
        )
        agent = AgentCore(
            backend,
            [tool("change_record", kind="write"), tool("get_record", kind="read")],
            "Ask first.",
        )
        state = agent.get_init_state()

        agent.generate_next_message(Message(role="user", content="Change my record."), state)
        write_turn = agent.generate_next_message(Message(role="user", content="Yes, proceed."), state)
        self.assertEqual(write_turn.assistant_message.tool_calls[0].name, "change_record")
        verify_turn = agent.generate_next_message(
            Message(role="tool", content="updated", tool_call_id="call_2_1"), state
        )

        self.assertEqual(state.plans[-1].phase, "verify")
        _, visible_tools = backend.requests[-1]
        self.assertEqual([tool.name for tool in visible_tools], ["get_record"])
        self.assertEqual(verify_turn.assistant_message.tool_calls[0].name, "get_record")

    def test_consumes_confirmation_after_a_write(self) -> None:
        backend = ScriptedBackend(
            [
                Generation(content="Please confirm that you want to proceed."),
                Generation(tool_calls=(ToolCall(name="change_record", arguments={}),)),
            ]
        )
        agent = AgentCore(backend, [tool("change_record", kind="write")], "Ask first.")
        state = agent.get_init_state()

        agent.generate_next_message(Message(role="user", content="Change my record."), state)
        agent.generate_next_message(Message(role="user", content="Yes, proceed."), state)

        self.assertFalse(state.awaiting_confirmation)

    def test_discover_phase_expands_to_all_read_tools_after_fact_collection(self) -> None:
        backend = ScriptedBackend(
            [
                Generation(tool_calls=(ToolCall(name="get_order", arguments={}),)),
                Generation(content="Please confirm the proposed update."),
            ]
        )
        agent = AgentCore(
            backend,
            [
                tool("get_order", kind="read"),
                tool("get_catalog", kind="read"),
                tool("change_record", kind="write"),
            ],
            "Gather facts before changes.",
            candidate_tool_limit=1,
        )
        state = agent.get_init_state()

        first = agent.generate_next_message(Message(role="user", content="Find my order."), state)
        agent.generate_next_message(
            Message(role="tool", content="order found", tool_call_id=first.assistant_message.tool_calls[0].call_id),
            state,
        )

        self.assertEqual(state.plans[-1].phase, "discover")
        _, visible_tools = backend.requests[-1]
        self.assertEqual([candidate.name for candidate in visible_tools], ["get_order", "get_catalog"])

    def test_ungrounded_write_enters_read_only_recovery(self) -> None:
        backend = ScriptedBackend(
            [
                Generation(tool_calls=(ToolCall(name="change_record", arguments={}),)),
                Generation(content="I will inspect the record."),
            ]
        )
        agent = AgentCore(
            backend,
            [
                ToolSpec(
                    name="change_record",
                    description="Change a record by ID.",
                    parameters={"type": "object", "properties": {"record_id": {"type": "string"}}},
                    kind="write",
                ),
                tool("get_record", kind="read"),
            ],
            "Use facts.",
        )
        state = agent.get_init_state()
        backend.generations[0] = Generation(
            tool_calls=(ToolCall(name="change_record", arguments={"record_id": "invented"}),)
        )

        rejected = agent.generate_next_message(Message(role="user", content="Change my record."), state)
        agent.generate_next_message(Message(role="user", content="Please continue."), state)

        self.assertIn("not grounded", rejected.rejected_calls[0].reason)
        self.assertTrue(state.recovery_required)
        self.assertEqual(state.plans[-1].phase, "recover")
        _, visible_tools = backend.requests[-1]
        self.assertEqual([candidate.name for candidate in visible_tools], ["get_record"])

    def test_failed_write_clears_confirmation_and_enters_recovery(self) -> None:
        backend = ScriptedBackend(
            [
                Generation(content="Please confirm that you want to proceed."),
                Generation(tool_calls=(ToolCall(name="change_record", arguments={}),)),
                Generation(content="I will inspect the current record."),
            ]
        )
        agent = AgentCore(
            backend,
            [tool("change_record", kind="write"), tool("get_record", kind="read")],
            "Use facts.",
        )
        state = agent.get_init_state()

        agent.generate_next_message(Message(role="user", content="Change my record."), state)
        write_turn = agent.generate_next_message(Message(role="user", content="Yes, proceed."), state)
        agent.generate_next_message(
            Message(
                role="tool",
                content="Error: update rejected",
                tool_call_id=write_turn.assistant_message.tool_calls[0].call_id,
            ),
            state,
        )

        self.assertFalse(state.awaiting_confirmation)
        self.assertTrue(state.recovery_required)
        self.assertEqual(state.plans[-1].phase, "recover")
        _, visible_tools = backend.requests[-1]
        self.assertEqual([candidate.name for candidate in visible_tools], ["get_record"])

    def test_retries_verification_when_model_claims_completion_without_a_read(self) -> None:
        backend = ScriptedBackend(
            [
                Generation(content="Please confirm that you want to proceed."),
                Generation(tool_calls=(ToolCall(name="change_record", arguments={}),)),
                Generation(content="The record has been updated."),
                Generation(tool_calls=(ToolCall(name="get_record", arguments={}),)),
            ]
        )
        agent = AgentCore(
            backend,
            [tool("change_record", kind="write"), tool("get_record", kind="read")],
            "Ask first.",
        )
        state = agent.get_init_state()

        agent.generate_next_message(Message(role="user", content="Change my record."), state)
        agent.generate_next_message(Message(role="user", content="Yes, proceed."), state)
        verify_turn = agent.generate_next_message(
            Message(role="tool", content="updated", tool_call_id="call_2_1"), state
        )

        self.assertEqual(verify_turn.assistant_message.tool_calls[0].name, "get_record")
        self.assertEqual(len(backend.requests), 4)
        retry_prompt, retry_tools = backend.requests[-1]
        self.assertIn("not yet verified", retry_prompt[-1].content)
        self.assertEqual([candidate.name for candidate in retry_tools], ["get_record"])

    def test_retries_verification_after_an_invalid_tool_call(self) -> None:
        backend = ScriptedBackend(
            [
                Generation(content="Please confirm that you want to proceed."),
                Generation(tool_calls=(ToolCall(name="change_record", arguments={}),)),
                Generation(content="The record has been updated."),
                Generation(tool_calls=(ToolCall(name="get_records", arguments={}),)),
                Generation(tool_calls=(ToolCall(name="get_record", arguments={}),)),
            ]
        )
        agent = AgentCore(
            backend,
            [tool("change_record", kind="write"), tool("get_record", kind="read")],
            "Ask first.",
        )
        state = agent.get_init_state()

        agent.generate_next_message(Message(role="user", content="Change my record."), state)
        agent.generate_next_message(Message(role="user", content="Yes, proceed."), state)
        verify_turn = agent.generate_next_message(
            Message(role="tool", content="updated", tool_call_id="call_2_1"), state
        )

        self.assertEqual(verify_turn.assistant_message.tool_calls[0].name, "get_record")
        self.assertEqual(len(backend.requests), 5)
        repair_prompt, repair_tools = backend.requests[-1]
        self.assertIn("get_records", repair_prompt[-1].content)
        self.assertIn("get_record", repair_prompt[-1].content)
        self.assertEqual([candidate.name for candidate in repair_tools], ["get_record"])

    def test_records_policy_history_and_valid_tool_call(self) -> None:
        backend = ScriptedBackend(
            [Generation(tool_calls=(ToolCall(name="get_order", arguments={"order_id": "o-1"}),))]
        )
        agent = AgentCore(backend, [order_lookup()], "Verify the order before changes.")
        state = agent.get_init_state()

        turn = agent.generate_next_message(Message(role="user", content="Where is order o-1?"), state)

        self.assertEqual(turn.assistant_message.tool_calls[0].name, "get_order")
        self.assertEqual(turn.assistant_message.tool_calls[0].call_id, "call_1_1")
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
        self.assertIn("valid available tool", turn.assistant_message.content)

    def test_turn_limit_stops_unbounded_loop(self) -> None:
        backend = ScriptedBackend([Generation(content="I need more information.")])
        agent = AgentCore(backend, [order_lookup()], "Ask for details.", max_decisions=1)
        state = agent.get_init_state()
        agent.generate_next_message(Message(role="user", content="Hello"), state)

        with self.assertRaises(TurnLimitReached):
            agent.generate_next_message(Message(role="user", content="Any update?"), state)
        self.assertTrue(state.terminated)

    def test_appends_multiple_tool_results_before_deciding(self) -> None:
        backend = ScriptedBackend([Generation(content="Both updates are complete.")])
        agent = AgentCore(backend, [order_lookup()], "Report completed changes.")
        state = agent.get_init_state()

        agent.generate_after_messages(
            [
                Message(role="tool", content="First tool result", tool_call_id="call_1_1"),
                Message(role="tool", content="Second tool result", tool_call_id="call_1_2"),
            ],
            state,
        )

        prompt, _ = backend.requests[0]
        self.assertEqual(prompt[-2].content, "First tool result")
        self.assertEqual(prompt[-1].content, "Second tool result")

    def test_tool_call_output_never_mixes_text_and_action(self) -> None:
        backend = ScriptedBackend(
            [
                Generation(
                    content="I will look it up.",
                    tool_calls=(ToolCall(name="get_order", arguments={"order_id": "o-1"}),),
                )
            ]
        )
        agent = AgentCore(backend, [order_lookup()], "Use tools for facts.")
        state = agent.get_init_state()

        turn = agent.generate_next_message(Message(role="user", content="Find o-1"), state)

        self.assertEqual(turn.assistant_message.content, "")
        self.assertEqual(turn.assistant_message.tool_calls[0].call_id, "call_1_1")

    def test_hides_thinking_but_keeps_raw_generation_in_state(self) -> None:
        raw = "<think>private chain of thought</think>\nThe order is confirmed."
        backend = ScriptedBackend([Generation(content=raw, raw_content=raw)])
        agent = AgentCore(backend, [order_lookup()], "Respond clearly.")
        state = agent.get_init_state()

        turn = agent.generate_next_message(Message(role="user", content="Status?"), state)

        self.assertEqual(turn.assistant_message.content, "The order is confirmed.")
        self.assertEqual(state.raw_generations, [raw])

    def test_compacts_model_context_without_changing_auditable_history(self) -> None:
        backend = ScriptedBackend([Generation(content="Done.")])
        agent = AgentCore(backend, [order_lookup()], "Follow policy.")
        state = agent.get_init_state(
            [
                Message(role="assistant", content="x" * 500),
                Message(
                    role="tool",
                    content=json.dumps(
                        {
                            "order_id": "o-1",
                            "user_id": "u-1",
                            "address": {"address1": "private"},
                            "items": [
                                {
                                    "name": "Keyboard",
                                    "product_id": "p-1",
                                    "item_id": "i-1",
                                    "price": 10,
                                    "options": {"switch": "linear"},
                                }
                            ],
                            "payment_history": [
                                {
                                    "transaction_type": "payment",
                                    "payment_method_id": "credit_card_123",
                                    "amount": 10,
                                }
                            ],
                        }
                    ),
                    tool_call_id="call_1_1",
                )
            ]
        )

        agent.generate_next_message(Message(role="user", content="Continue."), state)

        prompt, _ = backend.requests[0]
        self.assertLessEqual(len(prompt[-3].content), agent.MAX_ASSISTANT_CONTEXT_CHARS + 40)
        self.assertIn('"product_id":"p-1"', prompt[-2].content)
        self.assertIn('"payment_method_id":"credit_card_123"', prompt[-2].content)
        self.assertNotIn("address1", prompt[-2].content)
        self.assertEqual(len(state.messages[0].content), 500)


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
