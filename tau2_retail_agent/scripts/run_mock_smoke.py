#!/usr/bin/env python3
"""Run the local agent through one real τ² mock tool loop without an LLM.

This is an integration smoke test, not a benchmark result. The deterministic
backend proves that our τ² adapter, tool schemas, call IDs, tool execution, and
state propagation compose correctly before a Qwen model is introduced.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Sequence

from tau2.data_model.message import ToolMessage, UserMessage
from tau2.domains.mock.environment import get_environment

from tau2_retail_agent.models import Generation, Message, ToolCall, ToolSpec
from tau2_retail_agent.tau2_adapter import Tau2QwenAgent


class DeterministicMockBackend:
    """Exercises generic confirmation, write, verification, and final reply."""

    def generate(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Generation:
        tool_result_count = sum(message.role == "tool" for message in messages)
        if tool_result_count >= 2:
            return Generation(content="The task was created successfully.")
        if tool_result_count == 1:
            return Generation(tool_calls=(ToolCall(name="get_users", arguments={}),))
        if any(message.role == "user" and "yes" in message.content.lower() for message in messages):
            return Generation(
                tool_calls=(
                    ToolCall(
                        name="create_task",
                        arguments={"user_id": "user_1", "title": "Important Meeting"},
                    ),
                )
            )
        return Generation(content="I can create that task. Please confirm that you want to proceed.")


def serialize_tool_result(result: object) -> str:
    """Keep the official tool result visible to the next model decision."""

    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), ensure_ascii=False, default=str)
    if hasattr(result, "__dataclass_fields__"):
        return json.dumps(asdict(result), ensure_ascii=False, default=str)
    return json.dumps(result, ensure_ascii=False, default=str)


def main() -> None:
    environment = get_environment()
    agent = Tau2QwenAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        backend=DeterministicMockBackend(),
        max_decisions=4,
    )
    state = agent.get_init_state()

    confirmation, state = agent.generate_next_message(
        UserMessage.text("Please create an Important Meeting task for user_1."), state
    )
    if "confirm" not in (confirmation.content or "").lower():
        raise RuntimeError("Expected the agent to request confirmation before the write.")
    first, state = agent.generate_next_message(UserMessage.text("Yes, please proceed."), state)
    if not first.tool_calls or first.tool_calls[0].name != "create_task":
        raise RuntimeError("Expected the agent to request create_task.")

    tool_messages: list[ToolMessage] = []
    for call in first.tool_calls:
        result = environment.make_tool_call(
            call.name, requestor="assistant", **call.arguments
        )
        tool_messages.append(
            ToolMessage(
                id=call.id,
                role="tool",
                content=serialize_tool_result(result),
                requestor="assistant",
            )
        )

    verify, state = agent.generate_next_message(tool_messages[0], state)
    if not verify.tool_calls or verify.tool_calls[0].name != "get_users":
        raise RuntimeError("Expected a read-only verification after the write.")
    verified = environment.make_tool_call("get_users", requestor="assistant")
    final, state = agent.generate_next_message(
        ToolMessage(
            id=verify.tool_calls[0].id,
            role="tool",
            content=serialize_tool_result(verified),
            requestor="assistant",
        ),
        state,
    )
    created = environment.make_tool_call("get_users", requestor="assistant")
    created_task_ids = created[0].tasks
    if "task_2" not in created_task_ids:
        raise RuntimeError(f"Mock environment did not contain the created task: {created_task_ids}")
    if "created successfully" not in (final.content or "").lower():
        raise RuntimeError("The agent did not receive and communicate the tool result.")

    trace = {
        "tool_call": first.tool_calls[0].model_dump(),
        "tool_result": tool_messages[0].content,
        "verification_call": verify.tool_calls[0].model_dump(),
        "final_response": final.content,
        "agent_decisions": state.decisions,
        "created_task_ids": created_task_ids,
    }
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
