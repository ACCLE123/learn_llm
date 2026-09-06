"""Stateful, policy-aware core for a half-duplex tool agent."""

from __future__ import annotations

import json
from typing import Iterable

from .backend import ModelBackend, strip_qwen_thinking
from .models import (
    AgentState,
    AgentTurn,
    Generation,
    Message,
    ToolCall,
    ToolSpec,
    ToolValidationError,
)


class TurnLimitReached(RuntimeError):
    """The agent exceeded the configured decision budget for one task."""


class AgentCore:
    """Model-independent decision loop for the initial τ² Retail agent.

    The τ² adapter will feed it user/tool messages and translate the returned
    ``Message`` into τ²'s native message type. Keeping the core independent of
    τ² means its correctness can be tested before the benchmark is installed.
    """

    MAX_PROMPT_MESSAGES = 12
    MAX_ASSISTANT_CONTEXT_CHARS = 360
    MAX_TOOL_CONTEXT_CHARS = 1_200

    def __init__(
        self,
        backend: ModelBackend,
        tools: Iterable[ToolSpec],
        domain_policy: str,
        *,
        max_decisions: int = 12,
    ) -> None:
        if max_decisions < 1:
            raise ValueError("max_decisions must be at least one.")
        self.backend = backend
        self.tools = tuple(tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        if len(self.tools_by_name) != len(self.tools):
            raise ValueError("Tool names must be unique.")
        self.domain_policy = domain_policy
        self.max_decisions = max_decisions

    def get_init_state(self, message_history: Iterable[Message] = ()) -> AgentState:
        """Create state with explicit instructions and the immutable policy."""

        system_messages = [
            Message(
                role="system",
                content=(
                    "You are a customer-service tool agent. Follow the domain policy exactly. "
                    "Use only the provided tools for factual or state-changing operations. "
                    "Ask the user only for information not already present in the conversation "
                    "or a tool result, and never invent tool results. Keep user-facing replies "
                    "to at most two concise sentences; do not repeat a full tool result. For a "
                    "delivered-order exchange, use the order result's product_id and item_id. "
                    "When the user asks for a different configuration of the same product, call "
                    "get_product_details with that product_id to find candidate item IDs before "
                    "calling the exchange tool."
                ),
            ),
            Message(role="system", content=f"Domain policy:\n{self.domain_policy}"),
        ]
        # ``TransformersQwenBackend`` passes ``self.tools`` to Qwen's native
        # chat template, which renders the function schemas itself. Including
        # a JSON copy here would duplicate every Retail schema in the prompt.
        return AgentState(system_messages=system_messages, messages=list(message_history))

    def generate_next_message(self, incoming: Message, state: AgentState) -> AgentTurn:
        """Append one environment message and return one validated decision."""

        return self.generate_after_messages([incoming], state)

    def generate_after_messages(
        self, incoming_messages: Iterable[Message], state: AgentState
    ) -> AgentTurn:
        """Append one or more environment messages before one model decision.

        τ² may return several tool results in one turn. They all belong in the
        context before the model decides what to do next.
        """

        if state.terminated:
            raise RuntimeError("Cannot generate after the agent state is terminated.")
        if state.decisions >= self.max_decisions:
            state.terminated = True
            raise TurnLimitReached(f"Exceeded {self.max_decisions} agent decisions.")
        incoming = list(incoming_messages)
        if not incoming:
            raise ValueError("An agent turn requires at least one incoming message.")
        if any(message.role not in {"user", "tool"} for message in incoming):
            raise ValueError("An agent turn must be triggered by user or tool messages.")

        state.messages.extend(incoming)
        generation = self.backend.generate(self._model_history(state), self.tools)
        state.raw_generations.append(generation.raw_content or generation.content)
        assistant_message, rejected_calls = self._validate_generation(
            generation, decision_index=state.decisions + 1
        )
        state.messages.append(assistant_message)
        state.decisions += 1
        return AgentTurn(assistant_message=assistant_message, rejected_calls=tuple(rejected_calls))

    @staticmethod
    def terminate(state: AgentState) -> None:
        """Mark a completed or externally stopped interaction as terminal."""

        state.terminated = True

    def _validate_generation(
        self, generation: Generation, *, decision_index: int
    ) -> tuple[Message, list[ToolValidationError]]:
        rejected: list[ToolValidationError] = []
        accepted: list[ToolCall] = []
        for call_index, call in enumerate(generation.tool_calls, start=1):
            if call.name not in self.tools_by_name:
                rejected.append(ToolValidationError(call=call, reason="Unknown tool name."))
            elif not isinstance(call.arguments, dict):
                rejected.append(ToolValidationError(call=call, reason="Arguments must be an object."))
            else:
                accepted.append(
                    ToolCall(
                        name=call.name,
                        arguments=call.arguments,
                        call_id=call.call_id or f"call_{decision_index}_{call_index}",
                    )
                )

        if accepted:
            # τ²'s half-duplex protocol accepts either text or tool calls in a
            # message, never both. Keep the action and avoid a protocol error.
            content = ""
        elif not generation.content.strip() and rejected:
            content = "I could not form a valid tool request. Please restate the request."
        else:
            content = strip_qwen_thinking(generation.content)
        return Message(role="assistant", content=content, tool_calls=tuple(accepted)), rejected

    def _model_history(self, state: AgentState) -> list[Message]:
        """Create a compact model view without altering the auditable trajectory."""

        recent_messages = state.messages[-self.MAX_PROMPT_MESSAGES :]
        return [
            *state.system_messages,
            *(self._compact_for_prompt(message) for message in recent_messages),
        ]

    def _compact_for_prompt(self, message: Message) -> Message:
        if message.role == "assistant" and not message.tool_calls:
            return Message(
                role=message.role,
                content=self._truncate(message.content, self.MAX_ASSISTANT_CONTEXT_CHARS),
                tool_calls=message.tool_calls,
                tool_call_id=message.tool_call_id,
            )
        if message.role == "tool":
            return Message(
                role=message.role,
                content=self._compact_tool_result(message.content),
                tool_call_id=message.tool_call_id,
            )
        return message

    @staticmethod
    def _truncate(content: str, maximum_chars: int) -> str:
        if len(content) <= maximum_chars:
            return content
        head = content[:maximum_chars].rsplit(" ", maxsplit=1)[0]
        return f"{head} [earlier response truncated]"

    def _compact_tool_result(self, content: str) -> str:
        """Keep Retail order facts while dropping bulky, irrelevant fields."""

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return self._truncate(content, self.MAX_TOOL_CONTEXT_CHARS)
        if not isinstance(payload, dict):
            return self._truncate(content, self.MAX_TOOL_CONTEXT_CHARS)

        if "order_id" in payload and isinstance(payload.get("items"), list):
            items = [
                {
                    key: item[key]
                    for key in ("name", "product_id", "item_id", "price", "options")
                    if key in item
                }
                for item in payload["items"]
                if isinstance(item, dict)
            ]
            compact = {
                key: payload[key]
                for key in ("order_id", "user_id", "status", "exchange_items", "exchange_new_items")
                if key in payload
            }
            compact["items"] = items
            return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        return self._truncate(content, self.MAX_TOOL_CONTEXT_CHARS)
