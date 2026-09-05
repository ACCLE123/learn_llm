"""Stateful, policy-aware core for a half-duplex tool agent."""

from __future__ import annotations

import json
from typing import Iterable

from .backend import ModelBackend
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

        tool_summary = json.dumps([tool.as_schema() for tool in self.tools], ensure_ascii=False)
        system_messages = [
            Message(
                role="system",
                content=(
                    "You are a customer-service tool agent. Follow the domain policy exactly. "
                    "Use only the provided tools for factual or state-changing operations. "
                    "Ask the user for missing information and never invent tool results."
                ),
            ),
            Message(role="system", content=f"Domain policy:\n{self.domain_policy}"),
            Message(role="system", content=f"Available tool schemas:\n{tool_summary}"),
        ]
        return AgentState(system_messages=system_messages, messages=list(message_history))

    def generate_next_message(self, incoming: Message, state: AgentState) -> AgentTurn:
        """Append an environment message and return one validated decision."""

        if state.terminated:
            raise RuntimeError("Cannot generate after the agent state is terminated.")
        if state.decisions >= self.max_decisions:
            state.terminated = True
            raise TurnLimitReached(f"Exceeded {self.max_decisions} agent decisions.")
        if incoming.role not in {"user", "tool"}:
            raise ValueError("An agent turn must be triggered by a user or tool message.")

        state.messages.append(incoming)
        generation = self.backend.generate(state.history, self.tools)
        assistant_message, rejected_calls = self._validate_generation(generation)
        state.messages.append(assistant_message)
        state.decisions += 1
        return AgentTurn(assistant_message=assistant_message, rejected_calls=tuple(rejected_calls))

    @staticmethod
    def terminate(state: AgentState) -> None:
        """Mark a completed or externally stopped interaction as terminal."""

        state.terminated = True

    def _validate_generation(
        self, generation: Generation
    ) -> tuple[Message, list[ToolValidationError]]:
        rejected: list[ToolValidationError] = []
        accepted: list[ToolCall] = []
        for call in generation.tool_calls:
            if call.name not in self.tools_by_name:
                rejected.append(ToolValidationError(call=call, reason="Unknown tool name."))
            elif not isinstance(call.arguments, dict):
                rejected.append(ToolValidationError(call=call, reason="Arguments must be an object."))
            else:
                accepted.append(call)

        if not generation.content.strip() and not accepted and rejected:
            content = "I could not form a valid tool request. Please restate the request."
        else:
            content = generation.content.strip()
        return Message(role="assistant", content=content, tool_calls=tuple(accepted)), rejected
