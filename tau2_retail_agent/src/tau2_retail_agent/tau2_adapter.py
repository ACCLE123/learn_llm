"""Adapter from :mod:`tau2` messages to the model-independent agent core.

This module deliberately imports τ² at module load time. Core unit tests do
not import it; a real benchmark run must install τ² in the active environment.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    AssistantMessage,
    Message as TauMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall as TauToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

from .agent import AgentCore, AgentState, TurnLimitReached
from .backend import (
    GenerationTimeoutError,
    ModelBackend,
    ToolCallParseError,
    TransformersQwenBackend,
)
from .models import Message, ToolCall, ToolSpec


def tool_specs_from_tau2(tools: Iterable[Tool]) -> list[ToolSpec]:
    """Convert τ²'s executable tools into model-visible JSON schemas."""

    specs: list[ToolSpec] = []
    for tool in tools:
        function = tool.openai_schema["function"]
        specs.append(
            ToolSpec(
                name=function["name"],
                description=function.get("description", ""),
                parameters=function["parameters"],
            )
        )
    return specs


def _core_tool_calls(message: AssistantMessage) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(name=call.name, arguments=call.arguments, call_id=call.id or None)
        for call in (message.tool_calls or [])
    )


def tau2_history_to_core(message: TauMessage) -> Message:
    """Normalize a τ² history message into the backend's provider-neutral form."""

    if isinstance(message, SystemMessage):
        return Message(role="system", content=message.content or "")
    if isinstance(message, UserMessage):
        return Message(role="user", content=message.content or "")
    if isinstance(message, AssistantMessage):
        return Message(
            role="assistant",
            content=message.content or "",
            tool_calls=_core_tool_calls(message),
        )
    if isinstance(message, ToolMessage):
        return Message(
            role="tool",
            content=message.content or "",
            tool_call_id=message.id or None,
        )
    raise TypeError(f"Unsupported τ² history message: {type(message).__name__}")


def incoming_tau2_messages(message: ValidAgentInputMessage) -> list[Message]:
    """Return every incoming τ² event in the order the model must observe it."""

    if isinstance(message, MultiToolMessage):
        return [tau2_history_to_core(tool_message) for tool_message in message.tool_messages]
    return [tau2_history_to_core(message)]


def core_to_tau2_message(message: Message) -> AssistantMessage:
    """Translate a validated core decision into τ²'s half-duplex protocol."""

    tool_calls = [
        TauToolCall(
            id=call.call_id or "",
            name=call.name,
            arguments=call.arguments,
            requestor="assistant",
        )
        for call in message.tool_calls
    ]
    # τ² treats text and tool calls as mutually exclusive in text mode.
    return AssistantMessage.text(content=message.content, tool_calls=tool_calls or None)


class Tau2QwenAgent(HalfDuplexAgent[AgentState]):
    """A τ² half-duplex participant backed by :class:`AgentCore`."""

    STOP_TOKEN = "###STOP###"

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        backend: ModelBackend,
        *,
        max_decisions: int = 12,
    ) -> None:
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.core = AgentCore(
            backend=backend,
            tools=tool_specs_from_tau2(tools),
            domain_policy=domain_policy,
            max_decisions=max_decisions,
        )

    def get_init_state(
        self, message_history: Optional[list[TauMessage]] = None
    ) -> AgentState:
        history = [tau2_history_to_core(message) for message in (message_history or [])]
        return self.core.get_init_state(history)

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        try:
            turn = self.core.generate_after_messages(incoming_tau2_messages(message), state)
        except TurnLimitReached:
            # The returned text gives the user a clean stop while preserving a
            # terminal state for the runner and trace collector.
            return (
                AssistantMessage.text(
                    "I am unable to complete this request within the allowed steps."
                ),
                state,
            )
        except (GenerationTimeoutError, ToolCallParseError):
            self.core.terminate(state)
            return AssistantMessage.text(self.STOP_TOKEN), state
        return core_to_tau2_message(turn.assistant_message), state

    @classmethod
    def is_stop(cls, message: AssistantMessage) -> bool:
        return cls.STOP_TOKEN in (message.content or "")

    def stop(
        self,
        message: Optional[ValidAgentInputMessage] = None,
        state: Optional[AgentState] = None,
    ) -> None:
        if state is not None:
            self.core.terminate(state)


def create_tau2_qwen_agent(tools: list[Tool], domain_policy: str, **kwargs: Any) -> Tau2QwenAgent:
    """Factory usable with ``tau2.registry.register_agent_factory``.

    Tests and custom runners may provide a ``backend`` directly. CLI-style
    callers may instead pass ``model_name`` (or τ²'s ``agent_llm``/``llm``).
    """

    backend = kwargs.get("backend")
    if backend is None:
        model_name = kwargs.get("model_name") or kwargs.get("agent_llm") or kwargs.get("llm")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Provide either a backend or a local Qwen model_name.")
        backend = TransformersQwenBackend.from_pretrained(
            model_name,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            device_map=kwargs.get("device_map"),
            enable_thinking=kwargs.get("enable_thinking", False),
            max_generation_seconds=kwargs.get("max_generation_seconds", 90.0),
        )
    return Tau2QwenAgent(
        tools=tools,
        domain_policy=domain_policy,
        backend=backend,
        max_decisions=kwargs.get("max_decisions", 12),
    )
