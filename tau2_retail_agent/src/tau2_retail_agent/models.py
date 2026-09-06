"""Typed data exchanged by the agent core and a tool environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]
ToolKind = Literal["read", "write", "think", "generic"]
WorkflowPhase = Literal["discover", "act", "verify"]


@dataclass(frozen=True)
class ToolSpec:
    """A tool exposed by the environment using an OpenAI-compatible schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    kind: ToolKind = "generic"

    def as_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """One model-selected call. Arguments remain structured until execution."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class Message:
    """A normalized conversation entry, independent of a model provider."""

    role: MessageRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Generation:
    """The normalized result produced by a model backend for one agent turn."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    raw_content: str | None = None


@dataclass(frozen=True)
class ToolValidationError:
    """A non-executable tool call with a user-visible, traceable reason."""

    call: ToolCall
    reason: str


@dataclass(frozen=True)
class AgentTurn:
    """A completed decision step and any calls rejected before execution."""

    assistant_message: Message
    rejected_calls: tuple[ToolValidationError, ...] = ()


@dataclass(frozen=True)
class Observation:
    """A compact, tool-attributed fact returned by the environment."""

    tool_name: str | None
    tool_kind: ToolKind
    success: bool
    summary: str


@dataclass(frozen=True)
class WorkflowPlan:
    """Private plan for one agent decision; it is never shown to the user."""

    phase: WorkflowPhase
    candidate_tools: tuple[str, ...]
    requires_confirmation: bool
    observation_count: int


@dataclass
class AgentState:
    """Mutable state that must survive user and tool messages between turns."""

    system_messages: list[Message]
    messages: list[Message] = field(default_factory=list)
    decisions: int = 0
    terminated: bool = False
    raw_generations: list[str] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    plans: list[WorkflowPlan] = field(default_factory=list)
    awaiting_confirmation: bool = False
    verification_required: bool = False

    @property
    def history(self) -> list[Message]:
        """Return the complete promptable history in chronological order."""

        return [*self.system_messages, *self.messages]
