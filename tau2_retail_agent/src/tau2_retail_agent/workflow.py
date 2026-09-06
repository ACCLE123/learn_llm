"""Generic planning primitives for schema-driven tool agents."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import AgentState, ToolKind, ToolSpec, WorkflowPlan

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower().replace("_", " ")))


class ToolRetriever:
    """Small deterministic retriever over tool schemas and descriptions."""

    def __init__(self, tools: Iterable[ToolSpec]) -> None:
        self.tools = tuple(tools)

    def select(
        self,
        query: str,
        *,
        allowed_kinds: set[ToolKind] | None = None,
        limit: int = 4,
    ) -> tuple[ToolSpec, ...]:
        candidates = [
            tool for tool in self.tools if allowed_kinds is None or tool.kind in allowed_kinds
        ]
        if not candidates:
            candidates = list(self.tools)
        query_tokens = _tokens(query)

        def score(tool: ToolSpec) -> tuple[int, str]:
            name_tokens = _tokens(tool.name)
            description_tokens = _tokens(tool.description)
            parameter_tokens = _tokens(" ".join(tool.parameters.get("properties", {}).keys()))
            value = (
                4 * len(query_tokens & name_tokens)
                + 2 * len(query_tokens & parameter_tokens)
                + len(query_tokens & description_tokens)
            )
            if tool.kind == "read" and query_tokens & {"find", "lookup", "show", "list", "details"}:
                value += 3
            if tool.kind == "write" and query_tokens & {"change", "update", "cancel", "return", "exchange"}:
                value += 3
            return (-value, tool.name)

        return tuple(sorted(candidates, key=score)[:limit])


class PlanActObserveVerify:
    """Build a generic private plan from state, observations, and schemas."""

    def __init__(self, tools: Iterable[ToolSpec], *, candidate_limit: int = 4) -> None:
        self.tools = tuple(tools)
        self.retriever = ToolRetriever(self.tools)
        self.candidate_limit = candidate_limit

    def plan(self, state: AgentState) -> tuple[WorkflowPlan, tuple[ToolSpec, ...]]:
        phase = "verify" if state.verification_required else "act" if state.observations else "discover"
        allowed_kinds: set[ToolKind] | None = {"read"} if phase == "verify" else None
        visible_tools = self.retriever.select(
            self._query(state), allowed_kinds=allowed_kinds, limit=self.candidate_limit
        )
        plan = WorkflowPlan(
            phase=phase,
            candidate_tools=tuple(tool.name for tool in visible_tools),
            requires_confirmation=any(tool.kind == "write" for tool in visible_tools),
            observation_count=len(state.observations),
        )
        return plan, visible_tools

    @staticmethod
    def _query(state: AgentState) -> str:
        user_messages = [message.content for message in state.messages if message.role == "user"]
        observations = [observation.summary for observation in state.observations[-4:]]
        return "\n".join([*user_messages[-3:], *observations])


def is_explicit_confirmation(text: str) -> bool:
    normalized = text.lower()
    phrases = ("yes", "confirm", "confirmed", "proceed", "go ahead", "approve", "please do", "that works")
    negative_phrases = ("do not", "don't", "not proceed", "do not approve")
    return any(phrase in normalized for phrase in phrases) and not any(
        phrase in normalized for phrase in negative_phrases
    )


def is_confirmation_request(text: str) -> bool:
    normalized = text.lower()
    return any(phrase in normalized for phrase in ("confirm", "proceed", "approve", "go ahead"))
