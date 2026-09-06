"""Stateful Plan-Act-Observe-Verify core for a half-duplex tool agent."""

from __future__ import annotations

import json
from collections.abc import Iterable

from .backend import ModelBackend, strip_qwen_thinking
from .models import (
    AgentState,
    AgentTurn,
    Generation,
    Message,
    Observation,
    StructuredPlan,
    ToolCall,
    ToolSpec,
    ToolValidationError,
    WorkflowPlan,
)
from .workflow import PlanActObserveVerify, is_confirmation_request, is_explicit_confirmation


class TurnLimitReached(RuntimeError):
    """The agent exceeded the configured decision budget for one task."""


class AgentCore:
    """A generic, schema-driven Plan-Act-Observe-Verify tool agent.

    The workflow layer knows only tool schemas and their read/write metadata.
    It does not contain Retail entities, task IDs, product names, or benchmark
    answers. Qwen selects the next action from a small retrieved tool subset.
    """

    MAX_PROMPT_MESSAGES = 12
    MAX_ASSISTANT_CONTEXT_CHARS = 360
    MAX_TOOL_CONTEXT_CHARS = 1_200
    MAX_VERIFICATION_REPAIR_ATTEMPTS = 2

    def __init__(
        self,
        backend: ModelBackend,
        tools: Iterable[ToolSpec],
        domain_policy: str,
        *,
        max_decisions: int = 12,
        candidate_tool_limit: int = 4,
        enable_structured_planning: bool = False,
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
        self.enable_structured_planning = enable_structured_planning
        self.workflow = PlanActObserveVerify(self.tools, candidate_limit=candidate_tool_limit)

    def get_init_state(self, message_history: Iterable[Message] = ()) -> AgentState:
        """Create state with stable, domain-independent operating rules."""

        system_messages = [
            Message(
                role="system",
                content=(
                    "You are a tool-using service agent. Follow the domain policy exactly. "
                    "Use tools for facts and state changes; never invent a tool result. Ask only "
                    "for information that is missing from the conversation or observations. Before "
                    "any state-changing tool call, explain the proposed outcome and obtain explicit "
                    "user confirmation. After a state-changing tool succeeds, use a read tool to "
                    "verify the result before claiming completion. Ground every write argument in "
                    "the user request or a successful tool result. When a tool reports an error, "
                    "inspect current facts with a read tool before proposing another write. Keep "
                    "user-facing replies concise."
                ),
            ),
            Message(role="system", content=f"Domain policy:\n{self.domain_policy}"),
        ]
        return AgentState(system_messages=system_messages, messages=list(message_history))

    def generate_next_message(self, incoming: Message, state: AgentState) -> AgentTurn:
        return self.generate_after_messages([incoming], state)

    def generate_after_messages(
        self, incoming_messages: Iterable[Message], state: AgentState
    ) -> AgentTurn:
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
        self._record_observations(incoming, state)
        structured_plan = self._make_structured_plan(state)
        discovery_mode = structured_plan.mode if structured_plan else None
        plan, visible_tools = self.workflow.plan(state, discovery_mode=discovery_mode)
        state.plans.append(plan)
        generation = self.backend.generate(
            self._model_history(state, plan, structured_plan), visible_tools
        )
        state.raw_generations.append(generation.raw_content or generation.content)
        assistant_message, rejected_calls = self._validate_generation(
            generation,
            state=state,
            visible_tools=visible_tools,
            decision_index=state.decisions + 1,
        )
        if any("not grounded" in rejection.reason for rejection in rejected_calls):
            state.recovery_required = True
            state.awaiting_confirmation = False
        if plan.phase == "verify" and visible_tools and not assistant_message.tool_calls:
            for _ in range(self.MAX_VERIFICATION_REPAIR_ATTEMPTS):
                retry = self.backend.generate(
                    [
                        *self._model_history(state, plan),
                        self._verification_repair_message(rejected_calls, visible_tools),
                    ],
                    visible_tools,
                )
                state.raw_generations.append(retry.raw_content or retry.content)
                assistant_message, retry_rejections = self._validate_generation(
                    retry,
                    state=state,
                    visible_tools=visible_tools,
                    decision_index=state.decisions + 1,
                )
                rejected_calls.extend(retry_rejections)
                if assistant_message.tool_calls:
                    break
            if not assistant_message.tool_calls:
                assistant_message = Message(
                    role="assistant",
                    content="I could not verify the update, so I cannot confirm that it completed.",
                )
        elif plan.phase == "verify" and not visible_tools:
            assistant_message = Message(
                role="assistant",
                content="I could not verify the update because no read tool is available.",
            )
        if any(self.tools_by_name[call.name].kind == "write" for call in assistant_message.tool_calls):
            # Confirmation authorizes one concrete write decision, not an
            # unbounded sequence of later writes.
            state.awaiting_confirmation = False
        state.messages.append(assistant_message)
        if assistant_message.content and is_confirmation_request(assistant_message.content):
            state.awaiting_confirmation = True
        state.decisions += 1
        return AgentTurn(assistant_message=assistant_message, rejected_calls=tuple(rejected_calls))

    def _make_structured_plan(self, state: AgentState) -> StructuredPlan | None:
        """Ask the model for a compact private plan during normal discovery.

        Verify and recovery are controller-owned safety states, so they skip
        this generation entirely. An invalid plan fails safely to read mode.
        """

        if not self.enable_structured_planning or self.workflow.phase(state) != "discover":
            return None
        observations = "\n".join(
            f"- {'success' if observation.success else 'error'} {observation.tool_name}: "
            f"{self._truncate(observation.summary, 320)}"
            for observation in state.observations[-4:]
        ) or "- none"
        requests = "\n".join(
            self._truncate(message.content, 600)
            for message in state.messages
            if message.role == "user"
        ) or "- none"
        planner_messages = [
            Message(
                role="system",
                content=(
                    "You are the private planner for a tool agent. Return exactly one JSON object, "
                    "without markdown or explanation: {\"mode\": \"read\"|\"propose\", "
                    "\"missing_facts\": [string, ...]}. Choose read whenever verified facts are "
                    "still needed before proposing a state-changing operation. Choose propose only "
                    "when the current request can be described using known facts. Never invent facts."
                ),
            ),
            Message(
                role="user",
                content=f"User requests:\n{requests}\n\nObserved facts:\n{observations}",
            ),
        ]
        generation = self.backend.generate(planner_messages, ())
        structured_plan = self._parse_structured_plan(generation)
        state.structured_plans.append(structured_plan)
        return structured_plan

    @staticmethod
    def _parse_structured_plan(generation: Generation) -> StructuredPlan:
        raw_content = generation.raw_content or generation.content
        content = strip_qwen_thinking(generation.content)
        decoder = json.JSONDecoder()
        for start in (index for index, character in enumerate(content) if character == "{"):
            try:
                payload, _ = decoder.raw_decode(content[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or payload.get("mode") not in {"read", "propose"}:
                continue
            missing_facts = payload.get("missing_facts", [])
            if not isinstance(missing_facts, list) or not all(
                isinstance(fact, str) for fact in missing_facts
            ):
                continue
            return StructuredPlan(
                mode=payload["mode"],
                missing_facts=tuple(fact.strip() for fact in missing_facts if fact.strip()),
                raw_content=raw_content,
                valid=True,
            )
        return StructuredPlan(mode="read", missing_facts=(), raw_content=raw_content, valid=False)

    @staticmethod
    def terminate(state: AgentState) -> None:
        state.terminated = True

    def _record_observations(self, incoming: Iterable[Message], state: AgentState) -> None:
        for message in incoming:
            if message.role != "tool":
                continue
            tool_name = self._tool_name_for_result(message, state.messages)
            tool = self.tools_by_name.get(tool_name) if tool_name else None
            success = not message.content.lstrip().lower().startswith("error")
            state.observations.append(
                Observation(
                    tool_name=tool_name,
                    tool_kind=tool.kind if tool else "generic",
                    success=success,
                    summary=self._compact_tool_result(message.content),
                )
            )
            if not tool:
                continue
            if not success:
                state.recovery_required = True
                state.awaiting_confirmation = False
                if tool.kind == "write":
                    state.verification_required = False
            elif tool.kind == "write":
                state.recovery_required = False
                state.verification_required = True
            elif tool.kind == "read":
                state.recovery_required = False
                if state.verification_required:
                    state.verification_required = False

    @staticmethod
    def _tool_name_for_result(message: Message, history: Iterable[Message]) -> str | None:
        messages = list(history)
        for prior in reversed(messages[:-1]):
            for call in prior.tool_calls:
                if message.tool_call_id and call.call_id == message.tool_call_id:
                    return call.name
            if prior.tool_calls and not message.tool_call_id and len(prior.tool_calls) == 1:
                return prior.tool_calls[0].name
        return None

    def _validate_generation(
        self,
        generation: Generation,
        *,
        state: AgentState,
        visible_tools: Iterable[ToolSpec],
        decision_index: int,
    ) -> tuple[Message, list[ToolValidationError]]:
        visible_names = {tool.name for tool in visible_tools}
        rejected: list[ToolValidationError] = []
        accepted: list[ToolCall] = []
        for call_index, call in enumerate(generation.tool_calls, start=1):
            tool = self.tools_by_name.get(call.name)
            if tool is None:
                rejected.append(ToolValidationError(call=call, reason="Unknown tool name."))
            elif call.name not in visible_names:
                rejected.append(ToolValidationError(call=call, reason="Tool was not selected for this plan."))
            elif not isinstance(call.arguments, dict):
                rejected.append(ToolValidationError(call=call, reason="Arguments must be an object."))
            elif state.verification_required and tool.kind != "read":
                rejected.append(
                    ToolValidationError(
                        call=call,
                        reason="A successful state-changing action must be verified with a read tool.",
                    )
                )
            elif tool.kind == "write" and not self._write_arguments_are_grounded(call, state):
                rejected.append(
                    ToolValidationError(
                        call=call,
                        reason="Write arguments are not grounded in user input or successful tool observations.",
                    )
                )
            elif tool.kind == "write" and not self._write_is_confirmed(state):
                rejected.append(
                    ToolValidationError(
                        call=call,
                        reason="State-changing actions require explicit user confirmation.",
                    )
                )
            else:
                accepted.append(
                    ToolCall(
                        name=call.name,
                        arguments=call.arguments,
                        call_id=call.call_id or f"call_{decision_index}_{call_index}",
                    )
                )

        if accepted:
            content = ""
        elif rejected:
            requires_confirmation = any(
                "require explicit user confirmation" in rejection.reason for rejection in rejected
            )
            requires_evidence = any("not grounded" in rejection.reason for rejection in rejected)
            if requires_evidence:
                content = "I need to look up verified information before proposing that change."
            elif requires_confirmation:
                content = "I need to confirm the proposed change with you before processing it."
            else:
                content = "I need to select a valid available tool before proceeding."
        else:
            content = strip_qwen_thinking(generation.content)
        return Message(role="assistant", content=content, tool_calls=tuple(accepted)), rejected

    @staticmethod
    def _verification_repair_message(
        rejected_calls: Iterable[ToolValidationError], visible_tools: Iterable[ToolSpec]
    ) -> Message:
        allowed_names = ", ".join(tool.name for tool in visible_tools)
        rejected = list(rejected_calls)
        if rejected:
            latest = rejected[-1]
            correction = f"Previous tool call `{latest.call.name}` was rejected: {latest.reason}"
        else:
            correction = "No verification tool call was made."
        return Message(
            role="system",
            content=(
                "A state-changing tool has succeeded but is not yet verified. "
                f"{correction} Available read tool names are exactly: {allowed_names}. "
                "Call one available read tool now. Do not reply to the user or claim completion "
                "until that tool result is available."
            ),
        )

    @staticmethod
    def _write_is_confirmed(state: AgentState) -> bool:
        latest_user = next(
            (message.content for message in reversed(state.messages) if message.role == "user"),
            "",
        )
        return state.awaiting_confirmation and is_explicit_confirmation(latest_user)

    @staticmethod
    def _write_arguments_are_grounded(call: ToolCall, state: AgentState) -> bool:
        """Require write values to be present in user input or successful observations.

        This is a provider- and domain-neutral provenance check. It prevents a
        model from inventing identifiers while still allowing a user to supply
        an explicit value directly.
        """

        sources = [message.content.lower() for message in state.messages if message.role == "user"]
        sources.extend(
            observation.summary.lower() for observation in state.observations if observation.success
        )
        source_text = "\n".join(sources)
        return all(str(value).lower() in source_text for value in AgentCore._argument_values(call.arguments))

    @staticmethod
    def _argument_values(value: object) -> Iterable[object]:
        if isinstance(value, dict):
            for child in value.values():
                yield from AgentCore._argument_values(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from AgentCore._argument_values(child)
        elif value is not None:
            yield value

    def _model_history(
        self, state: AgentState, plan: WorkflowPlan, structured_plan: StructuredPlan | None = None
    ) -> list[Message]:
        recent_messages = state.messages[-self.MAX_PROMPT_MESSAGES :]
        planner_context = ""
        if structured_plan is not None:
            planner_context = (
                f" planner_mode={structured_plan.mode}; "
                f"missing_facts={list(structured_plan.missing_facts)}; "
                f"planner_valid={structured_plan.valid};"
            )
        plan_message = Message(
            role="system",
            content=(
                "Private workflow state: "
                f"phase={plan.phase}; observations={plan.observation_count}; "
                f"candidate_tools={list(plan.candidate_tools)}; "
                f"explicit_confirmation_required_for_writes={plan.requires_confirmation}. "
                f"{planner_context}"
                "Choose the next useful action from the candidate tools."
            ),
        )
        return [
            *state.system_messages,
            plan_message,
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
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return self._truncate(content, self.MAX_TOOL_CONTEXT_CHARS)
        if not isinstance(payload, dict):
            return self._truncate(content, self.MAX_TOOL_CONTEXT_CHARS)
        compact = {
            key: payload[key]
            for key in ("order_id", "user_id", "status", "name", "product_id", "item_id", "available")
            if key in payload
        }
        if "items" in payload and isinstance(payload["items"], list):
            compact["items"] = [
                {
                    key: item[key]
                    for key in ("name", "product_id", "item_id", "price", "options")
                    if key in item
                }
                for item in payload["items"]
                if isinstance(item, dict)
            ]
        if "variants" in payload and isinstance(payload["variants"], dict):
            compact["variants"] = [
                {
                    "item_id": variant.get("item_id", item_id),
                    "available": variant.get("available"),
                    "options": variant.get("options"),
                    "price": variant.get("price"),
                }
                for item_id, variant in payload["variants"].items()
                if isinstance(variant, dict) and variant.get("available") is True
            ]
        if "payment_history" in payload and isinstance(payload["payment_history"], list):
            compact["payment_history"] = [
                {
                    key: payment[key]
                    for key in ("transaction_type", "payment_method_id")
                    if key in payment
                }
                for payment in payload["payment_history"]
                if isinstance(payment, dict)
            ]
        serialized = json.dumps(compact or payload, ensure_ascii=False, separators=(",", ":"))
        return self._truncate(serialized, self.MAX_TOOL_CONTEXT_CHARS)
