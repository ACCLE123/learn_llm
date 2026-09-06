#!/usr/bin/env python3
"""Run the primary local Qwen checkpoint through one τ² mock task."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from tau2.data_model.message import MultiToolMessage, ToolMessage, UserMessage
from tau2.domains.mock.environment import get_environment

from tau2_retail_agent.backend import TransformersQwenBackend
from tau2_retail_agent.tau2_adapter import Tau2QwenAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-generation-seconds", type=float, default=90.0)
    parser.add_argument("--max-decisions", type=int, default=4)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def serialize_tool_result(result: object) -> str:
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), ensure_ascii=False, default=str)
    if hasattr(result, "__dataclass_fields__"):
        return json.dumps(asdict(result), ensure_ascii=False, default=str)
    return json.dumps(result, ensure_ascii=False, default=str)


def print_trace(trace: list[dict[str, object]], state: object) -> None:
    print(
        json.dumps(
            {
                "trace": trace,
                "raw_model_generations": getattr(state, "raw_generations", []),
                "workflow_plans": [asdict(plan) for plan in getattr(state, "plans", [])],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    print(f"Loading {args.model} ...", flush=True)
    backend = TransformersQwenBackend.from_pretrained(
        args.model,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        max_generation_seconds=args.max_generation_seconds,
    )
    print(f"Model device: {backend.device}; thinking: {args.enable_thinking}", flush=True)
    environment = get_environment()
    agent = Tau2QwenAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        backend=backend,
        max_decisions=args.max_decisions,
    )
    state = agent.get_init_state()
    incoming: UserMessage | ToolMessage | MultiToolMessage = UserMessage.text(
        "Please create an Important Meeting task for user_1."
    )
    trace: list[dict[str, object]] = []
    confirmation_sent = False

    for _ in range(args.max_decisions):
        response, state = agent.generate_next_message(incoming, state)
        trace.append(
            {
                "assistant_content": response.content,
                "tool_calls": [call.model_dump() for call in response.tool_calls or []],
            }
        )
        if not response.tool_calls:
            if state.awaiting_confirmation and not confirmation_sent:
                confirmation_sent = True
                incoming = UserMessage.text("Yes, please proceed with creating the task.")
                continue
            print_trace(trace, state)
            return

        results: list[ToolMessage] = []
        for call in response.tool_calls:
            try:
                result = environment.make_tool_call(
                    call.name, requestor="assistant", **call.arguments
                )
                results.append(
                    ToolMessage(
                        id=call.id,
                        role="tool",
                        content=serialize_tool_result(result),
                        requestor="assistant",
                    )
                )
            except Exception as error:
                results.append(
                    ToolMessage(
                        id=call.id,
                        role="tool",
                        content=str(error),
                        requestor="assistant",
                        error=True,
                    )
                )
        incoming = results[0] if len(results) == 1 else MultiToolMessage(role="tool", tool_messages=results)

    raise RuntimeError(f"Agent did not finish within {args.max_decisions} decisions: {trace}")


if __name__ == "__main__":
    main()
