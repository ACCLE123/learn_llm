#!/usr/bin/env python3
"""Profile the historically slow Retail tool-agent turn without any API calls.

This reconstructs the conversation state immediately before the sixth local
Qwen decision observed in Retail task 0. It deliberately does not construct a
τ² user simulator or contact DeepSeek, so an excessive duration isolates local
MPS/Transformers generation from benchmark orchestration.
"""

from __future__ import annotations

import argparse
import json

from tau2.domains.retail.environment import get_environment

from tau2_retail_agent.agent import AgentCore
from tau2_retail_agent.backend import GenerationTimeoutError, TransformersQwenBackend
from tau2_retail_agent.models import Message, ToolCall
from tau2_retail_agent.tau2_adapter import tool_specs_from_tau2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-generation-seconds", type=float, default=60.0)
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Profile all six recorded decisions using one loaded model instance.",
    )
    return parser.parse_args()


def recorded_history() -> list[Message]:
    """The deterministic dialogue prefix from task 0 before its slow turn."""

    return [
        Message(
            role="user",
            content=(
                "Hi, I hope you can help me out. I just received my order and I want to "
                "exchange two items from it. I’m not sure exactly how it all works, but I’d "
                "like to make sure everything I need gets taken care of in one go. Can you "
                "help me coordinate both exchanges?"
            ),
        ),
        Message(
            role="assistant",
            content=(
                "Sure, I'd be happy to help you with exchanging items. First, I'll need the "
                "order ID and the item IDs you want to exchange. Could you please provide "
                "those? Also, it would be helpful to know the new item IDs you'd like to "
                "exchange them for. Lastly, I'll need the payment method you'd like to use "
                "for any price differences."
            ),
        ),
        Message(
            role="user",
            content=(
                "Sure, the order number is W2378156. As for the items:\n\n"
                "1. I want to exchange the mechanical keyboard I received for the same one, "
                "but with clicky switches. Just to clarify, I’m looking for a keyboard that’s "
                "full size with RGB backlight and clicky switches—if that exact one isn’t "
                "available, I’d go without backlight on the same setup.\n\n"
                "2. I’d also like to swap the smart thermostat I got for the same model or "
                "equivalent, but one that works with Google Home instead of Apple HomeKit.\n\n"
                "I don't have the exact product or item IDs handy—sorry about that. Could you "
                "look them up from the order? For the payment method, I can use a card if any "
                "price difference comes up, but I’d prefer exchanges that don't cost extra if possible."
            ),
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=(ToolCall("get_order_details", {"order_id": "W2378156"}, "call_2_1"),),
        ),
        Message(role="tool", content="Error: Order not found", tool_call_id="call_2_1"),
        Message(
            role="assistant",
            content=(
                "I'm sorry, but I couldn't find the order with the ID you provided. Could you "
                "please double-check the order number or provide any other details that might help me locate it?"
            ),
        ),
        Message(
            role="user",
            content=(
                "I double-checked my receipt, and the order number is definitely W2378156. "
                "I ordered it using the name Yusuf Rossi, and my zip code is 19122. Is there "
                "another way you could look it up? Maybe you can check under my name or the "
                "shipping address listed on that order. Let me know if you need any other details, "
                "and I can try to help figure this out."
            ),
        ),
        Message(
            role="assistant",
            content="",
            tool_calls=(
                ToolCall(
                    "find_user_id_by_name_zip",
                    {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
                    "call_4_1",
                ),
            ),
        ),
        Message(role="tool", content="yusuf_rossi_9620", tool_call_id="call_4_1"),
        Message(
            role="assistant",
            content=(
                "I've found your user ID: yusuf_rossi_9620. Now, I'll need to check if there's "
                "any additional information I can use to locate the order. Could you please provide "
                "the shipping address associated with this order? This will help me verify the order "
                "details and locate it more accurately."
            ),
        ),
        Message(
            role="user",
            content=(
                "Of course! The shipping address on the order is where I live—it should be under "
                "my name, Yusuf Rossi, sent to my zip code 19122. I’m sorry, but I don’t have the "
                "exact street address written in front of me right now. Is there anything else you can "
                "use—like the email on file—or maybe the order can be pulled up another way with my name and zip?"
            ),
        ),
    ]


def main() -> None:
    args = parse_args()
    environment = get_environment()
    backend = TransformersQwenBackend.from_pretrained(
        args.model,
        max_new_tokens=args.max_new_tokens,
        max_generation_seconds=args.max_generation_seconds,
    )
    core = AgentCore(
        backend=backend,
        tools=tool_specs_from_tau2(environment.get_tools()),
        domain_policy=environment.get_policy(),
    )
    history = recorded_history()
    print(f"Profiling on {backend.device}; no DeepSeek request will be made.", flush=True)
    prefix_lengths = [1, 3, 5, 7, 9, 11] if args.sequence else [len(history)]
    results = []
    for decision, prefix_length in enumerate(prefix_lengths, start=1):
        state = core.get_init_state(history[:prefix_length])
        print(f"[profile] decision={decision}, history_messages={prefix_length}", flush=True)
        try:
            generation = backend.generate(state.history, core.tools)
        except GenerationTimeoutError as error:
            print(
                json.dumps(
                    {"result": "timeout", "decision": decision, "detail": str(error)},
                    indent=2,
                )
            )
            raise SystemExit(2) from error
        results.append(
            {
                "decision": decision,
                "content": generation.content,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in generation.tool_calls
                ],
            }
        )
    print(json.dumps({"result": "completed", "decisions": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
