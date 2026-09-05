"""Model backends used by the agent core.

The protocol lets the agent be unit-tested without a downloaded Qwen checkpoint.
``TransformersQwenBackend`` is deliberately lazy: importing this package does
not require torch or transformers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol, Sequence

from .models import Generation, Message, ToolCall, ToolSpec


class ModelBackend(Protocol):
    """Generate one agent decision from normalized history and tool schemas."""

    def generate(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Generation: ...


class ToolCallParseError(ValueError):
    """Raised when a Qwen tool-call block is not valid structured JSON."""


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_qwen_tool_calls(completion: str) -> Generation:
    """Parse Qwen's documented XML-wrapped function-call response format.

    Plain text is preserved as the assistant response. A malformed call is an
    explicit error instead of a guessed repair, because a repair can become an
    unintended business action.
    """

    calls: list[ToolCall] = []
    for block in _TOOL_CALL_PATTERN.findall(completion):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError as error:
            raise ToolCallParseError("Tool-call block is not valid JSON.") from error

        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            raise ToolCallParseError("Tool-call JSON must include a string 'name'.")
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ToolCallParseError("Tool-call arguments string is not valid JSON.") from error
        if not isinstance(arguments, dict):
            raise ToolCallParseError("Tool-call arguments must be a JSON object.")
        calls.append(ToolCall(name=payload["name"], arguments=arguments))

    content = _TOOL_CALL_PATTERN.sub("", completion).strip()
    return Generation(content=content, tool_calls=tuple(calls))


class TransformersQwenBackend:
    """A minimal local Transformers backend for a Qwen instruction checkpoint."""

    def __init__(self, model: object, tokenizer: object, max_new_tokens: int = 512) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        max_new_tokens: int = 512,
        device_map: str = "auto",
    ) -> "TransformersQwenBackend":
        """Load the model lazily so core tests stay dependency-free."""

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on local setup
            raise RuntimeError(
                "Install torch and transformers before loading a local Qwen model."
            ) from error

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
        )
        model.eval()
        return cls(model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)

    def generate(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Generation:
        """Render Qwen's native template and normalize the generated response."""

        rendered_messages = [
            {
                "role": message.role,
                "content": message.content,
                **(
                    {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                                },
                            }
                            for call in message.tool_calls
                        ]
                    }
                    if message.tool_calls
                    else {}
                ),
                **({"tool_call_id": message.tool_call_id} if message.tool_call_id else {}),
            }
            for message in messages
        ]
        prompt = self.tokenizer.apply_chat_template(
            rendered_messages,
            tools=[tool.as_schema() for tool in tools],
            tokenize=False,
            add_generation_prompt=True,
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        generated = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        completion = self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return parse_qwen_tool_calls(completion)
