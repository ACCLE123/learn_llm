"""Model backends used by the agent core.

The protocol lets the agent be unit-tested without a downloaded Qwen checkpoint.
``TransformersQwenBackend`` is deliberately lazy: importing this package does
not require torch or transformers.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Protocol, Sequence

from .models import Generation, Message, ToolCall, ToolSpec


class ModelBackend(Protocol):
    """Generate one agent decision from normalized history and tool schemas."""

    def generate(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Generation: ...


class ToolCallParseError(ValueError):
    """Raised when a Qwen tool-call block is not valid structured JSON."""


class GenerationTimeoutError(TimeoutError):
    """Raised when one local-model decision exceeds its configured budget."""


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINKING_PATTERN = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def strip_qwen_thinking(text: str) -> str:
    """Remove Qwen reasoning blocks from the text delivered to an end user.

    ``Generation.raw_content`` retains the original completion for local
    debugging and later trajectory research. An unfinished tag is redacted
    through the end of the completion as the conservative behaviour.
    """

    return _THINKING_PATTERN.sub("", text).replace("</think>", "").strip()


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
    return Generation(content=content, tool_calls=tuple(calls), raw_content=completion)


class TransformersQwenBackend:
    """A minimal local Transformers backend for a Qwen instruction checkpoint."""

    def __init__(
        self,
        model: object,
        tokenizer: object,
        max_new_tokens: int = 512,
        *,
        enable_thinking: bool = False,
        max_generation_seconds: float | None = 90.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.max_generation_seconds = max_generation_seconds

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        max_new_tokens: int = 512,
        device_map: str | None = None,
        enable_thinking: bool = False,
        max_generation_seconds: float | None = 90.0,
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
        resolved_device_map = device_map
        if resolved_device_map is None and torch.cuda.is_available():
            resolved_device_map = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            # Qwen checkpoints publish their intended dtype. Keeping it avoids
            # silently expanding a 4B model to roughly 16 GB of FP32 RAM on a
            # CPU-only machine.
            torch_dtype="auto",
            device_map=resolved_device_map,
        )
        if resolved_device_map is None and torch.backends.mps.is_available():
            model.to("mps")
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            max_generation_seconds=max_generation_seconds,
        )

    @property
    def device(self) -> str:
        """Return the device that owns the model's first parameter."""

        return str(next(self.model.parameters()).device)

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
            enable_thinking=self.enable_thinking,
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_tokens = inputs["input_ids"].shape[1]
        started_at = time.perf_counter()
        print(
            f"[qwen] generating on {device} "
            f"(prompt_tokens={prompt_tokens}, max_new_tokens={self.max_new_tokens})",
            flush=True,
        )
        generation_kwargs = {
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.max_generation_seconds is not None:
            generation_kwargs["max_time"] = self.max_generation_seconds
        generated = self.model.generate(
            **inputs,
            **generation_kwargs,
        )
        generated_tokens = generated.shape[1] - prompt_tokens
        elapsed_seconds = time.perf_counter() - started_at
        print(
            f"[qwen] generated {generated_tokens} tokens in {elapsed_seconds:.1f}s",
            flush=True,
        )
        if (
            self.max_generation_seconds is not None
            and elapsed_seconds >= self.max_generation_seconds
        ):
            raise GenerationTimeoutError(
                f"Qwen generation exceeded {self.max_generation_seconds:.1f}s "
                f"after producing {generated_tokens} tokens."
            )
        completion = self.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return parse_qwen_tool_calls(completion)
