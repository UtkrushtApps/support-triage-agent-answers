"""Provider wire-format translation for the mock LLM server.

Two directions:

* :func:`extract_prompt` turns an incoming request body (OpenAI Chat Completions
  or Anthropic Messages) into the canonical prompt string the matcher keys on.
* :func:`render_response` turns a provider-agnostic
  :class:`~infra.agent_runtime.fixtures.CannedResponse` into the response shape
  the corresponding SDK expects.

Canonicalization strips **volatile** fields that change every run — tool-call
ids, tool-result back-references, and JSON key ordering inside tool arguments —
because including them would defeat exact/normalized matching on a faithful
re-run. The semantic tier handles whatever drift remains.
"""
from __future__ import annotations

import json
from typing import Any

from .fixtures import CannedResponse

OPENAI = "openai"
ANTHROPIC = "anthropic"

# Volatile keys dropped from message/content blocks before building the key.
_VOLATILE_KEYS = frozenset({"id", "tool_call_id", "tool_use_id"})


def _canonical_json(value: Any) -> str:
    """Stable JSON dump (sorted keys) so key-ordering drift doesn't matter."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _canonical_args(arguments: Any) -> str:
    """Canonicalize tool-call arguments, which OpenAI sends as a JSON *string*."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments.strip()
    return _canonical_json(arguments)


def _strip_volatile(block: Any) -> Any:
    if isinstance(block, dict):
        return {k: _strip_volatile(v) for k, v in block.items() if k not in _VOLATILE_KEYS}
    if isinstance(block, list):
        return [_strip_volatile(item) for item in block]
    return block


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return _canonical_json(_strip_volatile(content))


def _openai_messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "")
        segments: list[str] = []
        content_str = _render_content(message.get("content"))
        if content_str:
            segments.append(content_str)
        for tool_call in message.get("tool_calls") or []:
            fn = tool_call.get("function", {})
            segments.append(f"call={fn.get('name', '')}({_canonical_args(fn.get('arguments', '{}'))})")
        lines.append(f"{role}: {' '.join(segments)}")
    return "\n".join(lines)


def _anthropic_to_prompt(body: dict[str, Any]) -> str:
    lines: list[str] = []
    system = body.get("system")
    if system:
        lines.append(f"system: {_render_content(system)}")
    for message in body.get("messages", []):
        role = message.get("role", "")
        lines.append(f"{role}: {_render_content(message.get('content'))}")
    return "\n".join(lines)


def extract_prompt(provider: str, body: dict[str, Any]) -> str:
    """Build the canonical match key from a request body."""
    if provider == OPENAI:
        return _openai_messages_to_prompt(body.get("messages", []))
    if provider == ANTHROPIC:
        return _anthropic_to_prompt(body)
    raise ValueError(f"unknown provider: {provider!r}")


def _render_openai(response: CannedResponse, model: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_mock_{i}",
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for i, tc in enumerate(response.tool_calls)
        ]
    elif response.content is None:
        message["content"] = ""
    finish = response.finish_reason or ("tool_calls" if response.tool_calls else "stop")
    return {
        "id": "chatcmpl-mock-0",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish, "logprobs": None}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _render_anthropic(response: CannedResponse, model: str) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    if response.content:
        blocks.append({"type": "text", "text": response.content})
    for i, tc in enumerate(response.tool_calls):
        blocks.append({"type": "tool_use", "id": f"toolu_mock_{i}", "name": tc.name, "input": tc.arguments})
    stop_reason = "tool_use" if response.tool_calls else "end_turn"
    return {
        "id": "msg_mock_0",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def render_response(provider: str, response: CannedResponse, model: str) -> dict[str, Any]:
    """Render a canned response into the provider's response wire shape."""
    if provider == OPENAI:
        return _render_openai(response, model)
    if provider == ANTHROPIC:
        return _render_anthropic(response, model)
    raise ValueError(f"unknown provider: {provider!r}")
