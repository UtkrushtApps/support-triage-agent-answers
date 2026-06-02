"""Fixture format + loader for the mock LLM server.

A *fixture* maps a recorded prompt to a provider-agnostic canned response. The
response is stored once and rendered into either OpenAI or Anthropic wire shape
at serve time (see :mod:`infra.agent_runtime.adapters`), so a fixture recorded
against one endpoint can replay on either.

On-disk format — one ``*.json`` file per fixture, or a JSON list of fixtures::

    {
      "id": "search_agent/turn-1",
      "provider": "openai",
      "prompt": "system: You are a recruiting assistant.\\nuser: Find the top backend candidate.",
      "response": {
        "content": null,
        "tool_calls": [{"name": "search_candidates", "arguments": {"role": "backend"}}],
        "finish_reason": "tool_calls"
      }
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .matching import Candidate


class FixtureError(ValueError):
    """Raised when a fixture file is malformed or ids collide."""


@dataclass(frozen=True)
class ToolCall:
    """A single tool/function invocation the model should emit."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        if "name" not in data:
            raise FixtureError(f"tool_call missing 'name': {data!r}")
        return cls(name=data["name"], arguments=dict(data.get("arguments") or {}))


@dataclass(frozen=True)
class CannedResponse:
    """Provider-agnostic response: optional text plus optional tool calls."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CannedResponse":
        return cls(
            content=data.get("content"),
            tool_calls=tuple(ToolCall.from_dict(tc) for tc in (data.get("tool_calls") or [])),
            finish_reason=data.get("finish_reason"),
        )


@dataclass(frozen=True)
class Fixture:
    """A recorded (prompt -> response) pair."""

    id: str
    prompt: str
    response: CannedResponse
    provider: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fixture":
        for required in ("id", "prompt", "response"):
            if required not in data:
                raise FixtureError(f"fixture missing '{required}': {data!r}")
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            response=CannedResponse.from_dict(data["response"]),
            provider=data.get("provider"),
        )

    def as_candidate(self) -> Candidate:
        return Candidate(key=self.id, prompt=self.prompt)


def _parse_file(path: Path) -> list[Fixture]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureError(f"{path}: invalid JSON ({exc})") from exc
    records = raw if isinstance(raw, list) else [raw]
    return [Fixture.from_dict(record) for record in records]


def load_fixtures(directory: str | Path) -> list[Fixture]:
    """Load every ``*.json`` fixture under *directory* (sorted, deterministic).

    Raises :class:`FixtureError` on malformed files or duplicate ids — a silent
    overwrite would make replay non-deterministic.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FixtureError(f"fixtures directory not found: {root}")

    fixtures: list[Fixture] = []
    seen: dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        for fixture in _parse_file(path):
            if fixture.id in seen:
                raise FixtureError(
                    f"duplicate fixture id {fixture.id!r} in {path} "
                    f"(already defined in {seen[fixture.id]})"
                )
            seen[fixture.id] = path
            fixtures.append(fixture)
    return fixtures
