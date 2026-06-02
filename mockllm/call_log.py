"""Call recording + call-ceiling kill-switch for the mock LLM server.

The recorded log is a generation-time gate signal: a task that should make
exactly N model calls can be asserted against ``count``; a runaway-loop task is
capped by ``max_calls`` so it fails fast instead of hanging or burning budget.
Every served and missed call is recorded; calls refused by the ceiling are
counted separately under ``killed``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_PREVIEW_CHARS = 240


def _preview(prompt: str) -> str:
    flat = " ".join(prompt.split())
    if len(flat) <= _PREVIEW_CHARS:
        return flat
    return flat[:_PREVIEW_CHARS] + "…"


@dataclass(frozen=True)
class CallRecord:
    """One served-or-missed call. Immutable; the log appends new records."""

    seq: int
    endpoint: str
    model: str
    tier: str
    matched_fixture: str | None
    similarity: float | None
    prompt_preview: str


class CallLog:
    """Append-only record of calls, with an optional hard ceiling."""

    def __init__(self, max_calls: int | None = None) -> None:
        self._max_calls = max_calls
        self._records: list[CallRecord] = []
        self._killed = 0

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def killed(self) -> int:
        return self._killed

    @property
    def max_calls(self) -> int | None:
        return self._max_calls

    def would_exceed(self) -> bool:
        """True when serving another call would breach the ceiling."""
        return self._max_calls is not None and len(self._records) >= self._max_calls

    def record(
        self,
        *,
        endpoint: str,
        model: str,
        tier: str,
        matched_fixture: str | None,
        similarity: float | None,
        prompt: str,
    ) -> CallRecord:
        record = CallRecord(
            seq=len(self._records),
            endpoint=endpoint,
            model=model,
            tier=tier,
            matched_fixture=matched_fixture,
            similarity=similarity,
            prompt_preview=_preview(prompt),
        )
        self._records.append(record)
        return record

    def record_killed(self) -> None:
        self._killed += 1

    def reset(self) -> None:
        self._records.clear()
        self._killed = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "killed": self._killed,
            "max_calls": self._max_calls,
            "calls": [asdict(record) for record in self._records],
        }
