"""Tiered prompt matching for the mock LLM server.

An incoming request's prompt is matched against the recorded fixtures through
four tiers, in order, stopping at the first hit:

1. ``exact``      — byte-for-byte equality.
2. ``normalized`` — equal after collapsing whitespace runs and stripping ends.
3. ``semantic``   — cosine similarity of embeddings ``>= threshold`` (skipped
   entirely when no embedder is supplied, so the server and its tests run
   without the optional ``fastembed`` dependency).
4. ``miss``       — nothing matched; the caller must fail **loudly** (HTTP 500
   with the closest candidates) rather than return a silent wrong answer.

Why tiers: an agent re-run sends *almost* the same prompts, but timestamps,
whitespace, JSON key ordering, and minor model phrasing drift. Exact match is
too brittle; pure semantic match is too loose. The ladder matches the cheap,
safe cases first and only falls back to fuzzy similarity — with a calibrated
threshold (see ``variance_experiment``) — for the rest.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

_WHITESPACE_RE = re.compile(r"\s+")

# How many nearest candidates to attach to a result for diagnostics.
_CLOSEST_N = 5


class MatchTier(str, Enum):
    """Which tier produced a match (or ``MISS`` if none did)."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    SEMANTIC = "semantic"
    MISS = "miss"


def normalize(text: str) -> str:
    """Collapse every run of whitespace to a single space and strip the ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into fixed-length float vectors.

    Implemented for real by :class:`infra.agent_runtime.embedding.FastEmbedder`
    and faked deterministically in tests.
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class Candidate:
    """A fixture's matchable prompt, tagged with the fixture id it belongs to."""

    key: str
    prompt: str


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching one incoming prompt against the candidate set."""

    tier: MatchTier
    candidate: Candidate | None = None
    similarity: float | None = None
    # (fixture_id, similarity) for the nearest candidates — populated on a
    # semantic hit or miss to make loud failures debuggable.
    closest: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def hit(self) -> bool:
        return self.candidate is not None


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Plain-Python cosine similarity (no numpy dependency)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class Matcher:
    """Matches prompts against a fixed candidate set using the four tiers.

    Normalized forms (and, when an embedder is supplied, embeddings) are
    precomputed once at construction so each ``match`` call is cheap.
    """

    def __init__(
        self,
        candidates: Sequence[Candidate],
        *,
        threshold: float = 0.9,
        embedder: Embedder | None = None,
    ) -> None:
        self._candidates: list[Candidate] = list(candidates)
        self._threshold = threshold
        self._embedder = embedder

        # First writer wins on collisions so the result is deterministic.
        self._by_exact: dict[str, Candidate] = {}
        self._by_normalized: dict[str, Candidate] = {}
        for candidate in self._candidates:
            self._by_exact.setdefault(candidate.prompt, candidate)
            self._by_normalized.setdefault(normalize(candidate.prompt), candidate)

        self._embeddings: list[list[float]] | None = None
        if embedder is not None and self._candidates:
            self._embeddings = embedder.embed([c.prompt for c in self._candidates])

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def semantic_enabled(self) -> bool:
        return self._embeddings is not None

    def match(self, prompt: str) -> MatchResult:
        exact = self._by_exact.get(prompt)
        if exact is not None:
            return MatchResult(MatchTier.EXACT, exact)

        normalized = self._by_normalized.get(normalize(prompt))
        if normalized is not None:
            return MatchResult(MatchTier.NORMALIZED, normalized)

        if self._embedder is not None and self._embeddings:
            return self._semantic_match(prompt)

        return MatchResult(MatchTier.MISS)

    def _semantic_match(self, prompt: str) -> MatchResult:
        assert self._embedder is not None and self._embeddings is not None
        query = self._embedder.embed([prompt])[0]
        scored = sorted(
            (
                (cosine_similarity(query, embedding), candidate)
                for embedding, candidate in zip(self._embeddings, self._candidates)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        closest = tuple((cand.key, round(sim, 4)) for sim, cand in scored[:_CLOSEST_N])
        best_sim, best_candidate = scored[0]
        if best_sim >= self._threshold:
            return MatchResult(MatchTier.SEMANTIC, best_candidate, best_sim, closest)
        return MatchResult(MatchTier.MISS, None, best_sim, closest)
