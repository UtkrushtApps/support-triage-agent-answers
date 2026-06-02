"""Optional fastembed-backed embedder for the semantic match tier.

Kept separate from :mod:`infra.agent_runtime.matching` so the matcher stays
dependency-free: the server and its unit tests run with no embedder (exact +
normalized tiers only), and the semantic tier switches on only when fastembed
is installed and an embedder is wired in.

The default model (``BAAI/bge-small-en-v1.5``, 384-dim) is small and downloads
once on first use; embeddings are deterministic for a fixed model, which is
what makes the semantic tier reproducible.
"""
from __future__ import annotations

from typing import Sequence

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedder:
    """Wraps ``fastembed.TextEmbedding`` behind the matcher's Embedder protocol.

    Both the import and the model load are deferred until the first ``embed``
    call so merely constructing one is cheap and import-safe.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None  # lazily constructed

    def _ensure_model(self):
        if self._model is None:
            from fastembed import TextEmbedding  # local import: optional dep

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        return [[float(x) for x in vector] for vector in model.embed(list(texts))]


def try_load_embedder(model_name: str = DEFAULT_MODEL) -> FastEmbedder | None:
    """Return a :class:`FastEmbedder` if fastembed is importable, else ``None``."""
    try:
        import fastembed  # noqa: F401  (probe only)
    except ImportError:
        return None
    return FastEmbedder(model_name)
