"""Cross-encoder reranking via sentence-transformers.

A bi-encoder (what dense retrieval uses) embeds the query and the document
separately and compares the two vectors, so a document's representation is
fixed before the query is known. A cross-encoder reads the pair jointly and
outputs one relevance score, which lets it use interactions between the two —
and makes it O(candidates) forward passes per query rather than one.

That is why it reranks a shortlist instead of scoring a corpus, and why its
latency belongs in the result rather than in a log line.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from recall.core.errors import ConfigurationError, RecallError
from recall.core.models import SearchResult
from recall.core.reranking.base import (
    preserve_retrieval_score,
    reranker_registry,
)
from recall.core.retrieval.base import rerank_positions

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerUnavailableError(RecallError):
    """The reranker's optional dependency is not installed."""


@reranker_registry.decorator("cross_encoder")
class CrossEncoderReranker:
    """Scores each (query, chunk) pair with a sentence-transformers CrossEncoder.

    The model is loaded lazily on first use, so constructing this — which
    configuration validation does at startup — never triggers a model download.
    """

    name = "cross_encoder"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ConfigurationError(f"reranking batch_size must be at least 1, got {batch_size}")
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._encoder: Any | None = None

    def _load(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RerankerUnavailableError(
                "The cross_encoder reranker needs sentence-transformers. "
                'Install it with `pip install "recall[local]"`.'
            ) from exc
        kwargs: dict[str, Any] = {"max_length": self.max_length}
        if self.device:
            kwargs["device"] = self.device
        self._encoder = CrossEncoder(self.model, **kwargs)
        return self._encoder

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Synchronous scoring. Runs off the event loop; see :meth:`rerank`."""
        encoder = self._load()
        scores = encoder.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(score) for score in scores]

    async def rerank(
        self, query: str, results: Sequence[SearchResult], *, top_k: int
    ) -> list[SearchResult]:
        if not results or top_k <= 0:
            return []

        pairs = [(query, result.content) for result in results]
        # A cross-encoder forward pass is CPU/GPU-bound and blocking. Running it
        # on the event loop would stall every other request for its duration,
        # and would make concurrent latency measurements meaningless.
        scores = await asyncio.to_thread(self._score, pairs)

        # Scores are raw model outputs, not probabilities and not on the same
        # scale as cosine similarity or BM25. Only the ordering is meaningful,
        # which is what the retrieval metrics consume.
        scored = [
            preserve_retrieval_score(result, score)
            for result, score in zip(results, scores, strict=True)
        ]
        scored.sort(key=lambda result: (-result.score, str(result.chunk_id)))
        return rerank_positions(scored[:top_k])
