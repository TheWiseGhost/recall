"""Embedder protocol, model provenance, and shared batching."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from recall.core.errors import EmbeddingError
from recall.core.models import RecallModel
from recall.core.registry import Registry

Vector = list[float]


class EmbeddingModelInfo(RecallModel):
    """Identity of the model that produced a set of vectors.

    Persisted with the vectors so an index can be validated (or invalidated)
    when configuration changes.
    """

    provider: str
    model: str
    dimensions: int
    normalized: bool = True
    # Cost per million input tokens, used by the evaluation layer. ``None``
    # means "free / local", not "unknown pricing we should guess at".
    cost_per_million_tokens: float | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Document and query embedding are separate methods because several models
    (BGE, E5, ...) require asymmetric prefixes.
    """

    @property
    def info(self) -> EmbeddingModelInfo: ...

    async def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    async def embed_query(self, query: str) -> Vector: ...


class EmbedderBase:
    """Base class handling batching, empty input, and dimension validation."""

    provider: str = "base"

    def __init__(self, *, model: str, dimensions: int, batch_size: int = 32) -> None:
        self._model = model
        self._dimensions = dimensions
        self.batch_size = max(1, batch_size)

    @property
    def info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider=self.provider, model=self._model, dimensions=self._dimensions
        )

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[Vector]:
        raise NotImplementedError

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Embed many texts, batching according to ``batch_size``."""
        if not texts:
            return []
        out: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors = await self._embed_batch(batch, is_query=False)
            if len(vectors) != len(batch):
                raise EmbeddingError(
                    f"{self.provider} returned {len(vectors)} vectors for {len(batch)} inputs"
                )
            out.extend(self._validate(v) for v in vectors)
        return out

    async def embed_query(self, query: str) -> Vector:
        """Embed a single query string."""
        vectors = await self._embed_batch([query], is_query=True)
        if not vectors:
            raise EmbeddingError(f"{self.provider} returned no vector for the query")
        return self._validate(vectors[0])

    def _validate(self, vector: Vector) -> Vector:
        if len(vector) != self._dimensions:
            raise EmbeddingError(
                f"{self.provider}:{self._model} produced a {len(vector)}-dimensional "
                f"vector but is configured for {self._dimensions}"
            )
        return vector


def l2_normalize(vector: Vector) -> Vector:
    """Scale ``vector`` to unit length; a zero vector is returned unchanged."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return list(vector)
    return [value / norm for value in vector]


embedder_registry: Registry[Embedder] = Registry("embedding provider")
