"""OpenAI embeddings.

Installed through the ``openai`` extra. The API key comes from configuration or
``OPENAI_API_KEY``; it is never persisted and never logged.
"""

from __future__ import annotations

from typing import Any

from recall.core.embeddings.base import (
    EmbedderBase,
    EmbeddingModelInfo,
    Vector,
    embedder_registry,
)
from recall.core.errors import (
    EmbeddingError,
    EmbeddingProviderUnavailableError,
    TransientError,
)

# Native dimensions. ``text-embedding-3-*`` also support truncation via the
# `dimensions` request parameter.
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# USD per million input tokens, for cost estimation only. Update as prices move;
# see docs/experiments/cost-model.md.
_MODEL_COST_PER_MTOK: dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}


@embedder_registry.decorator("openai")
class OpenAIEmbedder(EmbedderBase):
    """Wraps the OpenAI embeddings endpoint."""

    provider = "openai"

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        batch_size: int = 128,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        resolved = dimensions if dimensions is not None else _MODEL_DIMENSIONS.get(model)
        if resolved is None:
            raise EmbeddingError(
                f"Unknown OpenAI embedding model {model!r}; set embedding.dimensions explicitly"
            )
        super().__init__(model=model, dimensions=int(resolved), batch_size=batch_size)
        self._explicit_dimensions = dimensions is not None
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._client: Any | None = None

    @property
    def info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider=self.provider,
            model=self._model,
            dimensions=self._dimensions,
            cost_per_million_tokens=_MODEL_COST_PER_MTOK.get(self._model),
        )

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - depends on env
                raise EmbeddingProviderUnavailableError(
                    "The openai package is not installed. "
                    "Install it with: pip install 'recall[openai]'"
                ) from exc
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[Vector]:
        client = self._get_client()
        kwargs: dict[str, Any] = {"model": self._model, "input": texts}
        if self._explicit_dimensions and self._model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimensions
        try:
            response = await client.embeddings.create(**kwargs)
        except Exception as exc:
            if _is_transient(exc):
                raise TransientError(f"OpenAI embeddings temporarily unavailable: {exc}") from exc
            raise EmbeddingError(f"OpenAI embeddings request failed: {exc}") from exc
        # The API guarantees ordering by `index`, but sort defensively.
        items = sorted(response.data, key=lambda item: item.index)
        return [[float(x) for x in item.embedding] for item in items]


def _is_transient(exc: Exception) -> bool:
    """Classify a vendor exception as retryable without importing openai eagerly."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    name = type(exc).__name__
    return name in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
    }
