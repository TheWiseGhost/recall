"""Local embeddings via sentence-transformers / Hugging Face.

Installed through the ``local`` extra because it pulls in PyTorch. Encoding is
CPU/GPU-bound and synchronous, so it runs in a worker thread to keep the event
loop free.
"""

from __future__ import annotations

import asyncio
from typing import Any

from recall.core.embeddings.base import EmbedderBase, Vector, embedder_registry
from recall.core.errors import EmbeddingProviderUnavailableError

# Known models whose asymmetric query prefix materially affects quality.
_QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2": "query: ",
    "intfloat/e5-large-v2": "query: ",
}
_DOCUMENT_PREFIXES: dict[str, str] = {
    "intfloat/e5-base-v2": "passage: ",
    "intfloat/e5-large-v2": "passage: ",
}


@embedder_registry.decorator("sentence_transformers")
class SentenceTransformersEmbedder(EmbedderBase):
    """Wraps a ``sentence_transformers.SentenceTransformer`` model."""

    provider = "sentence_transformers"

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-base-en-v1.5",
        dimensions: int | None = None,
        batch_size: int = 32,
        device: str | None = None,
        normalize: bool = True,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        self._device = device
        self._normalize = normalize
        self._model_name = model
        self._encoder: Any | None = None
        self._query_prefix = (
            query_prefix if query_prefix is not None else _QUERY_PREFIXES.get(model, "")
        )
        self._document_prefix = (
            document_prefix if document_prefix is not None else _DOCUMENT_PREFIXES.get(model, "")
        )
        # Loading the model to read its dimension is expensive; allow config to
        # declare it and verify lazily on first encode.
        resolved = (
            dimensions
            if dimensions is not None
            else self._load().get_sentence_embedding_dimension()
        )
        super().__init__(model=model, dimensions=int(resolved), batch_size=batch_size)

    def _load(self) -> Any:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on env
                raise EmbeddingProviderUnavailableError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install 'recall[local]'"
                ) from exc
            self._encoder = SentenceTransformer(self._model_name, device=self._device)
        return self._encoder

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[Vector]:
        prefix = self._query_prefix if is_query else self._document_prefix
        prepared = [prefix + text for text in texts] if prefix else texts
        return await asyncio.to_thread(self._encode, prepared)

    def _encode(self, texts: list[str]) -> list[Vector]:
        encoder = self._load()
        arrays = encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in arrays]
