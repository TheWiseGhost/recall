"""Embedder base behaviour and the deterministic hashing embedder."""

from __future__ import annotations

import math

import pytest

from recall.core.embeddings.base import EmbedderBase, EmbeddingModelInfo, l2_normalize
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.errors import EmbeddingError


class TestL2Normalize:
    def test_produces_unit_length(self) -> None:
        assert math.isclose(sum(x * x for x in l2_normalize([3.0, 4.0])), 1.0)

    def test_zero_vector_is_returned_unchanged(self) -> None:
        assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


class TestEmbeddingModelInfo:
    def test_key_identifies_provider_and_model(self) -> None:
        info = EmbeddingModelInfo(provider="hash", model="hash-v1", dimensions=8)
        assert info.key == "hash:hash-v1"


class TestHashingEmbedder:
    async def test_dimensions_match_configuration(self) -> None:
        embedder = HashingEmbedder(dimensions=32)
        vectors = await embedder.embed_documents(["hello world"])
        assert len(vectors[0]) == 32

    async def test_is_deterministic(self) -> None:
        a = await HashingEmbedder(dimensions=32).embed_query("authentication tokens")
        b = await HashingEmbedder(dimensions=32).embed_query("authentication tokens")
        assert a == b

    async def test_vectors_are_normalized(self) -> None:
        vector = await HashingEmbedder(dimensions=64).embed_query("bearer token")
        assert math.isclose(sum(x * x for x in vector), 1.0, rel_tol=1e-9)

    async def test_empty_input_returns_empty_output(self) -> None:
        assert await HashingEmbedder(dimensions=8).embed_documents([]) == []

    async def test_batching_preserves_order_and_count(self) -> None:
        embedder = HashingEmbedder(dimensions=32, batch_size=2)
        texts = [f"document number {i}" for i in range(7)]
        vectors = await embedder.embed_documents(texts)
        assert len(vectors) == 7
        singles = [(await embedder.embed_documents([text]))[0] for text in texts]
        assert vectors == singles

    async def test_similar_text_scores_higher_than_unrelated_text(self) -> None:
        """Not a quality claim — just enough lexical signal to test plumbing."""
        embedder = HashingEmbedder(dimensions=512)
        query = await embedder.embed_query("how are bearer tokens verified")
        related, unrelated = await embedder.embed_documents(
            [
                "bearer tokens are verified by checking the signature and expiry",
                "the kitchen renovation used reclaimed oak flooring throughout",
            ]
        )
        cosine = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
        assert cosine(query, related) > cosine(query, unrelated)

    async def test_empty_string_is_handled(self) -> None:
        vector = await HashingEmbedder(dimensions=16).embed_query("")
        assert vector == [0.0] * 16

    def test_info_reports_provider_and_dimensions(self) -> None:
        info = HashingEmbedder(dimensions=16).info
        assert info.provider == "hash"
        assert info.dimensions == 16


class BrokenEmbedder(EmbedderBase):
    """Returns the wrong shape, to exercise the base class's validation."""

    provider = "broken"

    def __init__(self, *, count: int = 1, size: int = 4) -> None:
        super().__init__(model="broken", dimensions=4)
        self.count = count
        self.size = size

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        return [[0.0] * self.size for _ in range(self.count)]


class TestEmbedderValidation:
    async def test_wrong_vector_count_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="returned 1 vectors for 2 inputs"):
            await BrokenEmbedder(count=1).embed_documents(["a", "b"])

    async def test_wrong_dimensionality_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="configured for 4"):
            await BrokenEmbedder(size=8).embed_query("a")

    async def test_missing_query_vector_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="no vector"):
            await BrokenEmbedder(count=0).embed_query("a")
