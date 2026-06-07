"""Shared fixtures and in-memory fakes.

The fakes implement the same ports the PostgreSQL adapter does, so unit tests
exercise the real pipeline logic without a database.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest

from recall.core.embeddings.base import EmbeddingModelInfo, Vector
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.models import (
    Chunk,
    Document,
    SearchFilters,
    SearchResult,
    SourceItem,
    SourceType,
)

FIXTURES = Path(__file__).parent / "fixtures"


# --- fakes ------------------------------------------------------------------


class FakeDocumentRepository:
    """In-memory :class:`recall.core.ports.DocumentRepository`."""

    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, Document] = {}

    async def upsert(self, document: Document) -> Document:
        self.documents[document.id] = document
        return document

    async def get(self, document_id: uuid.UUID) -> Document | None:
        return self.documents.get(document_id)

    async def get_by_source(self, source_type: SourceType, source_id: str) -> Document | None:
        for document in self.documents.values():
            if document.source_type is source_type and document.source_id == source_id:
                return document
        return None

    async def list(
        self,
        *,
        source_types: Sequence[SourceType] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Document]:
        documents = list(self.documents.values())
        if source_types:
            allowed = set(source_types)
            documents = [d for d in documents if d.source_type in allowed]
        return documents[offset : offset + limit]

    async def count(self, *, source_types: Sequence[SourceType] | None = None) -> int:
        return len(await self.list(source_types=source_types, limit=10**9))

    async def checksums(self, source_type: SourceType) -> dict[str, str]:
        return {
            d.source_id: d.checksum for d in self.documents.values() if d.source_type is source_type
        }

    async def delete(self, document_id: uuid.UUID) -> bool:
        return self.documents.pop(document_id, None) is not None

    async def delete_missing(
        self, source_type: SourceType, present_source_ids: Sequence[str]
    ) -> list[uuid.UUID]:
        present = set(present_source_ids)
        stale = [
            d.id
            for d in self.documents.values()
            if d.source_type is source_type and d.source_id not in present
        ]
        for document_id in stale:
            del self.documents[document_id]
        return stale


class FakeStorage:
    """In-memory :class:`recall.core.ports.IngestStore` plus a vector index."""

    def __init__(self) -> None:
        self._documents = FakeDocumentRepository()
        self.chunks: dict[uuid.UUID, Chunk] = {}
        self.vectors: dict[uuid.UUID, Vector] = {}
        self.model: EmbeddingModelInfo | None = None
        # Records how many times each document was actually re-indexed, which
        # is what the incremental-sync tests assert on.
        self.index_calls: list[uuid.UUID] = []

    @property
    def documents(self) -> FakeDocumentRepository:
        return self._documents

    async def index_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        model: EmbeddingModelInfo,
    ) -> int:
        await self._documents.upsert(document)
        self.index_calls.append(document.id)
        self.model = model
        stale = [cid for cid, chunk in self.chunks.items() if chunk.document_id == document.id]
        for chunk_id in stale:
            del self.chunks[chunk_id]
            self.vectors.pop(chunk_id, None)
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.chunks[chunk.id] = chunk
            self.vectors[chunk.id] = list(vector)
        return len(chunks)

    async def query(
        self,
        vector: Vector,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
        model: EmbeddingModelInfo | None = None,
    ) -> list[SearchResult]:
        """Exact cosine search, so retriever tests do not need pgvector."""
        scored: list[tuple[float, uuid.UUID]] = []
        for chunk_id, stored in self.vectors.items():
            chunk = self.chunks[chunk_id]
            document = self._documents.documents[chunk.document_id]
            if (
                filters
                and filters.source_types
                and document.source_type not in filters.source_types
            ):
                continue
            scored.append((_cosine(vector, stored), chunk_id))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: list[SearchResult] = []
        for rank, (score, chunk_id) in enumerate(scored[:top_k], start=1):
            chunk = self.chunks[chunk_id]
            document = self._documents.documents[chunk.document_id]
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=score,
                    rank=rank,
                    metadata=dict(chunk.metadata),
                    document_title=document.title,
                    document_uri=document.uri,
                    source_type=document.source_type,
                )
            )
        return results


def _cosine(a: Vector, b: Vector) -> float:
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


class ListConnector:
    """Connector over a fixed list of documents, for pipeline tests."""

    def __init__(
        self, documents: Sequence[Document], *, source_type: SourceType = SourceType.MEMORY
    ) -> None:
        self.source_type = source_type
        self.documents = {d.source_id: d for d in documents}
        self.fetch_count: dict[str, int] = {}
        self.advertise_checksums = False

    def set(self, documents: Sequence[Document]) -> None:
        self.documents = {d.source_id: d for d in documents}

    async def discover(self) -> list[SourceItem]:
        return [
            SourceItem(
                source_id=document.source_id,
                source_type=self.source_type,
                uri=document.uri,
                title=document.title,
                checksum=document.checksum if self.advertise_checksums else None,
            )
            for document in self.documents.values()
        ]

    async def fetch(self, item: SourceItem) -> Document:
        self.fetch_count[item.source_id] = self.fetch_count.get(item.source_id, 0) + 1
        return self.documents[item.source_id]


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimensions=64)


@pytest.fixture
def storage() -> FakeStorage:
    return FakeStorage()


@pytest.fixture
def sample_document() -> Document:
    return Document.create(
        source_id="auth.md",
        source_type=SourceType.FILESYSTEM,
        title="Authentication",
        content=(
            "The API authenticates requests with a bearer token supplied in the "
            "Authorization header. Tokens are issued by the auth service and are "
            "valid for one hour. Every request is verified in three steps: the "
            "signature is checked, the expiry claim is compared against the current "
            "time, and the scope is matched against the endpoint requirement."
        ),
        uri="file:///corpus/auth.md",
        metadata={"file_type": "md", "author": "docs-team", "tags": ["security"]},
    )
