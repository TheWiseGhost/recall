"""Ports: the storage-side interfaces the domain depends on.

These live in ``core`` on purpose. Retrievers and pipelines are written against
these protocols, and ``recall.storage.postgres`` implements them. Adding a
Qdrant or Weaviate backend later means implementing :class:`VectorIndex` — no
change to any caller.
"""

from __future__ import annotations

import builtins
import uuid
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from recall.core.embeddings.base import EmbeddingModelInfo, Vector
from recall.core.models import Chunk, Document, SearchFilters, SearchResult, SourceType


@runtime_checkable
class DocumentRepository(Protocol):
    """Persistence for :class:`Document`."""

    async def upsert(self, document: Document) -> Document: ...

    async def get(self, document_id: uuid.UUID) -> Document | None: ...

    async def get_by_source(self, source_type: SourceType, source_id: str) -> Document | None: ...

    # NB: defining a method called `list` shadows the builtin for every
    # annotation later in this class body, hence `builtins.list` below.
    async def list(
        self,
        *,
        source_types: Sequence[SourceType] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> builtins.list[Document]: ...

    async def count(self, *, source_types: Sequence[SourceType] | None = None) -> int: ...

    async def checksums(self, source_type: SourceType) -> dict[str, str]:
        """Map ``source_id -> checksum`` for one source, for incremental sync."""
        ...

    async def delete(self, document_id: uuid.UUID) -> bool: ...

    async def delete_missing(
        self, source_type: SourceType, present_source_ids: Sequence[str]
    ) -> builtins.list[uuid.UUID]:
        """Delete documents of ``source_type`` absent from ``present_source_ids``.

        How a sync reflects deletions at the source. Returns the deleted IDs.
        """
        ...


@runtime_checkable
class ChunkRepository(Protocol):
    """Persistence for :class:`Chunk`."""

    async def replace_for_document(self, document_id: uuid.UUID, chunks: Sequence[Chunk]) -> int:
        """Atomically make ``chunks`` the complete chunk set for a document."""
        ...

    async def get(self, chunk_id: uuid.UUID) -> Chunk | None: ...

    async def get_many(self, chunk_ids: Sequence[uuid.UUID]) -> list[Chunk]: ...

    async def list_for_document(self, document_id: uuid.UUID) -> list[Chunk]: ...

    async def count(self) -> int: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Approximate-nearest-neighbour index over chunk embeddings."""

    async def upsert(
        self,
        chunk_ids: Sequence[uuid.UUID],
        vectors: Sequence[Vector],
        model: EmbeddingModelInfo,
    ) -> int: ...

    async def query(
        self,
        vector: Vector,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
        model: EmbeddingModelInfo | None = None,
    ) -> list[SearchResult]: ...

    async def delete_for_document(self, document_id: uuid.UUID) -> int: ...

    async def count(self) -> int: ...

    async def models(self) -> list[EmbeddingModelInfo]:
        """Which embedding models currently have vectors in the index."""
        ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Full-text index over chunk content, scored by term statistics.

    The mirror image of :class:`VectorIndex`. It exists as a port for the same
    reason: the scoring is inherently a storage-engine concern (it needs the
    inverted index and corpus-wide statistics), but retrievers must be able to
    consume it without importing SQLAlchemy.

    Implementations own their own scoring function and its parameters —
    ``k1``/``b`` are BM25's, not the port's.
    """

    name: str

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]: ...


@runtime_checkable
class IngestStore(Protocol):
    """The slice of storage the ingestion pipeline needs.

    Narrower than the full backend on purpose: it is the seam that lets the
    sync logic — the part most worth testing — run without a database.
    """

    @property
    def documents(self) -> DocumentRepository: ...

    async def index_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        model: EmbeddingModelInfo,
    ) -> int:
        """Write document, chunks and vectors atomically. Returns chunks written."""
        ...
