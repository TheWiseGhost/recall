"""pgvector-backed vector index."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from recall.core.embeddings.base import EmbeddingModelInfo, Vector
from recall.core.errors import DimensionMismatchError
from recall.core.models import SearchFilters, SearchResult, SourceType
from recall.storage.postgres.engine import session_scope
from recall.storage.postgres.filters import build_document_predicates
from recall.storage.postgres.models import ChunkEmbeddingRow, ChunkRow, DocumentRow
from recall.storage.postgres.repositories import upsert_embeddings


def embedding_rows(
    chunk_ids: Sequence[uuid.UUID],
    document_id: uuid.UUID,
    vectors: Sequence[Vector],
    model: EmbeddingModelInfo,
) -> list[dict[str, object]]:
    """Build ``chunk_embeddings`` row dicts, validating dimensions first."""
    if len(chunk_ids) != len(vectors):
        raise ValueError(f"got {len(chunk_ids)} chunk ids but {len(vectors)} vectors")
    rows: list[dict[str, object]] = []
    for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
        if len(vector) != model.dimensions:
            raise DimensionMismatchError(model.dimensions, len(vector))
        rows.append(
            {
                "chunk_id": chunk_id,
                "model_key": model.key,
                "document_id": document_id,
                "embedding": list(vector),
                "provider": model.provider,
                "model": model.model,
                "dimensions": model.dimensions,
            }
        )
    return rows


class PostgresVectorIndex:
    """Implements :class:`recall.core.ports.VectorIndex` on pgvector.

    Similarity is cosine distance. Vectors from normalising embedders are unit
    length, so ``score = 1 - distance`` is cosine similarity in ``[-1, 1]``,
    and in ``[0, 1]`` for the non-negative embedders we ship.

    Note on filtered search: pgvector applies the ANN index before predicates,
    so a highly selective filter can fall back to an exact scan. That is
    correct, just slower; see docs/architecture/storage.md.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def upsert(
        self,
        chunk_ids: Sequence[uuid.UUID],
        vectors: Sequence[Vector],
        model: EmbeddingModelInfo,
    ) -> int:
        if not chunk_ids:
            return 0
        async with session_scope(self._sessions) as session:
            document_ids = (
                await session.execute(
                    select(ChunkRow.id, ChunkRow.document_id).where(
                        ChunkRow.id.in_(list(chunk_ids))
                    )
                )
            ).all()
            lookup = {row[0]: row[1] for row in document_ids}
            rows: list[dict[str, object]] = []
            for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
                document_id = lookup.get(chunk_id)
                if document_id is None:
                    continue  # chunk was deleted concurrently
                rows.extend(embedding_rows([chunk_id], document_id, [vector], model))
            return await upsert_embeddings(session, rows)

    async def query(
        self,
        vector: Vector,
        *,
        top_k: int,
        filters: SearchFilters | None = None,
        model: EmbeddingModelInfo | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        if model is not None and len(vector) != model.dimensions:
            raise DimensionMismatchError(model.dimensions, len(vector))

        distance = ChunkEmbeddingRow.embedding.cosine_distance(list(vector)).label("distance")
        # Labels are explicit because `chunks` and `documents` both have a
        # `metadata` column; positional/derived keys would collide.
        statement = (
            select(
                ChunkEmbeddingRow.chunk_id.label("chunk_id"),
                ChunkEmbeddingRow.document_id.label("document_id"),
                ChunkRow.content.label("content"),
                ChunkRow.meta.label("chunk_metadata"),
                DocumentRow.title.label("document_title"),
                DocumentRow.uri.label("document_uri"),
                DocumentRow.source_type.label("source_type"),
                distance,
            )
            .join(ChunkRow, ChunkRow.id == ChunkEmbeddingRow.chunk_id)
            .join(DocumentRow, DocumentRow.id == ChunkEmbeddingRow.document_id)
            .order_by(distance)
            .limit(top_k)
        )
        if model is not None:
            statement = statement.where(ChunkEmbeddingRow.model_key == model.key)
        for predicate in build_document_predicates(filters):
            statement = statement.where(predicate)

        async with session_scope(self._sessions) as session:
            rows = (await session.execute(statement)).all()

        return [
            SearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                score=round(1.0 - float(row.distance), 6),
                rank=rank,
                metadata=dict(row.chunk_metadata or {}),
                document_title=row.document_title,
                document_uri=row.document_uri,
                source_type=SourceType(row.source_type),
                retriever="dense",
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def delete_for_document(self, document_id: uuid.UUID) -> int:
        async with session_scope(self._sessions) as session:
            result = await session.execute(
                delete(ChunkEmbeddingRow).where(ChunkEmbeddingRow.document_id == document_id)
            )
            return int(cast("CursorResult[Any]", result).rowcount or 0)

    async def count(self) -> int:
        async with session_scope(self._sessions) as session:
            statement = select(func.count()).select_from(ChunkEmbeddingRow)
            return int((await session.execute(statement)).scalar_one())

    async def models(self) -> list[EmbeddingModelInfo]:
        async with session_scope(self._sessions) as session:
            statement = select(
                ChunkEmbeddingRow.provider,
                ChunkEmbeddingRow.model,
                ChunkEmbeddingRow.dimensions,
            ).distinct()
            rows = (await session.execute(statement)).all()
        return [
            EmbeddingModelInfo(provider=row[0], model=row[1], dimensions=row[2]) for row in rows
        ]
