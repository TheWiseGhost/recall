"""PostgreSQL implementations of the storage ports."""

from __future__ import annotations

import builtins
import uuid
from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from recall.core.models import Chunk, Document, SourceType
from recall.storage.postgres.engine import session_scope
from recall.storage.postgres.mapping import to_chunk, to_chunk_row, to_document
from recall.storage.postgres.models import ChunkEmbeddingRow, ChunkRow, DocumentRow

# --- session-scoped primitives, shared by repositories and the unit of work ---


async def upsert_document(session: AsyncSession, document: Document) -> Document:
    """Insert or update a document by primary key. Idempotent.

    Note the ``meta`` key: the column is named ``metadata``, but that string is
    also the name of SQLAlchemy's ``Base.metadata``, so a string key would
    resolve to the wrong thing. Use the mapped attribute name here and Column
    objects in ``set_``.
    """
    table = DocumentRow.__table__
    statement = (
        pg_insert(DocumentRow)
        .values(
            id=document.id,
            source_id=document.source_id,
            source_type=document.source_type.value,
            title=document.title,
            content=document.content,
            uri=document.uri,
            meta=dict(document.metadata),
            checksum=document.checksum,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
        .on_conflict_do_update(
            index_elements=[table.c.id],
            set_={
                table.c.title: document.title,
                table.c.content: document.content,
                table.c.uri: document.uri,
                table.c.metadata: dict(document.metadata),
                table.c.checksum: document.checksum,
                table.c.updated_at: document.updated_at,
            },
        )
    )
    await session.execute(statement)
    return document


async def replace_chunks(
    session: AsyncSession, document_id: uuid.UUID, chunks: Sequence[Chunk]
) -> int:
    """Make ``chunks`` the complete chunk set for a document.

    Deleting first cascades to ``chunk_embeddings``, which is exactly what we
    want: stale vectors must never outlive the chunk they described.
    """
    await session.execute(delete(ChunkRow).where(ChunkRow.document_id == document_id))
    if not chunks:
        return 0
    session.add_all([to_chunk_row(chunk) for chunk in chunks])
    await session.flush()
    return len(chunks)


async def upsert_embeddings(
    session: AsyncSession,
    rows: Sequence[dict[str, object]],
) -> int:
    """Insert or replace vectors keyed by ``(chunk_id, model_key)``."""
    if not rows:
        return 0
    statement = pg_insert(ChunkEmbeddingRow).values(list(rows))
    statement = statement.on_conflict_do_update(
        index_elements=[ChunkEmbeddingRow.chunk_id, ChunkEmbeddingRow.model_key],
        set_={
            "embedding": statement.excluded.embedding,
            "document_id": statement.excluded.document_id,
            "provider": statement.excluded.provider,
            "model": statement.excluded.model,
            "dimensions": statement.excluded.dimensions,
        },
    )
    await session.execute(statement)
    return len(rows)


# --- repositories -----------------------------------------------------------


class PostgresDocumentRepository:
    """Implements :class:`recall.core.ports.DocumentRepository`."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def upsert(self, document: Document) -> Document:
        async with session_scope(self._sessions) as session:
            return await upsert_document(session, document)

    async def get(self, document_id: uuid.UUID) -> Document | None:
        async with session_scope(self._sessions) as session:
            row = await session.get(DocumentRow, document_id)
            return to_document(row) if row else None

    async def get_by_source(self, source_type: SourceType, source_id: str) -> Document | None:
        async with session_scope(self._sessions) as session:
            statement = select(DocumentRow).where(
                DocumentRow.source_type == source_type.value,
                DocumentRow.source_id == source_id,
            )
            row = (await session.execute(statement)).scalar_one_or_none()
            return to_document(row) if row else None

    async def list(
        self,
        *,
        source_types: Sequence[SourceType] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> builtins.list[Document]:
        async with session_scope(self._sessions) as session:
            statement = select(DocumentRow).order_by(DocumentRow.updated_at.desc(), DocumentRow.id)
            if source_types:
                statement = statement.where(
                    DocumentRow.source_type.in_([t.value for t in source_types])
                )
            statement = statement.limit(limit).offset(offset)
            rows = (await session.execute(statement)).scalars().all()
            return [to_document(row) for row in rows]

    async def count(self, *, source_types: Sequence[SourceType] | None = None) -> int:
        async with session_scope(self._sessions) as session:
            statement = select(func.count()).select_from(DocumentRow)
            if source_types:
                statement = statement.where(
                    DocumentRow.source_type.in_([t.value for t in source_types])
                )
            return int((await session.execute(statement)).scalar_one())

    async def checksums(self, source_type: SourceType) -> dict[str, str]:
        """``source_id -> checksum`` for one source type.

        One query for the whole source keeps incremental sync O(1) round trips
        rather than O(number of discovered items).
        """
        async with session_scope(self._sessions) as session:
            statement = select(DocumentRow.source_id, DocumentRow.checksum).where(
                DocumentRow.source_type == source_type.value
            )
            return {row[0]: row[1] for row in (await session.execute(statement)).all()}

    async def delete(self, document_id: uuid.UUID) -> bool:
        async with session_scope(self._sessions) as session:
            result = await session.execute(delete(DocumentRow).where(DocumentRow.id == document_id))
            return bool(cast("CursorResult[Any]", result).rowcount)

    async def delete_missing(
        self, source_type: SourceType, present_source_ids: Sequence[str]
    ) -> builtins.list[uuid.UUID]:
        """Delete documents of ``source_type`` whose source_id is not listed.

        This is how a sync reflects deletions at the source.
        """
        async with session_scope(self._sessions) as session:
            statement = select(DocumentRow.id, DocumentRow.source_id).where(
                DocumentRow.source_type == source_type.value
            )
            existing = (await session.execute(statement)).all()
            present = set(present_source_ids)
            stale = [row[0] for row in existing if row[1] not in present]
            if stale:
                await session.execute(delete(DocumentRow).where(DocumentRow.id.in_(stale)))
            return stale


class PostgresChunkRepository:
    """Implements :class:`recall.core.ports.ChunkRepository`."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def replace_for_document(self, document_id: uuid.UUID, chunks: Sequence[Chunk]) -> int:
        async with session_scope(self._sessions) as session:
            return await replace_chunks(session, document_id, chunks)

    async def get(self, chunk_id: uuid.UUID) -> Chunk | None:
        async with session_scope(self._sessions) as session:
            row = await session.get(ChunkRow, chunk_id)
            return to_chunk(row) if row else None

    async def get_many(self, chunk_ids: Sequence[uuid.UUID]) -> list[Chunk]:
        if not chunk_ids:
            return []
        async with session_scope(self._sessions) as session:
            statement = select(ChunkRow).where(ChunkRow.id.in_(list(chunk_ids)))
            rows = (await session.execute(statement)).scalars().all()
            by_id = {row.id: to_chunk(row) for row in rows}
            # Preserve the caller's ordering.
            return [by_id[cid] for cid in chunk_ids if cid in by_id]

    async def list_for_document(self, document_id: uuid.UUID) -> list[Chunk]:
        async with session_scope(self._sessions) as session:
            statement = (
                select(ChunkRow)
                .where(ChunkRow.document_id == document_id)
                .order_by(ChunkRow.position)
            )
            rows = (await session.execute(statement)).scalars().all()
            return [to_chunk(row) for row in rows]

    async def count(self) -> int:
        async with session_scope(self._sessions) as session:
            return int(
                (await session.execute(select(func.count()).select_from(ChunkRow))).scalar_one()
            )
