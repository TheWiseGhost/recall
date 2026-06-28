"""Storage facade.

Bundles the engine, the three repositories, and the one operation that must be
transactional: writing a document together with its chunks and their vectors.

That atomicity is the core reliability guarantee from the spec — a failed
embedding job must not leave a document indexed with stale or missing chunks.
Embedding happens *before* the transaction opens; the write is all-or-nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Self

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from recall.config.settings import DatabaseSettings, LexicalSettings
from recall.core.embeddings.base import EmbeddingModelInfo, Vector
from recall.core.models import Chunk, Document
from recall.storage.postgres.engine import (
    create_engine,
    create_session_factory,
    session_scope,
)
from recall.storage.postgres.lexical_index import PostgresBM25Index
from recall.storage.postgres.repositories import (
    PostgresChunkRepository,
    PostgresDocumentRepository,
    replace_chunks,
    upsert_document,
    upsert_embeddings,
)
from recall.storage.postgres.vector_index import PostgresVectorIndex, embedding_rows


class Storage:
    """Owns the engine and exposes the repositories."""

    def __init__(self, engine: AsyncEngine, *, lexical: LexicalSettings | None = None) -> None:
        self.engine = engine
        self.sessions: async_sessionmaker[AsyncSession] = create_session_factory(engine)
        self.documents = PostgresDocumentRepository(self.sessions)
        self.chunks = PostgresChunkRepository(self.sessions)
        self.vectors = PostgresVectorIndex(self.sessions)
        tuning = lexical or LexicalSettings()
        self.lexical = PostgresBM25Index(self.sessions, k1=tuning.k1, b=tuning.b)

    async def index_document(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        model: EmbeddingModelInfo,
    ) -> int:
        """Persist a document, replace its chunks, and write their vectors.

        One transaction. Returns the number of chunks written.
        """
        rows: list[dict[str, object]] = []
        if chunks:
            rows = embedding_rows([chunk.id for chunk in chunks], document.id, list(vectors), model)

        async with session_scope(self.sessions) as session:
            await upsert_document(session, document)
            written = await replace_chunks(session, document.id, chunks)
            await upsert_embeddings(session, rows)
            return written

    async def health(self) -> dict[str, object]:
        """Report connectivity, the pgvector extension, and current counts."""
        async with session_scope(self.sessions) as session:
            extension = (
                await session.execute(
                    text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                )
            ).scalar_one_or_none()
            documents = (await session.execute(text("SELECT count(*) FROM documents"))).scalar_one()
            chunks = (await session.execute(text("SELECT count(*) FROM chunks"))).scalar_one()
            vectors = (
                await session.execute(text("SELECT count(*) FROM chunk_embeddings"))
            ).scalar_one()
        return {
            "connected": True,
            "pgvector_version": extension,
            "documents": int(documents),
            "chunks": int(chunks),
            "vectors": int(vectors),
        }

    async def close(self) -> None:
        await self.engine.dispose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


def create_storage(
    settings: DatabaseSettings, *, lexical: LexicalSettings | None = None
) -> Storage:
    """Build a :class:`Storage` from database settings."""
    return Storage(create_engine(settings), lexical=lexical)
