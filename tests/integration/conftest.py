"""Fixtures for tests that need a real PostgreSQL + pgvector instance.

Point ``RECALL_TEST_DATABASE_URL`` at a database you are happy to have wiped;
otherwise the Compose default is used. If nothing is reachable the whole module
is skipped rather than failing, so ``pytest`` stays green on a laptop with no
Docker running.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text

from recall.config.settings import Settings
from recall.core.chunking.fixed import FixedSizeChunker
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.retrieval.dense import DenseRetriever
from recall.pipeline.ingest import IngestionPipeline
from recall.storage.postgres.storage import Storage, create_storage

TEST_DATABASE_URL = os.environ.get(
    "RECALL_TEST_DATABASE_URL",
    "postgresql+asyncpg://recall:recall@localhost:5432/recall_test",
)

# Small on purpose: fast to embed, and unrelated to any real model.
TEST_DIMENSIONS = 64


def test_settings() -> Settings:
    return Settings.from_mapping(
        {
            "database": {"url": TEST_DATABASE_URL},
            "embedding": {
                "provider": "hash",
                "model": "hash-v1",
                "dimensions": TEST_DIMENSIONS,
            },
            "chunking": {"strategy": "fixed", "chunk_size": 48, "overlap": 8},
        }
    )


async def _database_is_available(settings: Settings) -> bool:
    storage = create_storage(settings.database)
    try:
        async with storage.sessions() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        await storage.close()


@pytest_asyncio.fixture(scope="session")
async def settings() -> Settings:
    resolved = test_settings()
    if await _database_is_available(resolved):
        return resolved

    message = (
        f"No PostgreSQL at {TEST_DATABASE_URL}. "
        "Start one with `docker compose up -d postgres` and create the "
        "`recall_test` database, or set RECALL_TEST_DATABASE_URL."
    )
    # Skipping keeps `pytest` green on a laptop with no Docker running. In CI
    # that would make the whole integration suite pass vacuously, so fail
    # loudly instead.
    if os.environ.get("CI"):
        pytest.fail(message, pytrace=False)
    pytest.skip(message, allow_module_level=True)


@pytest_asyncio.fixture(scope="session")
async def migrated(settings: Settings) -> Settings:
    """Apply migrations once per session, from a clean schema."""
    import asyncio

    from recall.storage.postgres import migrate as migrations

    storage = create_storage(settings.database)
    try:
        async with storage.sessions() as session:
            await session.execute(text("DROP SCHEMA public CASCADE"))
            await session.execute(text("CREATE SCHEMA public"))
            await session.commit()
    finally:
        await storage.close()

    # Alembic's env.py calls asyncio.run, so it must not run on this loop.
    await asyncio.to_thread(migrations.upgrade, settings)
    return settings


@pytest_asyncio.fixture
async def storage(migrated: Settings) -> AsyncIterator[Storage]:
    """A Storage with empty tables. Truncation cascades to chunks and vectors."""
    instance = create_storage(migrated.database)
    async with instance.sessions() as session:
        await session.execute(text("TRUNCATE documents CASCADE"))
        await session.commit()
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimensions=TEST_DIMENSIONS)


@pytest.fixture
def chunker() -> FixedSizeChunker:
    return FixedSizeChunker(chunk_size=48, overlap=8)


@pytest.fixture
def pipeline(
    storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
) -> IngestionPipeline:
    return IngestionPipeline(storage=storage, chunker=chunker, embedder=embedder)


@pytest.fixture
def retriever(storage: Storage, embedder: HashingEmbedder) -> DenseRetriever:
    return DenseRetriever(embedder=embedder, index=storage.vectors)
