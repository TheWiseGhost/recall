"""Async engine and session management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from recall.config.settings import DatabaseSettings


def create_engine(settings: DatabaseSettings) -> AsyncEngine:
    """Build the async engine described by ``settings``."""
    connect_args: dict[str, object] = {}
    if settings.statement_timeout_ms > 0:
        # asyncpg has no per-statement timeout knob in the URL, so set it as a
        # server setting for every connection in the pool.
        connect_args["server_settings"] = {"statement_timeout": str(settings.statement_timeout_ms)}
    return create_async_engine(
        settings.url,
        echo=settings.echo,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory with autoflush off, so writes happen where we say."""
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False, class_=AsyncSession)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
