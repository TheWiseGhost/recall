"""Alembic environment.

Runs migrations through the async engine so Recall needs only one PostgreSQL
driver (asyncpg) rather than dragging in psycopg2 for migrations alone.
"""

from __future__ import annotations

import asyncio
from typing import Any

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from recall.config.settings import load_settings
from recall.storage.postgres.models import Base

config = context.config
target_metadata = Base.metadata

# `recall migrate` (and the test harness) construct the Alembic config
# programmatically and set these options from an already-loaded Settings.
# Only fall back to discovering configuration when running Alembic directly
# from the command line, where nothing has been set.
if not config.get_main_option("sqlalchemy.url", None):
    _settings = load_settings()
    config.set_main_option("sqlalchemy.url", _settings.database.url)
    config.set_main_option("recall.embedding_dimensions", str(_settings.embedding.dimensions))


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_object=_include_object,
    )


def _include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    # pgvector's ANN indexes are created with raw DDL; autogenerate should not
    # try to manage or drop them.
    return not (type_ == "index" and name is not None and name.endswith("_hnsw"))


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
