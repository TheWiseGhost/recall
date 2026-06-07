"""Programmatic access to Alembic, so migrations work from the CLI and Docker.

These functions are synchronous. ``env.py`` calls ``asyncio.run`` internally, so
they must never be invoked from inside a running event loop.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from recall.config.settings import Settings

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(settings: Settings) -> Config:
    """Build an Alembic config pointing at the packaged migrations."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    config.set_main_option("sqlalchemy.url", settings.database.url)
    # The initial migration needs the embedding dimension to size the pgvector
    # column. Passing it through the Alembic config keeps the migration reading
    # from the *same* Settings the caller resolved, rather than re-discovering
    # configuration and possibly disagreeing with it.
    config.set_main_option("recall.embedding_dimensions", str(settings.embedding.dimensions))
    return config


def upgrade(settings: Settings, revision: str = "head") -> None:
    """Apply migrations up to ``revision``."""
    command.upgrade(alembic_config(settings), revision)


def downgrade(settings: Settings, revision: str) -> None:
    """Revert migrations down to ``revision``."""
    command.downgrade(alembic_config(settings), revision)


def stamp(settings: Settings, revision: str = "head") -> None:
    """Record ``revision`` as applied without running it."""
    command.stamp(alembic_config(settings), revision)
