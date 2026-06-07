"""Storage adapters.

The interfaces these implement live in :mod:`recall.core.ports`. PostgreSQL +
pgvector is the only backend in v0.1; the abstraction exists so a dedicated
vector database can be added without touching retrieval code.
"""

from recall.storage.postgres import (
    PostgresChunkRepository,
    PostgresDocumentRepository,
    PostgresVectorIndex,
    Storage,
    create_storage,
)

__all__ = [
    "PostgresChunkRepository",
    "PostgresDocumentRepository",
    "PostgresVectorIndex",
    "Storage",
    "create_storage",
]
