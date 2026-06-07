"""Framework-independent domain layer.

Nothing in ``recall.core`` may import SQLAlchemy, FastAPI, Celery or Typer.
It defines the domain models and the protocols that adapters implement.
"""

from recall.core.models import (
    Chunk,
    Document,
    SearchFilters,
    SearchResult,
    SourceItem,
    SourceType,
    SyncResult,
)

__all__ = [
    "Chunk",
    "Document",
    "SearchFilters",
    "SearchResult",
    "SourceItem",
    "SourceType",
    "SyncResult",
]
