"""Deterministic identifier helpers.

Recall derives entity IDs deterministically so that ingesting the same source
twice produces the same primary keys. That is what makes ingestion and indexing
idempotent without an extra lookup table.
"""

from __future__ import annotations

import hashlib
import uuid

# A fixed, project-specific namespace. Do not change it: doing so renumbers
# every document in every existing database.
RECALL_NAMESPACE = uuid.UUID("6f6b8d24-1f2b-5a5e-9f2f-1b0e1f6a9c31")


def document_id(source_type: str, source_id: str) -> uuid.UUID:
    """Return the stable ID for a document identified by its source coordinates."""
    return uuid.uuid5(RECALL_NAMESPACE, f"document:{source_type}:{source_id}")


def chunk_id(document_id_: uuid.UUID, position: int, content_checksum: str) -> uuid.UUID:
    """Return the stable ID for a chunk.

    The content checksum participates so that re-chunking a *changed* document
    yields new chunk IDs rather than silently rewriting the meaning of an
    existing one.
    """
    return uuid.uuid5(RECALL_NAMESPACE, f"chunk:{document_id_}:{position}:{content_checksum}")


def checksum(*parts: str | bytes) -> str:
    """Return a hex SHA-256 digest over ``parts``.

    Parts are length-prefixed so that ``("ab", "c")`` and ``("a", "bc")`` hash
    differently.
    """
    digest = hashlib.sha256()
    for part in parts:
        raw = part.encode("utf-8") if isinstance(part, str) else part
        digest.update(str(len(raw)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(raw)
    return digest.hexdigest()
