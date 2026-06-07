"""Connector protocol.

A connector answers three questions:

``discover()``  what exists at the source, cheaply
``fetch(item)`` give me the normalized content of one item

Reconciliation (``sync``) is deliberately *not* part of this protocol. The
logic — compare checksums, re-chunk, re-embed, re-index, delete what vanished —
is identical for every source, so implementing it per connector would duplicate
the most correctness-sensitive code in the system. It lives once in
:class:`recall.pipeline.ingest.IngestionPipeline`, which consumes any
``Connector`` and returns a :class:`~recall.core.models.SyncResult`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from recall.core.models import Document, SourceItem, SourceType
from recall.core.registry import Registry


@runtime_checkable
class Connector(Protocol):
    """Reads an external source and yields canonical documents."""

    source_type: SourceType

    async def discover(self) -> list[SourceItem]:
        """List everything currently available at the source."""
        ...

    async def fetch(self, item: SourceItem) -> Document:
        """Load and normalize one discovered item."""
        ...


connector_registry: Registry[Connector] = Registry("connector")
