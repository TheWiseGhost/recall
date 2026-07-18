"""Ingestion and incremental synchronisation.

    discover
       |
    checksum known and unchanged? --> skip
       |
    fetch -> normalize -> checksum
       |
    still unchanged? --> skip
       |
    chunk -> embed -> index   (one transaction)

Two checksum comparisons, not one. The first uses whatever the connector could
learn cheaply during discovery and avoids the fetch entirely. The second is
authoritative: it compares the checksum of the *normalized content*, so a file
that was touched, re-saved, or re-downloaded byte-identically is correctly
detected as unchanged and is never re-embedded.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

from recall.connectors.base import Connector
from recall.core.chunking.base import Chunker
from recall.core.embeddings.base import Embedder
from recall.core.errors import RecallError
from recall.core.models import (
    Chunk,
    Document,
    SourceItem,
    SyncItemResult,
    SyncOutcome,
    SyncResult,
)
from recall.core.ports import IngestStore
from recall.observability.logging import get_logger
from recall.pipeline.retry import retry_async

_log = get_logger(__name__)


class IngestionPipeline:
    """Turns connector output into indexed, searchable chunks."""

    def __init__(
        self,
        *,
        storage: IngestStore,
        chunker: Chunker,
        embedder: Embedder,
        concurrency: int = 4,
        embed_attempts: int = 3,
    ) -> None:
        self.storage = storage
        self.chunker = chunker
        self.embedder = embedder
        self.concurrency = max(1, concurrency)
        self.embed_attempts = max(1, embed_attempts)

    # -- single document --------------------------------------------------
    async def index_document(self, document: Document) -> SyncItemResult:
        """Chunk, embed and index one already-fetched document."""
        chunks: list[Chunk] = await self.chunker.chunk(document)
        vectors = await self._embed_chunks(chunks)
        written = await self.storage.index_document(document, chunks, vectors, self.embedder.info)
        _log.info(
            "document_indexed",
            document_id=str(document.id),
            source_id=document.source_id,
            source_type=document.source_type.value,
            chunks=written,
            chunker=getattr(self.chunker, "name", "unknown"),
            embedding_model=self.embedder.info.key,
        )
        return SyncItemResult(
            source_id=document.source_id,
            outcome=SyncOutcome.UPDATED,
            document_id=document.id,
            chunks_written=written,
        )

    async def _embed_chunks(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        if not chunks:
            return []
        texts = [chunk.content for chunk in chunks]
        return await retry_async(
            lambda: self.embedder.embed_documents(texts),
            attempts=self.embed_attempts,
            description="embed_documents",
        )

    # -- sync -------------------------------------------------------------
    async def sync(
        self,
        connector: Connector,
        *,
        force: bool = False,
        prune: bool = True,
    ) -> SyncResult:
        """Reconcile ``connector``'s source with what is stored.

        Args:
            force: re-chunk and re-embed even when nothing changed. Use after
                changing the chunking strategy or embedding model.
            prune: delete stored documents that no longer exist at the source.
        """
        source_type = connector.source_type
        result = SyncResult(source_type=source_type)

        items = await connector.discover()
        known = await self.storage.documents.checksums(source_type)
        _log.info(
            "sync_started",
            source_type=source_type.value,
            discovered=len(items),
            already_stored=len(known),
            force=force,
        )

        semaphore = asyncio.Semaphore(self.concurrency)

        async def process(item: SourceItem) -> SyncItemResult:
            async with semaphore:
                return await self._process_item(connector, item, known, force=force)

        item_results = await asyncio.gather(
            *(process(item) for item in items), return_exceptions=False
        )
        result.items.extend(item_results)

        if prune:
            removed = await self.storage.documents.delete_missing(
                source_type, [item.source_id for item in items]
            )
            result.items.extend(
                SyncItemResult(
                    source_id=str(document_id),
                    outcome=SyncOutcome.DELETED,
                    document_id=document_id,
                )
                for document_id in removed
            )

        result.finished_at = datetime.now(UTC)
        _log.info(
            "sync_finished",
            source_type=source_type.value,
            created=result.created,
            updated=result.updated,
            unchanged=result.unchanged,
            deleted=result.deleted,
            failed=result.failed,
            chunks=result.chunks_written,
            duration_seconds=result.duration_seconds,
        )
        return result

    async def _process_item(
        self,
        connector: Connector,
        item: SourceItem,
        known: dict[str, str],
        *,
        force: bool,
    ) -> SyncItemResult:
        stored_checksum = known.get(item.source_id)
        is_new = stored_checksum is None

        # Pass 1: skip without fetching when discovery gave us a checksum.
        if (
            not force
            and item.checksum is not None
            and stored_checksum is not None
            and item.checksum == stored_checksum
        ):
            return SyncItemResult(source_id=item.source_id, outcome=SyncOutcome.UNCHANGED)

        try:
            document = await retry_async(
                lambda: connector.fetch(item), description=f"fetch:{item.source_id}"
            )
        except RecallError as exc:
            _log.warning("fetch_failed", source_id=item.source_id, error=str(exc))
            return SyncItemResult(
                source_id=item.source_id, outcome=SyncOutcome.FAILED, error=str(exc)
            )
        except Exception as exc:
            _log.exception("fetch_failed_unexpectedly", source_id=item.source_id)
            return SyncItemResult(
                source_id=item.source_id,
                outcome=SyncOutcome.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Pass 2: authoritative comparison on normalized content.
        if not force and stored_checksum is not None and document.checksum == stored_checksum:
            return SyncItemResult(
                source_id=item.source_id,
                outcome=SyncOutcome.UNCHANGED,
                document_id=document.id,
            )

        try:
            indexed = await self.index_document(document)
        except RecallError as exc:
            _log.warning("index_failed", source_id=item.source_id, error=str(exc))
            return SyncItemResult(
                source_id=item.source_id, outcome=SyncOutcome.FAILED, error=str(exc)
            )
        except Exception as exc:
            _log.exception("index_failed_unexpectedly", source_id=item.source_id)
            return SyncItemResult(
                source_id=item.source_id,
                outcome=SyncOutcome.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

        return indexed.model_copy(
            update={"outcome": SyncOutcome.CREATED if is_new else SyncOutcome.UPDATED}
        )
