"""Incremental synchronisation.

The single most important behaviour in the ingestion path: unchanged content
must never be re-fetched, re-chunked or re-embedded, and changed content always
must be.
"""

from __future__ import annotations

import pytest

from recall.core.chunking.fixed import FixedSizeChunker
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.errors import DocumentParseError, TransientError
from recall.core.models import Document, SourceItem, SourceType, SyncOutcome
from recall.pipeline.ingest import IngestionPipeline

from tests.conftest import FakeStorage, ListConnector


def make_document(source_id: str, content: str, title: str | None = None) -> Document:
    return Document.create(
        source_id=source_id,
        source_type=SourceType.MEMORY,
        title=title or source_id,
        content=content,
        uri=f"memory://{source_id}",
        metadata={"file_type": "md"},
    )


@pytest.fixture
def pipeline(storage: FakeStorage, embedder: HashingEmbedder) -> IngestionPipeline:
    return IngestionPipeline(
        storage=storage,
        chunker=FixedSizeChunker(chunk_size=32, overlap=4),
        embedder=embedder,
    )


@pytest.fixture
def connector() -> ListConnector:
    return ListConnector(
        [
            make_document("a.md", "authentication uses bearer tokens for every request"),
            make_document("b.md", "deployments roll one replica at a time"),
        ]
    )


class TestFirstSync:
    async def test_creates_every_document(
        self, pipeline: IngestionPipeline, connector: ListConnector
    ) -> None:
        result = await pipeline.sync(connector)
        assert result.created == 2
        assert result.updated == result.unchanged == result.failed == 0

    async def test_writes_chunks_and_vectors(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        result = await pipeline.sync(connector)
        assert result.chunks_written == len(storage.chunks) > 0
        assert set(storage.vectors) == set(storage.chunks)

    async def test_records_the_embedding_model(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        assert storage.model is not None
        assert storage.model.provider == "hash"


class TestUnchangedContent:
    async def test_second_sync_reindexes_nothing(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        storage.index_calls.clear()

        result = await pipeline.sync(connector)
        assert result.unchanged == 2
        assert result.created == result.updated == 0
        assert storage.index_calls == []

    async def test_discovery_checksums_avoid_the_fetch_entirely(
        self, pipeline: IngestionPipeline, connector: ListConnector
    ) -> None:
        connector.advertise_checksums = True
        await pipeline.sync(connector)
        fetches_after_first = dict(connector.fetch_count)

        await pipeline.sync(connector)
        assert connector.fetch_count == fetches_after_first

    async def test_without_discovery_checksums_it_fetches_but_does_not_reindex(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        storage.index_calls.clear()

        result = await pipeline.sync(connector)
        assert connector.fetch_count == {"a.md": 2, "b.md": 2}
        assert storage.index_calls == []
        assert result.unchanged == 2

    async def test_a_retitled_document_counts_as_changed(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        storage.index_calls.clear()
        connector.set(
            [
                make_document(
                    "a.md",
                    "authentication uses bearer tokens for every request",
                    title="Renamed",
                ),
                make_document("b.md", "deployments roll one replica at a time"),
            ]
        )
        result = await pipeline.sync(connector)
        assert result.updated == 1
        assert result.unchanged == 1


class TestChangedContent:
    async def test_edited_document_is_reindexed(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        storage.index_calls.clear()
        connector.set(
            [
                make_document("a.md", "authentication now uses mutual TLS instead of tokens"),
                make_document("b.md", "deployments roll one replica at a time"),
            ]
        )
        result = await pipeline.sync(connector)
        assert result.updated == 1
        assert result.unchanged == 1
        assert len(storage.index_calls) == 1

    async def test_stale_chunks_do_not_survive_a_rewrite(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        connector.set([make_document("a.md", " ".join(f"word{i}" for i in range(200)))])
        await pipeline.sync(connector)
        assert len(storage.chunks) > 1

        connector.set([make_document("a.md", "now it is short")])
        await pipeline.sync(connector)
        assert len(storage.chunks) == 1
        assert set(storage.vectors) == set(storage.chunks)

    async def test_new_document_is_created_while_others_are_unchanged(
        self, pipeline: IngestionPipeline, connector: ListConnector
    ) -> None:
        await pipeline.sync(connector)
        connector.documents["c.md"] = make_document("c.md", "a brand new runbook document")
        result = await pipeline.sync(connector)
        assert result.created == 1
        assert result.unchanged == 2


class TestForce:
    async def test_reindexes_everything(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        storage.index_calls.clear()

        result = await pipeline.sync(connector, force=True)
        assert result.updated == 2
        assert result.unchanged == 0
        assert len(storage.index_calls) == 2

    async def test_force_ignores_discovery_checksums(
        self, pipeline: IngestionPipeline, connector: ListConnector
    ) -> None:
        connector.advertise_checksums = True
        await pipeline.sync(connector)
        before = dict(connector.fetch_count)
        await pipeline.sync(connector, force=True)
        assert connector.fetch_count != before


class TestPruning:
    async def test_documents_removed_at_the_source_are_deleted(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        connector.set(
            [make_document("a.md", "authentication uses bearer tokens for every request")]
        )

        result = await pipeline.sync(connector)
        assert result.deleted == 1
        assert len(storage.documents.documents) == 1

    async def test_no_prune_keeps_them(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        connector.set(
            [make_document("a.md", "authentication uses bearer tokens for every request")]
        )

        result = await pipeline.sync(connector, prune=False)
        assert result.deleted == 0
        assert len(storage.documents.documents) == 2

    async def test_pruning_is_scoped_to_the_connector_source_type(
        self, pipeline: IngestionPipeline, storage: FakeStorage
    ) -> None:
        """A filesystem sync must not delete PDF documents."""
        pdf_connector = ListConnector(
            [
                Document.create(
                    source_id="manual.pdf",
                    source_type=SourceType.PDF,
                    title="Manual",
                    content="the printed manual explains the wiring",
                    uri="file:///manual.pdf",
                )
            ],
            source_type=SourceType.PDF,
        )
        memory_connector = ListConnector([make_document("a.md", "some memory content here")])

        await pipeline.sync(pdf_connector)
        await pipeline.sync(memory_connector)
        assert len(storage.documents.documents) == 2

        memory_connector.set([])
        result = await pipeline.sync(memory_connector)
        assert result.deleted == 1
        assert len(storage.documents.documents) == 1


class TestFailureIsolation:
    async def test_one_failing_document_does_not_abort_the_sync(
        self, storage: FakeStorage, embedder: HashingEmbedder
    ) -> None:
        class FlakyConnector(ListConnector):
            async def fetch(self, item: SourceItem) -> Document:
                if item.source_id == "bad.md":
                    raise DocumentParseError("cannot parse bad.md")
                return await super().fetch(item)

        connector = FlakyConnector(
            [
                make_document("good.md", "a perfectly readable document about tokens"),
                make_document("bad.md", "this one explodes on fetch"),
            ]
        )
        pipeline = IngestionPipeline(
            storage=storage, chunker=FixedSizeChunker(chunk_size=32, overlap=4), embedder=embedder
        )

        result = await pipeline.sync(connector)
        assert result.created == 1
        assert result.failed == 1
        failure = next(item for item in result.items if item.outcome is SyncOutcome.FAILED)
        assert "cannot parse bad.md" in (failure.error or "")

    async def test_unexpected_exceptions_are_recorded_not_raised(
        self, storage: FakeStorage, embedder: HashingEmbedder
    ) -> None:
        class ExplodingConnector(ListConnector):
            async def fetch(self, item: SourceItem) -> Document:
                raise RuntimeError("something unexpected")

        pipeline = IngestionPipeline(
            storage=storage, chunker=FixedSizeChunker(chunk_size=32, overlap=4), embedder=embedder
        )
        result = await pipeline.sync(ExplodingConnector([make_document("a.md", "content here")]))
        assert result.failed == 1
        assert "RuntimeError" in (result.items[0].error or "")

    async def test_transient_fetch_failures_are_retried(
        self, storage: FakeStorage, embedder: HashingEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr("recall.pipeline.retry.asyncio.sleep", no_sleep)

        attempts = 0

        class FlakyOnceConnector(ListConnector):
            async def fetch(self, item: SourceItem) -> Document:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise TransientError("network blip")
                return await super().fetch(item)

        pipeline = IngestionPipeline(
            storage=storage, chunker=FixedSizeChunker(chunk_size=32, overlap=4), embedder=embedder
        )
        result = await pipeline.sync(FlakyOnceConnector([make_document("a.md", "content here")]))
        assert result.created == 1
        assert attempts == 2


class TestIdempotency:
    async def test_chunk_ids_are_stable_across_syncs(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        first = set(storage.chunks)
        await pipeline.sync(connector, force=True)
        assert set(storage.chunks) == first

    async def test_document_ids_are_stable_across_syncs(
        self, pipeline: IngestionPipeline, connector: ListConnector, storage: FakeStorage
    ) -> None:
        await pipeline.sync(connector)
        first = set(storage.documents.documents)
        await pipeline.sync(connector, force=True)
        assert set(storage.documents.documents) == first

    async def test_empty_source_is_a_no_op(self, pipeline: IngestionPipeline) -> None:
        result = await pipeline.sync(ListConnector([]))
        assert result.items == []
        assert result.chunks_written == 0
