"""End to end: ingest a real directory, index it, search it, verify results.

This is the flow the Definition of Done describes, minus the CLI wrapper (which
is exercised in ``test_cli.py``).
"""

from __future__ import annotations

import pytest

from recall.connectors.filesystem import FilesystemConnector
from recall.core.models import SearchFilters, SourceType
from recall.core.retrieval.dense import DenseRetriever
from recall.pipeline.ingest import IngestionPipeline
from recall.pipeline.search import SearchService
from recall.storage.postgres.storage import Storage

from tests.conftest import FIXTURES

pytestmark = pytest.mark.integration

CORPUS = FIXTURES / "corpus"


@pytest.fixture
def connector() -> FilesystemConnector:
    return FilesystemConnector(root=CORPUS)


class TestIngestThenSearch:
    async def test_full_pipeline(
        self,
        pipeline: IngestionPipeline,
        connector: FilesystemConnector,
        retriever: DenseRetriever,
        storage: Storage,
    ) -> None:
        result = await pipeline.sync(connector)
        assert result.created == 4
        assert result.failed == 0
        assert result.chunks_written > 0

        assert await storage.documents.count() == 4
        assert await storage.chunks.count() == result.chunks_written
        assert await storage.vectors.count() == result.chunks_written

        response = await SearchService(retriever=retriever).search(
            "how are bearer tokens verified", top_k=3
        )
        assert response.results
        top = response.results[0]
        assert top.rank == 1
        assert top.score > 0
        assert top.content
        assert top.document_title
        assert top.document_uri.startswith("file://")
        assert top.source_type is SourceType.FILESYSTEM
        assert top.metadata["file_type"] in {"md", "txt", "json", "html"}
        assert response.timing.total_ms > 0

    async def test_results_are_relevant(
        self,
        pipeline: IngestionPipeline,
        connector: FilesystemConnector,
        retriever: DenseRetriever,
    ) -> None:
        """The authentication document should outrank the others for an auth query.

        The `hash` embedder carries lexical signal only; this asserts the
        plumbing ranks sensibly, not that any model is good.
        """
        await pipeline.sync(connector)
        results = await retriever.search("bearer token signature expiry verification", top_k=4)
        assert results[0].document_title == "Authentication"

    async def test_search_respects_filters(
        self,
        pipeline: IngestionPipeline,
        connector: FilesystemConnector,
        retriever: DenseRetriever,
    ) -> None:
        await pipeline.sync(connector)
        results = await retriever.search(
            "request", top_k=10, filters=SearchFilters(file_types=["json"])
        )
        assert results
        assert all(r.metadata["file_type"] == "json" for r in results)

    async def test_empty_index_returns_no_results(self, retriever: DenseRetriever) -> None:
        assert await retriever.search("anything at all", top_k=5) == []


class TestIncrementalSyncAgainstRealStorage:
    async def test_second_sync_changes_nothing(
        self, pipeline: IngestionPipeline, connector: FilesystemConnector, storage: Storage
    ) -> None:
        await pipeline.sync(connector)
        chunks_before = await storage.chunks.count()
        vectors_before = await storage.vectors.count()

        result = await pipeline.sync(connector)
        assert result.unchanged == 4
        assert result.created == result.updated == 0
        assert await storage.chunks.count() == chunks_before
        assert await storage.vectors.count() == vectors_before

    async def test_editing_a_file_reindexes_only_that_file(
        self, pipeline: IngestionPipeline, storage: Storage, tmp_path
    ) -> None:
        (tmp_path / "a.md").write_text("# A\n\noriginal content about tokens")
        (tmp_path / "b.md").write_text("# B\n\nunrelated content about deployments")
        connector = FilesystemConnector(root=tmp_path)

        await pipeline.sync(connector)
        original = await storage.documents.get_by_source(SourceType.FILESYSTEM, "a.md")
        assert original is not None

        (tmp_path / "a.md").write_text("# A\n\ncompletely rewritten content about mutual TLS")
        result = await pipeline.sync(connector)

        assert result.updated == 1
        assert result.unchanged == 1
        updated = await storage.documents.get_by_source(SourceType.FILESYSTEM, "a.md")
        assert updated is not None
        assert updated.checksum != original.checksum
        assert "mutual TLS" in updated.content

    async def test_deleting_a_file_prunes_it_and_its_vectors(
        self, pipeline: IngestionPipeline, storage: Storage, tmp_path
    ) -> None:
        (tmp_path / "a.md").write_text("# A\n\ncontent about tokens")
        (tmp_path / "b.md").write_text("# B\n\ncontent about deployments")
        connector = FilesystemConnector(root=tmp_path)

        await pipeline.sync(connector)
        assert await storage.documents.count() == 2

        (tmp_path / "b.md").unlink()
        result = await pipeline.sync(connector)

        assert result.deleted == 1
        assert await storage.documents.count() == 1
        # The cascade must have taken the orphaned chunks and vectors with it.
        assert await storage.chunks.count() == await storage.vectors.count()

    async def test_force_reindexes_without_changing_ids(
        self, pipeline: IngestionPipeline, connector: FilesystemConnector, storage: Storage
    ) -> None:
        await pipeline.sync(connector)
        documents_before = {d.id for d in await storage.documents.list(limit=100)}

        result = await pipeline.sync(connector, force=True)
        assert result.updated == 4
        assert {d.id for d in await storage.documents.list(limit=100)} == documents_before


class TestConcurrency:
    async def test_concurrent_searches_keep_separate_timings(
        self, pipeline: IngestionPipeline, connector: FilesystemConnector, retriever: DenseRetriever
    ) -> None:
        """The ambient timer is a context variable, so it must not leak between tasks."""
        import asyncio

        await pipeline.sync(connector)
        service = SearchService(retriever=retriever)
        responses = await asyncio.gather(
            *(service.search(f"query number {i}", top_k=2) for i in range(8))
        )
        assert len({r.request_id for r in responses}) == 8
        assert all(r.timing.embedding_ms > 0 and r.timing.retrieval_ms > 0 for r in responses)


class TestPDFIngestion:
    """PDFs and text files coexist: each connector owns its own source_type."""

    async def test_pdf_and_text_are_indexed_side_by_side(
        self, pipeline: IngestionPipeline, storage: Storage, tmp_path
    ) -> None:
        fitz = pytest.importorskip("fitz", reason="requires the `pdf` extra (PyMuPDF)")

        (tmp_path / "runbook.md").write_text("# Runbook\n\nrolling deployments replace replicas")
        document = fitz.open()
        page = document.new_page()
        page.insert_text(
            (72, 72), "Bearer tokens are verified against the public key.", fontsize=11
        )
        document.save(tmp_path / "security.pdf")
        document.close()

        from recall.connectors.pdf import PDFConnector

        text_result = await pipeline.sync(FilesystemConnector(root=tmp_path))
        pdf_result = await pipeline.sync(PDFConnector(root=tmp_path))

        assert text_result.created == 1
        assert pdf_result.created == 1
        # Neither sync pruned the other's documents.
        assert await storage.documents.count() == 2

        pdfs = await storage.documents.list(source_types=[SourceType.PDF], limit=10)
        assert [d.source_id for d in pdfs] == ["security.pdf"]

    async def test_pdf_chunks_are_searchable_and_filterable(
        self, pipeline: IngestionPipeline, retriever: DenseRetriever, tmp_path
    ) -> None:
        fitz = pytest.importorskip("fitz", reason="requires the `pdf` extra (PyMuPDF)")

        document = fitz.open()
        page = document.new_page()
        page.insert_text(
            (72, 72), "Bearer tokens are verified against the public key.", fontsize=11
        )
        document.save(tmp_path / "security.pdf")
        document.close()

        from recall.connectors.pdf import PDFConnector

        await pipeline.sync(PDFConnector(root=tmp_path))
        results = await retriever.search(
            "bearer token verification",
            top_k=5,
            filters=SearchFilters(source_types=[SourceType.PDF]),
        )
        assert results
        assert results[0].source_type is SourceType.PDF
        assert results[0].metadata["page_offsets"]
