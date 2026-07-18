"""Storage round-trips against a real PostgreSQL + pgvector."""

from __future__ import annotations

import uuid

import pytest

from recall.core.chunking.fixed import FixedSizeChunker
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.errors import DimensionMismatchError
from recall.core.models import Document, SearchFilters, SourceType
from recall.storage.postgres.storage import Storage

pytestmark = pytest.mark.integration


def make_document(
    source_id: str,
    content: str,
    *,
    source_type: SourceType = SourceType.FILESYSTEM,
    title: str = "Doc",
    **metadata: object,
) -> Document:
    return Document.create(
        source_id=source_id,
        source_type=source_type,
        title=title,
        content=content,
        uri=f"file:///{source_id}",
        metadata=dict(metadata),
    )


async def index(
    storage: Storage,
    document: Document,
    chunker: FixedSizeChunker,
    embedder: HashingEmbedder,
) -> int:
    chunks = await chunker.chunk(document)
    vectors = await embedder.embed_documents([c.content for c in chunks])
    return await storage.index_document(document, chunks, vectors, embedder.info)


class TestHealth:
    async def test_reports_pgvector_and_counts(self, storage: Storage) -> None:
        health = await storage.health()
        assert health["connected"] is True
        assert health["pgvector_version"]
        assert health["documents"] == 0


class TestDocumentRepository:
    async def test_upsert_and_get(self, storage: Storage) -> None:
        document = make_document("a.md", "body text", author="docs")
        await storage.documents.upsert(document)

        loaded = await storage.documents.get(document.id)
        assert loaded is not None
        assert loaded.content == "body text"
        assert loaded.metadata["author"] == "docs"
        assert loaded.source_type is SourceType.FILESYSTEM

    async def test_upsert_is_idempotent(self, storage: Storage) -> None:
        document = make_document("a.md", "body")
        await storage.documents.upsert(document)
        await storage.documents.upsert(document)
        assert await storage.documents.count() == 1

    async def test_upsert_updates_in_place(self, storage: Storage) -> None:
        await storage.documents.upsert(make_document("a.md", "first"))
        await storage.documents.upsert(make_document("a.md", "second"))
        loaded = await storage.documents.get(make_document("a.md", "second").id)
        assert loaded is not None and loaded.content == "second"
        assert await storage.documents.count() == 1

    async def test_get_by_source(self, storage: Storage) -> None:
        await storage.documents.upsert(make_document("a.md", "body"))
        found = await storage.documents.get_by_source(SourceType.FILESYSTEM, "a.md")
        assert found is not None
        assert await storage.documents.get_by_source(SourceType.PDF, "a.md") is None

    async def test_missing_document_returns_none(self, storage: Storage) -> None:
        assert await storage.documents.get(uuid.uuid4()) is None

    async def test_list_filters_and_paginates(self, storage: Storage) -> None:
        for i in range(5):
            await storage.documents.upsert(make_document(f"f{i}.md", "body"))
        await storage.documents.upsert(make_document("p.pdf", "body", source_type=SourceType.PDF))

        assert len(await storage.documents.list(limit=100)) == 6
        assert len(await storage.documents.list(limit=2)) == 2
        assert len(await storage.documents.list(limit=100, offset=4)) == 2
        pdfs = await storage.documents.list(source_types=[SourceType.PDF], limit=100)
        assert [d.source_id for d in pdfs] == ["p.pdf"]

    async def test_checksums_are_scoped_to_a_source_type(self, storage: Storage) -> None:
        await storage.documents.upsert(make_document("a.md", "body"))
        await storage.documents.upsert(make_document("p.pdf", "body", source_type=SourceType.PDF))
        assert set(await storage.documents.checksums(SourceType.FILESYSTEM)) == {"a.md"}
        assert set(await storage.documents.checksums(SourceType.PDF)) == {"p.pdf"}

    async def test_delete(self, storage: Storage) -> None:
        document = make_document("a.md", "body")
        await storage.documents.upsert(document)
        assert await storage.documents.delete(document.id) is True
        assert await storage.documents.delete(document.id) is False

    async def test_delete_missing_is_scoped_to_a_source_type(self, storage: Storage) -> None:
        await storage.documents.upsert(make_document("a.md", "body"))
        await storage.documents.upsert(make_document("b.md", "body"))
        await storage.documents.upsert(make_document("p.pdf", "body", source_type=SourceType.PDF))

        deleted = await storage.documents.delete_missing(SourceType.FILESYSTEM, ["a.md"])
        assert len(deleted) == 1
        assert await storage.documents.count() == 2
        assert await storage.documents.get_by_source(SourceType.PDF, "p.pdf") is not None


class TestChunkRepository:
    async def test_replace_writes_chunks(self, storage: Storage, chunker: FixedSizeChunker) -> None:
        document = make_document("a.md", " ".join(f"word{i}" for i in range(200)))
        await storage.documents.upsert(document)
        chunks = await chunker.chunk(document)
        written = await storage.chunks.replace_for_document(document.id, chunks)

        assert written == len(chunks) > 1
        stored = await storage.chunks.list_for_document(document.id)
        assert [c.position for c in stored] == list(range(len(chunks)))

    async def test_replace_removes_the_previous_set(
        self, storage: Storage, chunker: FixedSizeChunker
    ) -> None:
        document = make_document("a.md", " ".join(f"word{i}" for i in range(200)))
        await storage.documents.upsert(document)
        await storage.chunks.replace_for_document(document.id, await chunker.chunk(document))

        shorter = make_document("a.md", "now short")
        await storage.chunks.replace_for_document(shorter.id, await chunker.chunk(shorter))
        assert len(await storage.chunks.list_for_document(document.id)) == 1

    async def test_get_many_preserves_input_order(
        self, storage: Storage, chunker: FixedSizeChunker
    ) -> None:
        document = make_document("a.md", " ".join(f"word{i}" for i in range(200)))
        await storage.documents.upsert(document)
        chunks = await chunker.chunk(document)
        await storage.chunks.replace_for_document(document.id, chunks)

        wanted = [chunks[2].id, chunks[0].id]
        assert [c.id for c in await storage.chunks.get_many(wanted)] == wanted

    async def test_deleting_a_document_cascades_to_chunks(
        self, storage: Storage, chunker: FixedSizeChunker
    ) -> None:
        document = make_document("a.md", " ".join(f"word{i}" for i in range(200)))
        await storage.documents.upsert(document)
        await storage.chunks.replace_for_document(document.id, await chunker.chunk(document))

        await storage.documents.delete(document.id)
        assert await storage.chunks.list_for_document(document.id) == []

    async def test_metadata_round_trips(self, storage: Storage, chunker: FixedSizeChunker) -> None:
        document = make_document("a.md", "some content", file_type="md", tags=["security"])
        await storage.documents.upsert(document)
        await storage.chunks.replace_for_document(document.id, await chunker.chunk(document))
        chunk = (await storage.chunks.list_for_document(document.id))[0]
        assert chunk.metadata["file_type"] == "md"
        assert chunk.metadata["chunker"] == "fixed"


class TestVectorIndex:
    async def test_upsert_and_count(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        document = make_document("a.md", " ".join(f"word{i}" for i in range(200)))
        written = await index(storage, document, chunker, embedder)
        assert await storage.vectors.count() == written

    async def test_reindexing_does_not_duplicate_vectors(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        document = make_document("a.md", " ".join(f"word{i}" for i in range(200)))
        await index(storage, document, chunker, embedder)
        first = await storage.vectors.count()
        await index(storage, document, chunker, embedder)
        assert await storage.vectors.count() == first

    async def test_query_returns_ranked_results(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await index(
            storage,
            make_document("auth.md", "bearer tokens are verified by signature and expiry"),
            chunker,
            embedder,
        )
        await index(
            storage,
            make_document("kitchen.md", "reclaimed oak flooring throughout the kitchen"),
            chunker,
            embedder,
        )

        vector = await embedder.embed_query("how are bearer tokens verified")
        results = await storage.vectors.query(vector, top_k=2, model=embedder.info)
        assert len(results) == 2
        assert [r.rank for r in results] == [1, 2]
        assert results[0].score >= results[1].score
        assert "bearer tokens" in results[0].content

    async def test_results_carry_document_provenance(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await index(
            storage,
            make_document("auth.md", "bearer tokens", title="Authentication"),
            chunker,
            embedder,
        )
        vector = await embedder.embed_query("bearer tokens")
        result = (await storage.vectors.query(vector, top_k=1, model=embedder.info))[0]
        assert result.document_title == "Authentication"
        assert result.document_uri.endswith("auth.md")
        assert result.source_type is SourceType.FILESYSTEM

    async def test_scores_are_cosine_similarity(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        content = "an exactly identical sentence about tokens"
        await index(storage, make_document("a.md", content), chunker, embedder)
        vector = await embedder.embed_query(content)
        result = (await storage.vectors.query(vector, top_k=1, model=embedder.info))[0]
        assert result.score == pytest.approx(1.0, abs=1e-4)

    async def test_query_is_scoped_to_the_embedding_model(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await index(storage, make_document("a.md", "bearer tokens"), chunker, embedder)
        other = HashingEmbedder(model="other-model", dimensions=embedder.info.dimensions)
        vector = await embedder.embed_query("bearer tokens")
        assert await storage.vectors.query(vector, top_k=5, model=other.info) == []

    async def test_dimension_mismatch_is_rejected(
        self, storage: Storage, embedder: HashingEmbedder
    ) -> None:
        with pytest.raises(DimensionMismatchError):
            await storage.vectors.query([0.0, 1.0], top_k=1, model=embedder.info)

    async def test_deleting_a_document_cascades_to_vectors(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        document = make_document("a.md", "bearer tokens")
        await index(storage, document, chunker, embedder)
        await storage.documents.delete(document.id)
        assert await storage.vectors.count() == 0

    async def test_models_reports_what_is_indexed(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await index(storage, make_document("a.md", "bearer tokens"), chunker, embedder)
        models = await storage.vectors.models()
        assert [m.key for m in models] == ["hash:hash-v1"]


class TestMetadataFiltering:
    async def _seed(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await index(
            storage,
            make_document(
                "auth.md",
                "bearer tokens are verified by signature",
                file_type="md",
                author="docs-team",
                tags=["security", "api"],
            ),
            chunker,
            embedder,
        )
        await index(
            storage,
            make_document(
                "manual.pdf",
                "bearer tokens explained in the printed manual",
                source_type=SourceType.PDF,
                file_type="pdf",
                author="ops-team",
                tags=["reference"],
            ),
            chunker,
            embedder,
        )

    async def test_filter_by_source_type(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await self._seed(storage, chunker, embedder)
        vector = await embedder.embed_query("bearer tokens")
        results = await storage.vectors.query(
            vector,
            top_k=10,
            filters=SearchFilters(source_types=[SourceType.PDF]),
            model=embedder.info,
        )
        assert results and all(r.source_type is SourceType.PDF for r in results)

    async def test_filter_by_file_type(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await self._seed(storage, chunker, embedder)
        vector = await embedder.embed_query("bearer tokens")
        results = await storage.vectors.query(
            vector, top_k=10, filters=SearchFilters(file_types=["md"]), model=embedder.info
        )
        assert results and all(r.metadata["file_type"] == "md" for r in results)

    async def test_filter_by_author(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await self._seed(storage, chunker, embedder)
        vector = await embedder.embed_query("bearer tokens")
        results = await storage.vectors.query(
            vector, top_k=10, filters=SearchFilters(authors=["ops-team"]), model=embedder.info
        )
        assert len(results) == 1

    async def test_filter_by_tag(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await self._seed(storage, chunker, embedder)
        vector = await embedder.embed_query("bearer tokens")
        results = await storage.vectors.query(
            vector, top_k=10, filters=SearchFilters(tags=["security"]), model=embedder.info
        )
        assert len(results) == 1

    async def test_filter_by_arbitrary_metadata(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await self._seed(storage, chunker, embedder)
        vector = await embedder.embed_query("bearer tokens")
        results = await storage.vectors.query(
            vector,
            top_k=10,
            filters=SearchFilters(metadata={"author": "docs-team"}),
            model=embedder.info,
        )
        assert len(results) == 1

    async def test_filters_combine_conjunctively(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        await self._seed(storage, chunker, embedder)
        vector = await embedder.embed_query("bearer tokens")
        results = await storage.vectors.query(
            vector,
            top_k=10,
            filters=SearchFilters(source_types=[SourceType.PDF], authors=["docs-team"]),
            model=embedder.info,
        )
        assert results == []

    async def test_filtering_is_applied_before_top_k(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        """top_k must count *matching* rows, not rows that survive a post-filter."""
        await self._seed(storage, chunker, embedder)
        vector = await embedder.embed_query("bearer tokens")
        results = await storage.vectors.query(
            vector,
            top_k=1,
            filters=SearchFilters(source_types=[SourceType.PDF]),
            model=embedder.info,
        )
        assert len(results) == 1
        assert results[0].source_type is SourceType.PDF


class TestTransactionality:
    async def test_index_document_writes_everything_together(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        document = make_document("a.md", " ".join(f"word{i}" for i in range(200)))
        written = await index(storage, document, chunker, embedder)
        assert await storage.documents.count() == 1
        assert await storage.chunks.count() == written
        assert await storage.vectors.count() == written

    async def test_a_bad_vector_leaves_no_partial_state(
        self, storage: Storage, chunker: FixedSizeChunker, embedder: HashingEmbedder
    ) -> None:
        document = make_document("a.md", "content that will fail to index")
        chunks = await chunker.chunk(document)
        with pytest.raises(DimensionMismatchError):
            await storage.index_document(document, chunks, [[0.0, 1.0]], embedder.info)

        assert await storage.documents.count() == 0
        assert await storage.chunks.count() == 0
        assert await storage.vectors.count() == 0
