"""Chunking strategies against a live database.

Hierarchical chunking is the reason this file exists: ``chunks.parent_id`` is a
self-referencing foreign key, so the order chunks are written in is load-bearing
in a way no in-memory test can catch.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from recall.config.settings import Settings
from recall.core.chunking import HierarchicalChunker, SemanticChunker, SentenceChunker
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.models import Document, SourceType
from recall.core.tokenization import WhitespaceTokenCounter
from recall.pipeline.factory import build_chunker
from recall.pipeline.ingest import IngestionPipeline
from recall.storage.postgres.storage import Storage

pytestmark = pytest.mark.integration

WORDS = WhitespaceTokenCounter()

PROSE = " ".join(
    f"Sentence number {index} describes the deployment pipeline in some detail."
    for index in range(40)
)


def make_document() -> Document:
    return Document.create(
        source_id="prose.md",
        source_type=SourceType.FILESYSTEM,
        title="Prose",
        content=PROSE,
        uri="file:///prose.md",
    )


class TestHierarchicalPersistence:
    async def test_writes_both_levels_with_intact_parent_links(
        self, storage: Storage, embedder: HashingEmbedder
    ) -> None:
        document = make_document()
        chunker = HierarchicalChunker(
            parent_chunk_size=120, chunk_size=30, overlap=0, token_counter=WORDS
        )
        pipeline = IngestionPipeline(storage=storage, chunker=chunker, embedder=embedder)
        result = await pipeline.index_document(document)
        assert result.error is None
        assert result.chunks_written > 0

        async with storage.sessions() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT id, parent_id, metadata->>'chunk_level' AS level "
                        "FROM chunks WHERE document_id = :doc"
                    ),
                    {"doc": document.id},
                )
            ).all()

        parents = {row.id for row in rows if row.level == "parent"}
        children = [row for row in rows if row.level == "child"]
        assert parents and children
        # The foreign key would have rejected a dangling reference outright;
        # this asserts the links point where they should, not merely that they
        # point somewhere.
        assert all(child.parent_id in parents for child in children)

    async def test_deleting_a_document_removes_the_whole_hierarchy(
        self, storage: Storage, embedder: HashingEmbedder
    ) -> None:
        document = make_document()
        chunker = HierarchicalChunker(
            parent_chunk_size=120, chunk_size=30, overlap=0, token_counter=WORDS
        )
        pipeline = IngestionPipeline(storage=storage, chunker=chunker, embedder=embedder)
        await pipeline.index_document(document)
        await storage.documents.delete(document.id)

        assert await storage.chunks.list_for_document(document.id) == []

    async def test_reindexing_replaces_the_hierarchy_cleanly(
        self, storage: Storage, embedder: HashingEmbedder
    ) -> None:
        """`replace_chunks` deletes first; a parent row must not block on a child."""
        chunker = HierarchicalChunker(
            parent_chunk_size=120, chunk_size=30, overlap=0, token_counter=WORDS
        )
        pipeline = IngestionPipeline(storage=storage, chunker=chunker, embedder=embedder)
        await pipeline.index_document(make_document())
        first = len(await storage.chunks.list_for_document(make_document().id))

        shorter = Document.create(
            source_id="prose.md",
            source_type=SourceType.FILESYSTEM,
            title="Prose",
            content="Now it is short.",
            uri="file:///prose.md",
        )
        await pipeline.index_document(shorter)
        second = await storage.chunks.list_for_document(shorter.id)
        assert first > len(second) >= 1


class TestOtherStrategiesEndToEnd:
    async def test_sentence_chunking_is_searchable(
        self, storage: Storage, embedder: HashingEmbedder
    ) -> None:
        chunker = SentenceChunker(chunk_size=40, token_counter=WORDS)
        pipeline = IngestionPipeline(storage=storage, chunker=chunker, embedder=embedder)
        await pipeline.index_document(make_document())

        results = await storage.lexical.search("deployment pipeline", top_k=5)
        assert results
        assert all(r.content.rstrip().endswith(".") for r in results)

    async def test_semantic_chunking_is_searchable(
        self, storage: Storage, embedder: HashingEmbedder
    ) -> None:
        chunker = SemanticChunker(embedder=embedder, buffer_size=1, token_counter=WORDS)
        pipeline = IngestionPipeline(storage=storage, chunker=chunker, embedder=embedder)
        result = await pipeline.index_document(make_document())
        assert result.chunks_written > 0

        results = await storage.lexical.search("deployment pipeline", top_k=5)
        assert results

    async def test_content_length_is_generated_for_every_strategy(
        self, storage: Storage, embedder: HashingEmbedder
    ) -> None:
        """BM25 needs |D| whatever produced the chunk."""
        chunker = build_chunker(
            Settings.from_mapping({"chunking": {"strategy": "sentence"}}),
        )
        pipeline = IngestionPipeline(storage=storage, chunker=chunker, embedder=embedder)
        await pipeline.index_document(make_document())

        async with storage.sessions() as session:
            lengths = (
                (await session.execute(text("SELECT content_length FROM chunks"))).scalars().all()
            )
        assert lengths
        assert all(length > 0 for length in lengths)
