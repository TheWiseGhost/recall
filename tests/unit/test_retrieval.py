"""Dense retrieval and search timing."""

from __future__ import annotations

import uuid

from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.models import Chunk, Document, SearchFilters, SourceType
from recall.core.retrieval.base import Timer, rerank_positions, stage
from recall.core.retrieval.dense import DenseRetriever
from recall.pipeline.search import SearchService

from tests.conftest import FakeStorage


async def populate(storage: FakeStorage, embedder: HashingEmbedder) -> None:
    corpus = [
        ("auth.md", SourceType.FILESYSTEM, "Authentication", "bearer tokens verify the signature"),
        ("deploy.txt", SourceType.FILESYSTEM, "Deployment", "rolling deployments replace replicas"),
        ("guide.pdf", SourceType.PDF, "Guide", "token scope is matched against the endpoint"),
    ]
    for source_id, source_type, title, content in corpus:
        document = Document.create(
            source_id=source_id,
            source_type=source_type,
            title=title,
            content=content,
            uri=f"file:///{source_id}",
        )
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=document.id,
            content=content,
            position=0,
            token_count=len(content.split()),
            checksum="x",
        )
        vectors = await embedder.embed_documents([content])
        await storage.index_document(document, [chunk], vectors, embedder.info)


class TestTimer:
    def test_records_named_stages(self) -> None:
        timer = Timer()
        with timer.stage("embedding"):
            pass
        with timer.stage("retrieval"):
            pass
        timing = timer.to_timing()
        assert timing.embedding_ms >= 0
        assert timing.retrieval_ms >= 0
        assert timing.total_ms >= 0

    def test_repeated_stages_accumulate(self) -> None:
        timer = Timer()
        for _ in range(3):
            with timer.stage("retrieval"):
                pass
        assert len(timer.stages) == 1

    def test_ambient_stage_is_a_no_op_when_inactive(self) -> None:
        with stage("embedding"):
            pass  # must not raise

    def test_ambient_stage_records_against_the_active_timer(self) -> None:
        timer = Timer()
        with timer.activate(), stage("embedding"):
            pass
        assert "embedding" in timer.stages

    def test_unstaged_work_is_attributed_to_retrieval(self) -> None:
        assert Timer().to_timing().retrieval_ms >= 0


class TestRerankPositions:
    def test_ranks_are_one_based_and_sequential(self) -> None:
        from recall.core.models import SearchResult

        results = [
            SearchResult(
                chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), content="c", score=0.1, rank=99
            )
            for _ in range(3)
        ]
        assert [r.rank for r in rerank_positions(results)] == [1, 2, 3]


class TestDenseRetriever:
    async def test_returns_ranked_results(self, storage: FakeStorage) -> None:
        embedder = HashingEmbedder(dimensions=256)
        await populate(storage, embedder)
        retriever = DenseRetriever(embedder=embedder, index=storage)

        results = await retriever.search("how are bearer tokens verified", top_k=3)
        assert len(results) == 3
        assert [r.rank for r in results] == [1, 2, 3]
        assert results[0].score >= results[1].score >= results[2].score

    async def test_stamps_the_retriever_name(self, storage: FakeStorage) -> None:
        embedder = HashingEmbedder(dimensions=256)
        await populate(storage, embedder)
        retriever = DenseRetriever(embedder=embedder, index=storage)
        results = await retriever.search("tokens", top_k=1)
        assert results[0].retriever == "dense"

    async def test_top_k_limits_results(self, storage: FakeStorage) -> None:
        embedder = HashingEmbedder(dimensions=256)
        await populate(storage, embedder)
        retriever = DenseRetriever(embedder=embedder, index=storage)
        assert len(await retriever.search("tokens", top_k=1)) == 1

    async def test_non_positive_top_k_short_circuits(self, storage: FakeStorage) -> None:
        embedder = HashingEmbedder(dimensions=64)
        retriever = DenseRetriever(embedder=embedder, index=storage)
        assert await retriever.search("tokens", top_k=0) == []

    async def test_filters_are_applied(self, storage: FakeStorage) -> None:
        embedder = HashingEmbedder(dimensions=256)
        await populate(storage, embedder)
        retriever = DenseRetriever(embedder=embedder, index=storage)
        results = await retriever.search(
            "tokens", top_k=10, filters=SearchFilters(source_types=[SourceType.PDF])
        )
        assert results
        assert all(r.source_type is SourceType.PDF for r in results)

    async def test_carries_document_provenance(self, storage: FakeStorage) -> None:
        embedder = HashingEmbedder(dimensions=256)
        await populate(storage, embedder)
        retriever = DenseRetriever(embedder=embedder, index=storage)
        result = (await retriever.search("bearer tokens", top_k=1))[0]
        assert result.document_title
        assert result.document_uri
        assert result.source_type is not None


class TestSearchService:
    async def test_separates_embedding_from_retrieval_time(self, storage: FakeStorage) -> None:
        embedder = HashingEmbedder(dimensions=256)
        await populate(storage, embedder)
        service = SearchService(retriever=DenseRetriever(embedder=embedder, index=storage))

        response = await service.search("bearer tokens", top_k=2)
        assert response.timing.embedding_ms > 0
        assert response.timing.retrieval_ms > 0
        assert response.timing.total_ms >= (
            response.timing.embedding_ms + response.timing.retrieval_ms
        )

    async def test_response_echoes_the_query_and_strategy(self, storage: FakeStorage) -> None:
        embedder = HashingEmbedder(dimensions=256)
        await populate(storage, embedder)
        service = SearchService(retriever=DenseRetriever(embedder=embedder, index=storage))
        response = await service.search("tokens", top_k=2)
        assert response.query == "tokens"
        assert response.retrieval_strategy == "dense"
        assert response.top_k == 2
        assert response.request_id

    async def test_is_json_serializable(self, storage: FakeStorage) -> None:
        import json

        embedder = HashingEmbedder(dimensions=256)
        await populate(storage, embedder)
        service = SearchService(retriever=DenseRetriever(embedder=embedder, index=storage))
        response = await service.search("tokens", top_k=2)
        assert json.loads(json.dumps(response.model_dump(mode="json")))["query"] == "tokens"
