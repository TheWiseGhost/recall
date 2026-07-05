"""Hybrid retrieval against a live PostgreSQL, with real dense and BM25 legs.

The unit tests fuse scripted lists. These fuse the actual retrievers over an
actual index, which is the only place the wiring, the concurrent fan-out and
two real score distributions meet.

The embedder is ``hash`` — deterministic and dependency-free. That makes the
dense leg's *ranking* arbitrary rather than semantic, so nothing here asserts
retrieval quality, and none of it may be quoted as a quality result. What it
does establish is that the plumbing is correct.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from recall.config.settings import Settings
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.models import Chunk, Document, SearchFilters, SourceType
from recall.core.retrieval.base import Retriever
from recall.core.retrieval.dense import DenseRetriever
from recall.core.retrieval.fusion import ReciprocalRankFusion, WeightedScoreFusion
from recall.core.retrieval.hybrid import HybridRetriever
from recall.core.retrieval.lexical import BM25Retriever
from recall.pipeline.factory import build_fusion, build_retriever
from recall.pipeline.search import SearchService
from recall.storage.postgres.lexical_index import PostgresBM25Index
from recall.storage.postgres.storage import Storage

pytestmark = pytest.mark.integration

CORPUS: list[tuple[str, SourceType, str, str]] = [
    (
        "auth.md",
        SourceType.FILESYSTEM,
        "Authentication",
        "Authentication uses a bearer token. The service verifies the token signature.",
    ),
    (
        "deploy.md",
        SourceType.FILESYSTEM,
        "Deployment",
        "Rolling deployment replaces replicas one at a time while the service stays available.",
    ),
    (
        "observability.md",
        SourceType.FILESYSTEM,
        "Observability",
        "Prometheus scrapes metrics every fifteen seconds and every request is logged.",
    ),
    (
        "guide.pdf",
        SourceType.PDF,
        "Operator guide",
        "The service exposes a health endpoint. Authentication is not required for it.",
    ),
]


@pytest_asyncio.fixture
async def indexed(storage: Storage, embedder: HashingEmbedder) -> dict[str, uuid.UUID]:
    """One chunk per document, with vectors, so both legs have something to find."""
    ids: dict[str, uuid.UUID] = {}
    for source_id, source_type, title, content in CORPUS:
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
            checksum=document.checksum,
        )
        vectors = await embedder.embed_documents([content])
        await storage.index_document(document, [chunk], vectors, embedder.info)
        ids[source_id] = chunk.id
    return ids


@pytest.fixture
def legs(storage: Storage, embedder: HashingEmbedder) -> dict[str, Retriever]:
    return {
        "dense": DenseRetriever(embedder=embedder, index=storage.vectors),
        "bm25": BM25Retriever(index=PostgresBM25Index(storage.sessions)),
    }


class TestRealComponents:
    async def test_returns_the_union_of_both_legs(
        self, legs: dict[str, Retriever], indexed: dict[str, uuid.UUID]
    ) -> None:
        retriever = HybridRetriever(components=legs, fusion=ReciprocalRankFusion())
        results = await retriever.search("authentication token", top_k=10)

        dense_only = await legs["dense"].search("authentication token", top_k=30)
        bm25_only = await legs["bm25"].search("authentication token", top_k=30)
        expected = {r.chunk_id for r in dense_only} | {r.chunk_id for r in bm25_only}

        assert {r.chunk_id for r in results} == expected
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        assert all(r.retriever == "hybrid" for r in results)

    async def test_records_which_leg_found_each_hit(
        self, legs: dict[str, Retriever], indexed: dict[str, uuid.UUID]
    ) -> None:
        retriever = HybridRetriever(components=legs, fusion=ReciprocalRankFusion())
        results = await retriever.search("rolling deployment", top_k=10)

        assert results
        assert any("bm25" in r.component_scores for r in results)
        assert any("dense" in r.component_scores for r in results)
        # BM25 is selective; dense returns everything. A hit both legs found is
        # the case fusion exists to reward.
        both = [r for r in results if len(r.component_scores) == 2]
        assert both, "expected at least one chunk found by both legs"

    async def test_filters_reach_both_legs(
        self, legs: dict[str, Retriever], indexed: dict[str, uuid.UUID]
    ) -> None:
        retriever = HybridRetriever(components=legs, fusion=ReciprocalRankFusion())
        results = await retriever.search(
            "service authentication",
            top_k=10,
            filters=SearchFilters(source_types=[SourceType.PDF]),
        )
        assert results
        assert all(r.source_type is SourceType.PDF for r in results)

    async def test_bm25_only_terms_are_still_found(
        self, legs: dict[str, Retriever], indexed: dict[str, uuid.UUID]
    ) -> None:
        """The lexical leg contributes an exact match the hash embedder cannot."""
        retriever = HybridRetriever(components=legs, fusion=ReciprocalRankFusion())
        results = await retriever.search("Prometheus", top_k=10)
        top = results[0]
        assert top.chunk_id == indexed["observability.md"]
        assert "bm25" in top.component_scores

    @pytest.mark.parametrize("fusion", [ReciprocalRankFusion(), WeightedScoreFusion()])
    async def test_both_fusions_work_over_real_score_distributions(
        self, legs: dict[str, Retriever], indexed: dict[str, uuid.UUID], fusion: object
    ) -> None:
        retriever = HybridRetriever(components=legs, fusion=fusion)  # type: ignore[arg-type]
        results = await retriever.search("authentication token", top_k=5)
        assert results
        assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


class TestTiming:
    async def test_timing_is_internally_consistent(
        self, legs: dict[str, Retriever], indexed: dict[str, uuid.UUID]
    ) -> None:
        retriever = HybridRetriever(components=legs, fusion=ReciprocalRankFusion())
        response = await SearchService(retriever=retriever).search("token", top_k=5)

        timing = response.timing
        assert timing.total_ms > 0
        assert timing.fusion_ms > 0
        # Concurrent legs are merged with max, so the parts cannot exceed the
        # whole — a sum would have blown past total_ms.
        assert timing.retrieval_ms + timing.fusion_ms <= timing.total_ms + 1.0


class TestFactoryWiring:
    def test_builds_a_hybrid_from_settings(self, storage: Storage) -> None:
        settings = Settings.from_mapping(
            {
                "embedding": {"provider": "hash", "model": "hash-v1", "dimensions": 64},
                "hybrid": {"fusion": "rrf", "rrf_k": 42, "candidate_multiplier": 5},
            }
        )
        retriever = build_retriever(
            "hybrid",
            storage=storage,
            embedder=HashingEmbedder(dimensions=64),
            settings=settings,
        )
        assert isinstance(retriever, HybridRetriever)
        assert set(retriever.components) == {"dense", "bm25"}
        assert retriever.candidate_multiplier == 5
        assert isinstance(retriever.fusion, ReciprocalRankFusion)
        assert retriever.fusion.k == 42

    def test_fusion_strategy_follows_configuration(self) -> None:
        weighted = build_fusion(Settings.from_mapping({"hybrid": {"fusion": "weighted"}}))
        assert isinstance(weighted, WeightedScoreFusion)
        assert weighted.weights == {"dense": 0.65, "bm25": 0.35}

    def test_a_hybrid_naming_itself_is_rejected(self, storage: Storage) -> None:
        settings = Settings.from_mapping(
            {
                "embedding": {"provider": "hash", "model": "hash-v1", "dimensions": 64},
                "hybrid": {"components": ["hybrid"]},
            }
        )
        with pytest.raises(Exception, match="other than 'hybrid'"):
            build_retriever(
                "hybrid",
                storage=storage,
                embedder=HashingEmbedder(dimensions=64),
                settings=settings,
            )

    async def test_end_to_end_through_the_search_service(
        self, storage: Storage, indexed: dict[str, uuid.UUID]
    ) -> None:
        settings = Settings.from_mapping(
            {"embedding": {"provider": "hash", "model": "hash-v1", "dimensions": 64}}
        )
        retriever = build_retriever(
            "hybrid",
            storage=storage,
            embedder=HashingEmbedder(dimensions=64),
            settings=settings,
        )
        response = await SearchService(retriever=retriever).search("authentication", top_k=3)
        assert response.retrieval_strategy == "hybrid"
        assert response.results
        assert response.model_dump(mode="json")["results"][0]["component_ranks"]
