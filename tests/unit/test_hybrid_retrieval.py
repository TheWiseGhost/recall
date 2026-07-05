"""Hybrid retrieval: fan-out, fusion, timing and wiring."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence

import pytest

from recall.config.settings import Settings
from recall.core.errors import ConfigurationError
from recall.core.models import SearchFilters, SearchResult, SourceType
from recall.core.retrieval import create_retriever, retriever_registry
from recall.core.retrieval.base import Retriever, stage
from recall.core.retrieval.fusion import ReciprocalRankFusion, WeightedScoreFusion
from recall.core.retrieval.hybrid import HybridRetriever
from recall.pipeline.search import SearchService

A, B, C, D = (uuid.UUID(int=index) for index in range(1, 5))


class ScriptedRetriever:
    """A retriever that returns a fixed list, recording how it was called."""

    def __init__(
        self,
        name: str,
        results: Sequence[tuple[uuid.UUID, float]],
        *,
        delay: float = 0.0,
        stage_name: str = "retrieval",
    ) -> None:
        self.name = name
        self._results = list(results)
        self._delay = delay
        self._stage = stage_name
        self.calls: list[tuple[str, int, SearchFilters | None]] = []

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query, top_k, filters))
        with stage(self._stage):
            if self._delay:
                await asyncio.sleep(self._delay)
        return [
            SearchResult(
                chunk_id=chunk_id,
                document_id=uuid.UUID(int=100),
                content=f"chunk {chunk_id.int}",
                score=score,
                rank=rank,
                source_type=SourceType.MEMORY,
                retriever=self.name,
            )
            for rank, (chunk_id, score) in enumerate(self._results[:top_k], start=1)
        ]


def hybrid(**kwargs: object) -> HybridRetriever:
    components: dict[str, Retriever] = {
        "dense": ScriptedRetriever("dense", [(A, 0.9), (B, 0.8), (C, 0.7)]),
        "bm25": ScriptedRetriever("bm25", [(C, 9.0), (D, 8.0)]),
    }
    defaults: dict[str, object] = {"components": components, "fusion": ReciprocalRankFusion()}
    return HybridRetriever(**{**defaults, **kwargs})  # type: ignore[arg-type]


class TestRegistration:
    def test_hybrid_is_registered(self) -> None:
        assert "hybrid" in retriever_registry

    def test_resolves_through_the_registry(self) -> None:
        retriever = create_retriever(
            "hybrid",
            components={"dense": ScriptedRetriever("dense", [(A, 1.0)])},
            fusion=ReciprocalRankFusion(),
        )
        assert isinstance(retriever, HybridRetriever)
        assert retriever.name == "hybrid"


class TestFanOut:
    async def test_queries_every_component(self) -> None:
        components: dict[str, Retriever] = {
            "dense": ScriptedRetriever("dense", [(A, 0.9)]),
            "bm25": ScriptedRetriever("bm25", [(B, 9.0)]),
        }
        retriever = HybridRetriever(components=components, fusion=ReciprocalRankFusion())
        await retriever.search("tokens", top_k=5)
        for component in components.values():
            assert component.calls  # type: ignore[attr-defined]

    async def test_over_fetches_candidates(self) -> None:
        """Fusion can only see what a component returned; ask for more."""
        dense = ScriptedRetriever("dense", [(A, 0.9)])
        retriever = HybridRetriever(
            components={"dense": dense},
            fusion=ReciprocalRankFusion(),
            candidate_multiplier=4,
        )
        await retriever.search("tokens", top_k=5)
        assert dense.calls[0][1] == 20

    async def test_passes_filters_through_unchanged(self) -> None:
        dense = ScriptedRetriever("dense", [(A, 0.9)])
        filters = SearchFilters(source_types=[SourceType.PDF])
        retriever = HybridRetriever(components={"dense": dense}, fusion=ReciprocalRankFusion())
        await retriever.search("tokens", top_k=5, filters=filters)
        assert dense.calls[0][2] == filters

    async def test_components_run_concurrently(self) -> None:
        """Serial fan-out would lose the latency comparison for the wrong reason."""
        components: dict[str, Retriever] = {
            "dense": ScriptedRetriever("dense", [(A, 0.9)], delay=0.05),
            "bm25": ScriptedRetriever("bm25", [(B, 9.0)], delay=0.05),
        }
        retriever = HybridRetriever(components=components, fusion=ReciprocalRankFusion())
        started = asyncio.get_running_loop().time()
        await retriever.search("tokens", top_k=5)
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed < 0.09  # serial would be >= 0.10


class TestResults:
    async def test_fuses_into_one_ranking(self) -> None:
        results = await hybrid().search("tokens", top_k=10)
        assert {r.chunk_id for r in results} == {A, B, C, D}
        assert [r.rank for r in results] == [1, 2, 3, 4]

    async def test_stamps_hybrid_as_the_retriever(self) -> None:
        results = await hybrid().search("tokens", top_k=10)
        assert all(r.retriever == "hybrid" for r in results)

    async def test_keeps_component_provenance(self) -> None:
        by_id = {r.chunk_id: r for r in await hybrid().search("tokens", top_k=10)}
        assert by_id[C].component_scores == {"dense": 0.7, "bm25": 9.0}
        assert by_id[C].component_ranks == {"dense": 3, "bm25": 1}
        assert by_id[D].component_scores == {"bm25": 8.0}

    async def test_top_k_limits_the_fused_list(self) -> None:
        assert len(await hybrid().search("tokens", top_k=2)) == 2

    async def test_non_positive_top_k_short_circuits(self) -> None:
        dense = ScriptedRetriever("dense", [(A, 0.9)])
        retriever = HybridRetriever(components={"dense": dense}, fusion=ReciprocalRankFusion())
        assert await retriever.search("tokens", top_k=0) == []
        assert dense.calls == []

    async def test_the_fusion_strategy_is_swappable(self) -> None:
        rrf = await hybrid(fusion=ReciprocalRankFusion()).search("tokens", top_k=4)
        weighted = await hybrid(fusion=WeightedScoreFusion()).search("tokens", top_k=4)
        # Same candidates, different ordering — which is the whole reason the
        # choice is a configurable experimental variable.
        assert {r.chunk_id for r in rrf} == {r.chunk_id for r in weighted}
        assert [r.chunk_id for r in rrf] != [r.chunk_id for r in weighted]

    async def test_survives_a_component_returning_nothing(self) -> None:
        components: dict[str, Retriever] = {
            "dense": ScriptedRetriever("dense", [(A, 0.9)]),
            "bm25": ScriptedRetriever("bm25", []),
        }
        retriever = HybridRetriever(components=components, fusion=ReciprocalRankFusion())
        results = await retriever.search("tokens", top_k=5)
        assert [r.chunk_id for r in results] == [A]


class TestTiming:
    async def test_concurrent_stages_do_not_exceed_wall_clock(self) -> None:
        """Two 50 ms lookups that overlapped cost 50 ms, not 100 ms."""
        components: dict[str, Retriever] = {
            "dense": ScriptedRetriever("dense", [(A, 0.9)], delay=0.05),
            "bm25": ScriptedRetriever("bm25", [(B, 9.0)], delay=0.05),
        }
        retriever = HybridRetriever(components=components, fusion=ReciprocalRankFusion())
        response = await SearchService(retriever=retriever).search("tokens", top_k=5)

        assert response.timing.retrieval_ms == pytest.approx(50, abs=25)
        assert response.timing.retrieval_ms <= response.timing.total_ms

    async def test_records_fusion_separately_from_retrieval(self) -> None:
        response = await SearchService(retriever=hybrid()).search("tokens", top_k=5)
        assert response.retrieval_strategy == "hybrid"
        assert response.timing.fusion_ms > 0
        assert response.timing.retrieval_ms >= 0

    async def test_component_stage_breakdown_survives(self) -> None:
        """A dense component's embedding time must still be visible."""
        components: dict[str, Retriever] = {
            "dense": ScriptedRetriever("dense", [(A, 0.9)], delay=0.02, stage_name="embedding"),
            "bm25": ScriptedRetriever("bm25", [(B, 9.0)]),
        }
        retriever = HybridRetriever(components=components, fusion=ReciprocalRankFusion())
        response = await SearchService(retriever=retriever).search("tokens", top_k=5)
        assert response.timing.embedding_ms > 0


class TestConstruction:
    def test_rejects_no_components(self) -> None:
        with pytest.raises(ConfigurationError, match="at least one component"):
            HybridRetriever(components={}, fusion=ReciprocalRankFusion())

    def test_rejects_a_zero_candidate_multiplier(self) -> None:
        with pytest.raises(ConfigurationError, match="candidate_multiplier"):
            hybrid(candidate_multiplier=0)


class TestSettings:
    def test_weights_are_keyed_by_retriever_name(self) -> None:
        settings = Settings.from_mapping({"hybrid": {"dense_weight": 0.7, "lexical_weight": 0.3}})
        assert settings.hybrid.weights() == {"dense": 0.7, "bm25": 0.3}

    def test_an_unnamed_component_is_not_silently_zeroed(self) -> None:
        settings = Settings.from_mapping({"hybrid": {"components": ["dense", "bm25"]}})
        assert 0.0 not in settings.hybrid.weights().values()

    def test_rejects_duplicate_components(self) -> None:
        with pytest.raises(Exception, match="duplicates"):
            Settings.from_mapping({"hybrid": {"components": ["dense", "dense"]}})

    def test_rejects_an_unregistered_component(self) -> None:
        import tempfile
        from pathlib import Path

        from recall.config.settings import load_settings

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recall.yaml"
            path.write_text("hybrid:\n  components: [dense, telepathic]\n", encoding="utf-8")
            with pytest.raises(ConfigurationError, match="telepathic"):
                load_settings(path)

    def test_rejects_an_unregistered_fusion_strategy(self) -> None:
        with pytest.raises(Exception, match="fusion"):
            Settings.from_mapping({"hybrid": {"fusion": "telepathy"}})
