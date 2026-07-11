"""Reranking: the protocol, the identity reranker, and pipeline composition.

The cross-encoder itself is exercised with a stubbed model — downloading a real
one would make the suite depend on the network and on a multi-hundred-megabyte
artefact. What is tested here is everything around the model: the lazy import,
the error when the extra is missing, that scoring happens off the event loop,
and that the retrieval score survives.
"""

from __future__ import annotations

import asyncio
import sys
import types
import uuid
from collections.abc import Sequence
from typing import ClassVar

import pytest

from recall.config.settings import Settings
from recall.core.errors import ConfigurationError
from recall.core.models import SearchFilters, SearchResult
from recall.core.reranking import (
    CrossEncoderReranker,
    NoOpReranker,
    RerankerUnavailableError,
    create_reranker,
    reranker_registry,
)
from recall.core.retrieval.base import Retriever
from recall.pipeline.factory import build_reranker
from recall.pipeline.search import SearchService


def results(*scores: float) -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id=uuid.UUID(int=index),
            document_id=uuid.UUID(int=100),
            content=f"chunk {index}",
            score=score,
            rank=index,
        )
        for index, score in enumerate(scores, start=1)
    ]


class StubRetriever:
    """Returns as many results as asked for, recording the ask."""

    name = "stub"

    def __init__(self, available: int = 100) -> None:
        self.available = available
        self.calls: list[int] = []

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        self.calls.append(top_k)
        count = min(top_k, self.available)
        return results(*[1.0 - index / 1000 for index in range(count)])


class ReversingReranker:
    """Deterministic stand-in for a real reranker: reverses the candidates."""

    name = "reversing"

    def __init__(self) -> None:
        self.seen: list[int] = []

    async def rerank(
        self, query: str, candidates: Sequence[SearchResult], *, top_k: int
    ) -> list[SearchResult]:
        self.seen.append(len(candidates))
        reversed_results = [
            candidate.model_copy(update={"retrieval_score": candidate.score, "score": float(index)})
            for index, candidate in enumerate(reversed(list(candidates)))
        ]
        return [r.model_copy(update={"rank": i}) for i, r in enumerate(reversed_results[:top_k], 1)]


class TestRegistration:
    def test_both_strategies_are_registered(self) -> None:
        assert set(reranker_registry.names()) == {"cross_encoder", "none"}

    def test_resolve_through_the_registry(self) -> None:
        assert isinstance(create_reranker("none"), NoOpReranker)
        assert isinstance(create_reranker("cross_encoder"), CrossEncoderReranker)


class TestNoOpReranker:
    async def test_preserves_order(self) -> None:
        candidates = results(0.9, 0.5, 0.1)
        reranked = await NoOpReranker().rerank("q", candidates, top_k=3)
        assert [r.chunk_id for r in reranked] == [r.chunk_id for r in candidates]

    async def test_truncates_and_restamps_ranks(self) -> None:
        reranked = await NoOpReranker().rerank("q", results(0.9, 0.5, 0.1), top_k=2)
        assert [r.rank for r in reranked] == [1, 2]

    async def test_leaves_retrieval_score_unset(self) -> None:
        """It did not rerank, so there is no "before" distinct from "after"."""
        reranked = await NoOpReranker().rerank("q", results(0.9), top_k=1)
        assert reranked[0].retrieval_score is None


class TestSearchServiceComposition:
    async def test_no_reranker_leaves_retrieval_untouched(self) -> None:
        retriever = StubRetriever()
        response = await SearchService(retriever=retriever).search("q", top_k=5)
        assert retriever.calls == [5]  # no widening
        assert response.reranked is False
        assert response.reranking_strategy is None
        assert response.timing.reranking_ms == 0.0

    async def test_reranking_widens_the_candidate_pool(self) -> None:
        retriever = StubRetriever()
        reranker = ReversingReranker()
        service = SearchService(retriever=retriever, reranker=reranker, rerank_candidates=50)
        response = await service.search("q", top_k=5)

        assert retriever.calls == [50]
        assert reranker.seen == [50]
        assert response.candidates == 50
        assert len(response.results) == 5

    async def test_never_asks_for_fewer_candidates_than_top_k(self) -> None:
        """A reranker cannot fill a gap retrieval left."""
        retriever = StubRetriever()
        service = SearchService(
            retriever=retriever, reranker=ReversingReranker(), rerank_candidates=5
        )
        await service.search("q", top_k=20)
        assert retriever.calls == [20]

    async def test_reranker_reorders_the_results(self) -> None:
        service = SearchService(retriever=StubRetriever(), reranker=ReversingReranker())
        plain = await SearchService(retriever=StubRetriever()).search("q", top_k=3)
        reranked = await service.search("q", top_k=3)
        assert [r.chunk_id for r in reranked.results] != [r.chunk_id for r in plain.results]

    async def test_records_reranking_latency_and_strategy(self) -> None:
        service = SearchService(retriever=StubRetriever(), reranker=ReversingReranker())
        response = await service.search("q", top_k=3)
        assert response.reranked is True
        assert response.reranking_strategy == "reversing"
        assert response.timing.reranking_ms > 0
        assert response.timing.total_ms >= response.timing.reranking_ms

    async def test_ranks_are_sequential_after_reranking(self) -> None:
        service = SearchService(retriever=StubRetriever(), reranker=ReversingReranker())
        response = await service.search("q", top_k=4)
        assert [r.rank for r in response.results] == [1, 2, 3, 4]

    async def test_response_is_json_serializable(self) -> None:
        import json

        service = SearchService(retriever=StubRetriever(), reranker=ReversingReranker())
        response = await service.search("q", top_k=2)
        payload = json.loads(json.dumps(response.model_dump(mode="json")))
        assert payload["reranking_strategy"] == "reversing"
        assert payload["results"][0]["retrieval_score"] is not None


class TestFactoryWiring:
    def test_disabled_reranking_builds_nothing(self) -> None:
        assert build_reranker(Settings.from_mapping({"reranking": {"enabled": False}})) is None

    def test_enabled_none_is_a_real_component(self) -> None:
        """The control condition: widen the pool, do not reorder."""
        settings = Settings.from_mapping({"reranking": {"enabled": True, "strategy": "none"}})
        assert isinstance(build_reranker(settings), NoOpReranker)

    def test_cross_encoder_receives_its_settings(self) -> None:
        settings = Settings.from_mapping(
            {
                "reranking": {
                    "enabled": True,
                    "strategy": "cross_encoder",
                    "model": "cross-encoder/other",
                    "batch_size": 8,
                    "max_length": 256,
                }
            }
        )
        reranker = build_reranker(settings)
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.model == "cross-encoder/other"
        assert reranker.batch_size == 8
        assert reranker.max_length == 256

    def test_building_never_loads_the_model(self) -> None:
        """Configuration validation must not trigger a model download."""
        settings = Settings.from_mapping(
            {"reranking": {"enabled": True, "strategy": "cross_encoder"}}
        )
        reranker = build_reranker(settings)
        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker._encoder is None

    def test_rejects_an_unregistered_reranker(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from recall.config.settings import load_settings

        with TemporaryDirectory() as directory:
            path = Path(directory) / "recall.yaml"
            path.write_text("reranking:\n  strategy: telepathic\n", encoding="utf-8")
            with pytest.raises(ConfigurationError, match="telepathic"):
                load_settings(path)


class StubCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder."""

    instances: ClassVar[list[StubCrossEncoder]] = []

    def __init__(self, model: str, **kwargs: object) -> None:
        self.model = model
        self.kwargs = kwargs
        self.calls: list[list[tuple[str, str]]] = []
        self.threads: list[str] = []
        StubCrossEncoder.instances.append(self)

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        import threading

        self.calls.append(pairs)
        self.threads.append(threading.current_thread().name)
        # Score by content length, so the ordering is deterministic and
        # different from the retrieval ordering.
        return [float(len(document)) for _, document in pairs]


@pytest.fixture
def stub_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> type[StubCrossEncoder]:
    StubCrossEncoder.instances = []
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = StubCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return StubCrossEncoder


class TestCrossEncoderReranker:
    async def test_scores_and_reorders(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        reranker = CrossEncoderReranker()
        candidates = [
            SearchResult(
                chunk_id=uuid.UUID(int=index),
                document_id=uuid.UUID(int=100),
                content="x" * length,
                score=1.0 - index / 10,
                rank=index,
            )
            for index, length in enumerate([5, 50, 20], start=1)
        ]
        reranked = await reranker.rerank("q", candidates, top_k=3)
        # The stub scores by length, so the longest chunk must come first —
        # the opposite of the retrieval ordering it was handed.
        assert [len(r.content) for r in reranked] == [50, 20, 5]
        assert [r.rank for r in reranked] == [1, 2, 3]

    async def test_preserves_the_retrieval_score(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        candidates = results(0.9, 0.5)
        reranked = await CrossEncoderReranker().rerank("q", candidates, top_k=2)
        before = {r.chunk_id: r.score for r in candidates}
        for result in reranked:
            assert result.retrieval_score == before[result.chunk_id]
            assert result.score != result.retrieval_score

    async def test_pairs_the_query_with_every_candidate(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        await CrossEncoderReranker().rerank("how does auth work", results(0.9, 0.5), top_k=2)
        pairs = stub_sentence_transformers.instances[0].calls[0]
        assert [query for query, _ in pairs] == ["how does auth work"] * 2

    async def test_runs_off_the_event_loop(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        """A blocking forward pass on the loop would stall every other request."""
        await CrossEncoderReranker().rerank("q", results(0.9), top_k=1)
        thread = stub_sentence_transformers.instances[0].threads[0]
        assert thread != asyncio.get_running_loop().__class__.__name__
        assert "MainThread" not in thread

    async def test_loads_the_model_once(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        reranker = CrossEncoderReranker()
        await reranker.rerank("q", results(0.9), top_k=1)
        await reranker.rerank("q", results(0.8), top_k=1)
        assert len(stub_sentence_transformers.instances) == 1

    async def test_truncates_to_top_k(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        reranked = await CrossEncoderReranker().rerank("q", results(0.9, 0.5, 0.1), top_k=2)
        assert len(reranked) == 2

    async def test_empty_candidates_short_circuit(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        assert await CrossEncoderReranker().rerank("q", [], top_k=5) == []
        assert stub_sentence_transformers.instances == []

    async def test_non_positive_top_k_short_circuits(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        assert await CrossEncoderReranker().rerank("q", results(0.9), top_k=0) == []

    async def test_passes_model_options_through(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        reranker = CrossEncoderReranker(model="cross-encoder/x", device="cpu", max_length=128)
        await reranker.rerank("q", results(0.9), top_k=1)
        instance = stub_sentence_transformers.instances[0]
        assert instance.model == "cross-encoder/x"
        assert instance.kwargs == {"max_length": 128, "device": "cpu"}

    def test_rejects_a_zero_batch_size(self) -> None:
        with pytest.raises(ConfigurationError, match="batch_size"):
            CrossEncoderReranker(batch_size=0)

    async def test_missing_extra_gives_an_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def blocked(name: str, *args: object, **kwargs: object) -> object:
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # type: ignore[arg-type]
        monkeypatch.setattr(builtins, "__import__", blocked)

        with pytest.raises(RerankerUnavailableError, match=r"recall\[local\]"):
            await CrossEncoderReranker().rerank("q", results(0.9), top_k=1)


class TestEndToEndThroughTheService:
    async def test_cross_encoder_in_the_pipeline(
        self, stub_sentence_transformers: type[StubCrossEncoder]
    ) -> None:
        retriever: Retriever = StubRetriever()
        service = SearchService(
            retriever=retriever, reranker=CrossEncoderReranker(), rerank_candidates=20
        )
        response = await service.search("q", top_k=3)

        assert response.reranked is True
        assert response.reranking_strategy == "cross_encoder"
        assert response.candidates == 20
        assert len(response.results) == 3
        assert stub_sentence_transformers.instances[0].calls[0].__len__() == 20
        assert all(r.retrieval_score is not None for r in response.results)
