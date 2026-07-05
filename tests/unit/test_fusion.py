"""Rank fusion: RRF and weighted score fusion."""

from __future__ import annotations

import uuid

import pytest

from recall.core.errors import ConfigurationError, PluginNotFoundError
from recall.core.models import SearchResult
from recall.core.retrieval.fusion import (
    ReciprocalRankFusion,
    WeightedScoreFusion,
    create_fusion,
    fusion_registry,
)

# Stable IDs so ordering assertions can name a chunk.
A, B, C, D = (uuid.UUID(int=index) for index in range(1, 5))


def ranked(*pairs: tuple[uuid.UUID, float]) -> list[SearchResult]:
    """A ranked list from ``(chunk_id, score)`` pairs, in the given order."""
    return [
        SearchResult(
            chunk_id=chunk_id,
            document_id=uuid.UUID(int=100),
            content=f"chunk {chunk_id.int}",
            score=score,
            rank=rank,
        )
        for rank, (chunk_id, score) in enumerate(pairs, start=1)
    ]


class TestRegistration:
    def test_both_strategies_are_registered(self) -> None:
        assert set(fusion_registry.names()) == {"rrf", "weighted"}

    def test_unknown_strategy_names_the_alternatives(self) -> None:
        with pytest.raises(PluginNotFoundError, match="rrf"):
            create_fusion("telepathic")


class TestReciprocalRankFusion:
    def test_matches_the_formula(self) -> None:
        fusion = ReciprocalRankFusion(k=60)
        lists = {"dense": ranked((A, 0.9), (B, 0.8)), "bm25": ranked((B, 12.0), (A, 3.0))}
        results = {r.chunk_id: r.score for r in fusion.fuse(lists, top_k=10)}

        # Equal weights, rescaled to 0.5 each.
        assert results[A] == pytest.approx(0.5 / 61 + 0.5 / 62)
        assert results[B] == pytest.approx(0.5 / 62 + 0.5 / 61)

    def test_ignores_score_magnitude(self) -> None:
        """The point of RRF: BM25's unbounded scale must not swamp cosine."""
        fusion = ReciprocalRankFusion()
        modest = {"dense": ranked((A, 0.9)), "bm25": ranked((B, 2.0))}
        enormous = {"dense": ranked((A, 0.9)), "bm25": ranked((B, 9_999.0))}
        assert [r.chunk_id for r in fusion.fuse(modest, top_k=10)] == [
            r.chunk_id for r in fusion.fuse(enormous, top_k=10)
        ]

    def test_agreement_beats_a_single_strong_hit(self) -> None:
        """A chunk both retrievers like outranks one only a single retriever found."""
        fusion = ReciprocalRankFusion(k=60)
        lists = {
            "dense": ranked((A, 0.99), (B, 0.5)),
            "bm25": ranked((B, 10.0), (C, 9.0)),
        }
        assert fusion.fuse(lists, top_k=3)[0].chunk_id == B

    def test_smaller_k_sharpens_the_top(self) -> None:
        lists = {"dense": ranked((A, 1.0), (B, 0.9)), "bm25": ranked((B, 5.0), (A, 4.0))}
        flat = ReciprocalRankFusion(k=1000).fuse(lists, top_k=2)
        sharp = ReciprocalRankFusion(k=1).fuse(lists, top_k=2)
        assert sharp[0].score - sharp[1].score >= flat[0].score - flat[1].score

    def test_weights_shift_the_ranking(self) -> None:
        lists = {"dense": ranked((A, 0.9)), "bm25": ranked((B, 9.0))}
        dense_heavy = ReciprocalRankFusion(weights={"dense": 0.9, "bm25": 0.1})
        lexical_heavy = ReciprocalRankFusion(weights={"dense": 0.1, "bm25": 0.9})
        assert dense_heavy.fuse(lists, top_k=2)[0].chunk_id == A
        assert lexical_heavy.fuse(lists, top_k=2)[0].chunk_id == B

    def test_weights_are_scale_invariant(self) -> None:
        """0.65/0.35 and 65/35 must mean the same thing."""
        lists = {"dense": ranked((A, 0.9), (B, 0.1)), "bm25": ranked((B, 9.0), (C, 1.0))}
        small = ReciprocalRankFusion(weights={"dense": 0.65, "bm25": 0.35}).fuse(lists, top_k=5)
        large = ReciprocalRankFusion(weights={"dense": 65.0, "bm25": 35.0}).fuse(lists, top_k=5)
        assert [r.chunk_id for r in small] == [r.chunk_id for r in large]
        assert [r.score for r in small] == pytest.approx([r.score for r in large])

    def test_rejects_a_non_positive_k(self) -> None:
        with pytest.raises(ConfigurationError, match="rrf_k"):
            ReciprocalRankFusion(k=0)


class TestWeightedScoreFusion:
    def test_normalises_each_list_before_combining(self) -> None:
        fusion = WeightedScoreFusion(weights={"dense": 0.5, "bm25": 0.5})
        lists = {"dense": ranked((A, 1.0), (B, 0.0)), "bm25": ranked((B, 100.0), (A, 0.0))}
        results = {r.chunk_id: r.score for r in fusion.fuse(lists, top_k=10)}
        # Each list min-max maps to [0, 1], so both tie at 0.5 despite BM25's
        # raw scores being two orders of magnitude larger.
        assert results[A] == pytest.approx(0.5)
        assert results[B] == pytest.approx(0.5)

    def test_score_magnitude_matters(self) -> None:
        """The difference from RRF: a runaway top hit keeps its margin."""
        fusion = WeightedScoreFusion()
        lists = {
            "dense": ranked((A, 1.0), (B, 0.98), (C, 0.02)),
            "bm25": ranked((C, 9.0), (B, 8.9), (A, 0.1)),
        }
        by_id = {r.chunk_id: r.score for r in fusion.fuse(lists, top_k=3)}
        assert by_id[B] > by_id[A]
        assert by_id[B] > by_id[C]

    def test_identical_scores_do_not_divide_by_zero(self) -> None:
        fusion = WeightedScoreFusion()
        results = fusion.fuse({"dense": ranked((A, 0.5), (B, 0.5))}, top_k=5)
        assert [r.score for r in results] == pytest.approx([1.0, 1.0])

    def test_weights_shift_the_ranking(self) -> None:
        lists = {"dense": ranked((A, 1.0), (B, 0.0)), "bm25": ranked((B, 10.0), (A, 0.0))}
        dense_heavy = WeightedScoreFusion(weights={"dense": 0.9, "bm25": 0.1})
        assert dense_heavy.fuse(lists, top_k=2)[0].chunk_id == A

    def test_the_bottom_of_a_list_is_worth_nothing(self) -> None:
        """The documented cost of per-query min-max, pinned so it stays known.

        C is last in dense's list, so min-max scores it 0 there — it gets no
        credit for having been retrieved at all. RRF, which sees rank 3 rather
        than "worst", does not do this. It is why ``rrf`` is the default.
        """
        lists = {
            "dense": ranked((A, 0.9), (B, 0.8), (C, 0.7)),
            "bm25": ranked((C, 9.0), (D, 8.0)),
        }
        weighted = {r.chunk_id: r.score for r in WeightedScoreFusion().fuse(lists, top_k=10)}
        assert weighted[C] == pytest.approx(weighted[A])  # tied, despite C being found twice

        rrf = [r.chunk_id for r in ReciprocalRankFusion().fuse(lists, top_k=10)]
        assert rrf[0] == C  # found by both, and RRF rewards that

    def test_rejects_all_zero_weights(self) -> None:
        fusion = WeightedScoreFusion(weights={"dense": 0.0, "bm25": 0.0})
        with pytest.raises(ConfigurationError, match="must not all be zero"):
            fusion.fuse({"dense": ranked((A, 1.0)), "bm25": ranked((B, 1.0))}, top_k=2)

    def test_rejects_negative_weights(self) -> None:
        fusion = WeightedScoreFusion(weights={"dense": -1.0, "bm25": 2.0})
        with pytest.raises(ConfigurationError, match="non-negative"):
            fusion.fuse({"dense": ranked((A, 1.0)), "bm25": ranked((B, 1.0))}, top_k=2)


@pytest.mark.parametrize("strategy", ["rrf", "weighted"])
class TestSharedContract:
    def test_ranks_are_one_based_and_sequential(self, strategy: str) -> None:
        fusion = create_fusion(strategy)
        lists = {"dense": ranked((A, 0.9), (B, 0.5)), "bm25": ranked((C, 9.0), (A, 1.0))}
        assert [r.rank for r in fusion.fuse(lists, top_k=10)] == [1, 2, 3]

    def test_deduplicates_across_lists(self, strategy: str) -> None:
        fusion = create_fusion(strategy)
        lists = {"dense": ranked((A, 0.9)), "bm25": ranked((A, 9.0))}
        assert len(fusion.fuse(lists, top_k=10)) == 1

    def test_records_every_component_contribution(self, strategy: str) -> None:
        fusion = create_fusion(strategy)
        lists = {"dense": ranked((A, 0.9), (B, 0.4)), "bm25": ranked((B, 9.0))}
        by_id = {r.chunk_id: r for r in fusion.fuse(lists, top_k=10)}

        assert by_id[A].component_scores == {"dense": 0.9}
        assert by_id[A].component_ranks == {"dense": 1}
        assert by_id[B].component_scores == {"dense": 0.4, "bm25": 9.0}
        assert by_id[B].component_ranks == {"dense": 2, "bm25": 1}

    def test_top_k_truncates_after_fusing(self, strategy: str) -> None:
        """Truncating before fusion would discard evidence the fusion needs."""
        fusion = create_fusion(strategy)
        lists = {
            "dense": ranked((A, 0.9), (B, 0.8), (C, 0.7)),
            "bm25": ranked((C, 9.0), (D, 8.0)),
        }
        everything = fusion.fuse(lists, top_k=10)
        truncated = fusion.fuse(lists, top_k=2)
        assert len(truncated) == 2
        assert [r.chunk_id for r in truncated] == [r.chunk_id for r in everything[:2]]

    def test_non_positive_top_k_is_empty(self, strategy: str) -> None:
        assert create_fusion(strategy).fuse({"dense": ranked((A, 1.0))}, top_k=0) == []

    def test_empty_lists_are_empty(self, strategy: str) -> None:
        assert create_fusion(strategy).fuse({"dense": [], "bm25": []}, top_k=10) == []

    def test_a_single_component_passes_the_ranking_through(self, strategy: str) -> None:
        fusion = create_fusion(strategy)
        results = fusion.fuse({"dense": ranked((A, 0.9), (B, 0.5), (C, 0.1))}, top_k=10)
        assert [r.chunk_id for r in results] == [A, B, C]

    def test_ordering_is_deterministic_under_ties(self, strategy: str) -> None:
        fusion = create_fusion(strategy)
        lists = {"dense": ranked((A, 0.5), (B, 0.5)), "bm25": ranked((B, 1.0), (A, 1.0))}
        first = [r.chunk_id for r in fusion.fuse(lists, top_k=10)]
        assert first == [r.chunk_id for r in fusion.fuse(lists, top_k=10)]

    def test_preserves_provenance(self, strategy: str) -> None:
        fusion = create_fusion(strategy)
        results = fusion.fuse({"dense": ranked((A, 0.9))}, top_k=1)
        assert results[0].content == "chunk 1"
        assert results[0].document_id == uuid.UUID(int=100)
