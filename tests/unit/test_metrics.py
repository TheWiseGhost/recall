"""Retrieval metrics.

Values here are worked by hand in the assertions rather than compared against
the implementation's own output, because a metric that is wrong in the same way
as its test is worse than no test: it produces confident, reproducible,
incorrect science.
"""

from __future__ import annotations

import math

import pytest

from recall.core.evaluation.metrics import (
    DEFAULT_METRICS,
    dcg,
    evaluate,
    hit_rate_at_k,
    latency_summary,
    metric_registry,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
)

# "a" and "b" are relevant; "x", "y", "z" are not.
BINARY = {"a": 1, "b": 1}
GRADED = {"a": 3, "b": 1}


class TestPrecision:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], BINARY, 2) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(["x", "y"], BINARY, 2) == 0.0

    def test_half_relevant(self) -> None:
        assert precision_at_k(["a", "x"], BINARY, 2) == 0.5

    def test_denominator_is_k_not_the_list_length(self) -> None:
        """Returning three results when ten were asked for is not 100% precision."""
        assert precision_at_k(["a", "b"], BINARY, 10) == pytest.approx(0.2)

    def test_counts_positions_so_duplicates_count_twice(self) -> None:
        """Two chunks from one document occupy two of the k slots a user sees."""
        assert precision_at_k(["a", "a"], BINARY, 2) == 1.0

    def test_ignores_results_past_k(self) -> None:
        assert precision_at_k(["x", "a"], BINARY, 1) == 0.0

    def test_zero_k(self) -> None:
        assert precision_at_k(["a"], BINARY, 0) == 0.0


class TestRecall:
    def test_finds_everything(self) -> None:
        assert recall_at_k(["a", "b"], BINARY, 2) == 1.0

    def test_finds_half(self) -> None:
        assert recall_at_k(["a", "x"], BINARY, 2) == 0.5

    def test_counts_distinct_items_so_duplicates_do_not_inflate(self) -> None:
        """Finding one document three times has not found three documents."""
        assert recall_at_k(["a", "a", "a"], BINARY, 3) == 0.5

    def test_cannot_exceed_one(self) -> None:
        assert recall_at_k(["a", "a", "b", "b"], BINARY, 4) == 1.0

    def test_no_relevant_items_is_zero_not_a_crash(self) -> None:
        assert recall_at_k(["a"], {}, 5) == 0.0

    def test_respects_k(self) -> None:
        assert recall_at_k(["a", "b"], BINARY, 1) == 0.5


class TestHitRate:
    def test_one_hit_is_enough(self) -> None:
        assert hit_rate_at_k(["x", "y", "a"], BINARY, 3) == 1.0

    def test_no_hits(self) -> None:
        assert hit_rate_at_k(["x", "y"], BINARY, 2) == 0.0

    def test_a_hit_past_k_does_not_count(self) -> None:
        assert hit_rate_at_k(["x", "a"], BINARY, 1) == 0.0


class TestMRR:
    def test_first_position(self) -> None:
        assert mrr_at_k(["a", "x"], BINARY, 2) == 1.0

    def test_second_position(self) -> None:
        assert mrr_at_k(["x", "a"], BINARY, 2) == 0.5

    def test_third_position(self) -> None:
        assert mrr_at_k(["x", "y", "a"], BINARY, 3) == pytest.approx(1 / 3)

    def test_only_the_first_hit_matters(self) -> None:
        assert mrr_at_k(["x", "a", "b"], BINARY, 3) == 0.5

    def test_nothing_relevant_within_k(self) -> None:
        assert mrr_at_k(["x", "y", "a"], BINARY, 2) == 0.0


class TestNDCG:
    def test_perfect_ranking_is_one(self) -> None:
        assert ndcg_at_k(["a", "b"], BINARY, 2) == pytest.approx(1.0)

    def test_nothing_relevant_is_zero(self) -> None:
        assert ndcg_at_k(["x", "y"], BINARY, 2) == 0.0

    def test_worked_example(self) -> None:
        """One relevant item at rank 2, one relevant item total."""
        # DCG  = (2^1 - 1) / log2(3)
        # IDCG = (2^1 - 1) / log2(2) = 1
        assert ndcg_at_k(["x", "a"], {"a": 1}, 2) == pytest.approx(1 / math.log2(3))

    def test_graded_relevance_rewards_the_higher_grade_first(self) -> None:
        good_order = ndcg_at_k(["a", "b"], GRADED, 2)  # grade 3 then grade 1
        bad_order = ndcg_at_k(["b", "a"], GRADED, 2)
        assert good_order == pytest.approx(1.0)
        assert bad_order < good_order

    def test_exponential_gain_distinguishes_grades(self) -> None:
        """With a linear numerator, one grade-3 would equal three grade-1s."""
        assert dcg([3.0]) == pytest.approx(7.0)  # 2^3 - 1
        assert dcg([1.0]) == pytest.approx(1.0)

    def test_a_repeat_earns_gain_only_once(self) -> None:
        """Otherwise DCG can exceed the ideal and NDCG can exceed 1."""
        repeated = ndcg_at_k(["a", "a", "a"], {"a": 1}, 3)
        assert repeated == pytest.approx(1.0)
        assert repeated <= 1.0

    def test_never_exceeds_one_even_when_everything_repeats(self) -> None:
        assert ndcg_at_k(["a"] * 10, GRADED, 10) <= 1.0

    def test_ideal_uses_the_judgement_not_the_results(self) -> None:
        """Missing a relevant item must cost, not be normalised away."""
        assert ndcg_at_k(["a"], BINARY, 2) < 1.0

    def test_zero_k(self) -> None:
        assert ndcg_at_k(["a"], BINARY, 0) == 0.0


class TestEvaluate:
    def test_labels_carry_k(self) -> None:
        out = evaluate(["a", "x"], BINARY, 5)
        assert set(out) == {
            "precision@5",
            "recall@5",
            "hit_rate@5",
            "mrr@5",
            "ndcg@5",
        }

    def test_mrr_is_labelled_with_k(self) -> None:
        """It is computed over the top k, so calling it plain 'mrr' would mislead."""
        assert "mrr@10" in evaluate(["a"], BINARY, 10)

    def test_metric_subset(self) -> None:
        assert set(evaluate(["a"], BINARY, 3, ["mrr"])) == {"mrr@3"}

    def test_every_default_metric_is_registered(self) -> None:
        assert all(name in metric_registry for name in DEFAULT_METRICS)

    def test_empty_results_score_zero(self) -> None:
        assert all(value == 0.0 for value in evaluate([], BINARY, 5).values())


class TestPercentileAndLatency:
    def test_interpolates(self) -> None:
        assert percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)

    def test_p50_of_an_odd_count(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_p100_is_the_max(self) -> None:
        assert percentile([1.0, 5.0, 3.0], 1.0) == 5.0

    def test_single_sample(self) -> None:
        assert percentile([4.2], 0.99) == 4.2

    def test_empty(self) -> None:
        assert percentile([], 0.5) == 0.0

    def test_latency_summary_shape(self) -> None:
        summary = latency_summary([1.0, 2.0, 3.0, 4.0])
        assert set(summary) == {"p50", "p95", "p99", "mean", "min", "max"}
        assert summary["min"] == 1.0
        assert summary["max"] == 4.0
        assert summary["mean"] == 2.5

    def test_latency_summary_of_nothing_is_zeros(self) -> None:
        assert latency_summary([]) == {
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
