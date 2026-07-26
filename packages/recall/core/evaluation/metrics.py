"""Retrieval quality metrics.

Every metric here takes the same two things: a **ranked list of label keys** —
what the system returned, best first — and a **judgement**, mapping label key to
relevance grade. Everything about translating chunks into label keys happens
before this module, so the arithmetic can be read and checked on its own.

Three decisions are baked in. They change the numbers, so they are stated here
rather than buried:

**Recall is counted over distinct keys; precision is counted over positions.**
``Precision@K`` divides by ``K``, because a system that returns ten results and
gets three right has precision 0.3 whether or not the seven wrong ones are
duplicates. ``Recall@K`` divides by the number of relevant items in the
judgement and counts each one once, because finding the same relevant document
in three different chunks has not found three relevant documents.

**A repeated key earns gain only once, at its first occurrence.** In NDCG, a
document that appears at ranks 1, 2 and 3 would otherwise accumulate three
times the gain of a document found once — and could exceed the ideal DCG,
producing NDCG above 1. Later occurrences contribute nothing: they carry no
information the first did not.

**MRR is computed over the top ``K``**, not over an unbounded list, and is
reported as ``mrr@K``. A system is only ever asked for ``K`` results, so a
"true" MRR would require retrieving the whole corpus. Comparing ``mrr@10``
against a published unbounded MRR is not valid, which is why the ``@K`` is in
the name.

Binary relevance is graded relevance where every grade is 1, so there is one
implementation, not two.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from recall.core.registry import Registry

Judgement = Mapping[str, int]
"""``label key -> relevance grade``. Grades are positive integers; absent means 0."""


def _first_occurrences(keys: Sequence[str], k: int) -> list[tuple[int, str]]:
    """``(position, key)`` for the first appearance of each key within ``k``."""
    seen: set[str] = set()
    out: list[tuple[int, str]] = []
    for position, key in enumerate(keys[:k]):
        if key not in seen:
            seen.add(key)
            out.append((position, key))
    return out


def precision_at_k(ranked: Sequence[str], judgement: Judgement, k: int) -> float:
    """Fraction of the top ``k`` positions holding a relevant item.

    The denominator is ``k``, not ``len(ranked)``: a system asked for ten
    results that returns three is not thereby more precise.
    """
    if k <= 0:
        return 0.0
    hits = sum(1 for key in ranked[:k] if judgement.get(key, 0) > 0)
    return hits / k


def recall_at_k(ranked: Sequence[str], judgement: Judgement, k: int) -> float:
    """Fraction of the relevant items found in the top ``k``, counted once each."""
    relevant = {key for key, grade in judgement.items() if grade > 0}
    if not relevant:
        return 0.0
    found = {key for key in ranked[:k] if key in relevant}
    return len(found) / len(relevant)


def hit_rate_at_k(ranked: Sequence[str], judgement: Judgement, k: int) -> float:
    """1.0 when anything relevant appears in the top ``k``, else 0.0.

    Coarse on purpose. For a question-answering pipeline that only needs one
    good chunk, it is the metric that matches what the system is for.
    """
    return 1.0 if any(judgement.get(key, 0) > 0 for key in ranked[:k]) else 0.0


def mrr_at_k(ranked: Sequence[str], judgement: Judgement, k: int) -> float:
    """Reciprocal of the rank of the first relevant item, 0.0 if none in ``k``."""
    for position, key in enumerate(ranked[:k]):
        if judgement.get(key, 0) > 0:
            return 1.0 / (position + 1)
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the standard ``2^g - 1`` numerator.

    The exponential form is what makes NDCG sensitive to *graded* relevance:
    with a linear numerator, one grade-3 document is worth exactly three
    grade-1 documents, which is not what a grade of 3 is meant to say.
    """
    return float(
        sum((2.0**gain - 1.0) / math.log2(position + 2) for position, gain in enumerate(gains))
    )


def ndcg_at_k(ranked: Sequence[str], judgement: Judgement, k: int) -> float:
    """Normalised DCG over the top ``k``.

    The ideal ranking is the judgement's grades sorted descending — one entry
    per relevant item, so it is achievable by a system that returns each
    relevant document once.
    """
    if k <= 0:
        return 0.0
    gains = [0.0] * min(k, len(ranked))
    for position, key in _first_occurrences(ranked, k):
        gains[position] = float(judgement.get(key, 0))

    ideal = sorted((float(g) for g in judgement.values() if g > 0), reverse=True)[:k]
    best = dcg(ideal)
    if best == 0.0:
        return 0.0
    return dcg(gains) / best


# --- registry ---------------------------------------------------------------

# `Registry[float]` is exactly right here: `Registry[T].get()` returns
# `Callable[..., T]`, so a registry of floats is a registry of functions that
# return a metric value.
metric_registry: Registry[float] = Registry("metric")

# Registered under the names the experiment config uses. The ``@k`` suffix is
# added by the runner, which knows the sweep's k values.
metric_registry.register("precision_at_k", precision_at_k)
metric_registry.register("recall_at_k", recall_at_k)
metric_registry.register("hit_rate_at_k", hit_rate_at_k)
metric_registry.register("mrr", mrr_at_k)
metric_registry.register("ndcg_at_k", ndcg_at_k)

DEFAULT_METRICS: tuple[str, ...] = (
    "precision_at_k",
    "recall_at_k",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
)

# How each metric is labelled in results. `mrr` carries its @k too, for the
# reason in the module docstring.
METRIC_LABELS: dict[str, str] = {
    "precision_at_k": "precision@{k}",
    "recall_at_k": "recall@{k}",
    "hit_rate_at_k": "hit_rate@{k}",
    "mrr": "mrr@{k}",
    "ndcg_at_k": "ndcg@{k}",
}


def evaluate(
    ranked: Sequence[str],
    judgement: Judgement,
    k: int,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Compute ``metrics`` for one query, keyed by their ``@k`` labels."""
    out: dict[str, float] = {}
    for name in metrics:
        function = metric_registry.get(name)
        label = METRIC_LABELS.get(name, f"{name}@{{k}}").format(k=k)
        out[label] = float(function(ranked, judgement, k))
    return out


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile. ``fraction`` in ``[0, 1]``.

    Nearest-rank would report a p99 that is literally one of the observed
    samples, which for fewer than 100 queries is just the maximum. Interpolating
    at least degrades smoothly, and the query count is recorded alongside so a
    reader can judge whether a p99 over 30 queries means anything. (It does not.)
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(samples: Sequence[float]) -> dict[str, float]:
    """p50/p95/p99, mean, min and max over one stage's per-query latencies."""
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "p50": round(percentile(samples, 0.50), 3),
        "p95": round(percentile(samples, 0.95), 3),
        "p99": round(percentile(samples, 0.99), 3),
        "mean": round(sum(samples) / len(samples), 3),
        "min": round(min(samples), 3),
        "max": round(max(samples), 3),
    }
