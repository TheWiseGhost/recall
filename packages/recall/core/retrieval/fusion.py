"""Rank fusion: combining several ranked lists into one.

Two strategies, and the choice between them is itself an experimental variable.

**Reciprocal rank fusion** (``rrf``) throws the scores away and uses only
positions::

    score(d) = Σ  w_r / (k + rank_r(d))
               r

Discarding the scores is the point, not a shortcut. BM25 scores are unbounded
sums over query terms; cosine similarities live in ``[-1, 1]``. The two are not
on a common scale, and no fixed rescaling makes them comparable across queries —
a BM25 score of 8 means something different for a one-word query than for a
ten-word one. Ranks are the only signal both lists genuinely share, which is why
RRF is the default and why it is hard to beat.

**Weighted fusion** (``weighted``) keeps the scores and min-max normalises each
list before combining. It can express "this retriever was *much* more confident"
where RRF sees only "one position better", and it is the right choice when the
components' score distributions are known and stable. Its cost is stated
plainly in :class:`WeightedScoreFusion` — per-query min-max normalisation is a
lossy transform.

Both weight components by name, so ``dense_weight`` and ``lexical_weight``
apply to either strategy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from recall.core.errors import ConfigurationError
from recall.core.models import SearchResult
from recall.core.registry import Registry
from recall.core.retrieval.base import rerank_positions

# Cormack, Clarke & Buettcher (2009) found k=60 robust across collections. It
# damps the top of the curve: without it, rank 1 would be worth twice rank 2.
DEFAULT_RRF_K = 60

RankedLists = Mapping[str, Sequence[SearchResult]]


@runtime_checkable
class Fusion(Protocol):
    """Combines named ranked lists into a single ranking."""

    name: str

    def fuse(self, lists: RankedLists, *, top_k: int) -> list[SearchResult]: ...


def _normalized_weights(
    weights: Mapping[str, float] | None, names: Sequence[str]
) -> dict[str, float]:
    """Weights for ``names``, scaled to sum to 1.

    Scaling makes fused scores comparable between configurations: without it,
    doubling every weight would double every score and make two experiment runs
    look different when only the parameterisation changed.
    """
    resolved = {name: float((weights or {}).get(name, 1.0)) for name in names}
    if any(value < 0 for value in resolved.values()):
        raise ConfigurationError(f"fusion weights must be non-negative, got {resolved}")
    total = sum(resolved.values())
    if total <= 0:
        raise ConfigurationError("fusion weights must not all be zero")
    return {name: value / total for name, value in resolved.items()}


def _merge(
    lists: RankedLists,
    scores: Mapping[str, float],
    *,
    top_k: int,
) -> list[SearchResult]:
    """Materialise fused results, richest representation of each chunk wins.

    Every component's score and rank for a chunk is recorded on the result, so
    an experiment can ask which retriever actually found a hit — the question
    "is hybrid worth it?" is usually really "what does each side contribute?".
    """
    representative: dict[str, SearchResult] = {}
    component_scores: dict[str, dict[str, float]] = {}
    component_ranks: dict[str, dict[str, int]] = {}

    for retriever, results in lists.items():
        for result in results:
            key = str(result.chunk_id)
            representative.setdefault(key, result)
            component_scores.setdefault(key, {})[retriever] = result.score
            component_ranks.setdefault(key, {})[retriever] = result.rank

    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    fused = [
        representative[key].model_copy(
            update={
                "score": round(scores[key], 8),
                "component_scores": component_scores[key],
                "component_ranks": component_ranks[key],
            }
        )
        for key in ordered[:top_k]
    ]
    return rerank_positions(fused)


fusion_registry: Registry[Fusion] = Registry("fusion strategy")


@fusion_registry.decorator("rrf")
class ReciprocalRankFusion:
    """Rank-only fusion. Scale-free, and the sane default.

    Because it never looks at a score, it is immune to one component's scores
    drifting — a re-embedded corpus, a different BM25 ``b`` — in a way weighted
    fusion is not.
    """

    name = "rrf"

    def __init__(
        self, *, k: int = DEFAULT_RRF_K, weights: Mapping[str, float] | None = None
    ) -> None:
        if k < 1:
            raise ConfigurationError(f"rrf_k must be at least 1, got {k}")
        self.k = k
        self.weights = dict(weights or {})

    def fuse(self, lists: RankedLists, *, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []
        weights = _normalized_weights(self.weights, list(lists))
        scores: dict[str, float] = {}
        for retriever, results in lists.items():
            weight = weights[retriever]
            for result in results:
                # The result's own rank, not its index: a retriever is free to
                # return a list whose ranks are not 1..n.
                scores[str(result.chunk_id)] = scores.get(str(result.chunk_id), 0.0) + weight / (
                    self.k + result.rank
                )
        return _merge(lists, scores, top_k=top_k)


@fusion_registry.decorator("weighted")
class WeightedScoreFusion:
    """Score-based fusion over per-list min-max normalised scores.

    Normalisation is unavoidable — summing a cosine similarity and a BM25 score
    directly would let BM25's unbounded scale swamp the weights — but min-max is
    lossy in a way worth knowing about:

    * It is computed per query over the returned candidates, so the top result
      of every list is always 1.0 and the bottom always 0.0. A query where one
      retriever found nothing good still has a 1.0 in it.
    * It therefore erases absolute confidence. Two queries' fused scores are not
      comparable, and neither are two runs over different candidate depths.

    Ranking within one query is unaffected, which is what the retrieval metrics
    measure. Prefer ``rrf`` unless you specifically want score magnitudes to
    carry weight.
    """

    name = "weighted"

    def __init__(self, *, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(weights or {})

    @staticmethod
    def _normalize(results: Sequence[SearchResult]) -> dict[str, float]:
        if not results:
            return {}
        scores = [result.score for result in results]
        low, high = min(scores), max(scores)
        span = high - low
        if span <= 0:
            # Every candidate scored the same; nothing to discriminate on.
            return {str(result.chunk_id): 1.0 for result in results}
        return {str(result.chunk_id): (result.score - low) / span for result in results}

    def fuse(self, lists: RankedLists, *, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []
        weights = _normalized_weights(self.weights, list(lists))
        scores: dict[str, float] = {}
        for retriever, results in lists.items():
            weight = weights[retriever]
            for key, value in self._normalize(results).items():
                # A chunk absent from a list contributes 0 from it — see the
                # note on truncation bias in HybridRetriever.
                scores[key] = scores.get(key, 0.0) + weight * value
        return _merge(lists, scores, top_k=top_k)


def create_fusion(strategy: str, **kwargs: object) -> Fusion:
    """Instantiate the fusion strategy registered under ``strategy``."""
    return fusion_registry.create(strategy, **kwargs)
