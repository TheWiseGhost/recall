"""Reranker protocol and the identity reranker.

A reranker re-scores an already-retrieved candidate list. That ordering — cheap
recall-oriented retrieval first, expensive precision-oriented scoring second —
is the whole point: a cross-encoder that reads every query/document pair jointly
is far more accurate than a bi-encoder and far too slow to run over a corpus.

Reranking is therefore always a trade: latency for quality. Recall's job is to
make that trade measurable rather than assumed, which is why
:attr:`RetrievalTiming.reranking_ms` is a first-class field and why the
identity reranker below exists as a real, selectable component.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from recall.core.models import SearchResult
from recall.core.registry import Registry
from recall.core.retrieval.base import rerank_positions


@runtime_checkable
class Reranker(Protocol):
    """Re-scores and reorders retrieved candidates."""

    name: str

    async def rerank(
        self, query: str, results: Sequence[SearchResult], *, top_k: int
    ) -> list[SearchResult]: ...


reranker_registry: Registry[Reranker] = Registry("reranker")


def preserve_retrieval_score(result: SearchResult, score: float) -> SearchResult:
    """Replace ``score`` with a reranker's score, keeping the original.

    The retrieval score has to survive: a report that cannot compare the
    pre- and post-rerank orderings cannot say how much the reranker actually
    changed, only that it ran. ``retrieval_score`` is set once and never
    overwritten, so it always means "what retrieval thought".
    """
    update: dict[str, object] = {"score": score}
    if result.retrieval_score is None:
        update["retrieval_score"] = result.score
    return result.model_copy(update=update)


@reranker_registry.decorator("none")
class NoOpReranker:
    """Truncates to ``top_k`` without reordering.

    Not dead weight — it is the control. With ``reranking.enabled`` set and
    ``strategy: none``, retrieval still widens its candidate pool to ``top_n``
    and the pool is then truncated, so an experiment can separate "did the
    wider candidate pool help?" from "did the cross-encoder help?". Attributing
    the first effect to the second is an easy and expensive mistake.
    """

    name = "none"

    async def rerank(
        self, query: str, results: Sequence[SearchResult], *, top_k: int
    ) -> list[SearchResult]:
        return rerank_positions(list(results[:top_k]))
