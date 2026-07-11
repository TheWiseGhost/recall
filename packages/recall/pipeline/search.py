"""Search use case: retrieval, optional reranking, and timing instrumentation."""

from __future__ import annotations

from recall.core.models import RecallModel, SearchFilters, SearchResult
from recall.core.reranking.base import Reranker
from recall.core.retrieval.base import RetrievalTiming, Retriever, Timer, rerank_positions, stage
from recall.observability.logging import bind_request_id, get_logger

_log = get_logger(__name__)


class SearchResponse(RecallModel):
    """A search plus everything an experiment needs to interpret it."""

    query: str
    results: list[SearchResult]
    timing: RetrievalTiming
    retrieval_strategy: str
    top_k: int
    request_id: str
    reranked: bool = False
    reranking_strategy: str | None = None
    # How many candidates retrieval was asked for. Larger than top_k when
    # reranking is on, and part of what makes a run reproducible.
    candidates: int = 0


class SearchService:
    """Runs a query through retrieval and, optionally, a reranker.

    The two stages are composed here rather than inside a retriever because
    reranking changes what retrieval is asked for: the candidate pool has to
    widen to ``rerank_candidates`` for the reranker to have anything to
    reorder. Burying that in a retriever would make ``top_k`` mean different
    things depending on configuration, which is exactly the kind of silent
    change that corrupts a metric.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker | None = None,
        rerank_candidates: int = 50,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_candidates = max(1, rerank_candidates)

    async def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: SearchFilters | None = None,
        request_id: str | None = None,
    ) -> SearchResponse:
        rid = bind_request_id(request_id)
        strategy = getattr(self.retriever, "name", "unknown")
        timer = Timer()
        # Never fewer than top_k: a reranker cannot fill a gap retrieval left.
        candidates = max(top_k, self.rerank_candidates) if self.reranker else top_k

        # The retriever records its own stages (embedding vs. index lookup)
        # against this timer via the ambient-timer context variable.
        with timer.activate():
            results = await self.retriever.search(query, top_k=candidates, filters=filters)
            if self.reranker is not None:
                with stage("reranking"):
                    results = await self.reranker.rerank(query, results, top_k=top_k)
            else:
                results = rerank_positions(results[:top_k])

        timing = timer.to_timing()
        _log.info(
            "search_completed",
            operation="search",
            retrieval_strategy=strategy,
            reranking_strategy=getattr(self.reranker, "name", None),
            top_k=top_k,
            candidates=candidates,
            results=len(results),
            duration_ms=timing.total_ms,
            filtered=bool(filters and not filters.is_empty()),
            status="ok",
        )
        return SearchResponse(
            query=query,
            results=results,
            timing=timing,
            retrieval_strategy=strategy,
            top_k=top_k,
            request_id=rid,
            reranked=self.reranker is not None,
            reranking_strategy=getattr(self.reranker, "name", None),
            candidates=candidates,
        )
