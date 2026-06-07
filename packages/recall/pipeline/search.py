"""Search use case: retriever + timing instrumentation."""

from __future__ import annotations

from recall.core.models import RecallModel, SearchFilters, SearchResult
from recall.core.retrieval.base import RetrievalTiming, Retriever, Timer
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


class SearchService:
    """Runs a query through a retriever and records per-stage latency.

    Reranking and context selection slot in here in Milestone 2; the timing
    fields for them already exist so result files stay schema-compatible.
    """

    def __init__(self, *, retriever: Retriever) -> None:
        self.retriever = retriever

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

        # The retriever records its own stages (embedding vs. index lookup)
        # against this timer via the ambient-timer context variable.
        with timer.activate():
            results = await self.retriever.search(query, top_k=top_k, filters=filters)

        timing = timer.to_timing()
        _log.info(
            "search_completed",
            operation="search",
            retrieval_strategy=strategy,
            top_k=top_k,
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
        )
