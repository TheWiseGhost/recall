"""Hybrid retrieval: fan out to several retrievers, fuse their rankings.

The composition is deliberately generic. ``HybridRetriever`` knows nothing
about dense vectors or BM25 — it takes named :class:`Retriever` components and a
:class:`~recall.core.retrieval.fusion.Fusion`, so "dense + BM25" is just the
configuration that ships. Adding a third signal is a config change, and
"does a third signal help?" stays an answerable question rather than a rewrite.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from recall.core.errors import ConfigurationError
from recall.core.models import SearchFilters, SearchResult
from recall.core.retrieval.base import (
    Retriever,
    Timer,
    record_concurrent,
    rerank_positions,
    retriever_registry,
    stage,
)
from recall.core.retrieval.fusion import Fusion

# Fusion sees only what each component returned, so a chunk ranked first by one
# retriever but absent from another's truncated list contributes nothing from
# that other — it is scored as though the second retriever rejected it, when in
# fact it was never asked past position k. Over-fetching each component pushes
# that boundary out. 3x is cheap here because both components are one indexed
# query; it is not free on a remote reranking service.
DEFAULT_CANDIDATE_MULTIPLIER = 3


@retriever_registry.decorator("hybrid")
class HybridRetriever:
    """Runs components concurrently and fuses their rankings."""

    name = "hybrid"

    def __init__(
        self,
        *,
        components: Mapping[str, Retriever],
        fusion: Fusion,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        if not components:
            raise ConfigurationError("hybrid retrieval needs at least one component retriever")
        if candidate_multiplier < 1:
            raise ConfigurationError(
                f"candidate_multiplier must be at least 1, got {candidate_multiplier}"
            )
        self.components = dict(components)
        self.fusion = fusion
        self.candidate_multiplier = candidate_multiplier

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        candidate_k = top_k * self.candidate_multiplier

        # Concurrently: the components are independent queries, and a hybrid
        # search that cost the sum of its parts would lose the latency
        # comparison for reasons that have nothing to do with retrieval.
        timers: list[Timer] = []

        async def run(retriever: Retriever) -> list[SearchResult]:
            # Its own timer, so the components' stages do not interleave in a
            # shared one. They are merged with max afterwards, since they ran
            # in parallel.
            timer = Timer()
            timers.append(timer)
            with timer.activate():
                return await retriever.search(query, top_k=candidate_k, filters=filters)

        names = list(self.components)
        gathered = await asyncio.gather(*(run(self.components[name]) for name in names))
        record_concurrent(timers)

        with stage("fusion"):
            fused = self.fusion.fuse(dict(zip(names, gathered, strict=True)), top_k=top_k)

        # Component provenance stays on each result; `retriever` names what
        # produced the final ranking, which is what an experiment groups by.
        return rerank_positions(
            [result.model_copy(update={"retriever": self.name}) for result in fused]
        )
