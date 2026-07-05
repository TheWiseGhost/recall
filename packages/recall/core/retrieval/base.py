"""Retriever protocol and per-stage timing."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol, runtime_checkable

from recall.core.models import RecallModel, SearchFilters, SearchResult
from recall.core.registry import Registry


class RetrievalTiming(RecallModel):
    """Wall-clock breakdown of a single search, in milliseconds.

    Timing is a first-class result, not a logging afterthought: latency is one
    of the metrics experiments compare.
    """

    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    # Rank fusion, for hybrid retrieval. Separate from retrieval_ms so "is
    # hybrid worth its latency?" can distinguish the extra index query from the
    # cost of combining the results.
    fusion_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class Timer:
    """Accumulates named stage durations in milliseconds."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            self.stages[name] = self.stages.get(name, 0.0) + elapsed

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    @contextmanager
    def activate(self) -> Iterator[Timer]:
        """Make this the ambient timer for the current task.

        Uses a context variable, so concurrent searches each keep their own
        timings without any of them being threaded through call signatures.
        """
        token = _current_timer.set(self)
        try:
            yield self
        finally:
            _current_timer.reset(token)

    def merge_concurrent(self, others: Iterable[Timer]) -> None:
        """Fold in stage timings from work that ran in parallel with each other.

        Combined with ``max``, not ``sum``. Two 20 ms index lookups that
        overlapped cost 20 ms of wall clock; summing them would report 40 ms
        and exceed ``total_ms``, which is measured directly and would then
        contradict its own breakdown.
        """
        for other in others:
            for name, elapsed in other.stages.items():
                self.stages[name] = max(self.stages.get(name, 0.0), elapsed)

    def to_timing(self) -> RetrievalTiming:
        total = round(self.total_ms, 3)
        retrieval = self.stages.get("retrieval")
        if retrieval is None and not self.stages:
            # A retriever that declared no stages: attribute everything to it.
            retrieval = self.total_ms
        return RetrievalTiming(
            embedding_ms=round(self.stages.get("embedding", 0.0), 3),
            retrieval_ms=round(retrieval or 0.0, 3),
            fusion_ms=round(self.stages.get("fusion", 0.0), 3),
            reranking_ms=round(self.stages.get("reranking", 0.0), 3),
            generation_ms=round(self.stages.get("generation", 0.0), 3),
            total_ms=total,
        )


_current_timer: ContextVar[Timer | None] = ContextVar("recall_current_timer", default=None)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Record a named stage against the ambient timer, if one is active.

    A no-op when nothing is timing, so components can be instrumented
    unconditionally and still be usable standalone.
    """
    timer = _current_timer.get()
    if timer is None:
        yield
        return
    with timer.stage(name):
        yield


def record_concurrent(timers: Iterable[Timer]) -> None:
    """Merge timers from parallel sub-searches into the ambient timer.

    A no-op when nothing is timing, like :func:`stage`.
    """
    ambient = _current_timer.get()
    if ambient is not None:
        ambient.merge_concurrent(timers)


@runtime_checkable
class Retriever(Protocol):
    """Returns ranked chunks for a query."""

    name: str

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]: ...


def rerank_positions(results: list[SearchResult]) -> list[SearchResult]:
    """Re-stamp 1-based ranks after any reordering."""
    return [result.with_rank(index) for index, result in enumerate(results, start=1)]


retriever_registry: Registry[Retriever] = Registry("retriever")
