"""Domain models for evaluation.

These are the shapes that get written to ``results.json``, so they are a
contract: an experiment run months apart must produce comparable files.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from recall.core.models import RecallModel
from recall.core.retrieval.base import RetrievalTiming


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Granularity(StrEnum):
    """What a dataset's labels refer to."""

    DOCUMENT = "document"
    CHUNK = "chunk"


class EvaluationQuery(RecallModel):
    """One labelled query.

    ``relevant`` maps a label key to a relevance grade. Binary datasets are
    loaded as grade 1, so there is one code path rather than two.
    """

    query: str
    relevant: dict[str, int] = Field(default_factory=dict)
    granularity: Granularity = Granularity.DOCUMENT
    query_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_graded(self) -> bool:
        """True when the labels carry more than a yes/no."""
        return any(grade > 1 for grade in self.relevant.values())


class Dataset(RecallModel):
    """A labelled query set plus the provenance a reported number needs."""

    name: str
    path: str
    queries: list[EvaluationQuery]
    # From the sidecar `*.meta.json`. `kind` is "synthetic" or "curated" and is
    # reproduced in every report generated from the dataset.
    kind: str = "unknown"
    documents: int | None = None
    label_method: str | None = None
    version: str | None = None
    warning: str | None = None
    checksum: str = ""

    @property
    def is_graded(self) -> bool:
        return any(query.is_graded for query in self.queries)


class QueryOutcome(RecallModel):
    """What one query did, in one run. The unit ``results.json`` is a list of."""

    query: str
    query_id: str | None = None
    retrieved: list[str] = Field(default_factory=list)
    relevant: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    timing: RetrievalTiming = Field(default_factory=RetrievalTiming)
    # Chunk IDs, so a surprising number can be traced back to actual text.
    chunk_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class CostEstimate(RecallModel):
    """Estimated spend for a run.

    ``usd`` is ``None`` when the model carries no price — deliberately not
    ``0.0``, which would report a paid model as free. ``note`` says which case
    applies so a report never has to guess.
    """

    usd: float | None = None
    embedded_tokens: int = 0
    model: str = ""
    cost_per_million_tokens: float | None = None
    note: str = ""


class RunResult(RecallModel):
    """One point in the sweep: one configuration, evaluated over the dataset."""

    run_id: str
    # The swept variables that produced this point.
    parameters: dict[str, Any] = Field(default_factory=dict)
    outcomes: list[QueryOutcome] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, dict[str, float]] = Field(default_factory=dict)
    cost: CostEstimate = Field(default_factory=CostEstimate)
    queries: int = 0
    failed_queries: int = 0


class ExperimentResult(RecallModel):
    """Everything needed to re-run and to interpret an experiment.

    Provenance is not decoration. A metric without the git commit, the dataset
    version and the model versions that produced it is not reproducible, and an
    irreproducible number is not evidence.
    """

    experiment_id: str
    name: str
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    git_commit: str | None = None
    git_dirty: bool = False
    recall_version: str = ""
    dataset: Dataset | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    corpus: dict[str, int] = Field(default_factory=dict)
    runs: list[RunResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()
