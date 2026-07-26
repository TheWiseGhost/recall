"""Evaluation: metrics, cost estimation, and the shapes a result file holds.

Pure domain. Nothing here reads a file, opens a connection, or knows what a
retriever is — which is what makes the metric arithmetic checkable in isolation,
and what makes ``results.json`` a stable contract rather than a dump of
whatever the runner happened to have in scope.
"""

from recall.core.evaluation.cost import estimate_embedding_cost
from recall.core.evaluation.metrics import (
    DEFAULT_METRICS,
    METRIC_LABELS,
    Judgement,
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
from recall.core.evaluation.models import (
    CostEstimate,
    Dataset,
    EvaluationQuery,
    ExperimentResult,
    Granularity,
    QueryOutcome,
    RunResult,
)

__all__ = [
    "DEFAULT_METRICS",
    "METRIC_LABELS",
    "CostEstimate",
    "Dataset",
    "EvaluationQuery",
    "ExperimentResult",
    "Granularity",
    "Judgement",
    "QueryOutcome",
    "RunResult",
    "estimate_embedding_cost",
    "evaluate",
    "hit_rate_at_k",
    "latency_summary",
    "metric_registry",
    "mrr_at_k",
    "ndcg_at_k",
    "percentile",
    "precision_at_k",
    "recall_at_k",
]
