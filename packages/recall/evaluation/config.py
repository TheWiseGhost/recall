"""Experiment configuration: what to sweep, over what.

An experiment file is a project ``recall.yaml`` with two additions — a dataset,
and lists where a normal config has scalars. Each list is a sweep dimension, so
``strategies: [bm25, dense, hybrid]`` with ``top_k: [5, 10, 20]`` is nine runs
over one index.

**Which dimensions can be swept is deliberately explicit**, not "any list is a
sweep". Some config values are legitimately lists — ``hybrid.components`` is one
— and inferring intent from shape would turn a typo into a silently different
experiment. Sweeping something that would require rebuilding the index is
refused with an explanation rather than ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import Field, model_validator

from recall.core.errors import ConfigurationError
from recall.core.evaluation.metrics import DEFAULT_METRICS, metric_registry
from recall.core.models import RecallModel

# Config sections that change how the corpus is *indexed*, not how it is
# queried. Sweeping them needs a re-ingest per point, which the runner does not
# do yet.
#
# TODO / FUTURE: chunking sweeps by re-ingesting per point; embedding sweeps
# without a re-ingest, since vectors are keyed by (chunk_id, model_key) and one
# corpus can already hold several models at once.
_INDEX_TIME_SWEEPS: dict[str, str] = {
    "chunking.strategies": "chunking strategy",
    "chunking.chunk_sizes": "chunk size",
    "embedding.models": "embedding model",
    "embedding.providers": "embedding provider",
}


class DatasetRef(RecallModel):
    path: str


class ExperimentConfig(RecallModel):
    """A parsed experiment definition."""

    name: str
    dataset: DatasetRef
    # Human-authored, carried into the report untouched. Recall generates the
    # quantitative sections only; stating a hypothesis and drawing a conclusion
    # are a person's job.
    hypothesis: str | None = None

    retrieval_strategies: list[str] = Field(default_factory=lambda: ["dense"])
    reranking_strategies: list[str] = Field(default_factory=lambda: ["off"])
    top_k: list[int] = Field(default_factory=lambda: [10])
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))

    # Passed through to Settings, overriding the project's recall.yaml.
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.retrieval_strategies:
            raise ValueError("retrieval.strategies must name at least one strategy")
        if not self.top_k or any(value < 1 for value in self.top_k):
            raise ValueError("top_k must be a list of positive integers")
        for name in self.metrics:
            if name not in metric_registry:
                raise ValueError(
                    f"unknown metric {name!r}. Available: {', '.join(metric_registry.names())}"
                )
        return self

    @property
    def run_count(self) -> int:
        return len(self.retrieval_strategies) * len(self.reranking_strategies) * len(self.top_k)


def _as_list(value: Any, field: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    # A scalar where a sweep is allowed is just a sweep of one.
    return [value]


def _normalize_reranker(value: Any, filename: str) -> str:
    """Repair YAML's treatment of ``off`` as a boolean.

    YAML 1.1 parses bare ``off``, ``no`` and ``false`` as ``False`` — so
    ``strategies: [off, cross_encoder]`` arrives as ``[False, "cross_encoder"]``
    and would sweep a reranker literally named "False". Since ``off`` is the
    natural way to write "no reranking at all", it is normalised here rather
    than renamed to something less readable.
    """
    if value is False:
        return "off"
    if value is True:
        raise ConfigurationError(
            f"{filename}: reranking.strategies contains `on`/`yes`/`true`, which YAML "
            "read as a boolean and which is not a reranker name. Use `off` to disable "
            "reranking, `none` for the identity reranker, or a registered strategy."
        )
    return str(value)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Parse an experiment YAML file."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ConfigurationError(f"Experiment config not found: {resolved}")

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"{resolved} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{resolved} must contain a YAML mapping at the top level")

    for dotted, description in _INDEX_TIME_SWEEPS.items():
        section, key = dotted.split(".")
        if isinstance(raw.get(section), dict) and key in raw[section]:
            raise ConfigurationError(
                f"{resolved.name} sweeps {dotted}, which is not supported yet: varying "
                f"the {description} changes how the corpus is indexed, so each point "
                "would need its own ingest. Run one experiment per "
                f"{description} for now, with `recall ingest --force` between them."
            )

    retrieval = raw.get("retrieval") or {}
    reranking = raw.get("reranking") or {}
    if not isinstance(retrieval, dict) or not isinstance(reranking, dict):
        raise ConfigurationError(f"{resolved.name}: 'retrieval' and 'reranking' must be mappings")

    strategies = _as_list(retrieval.get("strategies") or retrieval.get("default"), "retrieval")
    rerankers = [
        _normalize_reranker(value, resolved.name)
        for value in _as_list(reranking.get("strategies"), "reranking")
    ]
    if not rerankers:
        # No explicit sweep: honour `enabled` + `strategy` the way a normal
        # recall.yaml would, as a sweep of one.
        enabled = bool(reranking.get("enabled"))
        single = reranking.get("strategy")
        rerankers = [_normalize_reranker(single, resolved.name)] if enabled and single else ["off"]

    # Everything except the sweep axes is a plain settings override.
    overrides = {
        section: dict(value)
        for section, value in raw.items()
        if section in {"chunking", "embedding", "hybrid", "lexical", "database", "logging"}
        and isinstance(value, dict)
    }
    if isinstance(reranking, dict):
        passthrough = {
            key: value
            for key, value in reranking.items()
            if key in {"model", "top_n", "device", "batch_size", "max_length"}
        }
        if passthrough:
            overrides["reranking"] = passthrough

    dataset = raw.get("dataset")
    if not isinstance(dataset, dict) or not dataset.get("path"):
        raise ConfigurationError(f"{resolved.name}: 'dataset.path' is required")

    try:
        return ExperimentConfig(
            name=str(raw.get("name") or resolved.stem),
            dataset=DatasetRef(path=str(dataset["path"])),
            hypothesis=raw.get("hypothesis"),
            retrieval_strategies=[str(value) for value in strategies] or ["dense"],
            reranking_strategies=[str(value) for value in rerankers],
            top_k=[int(value) for value in _as_list(raw.get("top_k"), "top_k")] or [10],
            metrics=[str(value) for value in _as_list(raw.get("metrics"), "metrics")]
            or list(DEFAULT_METRICS),
            overrides=overrides,
        )
    except Exception as exc:
        raise ConfigurationError(f"{resolved.name}: {exc}") from exc
