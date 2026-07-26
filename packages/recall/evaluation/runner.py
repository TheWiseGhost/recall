"""The experiment runner.

Takes an experiment config, runs every point in the sweep against one index,
and writes a result directory that can be re-read months later:

.. code-block:: text

    experiments/results/2026-08-03-hybrid-search/
      config.yaml    the resolved configuration, not the file as written
      results.json   per-query results for every run
      metrics.csv    aggregated metrics, one row per run
      report.md      the generated report

``config.yaml`` holds the *resolved* configuration on purpose. The file as
written may reference ``${VAR}``, may omit fields that took defaults, and may
sit next to a ``recall.yaml`` that has since changed. What is written is what
actually ran.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from recall import __version__
from recall.config.settings import Settings
from recall.core.errors import ConfigurationError, RecallError
from recall.core.evaluation.cost import estimate_embedding_cost
from recall.core.evaluation.metrics import evaluate, latency_summary
from recall.core.evaluation.models import (
    Dataset,
    ExperimentResult,
    QueryOutcome,
    RunResult,
)
from recall.core.tokenization import DEFAULT_TOKEN_COUNTER
from recall.evaluation.config import ExperimentConfig
from recall.evaluation.datasets import load_dataset
from recall.evaluation.labels import build_resolver
from recall.evaluation.report import render_report
from recall.observability.logging import get_logger
from recall.pipeline.factory import build_embedder, build_reranker, build_retriever
from recall.pipeline.search import SearchService
from recall.storage.postgres.storage import create_storage

_log = get_logger(__name__)


class ExperimentError(RecallError):
    """An experiment could not be run as configured."""


def git_provenance(root: Path | None = None) -> tuple[str | None, bool]:
    """``(commit, dirty)`` for the working tree, or ``(None, False)``.

    A metric without the commit that produced it cannot be reproduced, and the
    dirty flag matters just as much: a number from an uncommitted tree is not
    attributable to any commit at all, and the report says so.
    """
    directory = str(root or Path.cwd())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=directory,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=directory,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, False
    return commit or None, bool(status)


def merge_settings(base: Settings, overrides: dict[str, Any]) -> Settings:
    """Layer an experiment's overrides over the project settings."""
    data = base.model_dump(mode="python")
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(data.get(section), dict):
            data[section] = {**data[section], **values}
        else:
            data[section] = values
    try:
        return Settings.model_validate(data)
    except Exception as exc:
        raise ConfigurationError(f"experiment overrides are invalid: {exc}") from exc


def result_directory(root: Path, name: str, when: datetime) -> Path:
    """``<root>/results/<date>-<name>/``, suffixed if it already exists."""
    base = root / "results" / f"{when.date().isoformat()}-{name}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


class ExperimentRunner:
    """Runs one experiment config and writes its result directory."""

    def __init__(self, *, settings: Settings, config: ExperimentConfig) -> None:
        self.base_settings = settings
        self.config = config
        self.settings = merge_settings(settings, config.overrides)

    async def run(self, *, output_root: Path | None = None) -> tuple[ExperimentResult, Path]:
        started = datetime.now(UTC)
        dataset = load_dataset(self.config.dataset.path)
        commit, dirty = git_provenance()

        storage = create_storage(self.settings.database, lexical=self.settings.lexical)
        try:
            corpus = await self._corpus_snapshot(storage)
            documents = await self._document_labels(storage)
            resolver = build_resolver(dataset, documents)

            embedder = build_embedder(self.settings)
            result = ExperimentResult(
                experiment_id=f"{started.date().isoformat()}-{self.config.name}",
                name=self.config.name,
                started_at=started,
                git_commit=commit,
                git_dirty=dirty,
                recall_version=__version__,
                dataset=dataset,
                config=self.settings.model_dump(mode="json"),
                models={
                    "embedding": embedder.info.key,
                    "reranking": self.settings.reranking.model
                    if self.settings.reranking.enabled
                    else "",
                },
                corpus=corpus,
                notes=self._notes(dataset, dirty),
            )

            for strategy in self.config.retrieval_strategies:
                for reranker_name in self.config.reranking_strategies:
                    for k in self.config.top_k:
                        result.runs.append(
                            await self._run_point(
                                storage=storage,
                                embedder=embedder,
                                dataset=dataset,
                                resolver=resolver,
                                strategy=strategy,
                                reranker_name=reranker_name,
                                k=k,
                            )
                        )
        finally:
            await storage.close()

        result.finished_at = datetime.now(UTC)
        directory = self._write(result, output_root)
        return result, directory

    # -- one point in the sweep -------------------------------------------

    async def _run_point(
        self,
        *,
        storage: Any,
        embedder: Any,
        dataset: Dataset,
        resolver: Any,
        strategy: str,
        reranker_name: str,
        k: int,
    ) -> RunResult:
        settings = self._settings_for(reranker_name)
        retriever = build_retriever(strategy, storage=storage, embedder=embedder, settings=settings)
        service = SearchService(
            retriever=retriever,
            reranker=build_reranker(settings),
            rerank_candidates=settings.reranking.top_n,
        )

        run_id = f"{strategy}__rerank-{reranker_name}__k{k}"
        outcomes: list[QueryOutcome] = []
        tokens = 0

        for query in dataset.queries:
            tokens += DEFAULT_TOKEN_COUNTER.count(query.query)
            try:
                response = await service.search(query.query, top_k=k)
            except Exception as exc:  # one bad query must not lose the run
                _log.warning(
                    "experiment_query_failed", run=run_id, query=query.query, error=str(exc)
                )
                outcomes.append(
                    QueryOutcome(
                        query=query.query,
                        query_id=query.query_id,
                        relevant=dict(query.relevant),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            ranked = resolver.ranked_keys(response.results)
            outcomes.append(
                QueryOutcome(
                    query=query.query,
                    query_id=query.query_id,
                    retrieved=ranked,
                    relevant=dict(query.relevant),
                    metrics=evaluate(ranked, query.relevant, k, self.config.metrics),
                    timing=response.timing,
                    chunk_ids=[str(r.chunk_id) for r in response.results],
                )
            )

        return RunResult(
            run_id=run_id,
            parameters={
                "retrieval_strategy": strategy,
                "reranking_strategy": reranker_name,
                "top_k": k,
            },
            outcomes=outcomes,
            metrics=self._aggregate_metrics(outcomes),
            latency=self._aggregate_latency(outcomes),
            cost=estimate_embedding_cost(embedder.info, tokens),
            queries=len(outcomes),
            failed_queries=sum(1 for outcome in outcomes if outcome.error),
        )

    def _settings_for(self, reranker_name: str) -> Settings:
        """Settings with reranking switched to this point in the sweep."""
        if reranker_name == "off":
            reranking = self.settings.reranking.model_copy(update={"enabled": False})
        else:
            reranking = self.settings.reranking.model_copy(
                update={"enabled": True, "strategy": reranker_name}
            )
        return self.settings.model_copy(update={"reranking": reranking})

    # -- aggregation -------------------------------------------------------

    @staticmethod
    def _aggregate_metrics(outcomes: Sequence[QueryOutcome]) -> dict[str, float]:
        """Mean of each metric over the queries that produced one.

        Failed queries are excluded rather than scored zero. A crashed query
        measures the harness, not the retriever, and folding it in as a zero
        would understate quality for a reason unrelated to retrieval — the
        failure count is reported separately so it cannot hide.
        """
        scored = [outcome for outcome in outcomes if outcome.error is None]
        if not scored:
            return {}
        names = sorted({name for outcome in scored for name in outcome.metrics})
        return {
            name: round(sum(outcome.metrics.get(name, 0.0) for outcome in scored) / len(scored), 6)
            for name in names
        }

    @staticmethod
    def _aggregate_latency(outcomes: Sequence[QueryOutcome]) -> dict[str, dict[str, float]]:
        scored = [outcome for outcome in outcomes if outcome.error is None]
        if not scored:
            return {}
        stages = ("total_ms", "embedding_ms", "retrieval_ms", "fusion_ms", "reranking_ms")
        summaries: dict[str, dict[str, float]] = {}
        for stage in stages:
            samples = [float(getattr(outcome.timing, stage)) for outcome in scored]
            if any(sample > 0 for sample in samples) or stage == "total_ms":
                summaries[stage] = latency_summary(samples)
        return summaries

    # -- provenance and output --------------------------------------------

    async def _corpus_snapshot(self, storage: Any) -> dict[str, int]:
        health = await storage.health()
        snapshot = {
            "documents": int(health.get("documents", 0)),
            "chunks": int(health.get("chunks", 0)),
            "vectors": int(health.get("vectors", 0)),
        }
        if snapshot["chunks"] == 0:
            raise ExperimentError(
                "The index is empty. Every metric would be zero and that would look "
                "like a result. Run `recall ingest <corpus>` first."
            )
        return snapshot

    async def _document_labels(self, storage: Any) -> list[tuple[uuid.UUID, str]]:
        total = await storage.documents.count()
        documents = await storage.documents.list(limit=max(total, 1))
        return [(document.id, document.source_id) for document in documents]

    def _notes(self, dataset: Dataset, dirty: bool) -> list[str]:
        """Caveats that must travel with the numbers."""
        notes: list[str] = []
        if dataset.kind != "curated":
            notes.append(
                f"Dataset '{dataset.name}' is labelled {dataset.kind}. "
                "Results from it describe this dataset, not retrieval in general."
            )
        if dataset.warning:
            notes.append(dataset.warning)
        if len(dataset.queries) < 50:
            notes.append(
                f"Only {len(dataset.queries)} queries. Differences between runs of "
                "this size are usually noise; tail latency percentiles are not "
                "meaningful at all."
            )
        if self.settings.embedding.provider == "hash":
            notes.append(
                "The `hash` embedder is a hashing-trick baseline, not a semantic "
                "model. These numbers must not be used to make a quality claim."
            )
        if dirty:
            notes.append(
                "The working tree had uncommitted changes, so this run is not "
                "attributable to its recorded commit."
            )
        if not dataset.is_graded:
            notes.append(
                "Labels are binary, so NDCG reduces to a rank-discounted hit "
                "measure and cannot distinguish degrees of relevance."
            )
        return notes

    def _write(self, result: ExperimentResult, output_root: Path | None) -> Path:
        root = output_root or self.settings.experiments_dir
        directory = result_directory(Path(root), self.config.name, result.started_at)
        directory.mkdir(parents=True, exist_ok=True)

        (directory / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "experiment_id": result.experiment_id,
                    "name": result.name,
                    "hypothesis": self.config.hypothesis,
                    "dataset": {"path": self.config.dataset.path},
                    "retrieval": {"strategies": self.config.retrieval_strategies},
                    "reranking": {"strategies": self.config.reranking_strategies},
                    "top_k": self.config.top_k,
                    "metrics": self.config.metrics,
                    "resolved_settings": result.config,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        (directory / "results.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (directory / "metrics.csv").write_text(metrics_csv(result), encoding="utf-8")
        (directory / "report.md").write_text(
            render_report(result, hypothesis=self.config.hypothesis), encoding="utf-8"
        )
        return directory


def metrics_csv(result: ExperimentResult) -> str:
    """One row per run: parameters, metrics, latency and cost.

    Written by hand rather than through ``csv``: the column set is derived from
    the runs, and quoting is trivial because every value is a number or a name
    without commas.
    """
    if not result.runs:
        return "run_id\n"

    metric_names = sorted({name for run in result.runs for name in run.metrics})
    latency_columns = [
        f"{stage}_{statistic}"
        for stage in sorted({stage for run in result.runs for stage in run.latency})
        for statistic in ("p50", "p95", "p99")
    ]
    header = [
        "experiment_id",
        "run_id",
        "retrieval_strategy",
        "reranking_strategy",
        "top_k",
        "queries",
        "failed_queries",
        *metric_names,
        *latency_columns,
        "estimated_cost_usd",
        "embedded_tokens",
    ]

    lines = [",".join(header)]
    for run in result.runs:
        row: list[str] = [
            result.experiment_id,
            run.run_id,
            str(run.parameters.get("retrieval_strategy", "")),
            str(run.parameters.get("reranking_strategy", "")),
            str(run.parameters.get("top_k", "")),
            str(run.queries),
            str(run.failed_queries),
        ]
        row += [f"{run.metrics[name]:.6f}" if name in run.metrics else "" for name in metric_names]
        for column in latency_columns:
            stage, statistic = column.rsplit("_", 1)
            value = run.latency.get(stage, {}).get(statistic)
            row.append("" if value is None else f"{value:.3f}")
        # Empty, not 0 — an unpriced model has no cost, it does not have a zero.
        row.append("" if run.cost.usd is None else f"{run.cost.usd:.8f}")
        row.append(str(run.cost.embedded_tokens))
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"
