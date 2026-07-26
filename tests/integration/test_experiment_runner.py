"""The experiment runner, end to end against a live database.

Uses the `hash` embedder, so nothing here is a quality claim — what is being
tested is that the harness produces correct, complete, reproducible artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_asyncio
import yaml

from recall.config.settings import Settings
from recall.core.chunking.fixed import FixedSizeChunker
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.models import Document, SourceType
from recall.evaluation.benchmark import compare
from recall.evaluation.config import load_experiment_config
from recall.evaluation.labels import LabelResolutionError
from recall.evaluation.runner import ExperimentError, ExperimentRunner
from recall.pipeline.ingest import IngestionPipeline
from recall.storage.postgres.storage import Storage

pytestmark = pytest.mark.integration

CORPUS: list[tuple[str, str, str]] = [
    (
        "authentication.md",
        "Authentication",
        "Authentication uses a bearer token. The auth service verifies the token signature "
        "against its public key before any request is served.",
    ),
    (
        "deployment.md",
        "Deployment",
        "Rolling deployment replaces replicas one at a time. A failed readiness probe halts "
        "the rollout and the previous release stays live.",
    ),
    (
        "observability.md",
        "Observability",
        "Prometheus scrapes metrics every fifteen seconds. Every request is logged with a "
        "request identifier that survives across services.",
    ),
]

QUERIES = [
    {"query": "How are bearer tokens verified?", "relevant_documents": ["authentication.md"]},
    {"query": "What halts a rolling deployment?", "relevant_documents": ["deployment.md"]},
    {"query": "How often does Prometheus scrape?", "relevant_documents": ["observability.md"]},
]


@pytest_asyncio.fixture
async def corpus(storage: Storage, embedder: HashingEmbedder) -> Storage:
    pipeline = IngestionPipeline(
        storage=storage, chunker=FixedSizeChunker(chunk_size=64, overlap=8), embedder=embedder
    )
    for source_id, title, content in CORPUS:
        await pipeline.index_document(
            Document.create(
                source_id=source_id,
                source_type=SourceType.FILESYSTEM,
                title=title,
                content=content,
                uri=f"file:///{source_id}",
            )
        )
    return storage


@pytest.fixture
def experiment(tmp_path: Path, settings: Settings) -> tuple[Settings, Path]:
    """A dataset, its metadata, and an experiment config, all under tmp_path."""
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text("\n".join(json.dumps(query) for query in QUERIES) + "\n", encoding="utf-8")
    (tmp_path / "tiny.meta.json").write_text(
        json.dumps(
            {
                "name": "tiny",
                "kind": "synthetic",
                "documents": len(CORPUS),
                "label_method": "written for the integration suite",
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "exp.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "harness-check",
                "hypothesis": "Hybrid should not be worse than either component.",
                "dataset": {"path": str(dataset)},
                "retrieval": {"strategies": ["bm25", "dense", "hybrid"]},
                "top_k": [3],
            }
        ),
        encoding="utf-8",
    )
    return settings, config


async def run_experiment(
    settings: Settings, config_path: Path, output: Path
) -> tuple[object, Path]:
    runner = ExperimentRunner(settings=settings, config=load_experiment_config(config_path))
    return await runner.run(output_root=output)


class TestArtefacts:
    async def test_writes_all_four_files(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        _, directory = await run_experiment(settings, config_path, tmp_path / "out")

        for filename in ("config.yaml", "results.json", "metrics.csv", "report.md"):
            assert (directory / filename).is_file(), filename

    async def test_directory_is_named_by_date_and_experiment(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        result, directory = await run_experiment(settings, config_path, tmp_path / "out")
        assert directory.name.endswith("-harness-check")
        assert directory.parent.name == "results"
        assert result.experiment_id.endswith("-harness-check")  # type: ignore[attr-defined]

    async def test_results_json_round_trips(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        from recall.core.evaluation.models import ExperimentResult

        settings, config_path = experiment
        _, directory = await run_experiment(settings, config_path, tmp_path / "out")
        reloaded = ExperimentResult.model_validate_json(
            (directory / "results.json").read_text(encoding="utf-8")
        )
        assert len(reloaded.runs) == 3
        assert reloaded.dataset is not None
        assert reloaded.dataset.checksum

    async def test_metrics_csv_has_a_row_per_run(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        _, directory = await run_experiment(settings, config_path, tmp_path / "out")
        rows = (directory / "metrics.csv").read_text(encoding="utf-8").strip().split("\n")
        assert len(rows) == 4  # header + 3 runs
        assert "precision@3" in rows[0]

    async def test_config_yaml_records_the_resolved_settings(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        """What ran, not what was written — defaults and all."""
        settings, config_path = experiment
        _, directory = await run_experiment(settings, config_path, tmp_path / "out")
        written = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        assert written["resolved_settings"]["embedding"]["provider"] == "hash"
        assert written["resolved_settings"]["lexical"]["k1"] == 1.2
        assert written["hypothesis"]

    async def test_a_rerun_does_not_overwrite(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        _, first = await run_experiment(settings, config_path, tmp_path / "out")
        _, second = await run_experiment(settings, config_path, tmp_path / "out")
        assert first != second
        assert first.is_dir() and second.is_dir()


class TestResultContent:
    async def test_every_strategy_produces_a_run(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        result, _ = await run_experiment(settings, config_path, tmp_path / "out")
        strategies = {run.parameters["retrieval_strategy"] for run in result.runs}  # type: ignore[attr-defined]
        assert strategies == {"bm25", "dense", "hybrid"}

    async def test_metrics_are_computed_for_every_query(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        result, _ = await run_experiment(settings, config_path, tmp_path / "out")
        for run in result.runs:  # type: ignore[attr-defined]
            assert run.queries == len(QUERIES)
            assert run.failed_queries == 0
            assert set(run.metrics) == {
                "precision@3",
                "recall@3",
                "hit_rate@3",
                "mrr@3",
                "ndcg@3",
            }

    async def test_bm25_finds_the_obvious_lexical_matches(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        """Not a quality claim — a sanity check that labels resolve at all.

        If label resolution were broken, every metric would be 0 and the rest of
        this file would still pass.
        """
        settings, config_path = experiment
        result, _ = await run_experiment(settings, config_path, tmp_path / "out")
        bm25 = next(
            run
            for run in result.runs
            if run.parameters["retrieval_strategy"] == "bm25"  # type: ignore[attr-defined]
        )
        assert bm25.metrics["hit_rate@3"] > 0.0

    async def test_latency_is_recorded_per_stage(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        result, _ = await run_experiment(settings, config_path, tmp_path / "out")
        for run in result.runs:  # type: ignore[attr-defined]
            assert run.latency["total_ms"]["p50"] > 0
            assert set(run.latency["total_ms"]) >= {"p50", "p95", "p99"}

    async def test_cost_is_zero_for_a_local_embedder_not_unknown(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        result, _ = await run_experiment(settings, config_path, tmp_path / "out")
        for run in result.runs:  # type: ignore[attr-defined]
            assert run.cost.usd == 0.0
            assert run.cost.embedded_tokens > 0

    async def test_provenance_is_captured(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        result, _ = await run_experiment(settings, config_path, tmp_path / "out")
        assert result.recall_version  # type: ignore[attr-defined]
        assert result.corpus["documents"] == len(CORPUS)  # type: ignore[attr-defined]
        assert result.models["embedding"] == "hash:hash-v1"  # type: ignore[attr-defined]

    async def test_the_hash_embedder_caveat_is_recorded(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        """These numbers must never be quotable as a quality result."""
        settings, config_path = experiment
        result, _ = await run_experiment(settings, config_path, tmp_path / "out")
        assert any("hash" in note for note in result.notes)  # type: ignore[attr-defined]
        assert any("synthetic" in note for note in result.notes)  # type: ignore[attr-defined]

    async def test_the_report_carries_the_caveats(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        _, directory = await run_experiment(settings, config_path, tmp_path / "out")
        report = (directory / "report.md").read_text(encoding="utf-8")
        assert "synthetic" in report
        assert "## Analysis" in report
        assert "hash" in report


class TestGuardrails:
    async def test_an_empty_index_is_refused(
        self, storage: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        """All-zero metrics would otherwise look like a finding."""
        settings, config_path = experiment
        with pytest.raises(ExperimentError, match="index is empty"):
            await run_experiment(settings, config_path, tmp_path / "out")

    async def test_unresolvable_labels_are_refused(
        self, corpus: Storage, tmp_path: Path, settings: Settings
    ) -> None:
        dataset = tmp_path / "bad.jsonl"
        dataset.write_text(
            json.dumps({"query": "q", "relevant_documents": ["nonexistent.md"]}) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "bad.meta.json").write_text(
            json.dumps({"kind": "synthetic", "label_method": "x"}), encoding="utf-8"
        )
        config = tmp_path / "bad.yaml"
        config.write_text(
            yaml.safe_dump({"name": "bad", "dataset": {"path": str(dataset)}}), encoding="utf-8"
        )
        with pytest.raises(LabelResolutionError, match=r"nonexistent\.md"):
            await run_experiment(settings, config, tmp_path / "out")


class TestBenchmarkAgainstARealRun:
    async def test_a_run_does_not_regress_against_itself(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        baseline, _ = await run_experiment(settings, config_path, tmp_path / "out")
        current, _ = await run_experiment(settings, config_path, tmp_path / "out")

        comparison = compare(current, baseline, threshold=0.0)  # type: ignore[arg-type]
        assert comparison.passed
        assert not comparison.missing_runs
        assert comparison.unchanged > 0

    async def test_a_degraded_run_is_caught(
        self, corpus: Storage, experiment: tuple[Settings, Path], tmp_path: Path
    ) -> None:
        settings, config_path = experiment
        baseline, _ = await run_experiment(settings, config_path, tmp_path / "out")
        current, _ = await run_experiment(settings, config_path, tmp_path / "out")

        for run in baseline.runs:  # type: ignore[attr-defined]
            run.metrics = {name: min(1.0, value + 0.5) for name, value in run.metrics.items()}

        comparison = compare(current, baseline, threshold=0.02)  # type: ignore[arg-type]
        assert not comparison.passed
        assert comparison.regressions
