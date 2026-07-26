"""Dataset loading, label resolution, cost estimation, benchmarking, reports."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from recall.core.embeddings.base import EmbeddingModelInfo
from recall.core.errors import ConfigurationError
from recall.core.evaluation.cost import estimate_embedding_cost
from recall.core.evaluation.models import (
    CostEstimate,
    Dataset,
    EvaluationQuery,
    ExperimentResult,
    Granularity,
    QueryOutcome,
    RunResult,
)
from recall.core.models import SearchResult
from recall.evaluation.benchmark import BaselineError, compare, load_baseline
from recall.evaluation.config import load_experiment_config
from recall.evaluation.datasets import DatasetError, load_dataset
from recall.evaluation.labels import LabelResolutionError, build_resolver
from recall.evaluation.report import render_report
from recall.evaluation.runner import ExperimentRunner, metrics_csv, result_directory

META = {
    "name": "tiny",
    "kind": "synthetic",
    "documents": 2,
    "label_method": "hand-written for tests",
}


def write_dataset(directory: Path, lines: list[str], meta: dict | None = None) -> Path:
    path = directory / "tiny.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "tiny.meta.json").write_text(
        json.dumps(META if meta is None else meta), encoding="utf-8"
    )
    return path


class TestDatasetLoading:
    def test_binary_labels_become_grade_one(self, tmp_path: Path) -> None:
        path = write_dataset(tmp_path, ['{"query": "q", "relevant_documents": ["a.md"]}'])
        dataset = load_dataset(path)
        assert dataset.queries[0].relevant == {"a.md": 1}
        assert not dataset.is_graded

    def test_graded_labels_are_preserved(self, tmp_path: Path) -> None:
        path = write_dataset(
            tmp_path, ['{"query": "q", "relevant_documents": {"a.md": 3, "b.md": 1}}']
        )
        dataset = load_dataset(path)
        assert dataset.queries[0].relevant == {"a.md": 3, "b.md": 1}
        assert dataset.is_graded

    def test_chunk_level_labels(self, tmp_path: Path) -> None:
        path = write_dataset(tmp_path, ['{"query": "q", "relevant_chunks": ["c1"]}'])
        assert load_dataset(path).queries[0].granularity is Granularity.CHUNK

    def test_blank_lines_and_comments_are_skipped(self, tmp_path: Path) -> None:
        path = write_dataset(
            tmp_path,
            ["// a comment", "", '{"query": "q", "relevant_documents": ["a.md"]}'],
        )
        assert len(load_dataset(path).queries) == 1

    def test_records_a_checksum(self, tmp_path: Path) -> None:
        """A result file has to pin the exact labels it was scored against."""
        path = write_dataset(tmp_path, ['{"query": "q", "relevant_documents": ["a.md"]}'])
        first = load_dataset(path).checksum
        assert len(first) == 64
        path.write_text('{"query": "q2", "relevant_documents": ["a.md"]}\n', encoding="utf-8")
        assert load_dataset(path).checksum != first

    def test_missing_metadata_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.jsonl"
        path.write_text('{"query": "q", "relevant_documents": ["a.md"]}\n', encoding="utf-8")
        with pytest.raises(DatasetError, match=r"meta\.json"):
            load_dataset(path)

    def test_undeclared_kind_is_refused(self, tmp_path: Path) -> None:
        path = write_dataset(
            tmp_path, ['{"query": "q", "relevant_documents": ["a.md"]}'], meta={"kind": "vibes"}
        )
        with pytest.raises(DatasetError, match="curated"):
            load_dataset(path)

    def test_mixed_granularity_is_refused(self, tmp_path: Path) -> None:
        """Averaging metrics whose denominators mean different things is meaningless."""
        path = write_dataset(
            tmp_path,
            [
                '{"query": "q1", "relevant_documents": ["a.md"]}',
                '{"query": "q2", "relevant_chunks": ["c1"]}',
            ],
        )
        with pytest.raises(DatasetError, match="mixes"):
            load_dataset(path)

    def test_both_label_kinds_on_one_line_is_refused(self, tmp_path: Path) -> None:
        path = write_dataset(
            tmp_path,
            ['{"query": "q", "relevant_documents": ["a.md"], "relevant_chunks": ["c"]}'],
        )
        with pytest.raises(DatasetError, match="not both"):
            load_dataset(path)

    def test_duplicate_queries_are_refused(self, tmp_path: Path) -> None:
        """A repeated query would be weighted double in every average."""
        path = write_dataset(
            tmp_path,
            [
                '{"query": "same", "relevant_documents": ["a.md"]}',
                '{"query": "same", "relevant_documents": ["b.md"]}',
            ],
        )
        with pytest.raises(DatasetError, match="repeats"):
            load_dataset(path)

    def test_a_query_with_no_relevant_items_is_refused(self, tmp_path: Path) -> None:
        path = write_dataset(tmp_path, ['{"query": "q", "relevant_documents": {"a.md": 0}}'])
        with pytest.raises(DatasetError, match="no relevant items"):
            load_dataset(path)

    def test_missing_labels_are_refused(self, tmp_path: Path) -> None:
        path = write_dataset(tmp_path, ['{"query": "q"}'])
        with pytest.raises(DatasetError, match="relevant_documents"):
            load_dataset(path)

    def test_a_non_integer_grade_is_refused(self, tmp_path: Path) -> None:
        path = write_dataset(tmp_path, ['{"query": "q", "relevant_documents": {"a.md": "high"}}'])
        with pytest.raises(DatasetError, match="must be an integer"):
            load_dataset(path)

    def test_malformed_json_names_the_line(self, tmp_path: Path) -> None:
        path = write_dataset(tmp_path, ['{"query": "ok", "relevant_documents": ["a"]}', "{oops"])
        with pytest.raises(DatasetError, match="line 2"):
            load_dataset(path)

    def test_empty_dataset_is_refused(self, tmp_path: Path) -> None:
        path = write_dataset(tmp_path, ["// only a comment"])
        with pytest.raises(DatasetError, match="no queries"):
            load_dataset(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "absent.jsonl")

    def test_the_shipped_example_loads(self) -> None:
        dataset = load_dataset("experiments/datasets/example_tiny.jsonl")
        assert dataset.kind == "synthetic"
        assert len(dataset.queries) == 10


def make_dataset(labels: list[str], granularity: Granularity = Granularity.DOCUMENT) -> Dataset:
    return Dataset(
        name="d",
        path="d.jsonl",
        kind="synthetic",
        queries=[
            EvaluationQuery(
                query="q", relevant={label: 1 for label in labels}, granularity=granularity
            )
        ],
    )


class TestLabelResolution:
    def test_exact_source_ids_resolve(self) -> None:
        doc = uuid.uuid4()
        resolver = build_resolver(make_dataset(["guides/a.md"]), [(doc, "guides/a.md")])
        result = SearchResult(chunk_id=uuid.uuid4(), document_id=doc, content="", score=1.0, rank=1)
        assert resolver.key(result) == "guides/a.md"

    def test_basenames_resolve_when_unambiguous(self) -> None:
        """Datasets are commonly written with bare filenames."""
        doc = uuid.uuid4()
        resolver = build_resolver(make_dataset(["a.md"]), [(doc, "guides/a.md")])
        result = SearchResult(chunk_id=uuid.uuid4(), document_id=doc, content="", score=1.0, rank=1)
        assert resolver.key(result) == "a.md"

    def test_an_ambiguous_basename_is_refused(self) -> None:
        """Silently picking one of two index.md files would produce untrustworthy numbers."""
        with pytest.raises(LabelResolutionError, match="more than one document"):
            build_resolver(
                make_dataset(["index.md"]),
                [(uuid.uuid4(), "a/index.md"), (uuid.uuid4(), "b/index.md")],
            )

    def test_an_unresolvable_label_is_refused(self) -> None:
        """The dangerous case: it would score zero and look like a finding."""
        with pytest.raises(LabelResolutionError, match="match no ingested document"):
            build_resolver(make_dataset(["ghost.md"]), [(uuid.uuid4(), "real.md")])

    def test_the_error_names_what_is_available(self) -> None:
        with pytest.raises(LabelResolutionError, match=r"real\.md"):
            build_resolver(make_dataset(["ghost.md"]), [(uuid.uuid4(), "real.md")])

    def test_chunk_granularity_uses_chunk_ids(self) -> None:
        resolver = build_resolver(make_dataset(["c1"], Granularity.CHUNK), [])
        chunk = uuid.uuid4()
        result = SearchResult(
            chunk_id=chunk, document_id=uuid.uuid4(), content="", score=1.0, rank=1
        )
        assert resolver.key(result) == str(chunk)

    def test_ranked_keys_preserve_duplicates(self) -> None:
        """Precision counts positions, so two chunks from one document are two."""
        doc = uuid.uuid4()
        resolver = build_resolver(make_dataset(["a.md"]), [(doc, "a.md")])
        results = [
            SearchResult(chunk_id=uuid.uuid4(), document_id=doc, content="", score=1.0, rank=i)
            for i in (1, 2)
        ]
        assert resolver.ranked_keys(results) == ["a.md", "a.md"]


class TestCostEstimation:
    def test_a_priced_model_is_computed(self) -> None:
        info = EmbeddingModelInfo(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1536,
            cost_per_million_tokens=0.02,
        )
        estimate = estimate_embedding_cost(info, 1_000_000)
        assert estimate.usd == pytest.approx(0.02)

    def test_a_local_model_is_free(self) -> None:
        info = EmbeddingModelInfo(provider="hash", model="hash-v1", dimensions=64)
        estimate = estimate_embedding_cost(info, 10_000)
        assert estimate.usd == 0.0
        assert "locally" in estimate.note

    def test_an_unpriced_remote_model_is_not_reported_as_free(self) -> None:
        """Reporting a paid model as $0 would make an expensive setup look cheap."""
        info = EmbeddingModelInfo(provider="openai", model="brand-new", dimensions=99)
        estimate = estimate_embedding_cost(info, 10_000)
        assert estimate.usd is None
        assert "not estimated" in estimate.note

    def test_no_model_at_all(self) -> None:
        assert estimate_embedding_cost(None, 0).usd is None


def make_result(name: str, metrics: dict[str, float], checksum: str = "abc") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=f"2026-01-01-{name}",
        name=name,
        dataset=Dataset(
            name="d",
            path="d.jsonl",
            kind="synthetic",
            checksum=checksum,
            queries=[EvaluationQuery(query="q", relevant={"a.md": 1})],
        ),
        runs=[
            RunResult(
                run_id="dense__rerank-off__k10",
                parameters={
                    "retrieval_strategy": "dense",
                    "reranking_strategy": "off",
                    "top_k": 10,
                },
                metrics=metrics,
                latency={"total_ms": {"p50": 5.0, "p95": 9.0, "p99": 10.0}},
                cost=CostEstimate(usd=0.0, embedded_tokens=42, model="hash:hash-v1"),
                queries=1,
                outcomes=[QueryOutcome(query="q", metrics=metrics)],
            )
        ],
    )


class TestBenchmark:
    def test_identical_runs_pass(self) -> None:
        result = make_result("a", {"mrr@10": 0.8})
        comparison = compare(result, make_result("a", {"mrr@10": 0.8}))
        assert comparison.passed
        assert comparison.unchanged == 1

    def test_a_drop_beyond_the_threshold_fails(self) -> None:
        comparison = compare(
            make_result("a", {"mrr@10": 0.70}), make_result("a", {"mrr@10": 0.80}), threshold=0.02
        )
        assert not comparison.passed
        assert comparison.regressions[0].delta == pytest.approx(-0.10)

    def test_a_drop_within_the_threshold_passes(self) -> None:
        comparison = compare(
            make_result("a", {"mrr@10": 0.79}), make_result("a", {"mrr@10": 0.80}), threshold=0.02
        )
        assert comparison.passed

    def test_improvements_are_reported_but_do_not_fail(self) -> None:
        comparison = compare(make_result("a", {"mrr@10": 0.95}), make_result("a", {"mrr@10": 0.80}))
        assert comparison.passed
        assert comparison.improvements

    def test_a_changed_dataset_invalidates_the_comparison(self) -> None:
        """A relabel would otherwise be reported as a regression."""
        with pytest.raises(BaselineError, match="dataset changed"):
            compare(
                make_result("a", {"mrr@10": 0.5}, checksum="new"),
                make_result("a", {"mrr@10": 0.9}, checksum="old"),
            )

    def test_runs_are_matched_by_id(self) -> None:
        current = make_result("a", {"mrr@10": 0.8})
        baseline = make_result("a", {"mrr@10": 0.8})
        baseline.runs[0].run_id = "bm25__rerank-off__k5"
        comparison = compare(current, baseline)
        assert comparison.missing_runs == ["bm25__rerank-off__k5"]
        assert comparison.new_runs == ["dense__rerank-off__k10"]
        assert comparison.passed  # nothing comparable, so nothing regressed

    def test_a_negative_threshold_is_refused(self) -> None:
        with pytest.raises(BaselineError, match="non-negative"):
            compare(make_result("a", {}), make_result("a", {}), threshold=-1.0)

    def test_relative_change_is_none_at_a_zero_baseline(self) -> None:
        comparison = compare(make_result("a", {"mrr@10": 0.5}), make_result("a", {"mrr@10": 0.0}))
        assert comparison.improvements[0].relative is None

    def test_load_baseline_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(make_result("a", {"mrr@10": 0.8}).model_dump(mode="json")), encoding="utf-8"
        )
        assert load_baseline(path).runs[0].metrics == {"mrr@10": 0.8}

    def test_missing_baseline_explains_how_to_make_one(self, tmp_path: Path) -> None:
        with pytest.raises(BaselineError, match="recall experiment"):
            load_baseline(tmp_path / "absent.json")

    def test_a_non_result_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        path.write_text('{"not": "a result"}', encoding="utf-8")
        with pytest.raises(BaselineError, match="not a Recall experiment result"):
            load_baseline(path)


class TestMetricsCsv:
    def test_one_row_per_run(self) -> None:
        rows = metrics_csv(make_result("a", {"mrr@10": 0.8})).strip().split("\n")
        assert len(rows) == 2
        assert rows[0].startswith("experiment_id,run_id,")

    def test_an_unpriced_cost_is_blank_not_zero(self) -> None:
        result = make_result("a", {"mrr@10": 0.8})
        result.runs[0].cost = CostEstimate(usd=None, embedded_tokens=10)
        body = metrics_csv(result).strip().split("\n")[1]
        assert body.endswith(",,10")

    def test_no_runs(self) -> None:
        result = make_result("a", {})
        result.runs = []
        assert metrics_csv(result) == "run_id\n"


class TestReport:
    def test_includes_provenance_and_caveats(self) -> None:
        result = make_result("a", {"mrr@10": 0.8})
        result.git_commit = "deadbeef"
        result.notes = ["a caveat"]
        markdown = render_report(result, hypothesis="hybrid wins")

        assert "hybrid wins" in markdown
        assert "deadbeef" in markdown
        assert "a caveat" in markdown
        assert "mrr@10" in markdown

    def test_labels_a_synthetic_dataset_prominently(self) -> None:
        markdown = render_report(make_result("a", {"mrr@10": 0.8}))
        assert "synthetic" in markdown.split("## Setup")[0]

    def test_flags_a_dirty_tree(self) -> None:
        result = make_result("a", {"mrr@10": 0.8})
        result.git_commit = "deadbeef"
        result.git_dirty = True
        assert "dirty tree" in render_report(result)

    def test_leaves_the_analysis_to_a_human(self) -> None:
        """The tool lays out what was measured; it does not conclude."""
        markdown = render_report(make_result("a", {"mrr@10": 0.8}))
        assert "## Analysis" in markdown
        assert "To be written" in markdown

    def test_says_when_no_hypothesis_was_given(self) -> None:
        assert "Not stated" in render_report(make_result("a", {"mrr@10": 0.8}))


class TestExperimentConfig:
    def write(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "exp.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_parses_the_target_shape(self) -> None:
        config = load_experiment_config("experiments/configs/002-bm25-vs-dense-vs-hybrid.yaml")
        assert config.retrieval_strategies == ["bm25", "dense", "hybrid"]
        assert config.top_k == [5, 10, 20]
        assert config.run_count == 9
        assert config.hypothesis

    def test_a_scalar_is_a_sweep_of_one(self, tmp_path: Path) -> None:
        path = self.write(
            tmp_path,
            "name: x\ndataset:\n  path: d.jsonl\nretrieval:\n  strategies: dense\ntop_k: 5\n",
        )
        config = load_experiment_config(path)
        assert config.retrieval_strategies == ["dense"]
        assert config.top_k == [5]
        assert config.run_count == 1

    def test_index_time_sweeps_are_refused_with_an_explanation(self, tmp_path: Path) -> None:
        path = self.write(
            tmp_path,
            "name: x\ndataset:\n  path: d.jsonl\nchunking:\n  strategies: [fixed, semantic]\n",
        )
        with pytest.raises(ConfigurationError, match="own ingest"):
            load_experiment_config(path)

    def test_an_unknown_metric_is_refused(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "name: x\ndataset:\n  path: d.jsonl\nmetrics: [vibes_at_k]\n")
        with pytest.raises(ConfigurationError, match="vibes_at_k"):
            load_experiment_config(path)

    def test_a_missing_dataset_is_refused(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, "name: x\n")
        with pytest.raises(ConfigurationError, match=r"dataset\.path"):
            load_experiment_config(path)

    def test_reranking_sweep(self, tmp_path: Path) -> None:
        path = self.write(
            tmp_path,
            "name: x\ndataset:\n  path: d.jsonl\nreranking:\n  strategies: [off, none, cross_encoder]\n",
        )
        assert load_experiment_config(path).reranking_strategies == ["off", "none", "cross_encoder"]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_experiment_config(tmp_path / "absent.yaml")


class TestResultDirectory:
    def test_named_by_date_and_experiment(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        when = datetime(2026, 8, 3, tzinfo=UTC)
        assert result_directory(tmp_path, "hybrid", when).name == "2026-08-03-hybrid"

    def test_never_overwrites_an_existing_run(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        when = datetime(2026, 8, 3, tzinfo=UTC)
        first = result_directory(tmp_path, "hybrid", when)
        first.mkdir(parents=True)
        assert result_directory(tmp_path, "hybrid", when).name == "2026-08-03-hybrid-2"


class TestAggregation:
    def test_failed_queries_are_excluded_not_scored_zero(self) -> None:
        """A crashed query measures the harness, not the retriever."""
        outcomes = [
            QueryOutcome(query="a", metrics={"mrr@10": 1.0}),
            QueryOutcome(query="b", error="boom"),
        ]
        assert ExperimentRunner._aggregate_metrics(outcomes) == {"mrr@10": 1.0}

    def test_all_failed_produces_no_metrics(self) -> None:
        outcomes = [QueryOutcome(query="a", error="boom")]
        assert ExperimentRunner._aggregate_metrics(outcomes) == {}

    def test_means_are_over_scored_queries(self) -> None:
        outcomes = [
            QueryOutcome(query="a", metrics={"mrr@10": 1.0}),
            QueryOutcome(query="b", metrics={"mrr@10": 0.0}),
        ]
        assert ExperimentRunner._aggregate_metrics(outcomes) == {"mrr@10": 0.5}
