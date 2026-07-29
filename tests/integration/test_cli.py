"""The CLI, driven through Typer's runner against a real database.

This is the Definition of Done path: init -> migrate -> ingest -> search.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from recall.cli.main import app
from recall.config.settings import Settings

from tests.integration.conftest import TEST_DIMENSIONS

pytestmark = pytest.mark.integration

runner = CliRunner()


@pytest.fixture
def project(migrated: Settings, storage, tmp_path: Path) -> Iterator[Path]:
    """A scaffolded project wired to the (already migrated, empty) test database.

    ``storage`` is requested purely for its truncation side effect.
    """
    result = runner.invoke(app, ["init", str(tmp_path), "--embedding-provider", "hash"])
    assert result.exit_code == 0, result.output

    config = tmp_path / "recall.yaml"
    config.write_text(
        config.read_text()
        .replace(
            "${DATABASE_URL:-postgresql+asyncpg://recall:recall@localhost:5432/recall}",
            migrated.database.url,
        )
        .replace("dimensions: 384", f"dimensions: {TEST_DIMENSIONS}")
    )
    yield tmp_path


def run(project: Path, *args: str):
    return runner.invoke(app, [*args, "--config", str(project / "recall.yaml")])


class TestInit:
    def test_scaffolds_the_expected_files(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "recall.yaml").is_file()
        assert (tmp_path / ".env.example").is_file()
        assert (tmp_path / ".gitignore").is_file()
        assert (tmp_path / "examples" / "documents" / "authentication.md").is_file()

    def test_does_not_overwrite_without_force(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        (tmp_path / "recall.yaml").write_text("# edited by hand\n")
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "recall.yaml").read_text() == "# edited by hand\n"

    def test_force_overwrites(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        (tmp_path / "recall.yaml").write_text("# edited by hand\n")
        runner.invoke(app, ["init", str(tmp_path), "--force"])
        assert "embedding:" in (tmp_path / "recall.yaml").read_text()

    def test_rejects_an_unknown_provider(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["init", str(tmp_path), "--embedding-provider", "telepathy"])
        assert result.exit_code == 1

    def test_written_config_is_loadable(self, tmp_path: Path) -> None:
        from recall.config.settings import load_settings

        runner.invoke(app, ["init", str(tmp_path), "--embedding-provider", "hash"])
        settings = load_settings(tmp_path / "recall.yaml")
        assert settings.embedding.provider == "hash"
        assert settings.chunking.strategy == "fixed"


class TestVersionAndComponents:
    def test_version(self) -> None:
        from recall import __version__

        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_connectors_lists_every_registry(self) -> None:
        result = runner.invoke(app, ["connectors"])
        assert result.exit_code == 0
        for name in (
            "filesystem",
            "pdf",
            "fixed",
            "sentence",
            "semantic",
            "hierarchical",
            "hash",
            "dense",
            "bm25",
            "hybrid",
            "rrf",
            "weighted",
            "cross_encoder",
        ):
            assert name in result.output, name


class TestStatus:
    def test_reports_a_healthy_database(self, project: Path) -> None:
        result = run(project, "status")
        assert result.exit_code == 0, result.output
        assert "connected" in result.output
        assert "hash:hash-v1" in result.output

    def test_reports_an_unreachable_database(self, project: Path) -> None:
        config = project / "recall.yaml"
        config.write_text(config.read_text().replace("@localhost:5432", "@localhost:65432"))
        result = run(project, "status")
        assert result.exit_code == 1
        assert "unreachable" in result.output


class TestIngestAndSearch:
    def test_definition_of_done_flow(self, project: Path) -> None:
        ingest = run(project, "ingest", str(project / "examples" / "documents"))
        assert ingest.exit_code == 0, ingest.output
        # The summary table prints every outcome label, so assert on the count.
        assert "created=1" in ingest.output
        assert "failed=0" in ingest.output

        listing = run(project, "documents", "list")
        assert "(1 total)" in listing.output

        search = run(project, "search", "How does authentication work?")
        assert search.exit_code == 0, search.output
        assert "Authentication" in search.output
        assert "ms" in search.output

    def test_json_output_carries_scores_and_timing(self, project: Path) -> None:
        run(project, "ingest", str(project / "examples" / "documents"))
        result = run(project, "search", "authentication tokens", "--json", "--top-k", "3")
        assert result.exit_code == 0, result.output

        payload = json.loads(result.stdout)
        assert payload["query"] == "authentication tokens"
        assert payload["retrieval_strategy"] == "dense"
        assert 0 < len(payload["results"]) <= 3
        first = payload["results"][0]
        assert set(first) >= {"chunk_id", "document_id", "content", "score", "rank", "metadata"}
        assert first["rank"] == 1
        assert payload["timing"]["total_ms"] > 0
        assert payload["timing"]["embedding_ms"] > 0

    def test_second_ingest_is_incremental(self, project: Path) -> None:
        run(project, "ingest", str(project / "examples" / "documents"))
        again = run(project, "ingest", str(project / "examples" / "documents"))
        assert again.exit_code == 0
        assert "unchanged" in again.output

    def test_missing_path_fails_clearly(self, project: Path) -> None:
        result = run(project, "ingest", str(project / "nowhere"))
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_unknown_connector_is_rejected(self, project: Path) -> None:
        result = run(project, "ingest", str(project / "examples"), "--type", "telepathy")
        assert result.exit_code == 1

    def test_search_on_an_empty_index_says_so(self, project: Path) -> None:
        result = run(project, "search", "anything")
        assert result.exit_code == 0
        assert "no results" in result.output

    def test_invalid_source_type_filter_is_rejected(self, project: Path) -> None:
        result = run(project, "search", "x", "--source-type", "telepathy")
        assert result.exit_code == 1


class TestDocuments:
    def test_list_and_show(self, project: Path) -> None:
        run(project, "ingest", str(project / "examples" / "documents"))

        listing = run(project, "documents", "list")
        assert listing.exit_code == 0
        assert "Authentication" in listing.output

        payload = json.loads(run(project, "search", "authentication", "--json").stdout)
        document_id = payload["results"][0]["document_id"]

        shown = run(project, "documents", "show", document_id, "--chunks")
        assert shown.exit_code == 0
        assert "Authentication" in shown.output
        assert "chunks" in shown.output

    def test_show_rejects_a_non_uuid(self, project: Path) -> None:
        result = run(project, "documents", "show", "not-a-uuid")
        assert result.exit_code == 1
        assert "not a valid UUID" in result.output

    def test_show_reports_a_missing_document(self, project: Path) -> None:
        result = run(project, "documents", "show", "00000000-0000-0000-0000-000000000000")
        assert result.exit_code == 1
        assert "No document" in result.output


class TestRetrievalStrategies:
    """Every strategy must be reachable from the CLI, not just from Python."""

    @pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
    def test_search_with_each_strategy(self, project: Path, strategy: str) -> None:
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0

        result = run(project, "search", "authentication", "--strategy", strategy, "--json")
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["retrieval_strategy"] == strategy
        assert payload["results"]

    def test_hybrid_records_component_provenance(self, project: Path) -> None:
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0
        result = run(project, "search", "authentication", "--strategy", "hybrid", "--json")
        payload = json.loads(result.stdout)
        assert payload["results"][0]["component_ranks"]

    def test_unknown_strategy_lists_the_alternatives(self, project: Path) -> None:
        result = run(project, "search", "q", "--strategy", "telepathy")
        assert result.exit_code != 0
        assert "bm25" in result.output

    def test_rerank_off_is_accepted(self, project: Path) -> None:
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0
        result = run(project, "search", "authentication", "--rerank", "off", "--json")
        assert result.exit_code == 0
        assert json.loads(result.stdout)["reranked"] is False

    def test_rerank_none_widens_the_candidate_pool(self, project: Path) -> None:
        """The control condition: pool widens, order is untouched."""
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0
        result = run(project, "search", "authentication", "--rerank", "none", "--json")
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["reranked"] is True
        assert payload["reranking_strategy"] == "none"
        assert payload["candidates"] > payload["top_k"]

    def test_unknown_reranker_is_rejected(self, project: Path) -> None:
        result = run(project, "search", "q", "--rerank", "telepathy")
        assert result.exit_code != 0
        assert "cross_encoder" in result.output


class TestExperimentCommands:
    def write_experiment(self, project: Path) -> Path:
        dataset = project / "queries.jsonl"
        dataset.write_text(
            "\n".join(
                json.dumps(payload)
                for payload in (
                    {
                        "query": "How does authentication work?",
                        "relevant_documents": ["authentication.md"],
                    },
                    {
                        "query": "How are bearer tokens verified?",
                        "relevant_documents": ["authentication.md"],
                    },
                )
            )
            + "\n",
            encoding="utf-8",
        )
        (project / "queries.meta.json").write_text(
            json.dumps({"kind": "synthetic", "documents": 1, "label_method": "for the CLI tests"}),
            encoding="utf-8",
        )
        config = project / "exp.yaml"
        config.write_text(
            "name: cli-check\n"
            f"dataset:\n  path: {dataset}\n"
            "retrieval:\n  strategies: [bm25, dense]\n"
            "top_k: [3]\n",
            encoding="utf-8",
        )
        return config

    def test_experiment_writes_a_result_directory(self, project: Path) -> None:
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0
        config = self.write_experiment(project)

        result = run(project, "experiment", str(config), "--output", str(project / "out"))
        assert result.exit_code == 0, result.output

        directories = list((project / "out" / "results").glob("*-cli-check"))
        assert len(directories) == 1
        for filename in ("config.yaml", "results.json", "metrics.csv", "report.md"):
            assert (directories[0] / filename).is_file(), filename

    def test_experiment_warns_about_a_synthetic_dataset(self, project: Path) -> None:
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0
        config = self.write_experiment(project)
        result = run(project, "experiment", str(config), "--output", str(project / "out"))
        assert "synthetic" in result.output

    def test_experiment_on_an_empty_index_fails_clearly(self, project: Path) -> None:
        config = self.write_experiment(project)
        result = run(project, "experiment", str(config), "--output", str(project / "out"))
        assert result.exit_code != 0
        assert "empty" in result.output

    def test_report_regenerates_from_a_result_directory(self, project: Path) -> None:
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0
        config = self.write_experiment(project)
        run(project, "experiment", str(config), "--output", str(project / "out"))
        directory = next((project / "out" / "results").glob("*-cli-check"))

        (directory / "report.md").unlink()
        result = run(project, "report", str(directory))
        assert result.exit_code == 0, result.output
        assert (directory / "report.md").is_file()
        assert "## Analysis" in (directory / "report.md").read_text(encoding="utf-8")

    def test_report_on_a_missing_experiment_fails_clearly(self, project: Path) -> None:
        result = run(project, "report", "no-such-experiment")
        assert result.exit_code != 0

    def test_benchmark_passes_against_its_own_baseline(self, project: Path) -> None:
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0
        config = self.write_experiment(project)
        run(project, "experiment", str(config), "--output", str(project / "out"))
        baseline = next((project / "out" / "results").glob("*-cli-check")) / "results.json"

        result = run(
            project,
            "benchmark",
            str(config),
            "--baseline",
            str(baseline),
            "--output",
            str(project / "out"),
        )
        assert result.exit_code == 0, result.output
        assert "PASSED" in result.output

    def test_benchmark_fails_on_a_regression(self, project: Path) -> None:
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0
        config = self.write_experiment(project)
        run(project, "experiment", str(config), "--output", str(project / "out"))
        baseline_path = next((project / "out" / "results").glob("*-cli-check")) / "results.json"

        # Inflate the baseline so the real run looks like a regression.
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        for run_payload in payload["runs"]:
            run_payload["metrics"] = {
                name: min(1.0, value + 0.5) for name, value in run_payload["metrics"].items()
            }
        inflated = project / "inflated.json"
        inflated.write_text(json.dumps(payload), encoding="utf-8")

        result = run(
            project,
            "benchmark",
            str(config),
            "--baseline",
            str(inflated),
            "--output",
            str(project / "out"),
        )
        assert result.exit_code == 1
        assert "FAILED" in result.output

    def test_benchmark_without_a_baseline_says_how_to_make_one(self, project: Path) -> None:
        assert run(project, "ingest", str(project / "examples" / "documents")).exit_code == 0
        config = self.write_experiment(project)
        result = run(project, "benchmark", str(config), "--output", str(project / "out"))
        assert result.exit_code == 0
        assert "--baseline" in result.output
