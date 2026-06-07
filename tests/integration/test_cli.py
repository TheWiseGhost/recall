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
        for name in ("filesystem", "pdf", "fixed", "hash", "dense"):
            assert name in result.output


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
