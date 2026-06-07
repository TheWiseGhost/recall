"""The filesystem connector."""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.connectors.filesystem import FilesystemConnector
from recall.core.errors import DocumentParseError
from recall.core.models import SourceType

from tests.conftest import FIXTURES

CORPUS = FIXTURES / "corpus"


class TestDiscovery:
    async def test_finds_supported_files(self) -> None:
        items = await FilesystemConnector(root=CORPUS).discover()
        assert {item.source_id for item in items} == {
            "authentication.md",
            "deployment.txt",
            "service.json",
            "index.html",
        }

    async def test_skips_unsupported_extensions(self) -> None:
        items = await FilesystemConnector(root=CORPUS).discover()
        assert not any(item.source_id.endswith(".rst") for item in items)

    async def test_excludes_noise_directories(self) -> None:
        items = await FilesystemConnector(root=CORPUS).discover()
        assert not any("node_modules" in item.source_id for item in items)

    async def test_source_ids_are_relative_to_the_root(self) -> None:
        items = await FilesystemConnector(root=CORPUS).discover()
        assert all(not Path(item.source_id).is_absolute() for item in items)

    async def test_extensions_are_configurable(self) -> None:
        items = await FilesystemConnector(root=CORPUS, extensions=[".json"]).discover()
        assert [item.source_id for item in items] == ["service.json"]

    async def test_accepts_a_bare_extension(self) -> None:
        items = await FilesystemConnector(root=CORPUS, extensions=["json"]).discover()
        assert [item.source_id for item in items] == ["service.json"]

    async def test_accepts_a_single_file_as_root(self) -> None:
        items = await FilesystemConnector(root=CORPUS / "deployment.txt").discover()
        assert len(items) == 1
        assert items[0].source_id == "deployment.txt"

    async def test_size_limit_excludes_large_files(self) -> None:
        items = await FilesystemConnector(root=CORPUS, max_file_bytes=1).discover()
        assert items == []

    async def test_non_recursive_stays_in_the_top_directory(self, tmp_path: Path) -> None:
        (tmp_path / "top.md").write_text("top")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "deep.md").write_text("deep")
        items = await FilesystemConnector(root=tmp_path, recursive=False).discover()
        assert [item.source_id for item in items] == ["top.md"]

    async def test_missing_root_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentParseError, match="does not exist"):
            await FilesystemConnector(root=tmp_path / "nope").discover()

    async def test_discovery_is_ordered(self) -> None:
        connector = FilesystemConnector(root=CORPUS)
        first = [item.source_id for item in await connector.discover()]
        second = [item.source_id for item in await connector.discover()]
        assert first == second == sorted(first)


class TestFetch:
    async def _fetch(self, source_id: str, **kwargs: object):
        connector = FilesystemConnector(root=CORPUS, **kwargs)  # type: ignore[arg-type]
        items = await connector.discover()
        item = next(i for i in items if i.source_id == source_id)
        return await connector.fetch(item)

    async def test_markdown_title_comes_from_the_heading(self) -> None:
        document = await self._fetch("authentication.md")
        assert document.title == "Authentication"
        assert document.source_type is SourceType.FILESYSTEM

    async def test_html_title_comes_from_the_title_tag(self) -> None:
        document = await self._fetch("index.html")
        assert document.title == "Observability Guide"
        assert "Prometheus" in document.content
        assert "should not be extracted" not in document.content

    async def test_json_is_flattened(self) -> None:
        document = await self._fetch("service.json")
        assert document.title == "billing-service"
        assert "owner: payments-team" in document.content

    async def test_plain_text_falls_back_to_the_stem(self) -> None:
        document = await self._fetch("deployment.txt")
        assert document.title == "deployment"

    async def test_metadata_carries_file_type(self) -> None:
        document = await self._fetch("authentication.md")
        assert document.metadata["file_type"] == "md"
        assert document.metadata["filename"] == "authentication.md"
        assert document.metadata["char_count"] == len(document.content)

    async def test_uri_is_a_file_url(self) -> None:
        document = await self._fetch("authentication.md")
        assert document.uri.startswith("file://")

    async def test_identical_content_yields_identical_checksums(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# Same\n\nidentical body")
        connector = FilesystemConnector(root=tmp_path)
        items = await connector.discover()
        first = await connector.fetch(items[0])

        # Rewrite the same bytes: mtime changes, content does not.
        (tmp_path / "a.md").write_text("# Same\n\nidentical body")
        second = await connector.fetch((await connector.discover())[0])

        assert first.checksum == second.checksum
        assert first.id == second.id

    async def test_changed_content_changes_the_checksum(self, tmp_path: Path) -> None:
        path = tmp_path / "a.md"
        path.write_text("# Same\n\noriginal")
        connector = FilesystemConnector(root=tmp_path)
        first = await connector.fetch((await connector.discover())[0])
        path.write_text("# Same\n\nedited")
        second = await connector.fetch((await connector.discover())[0])
        assert first.checksum != second.checksum
        assert first.id == second.id  # same source, so same document

    async def test_deleted_file_reports_clearly(self, tmp_path: Path) -> None:
        path = tmp_path / "a.md"
        path.write_text("body")
        connector = FilesystemConnector(root=tmp_path)
        item = (await connector.discover())[0]
        path.unlink()
        with pytest.raises(DocumentParseError, match="disappeared"):
            await connector.fetch(item)

    async def test_undecodable_bytes_do_not_crash(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_bytes(b"valid text \xff\xfe then more")
        connector = FilesystemConnector(root=tmp_path)
        document = await connector.fetch((await connector.discover())[0])
        assert "valid text" in document.content
