"""The PDF connector.

Fixtures are generated with PyMuPDF at test time rather than committed as
binaries, so the corpus stays readable in review and in git history.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recall.connectors.pdf import PDFConnector, page_for_offset
from recall.core.errors import DocumentParseError
from recall.core.models import SourceType

fitz = pytest.importorskip("fitz", reason="requires the `pdf` extra (PyMuPDF)")


def write_pdf(path: Path, pages: list[str], *, title: str | None = None) -> Path:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    if title:
        document.set_metadata({"title": title, "author": "Test Author"})
    document.save(path)
    document.close()
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    write_pdf(
        tmp_path / "handbook.pdf",
        [
            "Page one covers authentication and bearer tokens.",
            "Page two covers deployment and rollback procedures.",
            "Page three covers observability and metrics collection.",
        ],
        title="Engineering Handbook",
    )
    (tmp_path / "notes.md").write_text("# Not a PDF\n\nshould be ignored")
    return tmp_path


class TestDiscovery:
    async def test_finds_only_pdfs(self, corpus: Path) -> None:
        items = await PDFConnector(root=corpus).discover()
        assert [item.source_id for item in items] == ["handbook.pdf"]

    async def test_source_type_is_pdf(self, corpus: Path) -> None:
        items = await PDFConnector(root=corpus).discover()
        assert items[0].source_type is SourceType.PDF


class TestFetch:
    async def _fetch(self, corpus: Path, **kwargs: object):
        connector = PDFConnector(root=corpus, **kwargs)  # type: ignore[arg-type]
        return await connector.fetch((await connector.discover())[0])

    async def test_extracts_text_from_every_page(self, corpus: Path) -> None:
        document = await self._fetch(corpus)
        assert "authentication" in document.content
        assert "deployment" in document.content
        assert "observability" in document.content

    async def test_title_comes_from_pdf_metadata(self, corpus: Path) -> None:
        document = await self._fetch(corpus)
        assert document.title == "Engineering Handbook"

    async def test_author_is_captured(self, corpus: Path) -> None:
        document = await self._fetch(corpus)
        assert document.metadata["author"] == "Test Author"

    async def test_records_page_count(self, corpus: Path) -> None:
        document = await self._fetch(corpus)
        assert document.metadata["page_count"] == 3
        assert document.metadata["pages_extracted"] == 3

    async def test_page_offsets_are_recorded(self, corpus: Path) -> None:
        document = await self._fetch(corpus)
        offsets = document.metadata["page_offsets"]
        assert [entry[0] for entry in offsets] == [1, 2, 3]
        assert [entry[1] for entry in offsets] == sorted(entry[1] for entry in offsets)

    async def test_max_pages_limits_extraction(self, corpus: Path) -> None:
        document = await self._fetch(corpus, max_pages=2)
        assert document.metadata["pages_extracted"] == 2
        assert "observability" not in document.content

    async def test_falls_back_to_the_filename_without_a_title(self, tmp_path: Path) -> None:
        write_pdf(tmp_path / "untitled.pdf", ["Some body text about tokens."])
        connector = PDFConnector(root=tmp_path)
        document = await connector.fetch((await connector.discover())[0])
        assert document.title == "untitled"

    async def test_a_pdf_with_no_text_is_reported(self, tmp_path: Path) -> None:
        document = fitz.open()
        document.new_page()
        document.save(tmp_path / "blank.pdf")
        document.close()

        connector = PDFConnector(root=tmp_path)
        with pytest.raises(DocumentParseError, match="No extractable text"):
            await connector.fetch((await connector.discover())[0])

    async def test_a_corrupt_file_is_reported(self, tmp_path: Path) -> None:
        (tmp_path / "broken.pdf").write_bytes(b"this is definitely not a pdf")
        connector = PDFConnector(root=tmp_path)
        with pytest.raises(DocumentParseError):
            await connector.fetch((await connector.discover())[0])

    async def test_checksum_is_stable_across_fetches(self, corpus: Path) -> None:
        first = await self._fetch(corpus)
        second = await self._fetch(corpus)
        assert first.checksum == second.checksum
        assert first.id == second.id


class TestPageForOffset:
    def test_maps_offsets_to_pages(self) -> None:
        offsets = [[1, 0], [2, 100], [3, 250]]
        assert page_for_offset(offsets, 0) == 1
        assert page_for_offset(offsets, 99) == 1
        assert page_for_offset(offsets, 100) == 2
        assert page_for_offset(offsets, 300) == 3

    def test_returns_none_without_offsets(self) -> None:
        assert page_for_offset([], 10) is None

    def test_chunk_offsets_map_to_real_pages(self, corpus: Path) -> None:
        """Chunk offsets plus page offsets are enough to cite a page number."""
        import asyncio

        from recall.core.chunking.fixed import FixedSizeChunker

        async def load():
            connector = PDFConnector(root=corpus)
            return await connector.fetch((await connector.discover())[0])

        document = asyncio.run(load())
        chunks = FixedSizeChunker(chunk_size=12, overlap=2).chunk(document)
        offsets = document.metadata["page_offsets"]
        pages = [page_for_offset(offsets, c.start_char or 0) for c in chunks]
        assert all(page is not None for page in pages)
        assert pages == sorted(pages)  # chunks advance through the document
