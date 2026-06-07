"""PDF connector backed by PyMuPDF.

Why PyMuPDF: it is the fastest pure-extraction option with good layout fidelity
and no external binaries (unlike poppler/pdftotext), and it exposes page-level
text, which Recall needs so every chunk can carry a page number.

Page boundaries are recorded as ``page_offsets`` in document metadata:
``[[page_number, start_char], ...]``. Chunkers preserve document metadata, so a
chunk's character offsets can be mapped back to a page for citation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from recall.connectors.base import connector_registry
from recall.connectors.filesystem import DEFAULT_EXCLUDE_DIRS, FilesystemConnector
from recall.connectors.text_extraction import normalize_whitespace
from recall.core.errors import DocumentParseError
from recall.core.models import Document, SourceItem, SourceType

DEFAULT_MAX_PDF_BYTES = 100 * 1024 * 1024


@connector_registry.decorator("pdf")
class PDFConnector(FilesystemConnector):
    """Discovers ``.pdf`` files on disk and extracts per-page text."""

    source_type = SourceType.PDF

    def __init__(
        self,
        *,
        root: str | Path,
        recursive: bool = True,
        exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
        max_file_bytes: int = DEFAULT_MAX_PDF_BYTES,
        follow_symlinks: bool = False,
        max_pages: int | None = None,
        extensions: Sequence[str] = (".pdf",),
    ) -> None:
        super().__init__(
            root=root,
            extensions=extensions,
            recursive=recursive,
            exclude_dirs=exclude_dirs,
            max_file_bytes=max_file_bytes,
            follow_symlinks=follow_symlinks,
        )
        self.max_pages = max_pages

    async def fetch(self, item: SourceItem) -> Document:
        return await asyncio.to_thread(self._fetch_sync, item)

    def _fetch_sync(self, item: SourceItem) -> Document:
        path = Path(item.metadata.get("path", ""))
        if not path.is_file():
            raise DocumentParseError(f"File disappeared before fetch: {path}")

        try:
            import fitz  # PyMuPDF
        except ImportError as exc:  # pragma: no cover - depends on env
            raise DocumentParseError(
                "PyMuPDF is not installed. Install it with: pip install 'recall[pdf]'"
            ) from exc

        try:
            document = fitz.open(path)
        except Exception as exc:
            raise DocumentParseError(f"Could not open PDF {path}: {exc}") from exc

        try:
            pdf_metadata = dict(document.metadata or {})
            page_count = document.page_count
            limit = page_count if self.max_pages is None else min(page_count, self.max_pages)

            parts: list[str] = []
            page_offsets: list[list[int]] = []
            cursor = 0
            for page_number in range(limit):
                text = normalize_whitespace(document.load_page(page_number).get_text("text"))
                if not text:
                    continue
                page_offsets.append([page_number + 1, cursor])
                parts.append(text)
                cursor += len(text) + 2  # the "\n\n" join separator
        finally:
            document.close()

        content = "\n\n".join(parts)
        if not content.strip():
            raise DocumentParseError(
                f"No extractable text in {path}. It may be a scanned PDF; OCR is not "
                "supported yet (TODO / FUTURE)."
            )

        stat = path.stat()
        metadata = dict(item.metadata)
        metadata.update(
            {
                "filename": path.name,
                "extension": ".pdf",
                "file_type": "pdf",
                "page_count": page_count,
                "pages_extracted": len(page_offsets),
                "page_offsets": page_offsets,
                "char_count": len(content),
            }
        )
        for key in ("author", "subject", "keywords", "creator", "producer"):
            value = pdf_metadata.get(key)
            if isinstance(value, str) and value.strip():
                metadata[key] = value.strip()

        pdf_title = pdf_metadata.get("title")
        title = pdf_title.strip() if isinstance(pdf_title, str) and pdf_title.strip() else path.stem

        return Document.create(
            source_id=item.source_id,
            source_type=self.source_type,
            title=title,
            content=content,
            uri=path.as_uri(),
            metadata=metadata,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        )


def page_for_offset(page_offsets: Sequence[Sequence[int]], char_offset: int) -> int | None:
    """Map a character offset back to a 1-based page number.

    ``page_offsets`` is the ``[[page, start_char], ...]`` list stored in PDF
    document metadata.
    """
    page: int | None = None
    for entry in page_offsets:
        if len(entry) < 2:
            continue
        number, start = int(entry[0]), int(entry[1])
        if start <= char_offset:
            page = number
        else:
            break
    return page
