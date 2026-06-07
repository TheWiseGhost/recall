"""Filesystem connector for ``.txt``, ``.md``, ``.json`` and ``.html``.

Discovery is cheap: it stats files and uses ``(size, mtime)`` as a provisional
change signal. The authoritative signal is still the content checksum computed
after fetch, so a touched-but-unchanged file is detected as unchanged.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from recall.connectors.base import connector_registry
from recall.connectors.text_extraction import (
    extract_html,
    extract_json,
    extract_markdown,
    extract_text,
)
from recall.core.errors import DocumentParseError, UnsupportedFileTypeError
from recall.core.models import Document, SourceItem, SourceType

_EXTRACTORS = {
    ".txt": extract_text,
    ".text": extract_text,
    ".log": extract_text,
    ".md": extract_markdown,
    ".markdown": extract_markdown,
    ".rst": extract_text,
    ".json": extract_json,
    ".jsonl": extract_text,
    ".html": extract_html,
    ".htm": extract_html,
}

DEFAULT_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".json", ".html")

DEFAULT_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
    }
)

# Guardrail against accidentally embedding a multi-gigabyte log file.
DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024


@connector_registry.decorator("filesystem")
class FilesystemConnector:
    """Walks a directory tree (or a single file) and emits documents."""

    source_type = SourceType.FILESYSTEM

    def __init__(
        self,
        *,
        root: str | Path,
        extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        recursive: bool = True,
        exclude_dirs: Iterable[str] = DEFAULT_EXCLUDE_DIRS,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        follow_symlinks: bool = False,
        encoding: str = "utf-8",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.extensions = tuple(self._normalize_ext(e) for e in extensions)
        self.recursive = recursive
        self.exclude_dirs = frozenset(exclude_dirs)
        self.max_file_bytes = max_file_bytes
        self.follow_symlinks = follow_symlinks
        self.encoding = encoding

    @staticmethod
    def _normalize_ext(ext: str) -> str:
        ext = ext.strip().lower()
        return ext if ext.startswith(".") else f".{ext}"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    # -- discovery --------------------------------------------------------
    async def discover(self) -> list[SourceItem]:
        return await asyncio.to_thread(self._discover_sync)

    def _discover_sync(self) -> list[SourceItem]:
        if not self.root.exists():
            raise DocumentParseError(f"Path does not exist: {self.root}")

        paths = [self.root] if self.root.is_file() else list(self._walk(self.root))
        items: list[SourceItem] = []
        for path in sorted(paths):
            if not self.supports(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size > self.max_file_bytes:
                continue
            items.append(
                SourceItem(
                    source_id=self._source_id(path),
                    source_type=self.source_type,
                    uri=path.as_uri(),
                    title=path.stem,
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                    metadata={
                        "path": str(path),
                        "file_type": path.suffix.lower().lstrip("."),
                        "size_bytes": stat.st_size,
                        "directory": str(path.parent),
                    },
                )
            )
        return items

    def _walk(self, root: Path) -> Iterable[Path]:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=self.follow_symlinks):
            # Prune in place so os.walk never descends into excluded trees.
            dirnames[:] = [
                d for d in dirnames if d not in self.exclude_dirs and not d.startswith(".")
            ]
            for filename in filenames:
                yield Path(dirpath) / filename
            if not self.recursive:
                dirnames[:] = []

    def _source_id(self, path: Path) -> str:
        """Path relative to the root, so moving the corpus does not renumber it."""
        # When the root *is* the file, relative_to would yield "."; use the
        # filename so ingesting a directory and ingesting one file out of it
        # produce the same source id (and therefore the same document).
        base = self.root.parent if self.root.is_file() else self.root
        try:
            return str(path.relative_to(base))
        except ValueError:
            return str(path)

    # -- fetching ---------------------------------------------------------
    async def fetch(self, item: SourceItem) -> Document:
        return await asyncio.to_thread(self._fetch_sync, item)

    def _fetch_sync(self, item: SourceItem) -> Document:
        path = Path(item.metadata.get("path", ""))
        if not path.is_file():
            raise DocumentParseError(f"File disappeared before fetch: {path}")

        suffix = path.suffix.lower()
        extractor = _EXTRACTORS.get(suffix)
        if extractor is None:
            raise UnsupportedFileTypeError(str(path), suffix)

        try:
            raw = path.read_text(encoding=self.encoding, errors="replace")
        except OSError as exc:
            raise DocumentParseError(f"Could not read {path}: {exc}") from exc

        content, extracted_title = extractor(raw)
        stat = path.stat()
        metadata = dict(item.metadata)
        metadata.update(
            {
                "filename": path.name,
                "extension": suffix,
                "char_count": len(content),
            }
        )
        return Document.create(
            source_id=item.source_id,
            source_type=self.source_type,
            title=extracted_title or item.title or path.stem,
            content=content,
            uri=path.as_uri(),
            metadata=metadata,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        )
