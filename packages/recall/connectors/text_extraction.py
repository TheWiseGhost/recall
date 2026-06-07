"""Plain-text extraction for the formats the filesystem connector supports.

HTML is handled with the standard library rather than BeautifulSoup: the job is
"strip tags, drop script/style, keep readable text", which does not justify a
dependency.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any

_WHITESPACE_RUN = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")

_SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
_BLOCK_TAGS = {
    "p",
    "div",
    "br",
    "li",
    "tr",
    "section",
    "article",
    "header",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
    "table",
    "ul",
    "ol",
}


class _HTMLTextExtractor(HTMLParser):
    """Collects visible text and the document title."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str | None = None
        self._skip_depth = 0
        self._in_title = False

    def _break(self) -> None:
        """Emit a line break, but never two in a row.

        ``</p><p>`` fires on both the closing and opening tag; without this the
        output would gain a blank line at every block boundary.
        """
        if self.parts and self.parts[-1] == "\n":
            return
        self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self._break()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self._break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title = (self.title or "") + data.strip()
            return
        self.parts.append(data)


def normalize_whitespace(text: str) -> str:
    """Collapse horizontal whitespace and cap consecutive blank lines at one."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RUN.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def extract_html(raw: str) -> tuple[str, str | None]:
    """Return ``(text, title)`` for an HTML document."""
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    parser.close()
    return normalize_whitespace("".join(parser.parts)), (parser.title or None)


def extract_json(raw: str) -> tuple[str, str | None]:
    """Flatten JSON into ``key: value`` lines so it embeds meaningfully.

    Malformed JSON falls back to the raw text rather than failing ingestion.
    """
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError:
        return normalize_whitespace(raw), None

    lines: list[str] = []
    _flatten(data, "", lines)
    title = None
    if isinstance(data, dict):
        for key in ("title", "name", "subject", "heading"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                title = value.strip()
                break
    return "\n".join(lines), title


def _flatten(value: Any, prefix: str, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _flatten(item, f"{prefix}[{index}]", out)
    elif value is not None:
        out.append(f"{prefix}: {value}" if prefix else str(value))


def extract_markdown(raw: str) -> tuple[str, str | None]:
    """Keep Markdown as-is; only lift the first ATX heading as a title."""
    title: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip()
            if candidate:
                title = candidate
                break
        if stripped:
            break
    return normalize_whitespace(raw), title


def extract_text(raw: str) -> tuple[str, str | None]:
    """Plain text passes through with whitespace normalisation only."""
    return normalize_whitespace(raw), None
