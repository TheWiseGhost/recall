"""Fixed-size chunking with configurable overlap.

The baseline strategy every other strategy is measured against. Windows are
sized in *tokens* (per the configured :class:`TokenCounter`) but cut on token
boundaries in the original text, so chunk content never splits a word and
character offsets always map back into the source document.
"""

from __future__ import annotations

import re
from typing import Any

from recall.core.chunking.base import ChunkerBase, chunker_registry
from recall.core.errors import ConfigurationError
from recall.core.models import Document
from recall.core.tokenization import TokenCounter

_TOKEN_SPAN_RE = re.compile(r"\S+", re.UNICODE)


@chunker_registry.decorator("fixed")
class FixedSizeChunker(ChunkerBase):
    """Sliding window of ``chunk_size`` tokens advancing by ``chunk_size - overlap``."""

    name = "fixed"

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        overlap: int = 64,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__(token_counter=token_counter)
        if chunk_size <= 0:
            raise ConfigurationError("chunk_size must be positive")
        if overlap < 0:
            raise ConfigurationError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ConfigurationError(
                f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}); "
                "otherwise the window never advances"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def params(self) -> dict[str, Any]:
        return {"chunk_size": self.chunk_size, "overlap": self.overlap}

    def split(self, document: Document) -> list[tuple[str, int, int]]:
        text = document.content
        # (start, end, token_weight) for each whitespace-delimited word.
        words: list[tuple[int, int, int]] = [
            (m.start(), m.end(), max(1, self.token_counter.count(m.group())))
            for m in _TOKEN_SPAN_RE.finditer(text)
        ]
        if not words:
            return []

        spans: list[tuple[str, int, int]] = []
        start_index = 0
        total = len(words)

        while start_index < total:
            budget = 0
            end_index = start_index
            while end_index < total:
                weight = words[end_index][2]
                # Always take at least one word, even if it alone exceeds the budget.
                if budget + weight > self.chunk_size and end_index > start_index:
                    break
                budget += weight
                end_index += 1

            start_char = words[start_index][0]
            end_char = words[end_index - 1][1]
            spans.append((text[start_char:end_char], start_char, end_char))

            if end_index >= total:
                break

            start_index = self._next_start(words, start_index, end_index)

        return spans

    def _next_start(
        self, words: list[tuple[int, int, int]], start_index: int, end_index: int
    ) -> int:
        """Walk back from ``end_index`` until ``overlap`` tokens are covered."""
        if self.overlap == 0:
            return end_index
        carried = 0
        cursor = end_index
        while cursor > start_index + 1 and carried < self.overlap:
            carried += words[cursor - 1][2]
            cursor -= 1
        # Guarantee forward progress: a window must never restart where it began.
        return max(cursor, start_index + 1)
