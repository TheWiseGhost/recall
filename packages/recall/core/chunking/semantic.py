"""Semantic chunking: boundaries where the meaning shifts.

Fixed-size and sentence chunking both cut on a budget. Semantic chunking cuts
where the text changes subject: it embeds each sentence, measures the cosine
distance between consecutive sentences, and breaks where that distance spikes.
A chunk then covers one topic instead of one token count.

The cost is honest and worth stating: this embeds every sentence in the corpus
*at ingest time*, on top of embedding the chunks it produces. On a paid API
that roughly doubles ingestion cost. Whether the retrieval quality pays for it
is exactly the kind of question Recall exists to answer rather than assume —
experiment 001.

The breakpoint threshold is a **percentile of the distances in this document**,
not an absolute distance. Absolute cosine distances are not comparable across
embedding models or across documents of different styles, so a fixed threshold
would silently mean something different for every model someone swaps in.
"""

from __future__ import annotations

import math
from typing import Any

from recall.core.chunking.base import ChunkerBase, Span, chunker_registry
from recall.core.chunking.sentences import split_sentences
from recall.core.embeddings.base import Embedder, Vector
from recall.core.errors import ConfigurationError
from recall.core.models import Document
from recall.core.tokenization import TokenCounter


def cosine_distance(left: Vector, right: Vector) -> float:
    """``1 - cosine similarity``, in ``[0, 2]``. Zero vectors are maximally far."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0.0:
        return 1.0
    return 1.0 - dot / norm


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of ``values``. ``fraction`` in ``[0, 1]``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@chunker_registry.decorator("semantic")
class SemanticChunker(ChunkerBase):
    """Breaks where consecutive sentences are semantically far apart.

    ``buffer_size`` widens what gets embedded: with ``buffer_size=1`` each
    sentence is embedded together with its immediate neighbours. A lone
    sentence often embeds to something noisy — pronouns, a bare clause — and
    comparing two noisy vectors produces spurious breakpoints. Widening the
    window is what makes the distance signal usable.

    ``max_chunk_size`` is a backstop, not a target: a long stretch of on-topic
    prose would otherwise become one enormous chunk that no retriever can rank
    usefully. When a group exceeds it, it is cut at the next-largest distance
    inside the group rather than at an arbitrary token offset.
    """

    name = "semantic"

    def __init__(
        self,
        *,
        embedder: Embedder,
        breakpoint_percentile: float = 0.95,
        buffer_size: int = 1,
        max_chunk_size: int = 1024,
        min_sentences: int = 1,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__(token_counter=token_counter)
        if not 0.0 < breakpoint_percentile < 1.0:
            raise ConfigurationError(
                f"breakpoint_percentile must be strictly between 0 and 1, "
                f"got {breakpoint_percentile}"
            )
        if buffer_size < 0:
            raise ConfigurationError("buffer_size must be non-negative")
        if max_chunk_size <= 0:
            raise ConfigurationError("max_chunk_size must be positive")
        if min_sentences < 1:
            raise ConfigurationError("min_sentences must be at least 1")
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.max_chunk_size = max_chunk_size
        self.min_sentences = min_sentences

    def params(self) -> dict[str, Any]:
        return {
            "breakpoint_percentile": self.breakpoint_percentile,
            "buffer_size": self.buffer_size,
            "max_chunk_size": self.max_chunk_size,
            "min_sentences": self.min_sentences,
            # The boundaries depend on the model, so a chunk produced by one
            # model is not reproducible with another. Record which was used.
            "embedding_model": self.embedder.info.key,
        }

    def _windows(self, sentences: list[Span]) -> list[str]:
        """Each sentence widened by ``buffer_size`` neighbours on both sides."""
        if self.buffer_size == 0:
            return [text for text, _, _ in sentences]
        windows: list[str] = []
        for index in range(len(sentences)):
            low = max(0, index - self.buffer_size)
            high = min(len(sentences), index + self.buffer_size + 1)
            windows.append(" ".join(sentences[position][0] for position in range(low, high)))
        return windows

    async def split_async(self, document: Document) -> list[Span]:
        text = document.content
        sentences = split_sentences(text)
        if len(sentences) <= self.min_sentences:
            # Nothing to decide: one span covering everything.
            return [(s, start, end) for s, start, end in sentences[:1]] or []

        vectors = await self.embedder.embed_documents(self._windows(sentences))
        distances = [
            cosine_distance(vectors[index], vectors[index + 1]) for index in range(len(vectors) - 1)
        ]
        threshold = percentile(distances, self.breakpoint_percentile)

        # `distances[i]` is the gap between sentence i and i+1, so a breakpoint
        # at i means "start a new chunk at sentence i+1".
        breakpoints = {
            index + 1 for index, distance in enumerate(distances) if distance > threshold
        }
        spans: list[Span] = []
        for low, high in self._group(sentences, breakpoints, distances):
            start, end = sentences[low][1], sentences[high - 1][2]
            spans.append((text[start:end], start, end))
        return spans

    def _group(
        self, sentences: list[Span], breakpoints: set[int], distances: list[float]
    ) -> list[tuple[int, int]]:
        """Sentence index ranges, honouring breakpoints then ``max_chunk_size``."""
        raw: list[tuple[int, int]] = []
        start = 0
        for index in range(1, len(sentences) + 1):
            at_boundary = index == len(sentences) or index in breakpoints
            long_enough = index - start >= self.min_sentences or index == len(sentences)
            if at_boundary and long_enough:
                raw.append((start, index))
                start = index
        if start < len(sentences):
            raw.append((start, len(sentences)))

        final: list[tuple[int, int]] = []
        for group in raw:
            final.extend(self._enforce_max(sentences, group, distances))
        return final

    def _enforce_max(
        self, sentences: list[Span], group: tuple[int, int], distances: list[float]
    ) -> list[tuple[int, int]]:
        """Recursively cut an oversized group at its weakest internal seam."""
        low, high = group
        tokens = sum(self.token_counter.count(sentences[index][0]) for index in range(low, high))
        if tokens <= self.max_chunk_size or high - low <= 1:
            return [group]

        # Cut at the largest remaining distance inside the group, so the split
        # still lands on a topic seam rather than a token offset.
        interior = range(low, high - 1)
        cut = max(interior, key=lambda index: distances[index]) + 1
        if cut <= low or cut >= high:  # pragma: no cover - defensive
            cut = low + (high - low) // 2
        return self._enforce_max(sentences, (low, cut), distances) + self._enforce_max(
            sentences, (cut, high), distances
        )
