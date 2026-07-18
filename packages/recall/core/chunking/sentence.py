"""Sentence-aware chunking.

Fixed-size chunking cuts wherever the token budget runs out, which routinely
severs a sentence — and a half-sentence embeds to something that means neither
half. This packs whole sentences into windows instead, trading exact chunk
sizes for chunks that are self-contained.

Whether that trade helps is a corpus-dependent question, which is why both
strategies exist and are selectable by name.
"""

from __future__ import annotations

from typing import Any

from recall.core.chunking.base import ChunkerBase, Span, chunker_registry
from recall.core.chunking.sentences import split_sentences
from recall.core.errors import ConfigurationError
from recall.core.models import Document
from recall.core.tokenization import TokenCounter


@chunker_registry.decorator("sentence")
class SentenceChunker(ChunkerBase):
    """Packs whole sentences into windows of at most ``chunk_size`` tokens.

    Overlap is counted in *sentences* rather than tokens: the unit that makes
    the strategy worth having is the sentence, and "carry one sentence of
    context forward" is a statement someone can reason about, where "carry 64
    tokens" would cut mid-sentence again and undo the point.

    A single sentence longer than ``chunk_size`` becomes its own oversized
    chunk rather than being split. Splitting it would reintroduce exactly the
    severed-sentence problem; an oversized chunk is visible in ``token_count``,
    and silently truncating it would not be.
    """

    name = "sentence"

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        overlap_sentences: int = 1,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__(token_counter=token_counter)
        if chunk_size <= 0:
            raise ConfigurationError("chunk_size must be positive")
        if overlap_sentences < 0:
            raise ConfigurationError("overlap_sentences must be non-negative")
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def params(self) -> dict[str, Any]:
        return {"chunk_size": self.chunk_size, "overlap_sentences": self.overlap_sentences}

    def split(self, document: Document) -> list[Span]:
        text = document.content
        sentences = split_sentences(text)
        if not sentences:
            return []

        costs = [max(1, self.token_counter.count(sentence)) for sentence, _, _ in sentences]
        spans: list[Span] = []
        start_index = 0
        total = len(sentences)

        while start_index < total:
            budget = 0
            end_index = start_index
            while end_index < total:
                # Always take at least one sentence, even when it alone busts
                # the budget — see the class docstring.
                if budget + costs[end_index] > self.chunk_size and end_index > start_index:
                    break
                budget += costs[end_index]
                end_index += 1

            start_char = sentences[start_index][1]
            end_char = sentences[end_index - 1][2]
            spans.append((text[start_char:end_char], start_char, end_char))

            if end_index >= total:
                break
            # Guarantee forward progress: a window must never restart where it
            # began, however large the overlap is set.
            start_index = max(end_index - self.overlap_sentences, start_index + 1)

        return spans
