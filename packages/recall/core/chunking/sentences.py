"""Dependency-free sentence segmentation.

Both the sentence and semantic chunkers need sentence boundaries, so the rules
live here once rather than drifting between them.

This is a heuristic, not a trained model. Recall does not pull in NLTK or spaCy
for it: both are large, both want a download step, and the failure mode of a
missed boundary here is a slightly differently-shaped chunk rather than a wrong
answer. If a corpus needs better segmentation, pass a different splitter to the
chunker — that is why this is a function and not inlined.

**Known limits**, in the spirit of not making anyone rediscover them:

* Abbreviations are handled by an explicit list. One that is not on it (a
  domain-specific ``Prot.``, ``Sec.``) will end a sentence early.
* Decimal numbers, ellipses and initials are handled; ``etc.`` at the genuine
  end of a sentence is not, and will merge with the next.
* No support for languages that do not delimit sentences with ``.?!``.
"""

from __future__ import annotations

import re

Span = tuple[str, int, int]

# Tokens that end in a period without ending a sentence. Matched
# case-sensitively against the word immediately before the period.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "cf",
        "al",
        "fig",
        "eq",
        "no",
        "vol",
        "ch",
        "pp",
        "approx",
        "inc",
        "ltd",
        "co",
        "dept",
        "est",
        "min",
        "max",
        "ca",
    }
)

# A blank line is a hard boundary regardless of punctuation: it separates
# paragraphs, list items and headings, none of which reliably end in a period.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")

# Sentence-final punctuation, optionally followed by a closing quote or
# bracket, then whitespace. The class includes typographic quotes because real
# prose uses them and a boundary before one would leave the quote orphaned at
# the start of the next chunk.
_BOUNDARY = re.compile(r'([.!?]+)(["\'\)\]»”’]*)(\s+)')  # noqa: RUF001

# The word immediately preceding a candidate boundary.
_PRECEDING_WORD = re.compile(r"([\w.]+)$", re.UNICODE)


def _is_false_boundary(text: str, punctuation_start: int) -> bool:
    """True when a period does not actually end a sentence."""
    before = text[:punctuation_start]
    match = _PRECEDING_WORD.search(before)
    if match is None:
        return False
    word = match.group(1).lower().rstrip(".")

    if word in _ABBREVIATIONS:
        return True
    # A single letter: an initial ("J. R. R. Tolkien"), not a sentence end.
    if len(word) == 1 and word.isalpha():
        return True
    # A bare number: "1. First item", "3.14".
    return bool(word.isdigit())


def split_sentences(text: str) -> list[Span]:
    """Split ``text`` into ``(sentence, start, end)`` spans in document order.

    Spans are stripped of surrounding whitespace and cover the whole text
    except that whitespace, so ``start``/``end`` always index back into the
    original document.
    """
    if not text.strip():
        return []

    boundaries: list[int] = []
    for match in _PARAGRAPH_BREAK.finditer(text):
        boundaries.append(match.start())

    for match in _BOUNDARY.finditer(text):
        if _is_false_boundary(text, match.start(1)):
            continue
        # Cut after the punctuation and any closing quote, before the space.
        boundaries.append(match.end(2))

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for boundary in sorted(set(boundaries)):
        if boundary <= cursor:
            continue
        ranges.append((cursor, boundary))
        cursor = boundary
    if cursor < len(text):
        ranges.append((cursor, len(text)))

    stripped: list[Span] = []
    for start, end in ranges:
        raw = text[start:end]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        real_start, real_end = start + lead, end - trail
        if real_end > real_start:
            stripped.append((text[real_start:real_end], real_start, real_end))
    return stripped
