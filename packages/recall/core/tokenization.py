"""Token counting.

Chunk sizes are expressed in tokens, but the *right* tokenizer depends on the
embedding model. Rather than pull in ``tiktoken`` (and a network download) for
the MVP, Recall counts tokens through a protocol with a dependency-free default
that approximates subword tokenizers closely enough for chunk sizing.

If you need exactness for a specific model, implement :class:`TokenCounter` and
pass it to the chunker.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

# Words, standalone punctuation, and runs of whitespace-free symbols.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Subword tokenizers split long words into pieces. Empirically ~4 characters
# per token for English prose; we apply that only to words long enough for a
# BPE merge to have split them.
_CHARS_PER_SUBWORD = 4
_MIN_SUBWORD_SPLIT_LEN = 6


@runtime_checkable
class TokenCounter(Protocol):
    """Estimates the number of tokens in a piece of text."""

    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    """Dependency-free approximation of a subword tokenizer.

    Deterministic and fast, which also makes chunking unit tests stable.
    """

    name = "approx"

    def count(self, text: str) -> int:
        total = 0
        for match in _TOKEN_RE.finditer(text):
            token = match.group()
            if len(token) >= _MIN_SUBWORD_SPLIT_LEN:
                total += max(1, round(len(token) / _CHARS_PER_SUBWORD))
            else:
                total += 1
        return total


class WhitespaceTokenCounter:
    """Counts whitespace-delimited words. Useful when you want 1 token == 1 word."""

    name = "whitespace"

    def count(self, text: str) -> int:
        return len(text.split())


DEFAULT_TOKEN_COUNTER: TokenCounter = ApproxTokenCounter()
