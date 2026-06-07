"""A deterministic, dependency-free embedder.

This is not a good semantic model and it is not pretending to be one. It exists
because a retrieval framework needs an embedder that:

* installs with zero extra dependencies,
* is byte-for-byte deterministic across machines and runs,
* is fast enough to embed a corpus inside a unit test.

It implements the hashing trick over character n-grams and word unigrams, so it
does carry real lexical signal — enough for end-to-end plumbing tests and quick
local demos. Do not use it to make quality claims.
"""

from __future__ import annotations

import hashlib
import re

from recall.core.embeddings.base import EmbedderBase, Vector, embedder_registry, l2_normalize

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@embedder_registry.decorator("hash")
class HashingEmbedder(EmbedderBase):
    """Hashing-trick embedder over word unigrams and character trigrams."""

    provider = "hash"

    def __init__(
        self,
        *,
        model: str = "hash-v1",
        dimensions: int = 384,
        char_ngram: int = 3,
        batch_size: int = 256,
    ) -> None:
        super().__init__(model=model, dimensions=dimensions, batch_size=batch_size)
        self.char_ngram = char_ngram

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> list[Vector]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Vector:
        vector = [0.0] * self._dimensions
        lowered = text.lower()

        for word in _WORD_RE.findall(lowered):
            self._add(vector, f"w:{word}", 1.0)
            if len(word) > self.char_ngram:
                padded = f"^{word}$"
                for i in range(len(padded) - self.char_ngram + 1):
                    self._add(vector, f"c:{padded[i : i + self.char_ngram]}", 0.35)

        return l2_normalize(vector)

    def _add(self, vector: Vector, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % self._dimensions
        # Signed hashing keeps collisions from systematically inflating scores.
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * weight
