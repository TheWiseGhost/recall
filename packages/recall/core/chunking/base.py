"""Chunker protocol and shared construction helpers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from recall.core.ids import checksum as _checksum
from recall.core.ids import chunk_id as _chunk_id
from recall.core.models import Chunk, Document
from recall.core.registry import Registry
from recall.core.tokenization import DEFAULT_TOKEN_COUNTER, TokenCounter


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrievable chunks."""

    name: str

    def chunk(self, document: Document) -> list[Chunk]: ...


class ChunkerBase:
    """Convenience base that handles ID derivation and metadata propagation.

    Subclasses implement :meth:`split`, returning ``(content, start, end)``
    triples in document order; the base class turns those into chunks.
    """

    name: str = "base"

    def __init__(self, *, token_counter: TokenCounter | None = None) -> None:
        self.token_counter = token_counter or DEFAULT_TOKEN_COUNTER

    # -- to implement -----------------------------------------------------
    def split(self, document: Document) -> list[tuple[str, int, int]]:
        """Return ``(text, start_char, end_char)`` spans for ``document``."""
        raise NotImplementedError

    def params(self) -> dict[str, Any]:
        """Strategy parameters recorded on every chunk, for reproducibility."""
        return {}

    # -- public API -------------------------------------------------------
    def chunk(self, document: Document) -> list[Chunk]:
        """Split ``document`` and materialise :class:`Chunk` objects."""
        params = self.params()
        chunks: list[Chunk] = []
        for position, (text, start, end) in enumerate(self.split(document)):
            content_checksum = _checksum(text)
            chunks.append(
                Chunk(
                    id=_chunk_id(document.id, position, content_checksum),
                    document_id=document.id,
                    parent_id=None,
                    content=text,
                    metadata=self.build_metadata(document, params),
                    position=position,
                    token_count=self.token_counter.count(text),
                    checksum=content_checksum,
                    start_char=start,
                    end_char=end,
                )
            )
        return chunks

    def build_metadata(self, document: Document, params: dict[str, Any]) -> dict[str, Any]:
        """Chunk metadata: source metadata first, then chunker provenance.

        Source keys are preserved so metadata filtering works against chunks
        without a join back to the document.
        """
        metadata: dict[str, Any] = dict(document.metadata)
        metadata["chunker"] = self.name
        if params:
            metadata["chunker_params"] = params
        return metadata


chunker_registry: Registry[Chunker] = Registry("chunker")
