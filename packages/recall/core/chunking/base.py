"""Chunker protocol and shared construction helpers."""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

from recall.core.ids import checksum as _checksum
from recall.core.ids import chunk_id as _chunk_id
from recall.core.models import Chunk, Document
from recall.core.registry import Registry
from recall.core.tokenization import DEFAULT_TOKEN_COUNTER, TokenCounter

Span = tuple[str, int, int]
"""``(text, start_char, end_char)`` — a span of a document, in document order."""


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrievable chunks.

    Asynchronous because chunking is not always pure text manipulation:
    semantic chunking has to embed candidate sentences to find its boundaries,
    and a future LLM-guided chunker would call out too. Strategies that need
    neither simply do not await anything.
    """

    name: str

    async def chunk(self, document: Document) -> list[Chunk]: ...


class ChunkerBase:
    """Convenience base that handles ID derivation and metadata propagation.

    Subclasses implement :meth:`split`, returning ``(content, start, end)``
    triples in document order; the base class turns those into chunks. When
    splitting needs I/O, override :meth:`split_async` instead — it defaults to
    delegating to :meth:`split`, so the common synchronous case stays a single
    plain method.
    """

    name: str = "base"

    def __init__(self, *, token_counter: TokenCounter | None = None) -> None:
        self.token_counter = token_counter or DEFAULT_TOKEN_COUNTER

    # -- to implement -----------------------------------------------------
    def split(self, document: Document) -> list[Span]:
        """Return ``(text, start_char, end_char)`` spans for ``document``."""
        raise NotImplementedError

    async def split_async(self, document: Document) -> list[Span]:
        """Async splitting hook. Defaults to the synchronous :meth:`split`."""
        return self.split(document)

    def params(self) -> dict[str, Any]:
        """Strategy parameters recorded on every chunk, for reproducibility."""
        return {}

    # -- public API -------------------------------------------------------
    async def chunk(self, document: Document) -> list[Chunk]:
        """Split ``document`` and materialise :class:`Chunk` objects."""
        return self.materialize(document, await self.split_async(document))

    def materialize(
        self,
        document: Document,
        spans: list[Span],
        *,
        first_position: int = 0,
        parent_id: uuid.UUID | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Turn spans into chunks, deriving IDs and propagating metadata.

        ``first_position`` and ``parent_id`` exist for hierarchical chunking,
        which emits two levels from one document and must keep positions unique
        across both — chunk IDs fold in the position, so a reused position plus
        identical text would collide.
        """
        params = self.params()
        metadata = self.build_metadata(document, params)
        if extra_metadata:
            metadata = {**metadata, **extra_metadata}
        chunks: list[Chunk] = []
        for offset, (text, start, end) in enumerate(spans):
            position = first_position + offset
            content_checksum = _checksum(text)
            chunks.append(
                Chunk(
                    id=_chunk_id(document.id, position, content_checksum),
                    document_id=document.id,
                    parent_id=parent_id,
                    content=text,
                    metadata=dict(metadata),
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
