"""Parent/child chunking.

Retrieval wants small chunks: a precise match is easier to find in a focused
span, and a big chunk dilutes its own embedding. Answering wants large ones: a
matched sentence is often useless without the paragraph around it.

Hierarchical chunking refuses the trade by emitting both. Small child chunks
are the retrieval units; each carries ``parent_id`` pointing at the larger span
it came from, so a retrieved child can be expanded to its parent before the
text is used.

The expansion step itself is context selection and lands in Milestone 4. What
this chunker does is record the structure — ``chunks.parent_id`` has existed
since the initial migration for exactly this.
"""

from __future__ import annotations

from typing import Any

from recall.core.chunking.base import ChunkerBase, Span, chunker_registry
from recall.core.chunking.fixed import FixedSizeChunker
from recall.core.errors import ConfigurationError
from recall.core.models import Chunk, Document
from recall.core.tokenization import TokenCounter


@chunker_registry.decorator("hierarchical")
class HierarchicalChunker(ChunkerBase):
    """Emits large parent chunks and the small child chunks inside them.

    Children are produced by re-chunking each parent's text, so a child is
    always wholly contained in its parent and character offsets stay valid
    against the original document.

    Positions are assigned children-first in document order, then parents.
    Children are the retrieval units and their positions have to be their
    reading order, because neighbour expansion and offset arithmetic depend on
    it. Positions are unique across both levels because a chunk's ID folds in
    its position — a parent and its only child can have byte-identical text,
    and reusing the position would collide their IDs.
    """

    name = "hierarchical"

    def __init__(
        self,
        *,
        parent_chunk_size: int = 2048,
        chunk_size: int = 512,
        overlap: int = 64,
        token_counter: TokenCounter | None = None,
    ) -> None:
        super().__init__(token_counter=token_counter)
        if parent_chunk_size <= chunk_size:
            raise ConfigurationError(
                f"parent_chunk_size ({parent_chunk_size}) must be larger than "
                f"chunk_size ({chunk_size}); otherwise the hierarchy is flat"
            )
        self.parent_chunk_size = parent_chunk_size
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._parents = FixedSizeChunker(
            chunk_size=parent_chunk_size, overlap=0, token_counter=self.token_counter
        )
        # Parents do not overlap: overlapping parents would assign the same
        # text to two hierarchies and double-count it on expansion.
        self._children = FixedSizeChunker(
            chunk_size=chunk_size, overlap=overlap, token_counter=self.token_counter
        )

    def params(self) -> dict[str, Any]:
        return {
            "parent_chunk_size": self.parent_chunk_size,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }

    def split(self, document: Document) -> list[Span]:
        """The child spans alone. :meth:`chunk` is what emits both levels."""
        return [span for _, children in self._hierarchy(document) for span in children]

    def _hierarchy(self, document: Document) -> list[tuple[Span, list[Span]]]:
        """``(parent_span, child_spans)`` pairs, offsets absolute in the document."""
        pairs: list[tuple[Span, list[Span]]] = []
        for parent_text, parent_start, parent_end in self._parents.split(document):
            # Re-chunk the parent's own text, then shift the offsets back into
            # the document's coordinate space.
            inner = document.model_copy(update={"content": parent_text})
            children = [
                (text, parent_start + start, parent_start + end)
                for text, start, end in self._children.split(inner)
            ]
            pairs.append(((parent_text, parent_start, parent_end), children or []))
        return pairs

    async def chunk(self, document: Document) -> list[Chunk]:
        hierarchy = self._hierarchy(document)
        if not hierarchy:
            return []

        child_count = sum(len(children) for _, children in hierarchy)

        # Parents are materialised first so their IDs exist to be referenced,
        # but they are positioned after the children.
        parents: list[Chunk] = []
        for index, (parent_span, children) in enumerate(hierarchy):
            parents.extend(
                self.materialize(
                    document,
                    [parent_span],
                    first_position=child_count + index,
                    extra_metadata={
                        "chunk_level": "parent",
                        "parent_index": index,
                        "child_count": len(children),
                    },
                )
            )

        children_chunks: list[Chunk] = []
        position = 0
        for index, (_, children) in enumerate(hierarchy):
            children_chunks.extend(
                self.materialize(
                    document,
                    children,
                    first_position=position,
                    parent_id=parents[index].id,
                    extra_metadata={"chunk_level": "child", "parent_index": index},
                )
            )
            position += len(children)

        # Parents lead the returned list even though they are positioned after
        # the children. The storage layer inserts in list order, and
        # `chunks.parent_id` is a self-referencing foreign key: a row cannot
        # point at one that has not been written yet.
        return [*parents, *children_chunks]
