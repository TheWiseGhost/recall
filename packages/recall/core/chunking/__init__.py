"""Chunking strategies.

A chunker turns one :class:`~recall.core.models.Document` into an ordered list
of :class:`~recall.core.models.Chunk` objects. Strategy choice is one of the
main experimental variables in Recall, so chunkers are resolved by name.

Importing the concrete modules is what populates the registry.
"""

from recall.core.chunking.base import Chunker, ChunkerBase, Span, chunker_registry
from recall.core.chunking.fixed import FixedSizeChunker
from recall.core.chunking.hierarchical import HierarchicalChunker
from recall.core.chunking.semantic import SemanticChunker
from recall.core.chunking.sentence import SentenceChunker
from recall.core.chunking.sentences import split_sentences

__all__ = [
    "Chunker",
    "ChunkerBase",
    "FixedSizeChunker",
    "HierarchicalChunker",
    "SemanticChunker",
    "SentenceChunker",
    "Span",
    "chunker_registry",
    "create_chunker",
    "split_sentences",
]


def create_chunker(strategy: str, **kwargs: object) -> Chunker:
    """Instantiate the chunker registered under ``strategy``."""
    return chunker_registry.create(strategy, **kwargs)
