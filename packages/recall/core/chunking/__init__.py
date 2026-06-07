"""Chunking strategies.

A chunker turns one :class:`~recall.core.models.Document` into an ordered list
of :class:`~recall.core.models.Chunk` objects. Strategy choice is one of the
main experimental variables in Recall, so chunkers are resolved by name.
"""

from recall.core.chunking.base import Chunker, ChunkerBase, chunker_registry
from recall.core.chunking.fixed import FixedSizeChunker

# Importing the concrete modules is what populates the registry.
__all__ = [
    "Chunker",
    "ChunkerBase",
    "FixedSizeChunker",
    "chunker_registry",
    "create_chunker",
]


def create_chunker(strategy: str, **kwargs: object) -> Chunker:
    """Instantiate the chunker registered under ``strategy``."""
    return chunker_registry.create(strategy, **kwargs)
