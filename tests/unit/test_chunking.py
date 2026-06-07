"""Fixed-size chunking."""

from __future__ import annotations

import pytest

from recall.core.chunking import chunker_registry, create_chunker
from recall.core.chunking.fixed import FixedSizeChunker
from recall.core.errors import ConfigurationError
from recall.core.models import Document, SourceType
from recall.core.tokenization import WhitespaceTokenCounter


def make_document(content: str, **metadata: object) -> Document:
    return Document.create(
        source_id="doc.md",
        source_type=SourceType.FILESYSTEM,
        title="Doc",
        content=content,
        uri="file:///doc.md",
        metadata=dict(metadata),
    )


def words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


class TestConfiguration:
    def test_rejects_overlap_at_or_above_chunk_size(self) -> None:
        with pytest.raises(ConfigurationError, match="never advances"):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(ConfigurationError):
            FixedSizeChunker(chunk_size=0)

    def test_is_registered(self) -> None:
        assert "fixed" in chunker_registry
        assert isinstance(create_chunker("fixed", chunk_size=32, overlap=4), FixedSizeChunker)


class TestSplitting:
    def test_empty_document_produces_no_chunks(self) -> None:
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        assert chunker.chunk(make_document("   \n\n  ")) == []

    def test_short_document_is_one_chunk(self) -> None:
        chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        chunks = chunker.chunk(make_document("just a few words"))
        assert len(chunks) == 1
        assert chunks[0].content == "just a few words"
        assert chunks[0].position == 0

    def test_respects_chunk_size(self) -> None:
        chunker = FixedSizeChunker(chunk_size=10, overlap=0, token_counter=WhitespaceTokenCounter())
        chunks = chunker.chunk(make_document(words(35)))
        assert [len(c.content.split()) for c in chunks] == [10, 10, 10, 5]

    def test_overlap_repeats_trailing_tokens(self) -> None:
        chunker = FixedSizeChunker(chunk_size=10, overlap=3, token_counter=WhitespaceTokenCounter())
        chunks = chunker.chunk(make_document(words(20)))
        first = chunks[0].content.split()
        second = chunks[1].content.split()
        assert first[-3:] == second[:3]

    def test_positions_are_sequential(self) -> None:
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        chunks = chunker.chunk(make_document(words(60)))
        assert [c.position for c in chunks] == list(range(len(chunks)))

    def test_offsets_map_back_into_the_source(self) -> None:
        content = words(40)
        chunker = FixedSizeChunker(chunk_size=8, overlap=2)
        for chunk in chunker.chunk(make_document(content)):
            assert chunk.start_char is not None and chunk.end_char is not None
            assert content[chunk.start_char : chunk.end_char] == chunk.content

    def test_never_splits_a_word(self) -> None:
        chunker = FixedSizeChunker(chunk_size=5, overlap=1)
        for chunk in chunker.chunk(make_document(words(50))):
            for token in chunk.content.split():
                assert token.startswith("w") and token[1:].isdigit()

    def test_covers_every_token(self) -> None:
        content = words(53)
        chunker = FixedSizeChunker(chunk_size=7, overlap=2, token_counter=WhitespaceTokenCounter())
        seen: set[str] = set()
        for chunk in chunker.chunk(make_document(content)):
            seen.update(chunk.content.split())
        assert seen == set(content.split())

    def test_terminates_on_a_single_oversized_token(self) -> None:
        """One word longer than the budget must still be emitted, once."""
        content = "a" * 5000
        chunker = FixedSizeChunker(chunk_size=16, overlap=4)
        chunks = chunker.chunk(make_document(content))
        assert len(chunks) == 1
        assert chunks[0].content == content

    def test_zero_overlap_does_not_repeat(self) -> None:
        chunker = FixedSizeChunker(chunk_size=5, overlap=0, token_counter=WhitespaceTokenCounter())
        chunks = chunker.chunk(make_document(words(20)))
        rendered = [c.content.split() for c in chunks]
        flattened = [token for group in rendered for token in group]
        assert len(flattened) == len(set(flattened)) == 20


class TestChunkMetadata:
    def test_source_metadata_is_preserved(self) -> None:
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        chunks = chunker.chunk(make_document(words(30), file_type="md", author="docs"))
        assert all(c.metadata["file_type"] == "md" for c in chunks)
        assert all(c.metadata["author"] == "docs" for c in chunks)

    def test_records_strategy_provenance(self) -> None:
        chunker = FixedSizeChunker(chunk_size=10, overlap=2)
        chunk = chunker.chunk(make_document(words(30)))[0]
        assert chunk.metadata["chunker"] == "fixed"
        assert chunk.metadata["chunker_params"] == {"chunk_size": 10, "overlap": 2}

    def test_token_counts_are_recorded(self) -> None:
        chunker = FixedSizeChunker(chunk_size=10, overlap=0, token_counter=WhitespaceTokenCounter())
        chunks = chunker.chunk(make_document(words(25)))
        assert [c.token_count for c in chunks] == [10, 10, 5]


class TestChunkIdentity:
    def test_same_document_yields_same_chunk_ids(self) -> None:
        document = make_document(words(40))
        chunker = FixedSizeChunker(chunk_size=8, overlap=2)
        assert [c.id for c in chunker.chunk(document)] == [c.id for c in chunker.chunk(document)]

    def test_changed_content_yields_different_chunk_ids(self) -> None:
        chunker = FixedSizeChunker(chunk_size=8, overlap=2)
        first = chunker.chunk(make_document(words(40)))
        second = chunker.chunk(make_document(words(40) + " extra"))
        assert first[-1].id != second[-1].id

    def test_all_chunks_belong_to_their_document(self) -> None:
        document = make_document(words(40))
        for chunk in FixedSizeChunker(chunk_size=8, overlap=2).chunk(document):
            assert chunk.document_id == document.id
