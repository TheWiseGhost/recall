"""Sentence, semantic and hierarchical chunking."""

from __future__ import annotations

from itertools import pairwise

import pytest

from recall.config.settings import Settings
from recall.core.chunking import (
    HierarchicalChunker,
    SemanticChunker,
    SentenceChunker,
    chunker_registry,
    create_chunker,
    split_sentences,
)
from recall.core.chunking.semantic import cosine_distance, percentile
from recall.core.embeddings.base import EmbeddingModelInfo, Vector
from recall.core.embeddings.hashing import HashingEmbedder
from recall.core.errors import ConfigurationError
from recall.core.models import Document, SourceType
from recall.core.tokenization import WhitespaceTokenCounter
from recall.pipeline.factory import build_chunker

WORDS = WhitespaceTokenCounter()


def make_document(content: str) -> Document:
    return Document.create(
        source_id="doc.md",
        source_type=SourceType.FILESYSTEM,
        title="Doc",
        content=content,
        uri="file:///doc.md",
        metadata={"file_type": "md"},
    )


class TestSentenceSegmentation:
    def test_splits_on_terminal_punctuation(self) -> None:
        spans = split_sentences("One thing. Two things! Three things?")
        assert [text for text, _, _ in spans] == [
            "One thing.",
            "Two things!",
            "Three things?",
        ]

    def test_offsets_index_back_into_the_source(self) -> None:
        text = "First sentence here. Second sentence here."
        for sentence, start, end in split_sentences(text):
            assert text[start:end] == sentence

    @pytest.mark.parametrize(
        "text",
        [
            "Ask Dr. Smith about it. Then leave.",
            "Use e.g. this one. Then leave.",
            "See Fig. 4 for detail. Then leave.",
            "Pi is 3.14 exactly. Then leave.",
            "J. R. R. Tolkien wrote it. Then leave.",
        ],
    )
    def test_abbreviations_and_numbers_do_not_end_sentences(self, text: str) -> None:
        assert len(split_sentences(text)) == 2

    def test_a_blank_line_is_a_boundary_without_punctuation(self) -> None:
        """Headings and list items rarely end in a period."""
        spans = split_sentences("# A heading\n\nSome body text follows")
        assert len(spans) == 2
        assert spans[0][0] == "# A heading"

    def test_closing_quotes_stay_with_their_sentence(self) -> None:
        spans = split_sentences('He said "go now." Then he left.')
        assert spans[0][0].endswith('"')

    def test_blank_input_produces_nothing(self) -> None:
        assert split_sentences("   \n\n  ") == []

    def test_text_without_punctuation_is_one_sentence(self) -> None:
        assert len(split_sentences("no terminal punctuation here")) == 1


class TestSentenceChunker:
    def test_is_registered(self) -> None:
        assert "sentence" in chunker_registry
        assert isinstance(create_chunker("sentence"), SentenceChunker)

    async def test_never_splits_a_sentence(self) -> None:
        """The entire reason this strategy exists."""
        sentences = [f"Sentence number {index} has several words in it." for index in range(20)]
        document = make_document(" ".join(sentences))
        chunks = await SentenceChunker(chunk_size=20, token_counter=WORDS).chunk(document)

        for chunk in chunks:
            # Every chunk starts at a sentence start and ends at a sentence end.
            assert chunk.content[0].isupper()
            assert chunk.content.rstrip().endswith(".")

    async def test_respects_the_token_budget(self) -> None:
        document = make_document(" ".join(f"Word{i} is here." for i in range(40)))
        chunker = SentenceChunker(chunk_size=12, overlap_sentences=0, token_counter=WORDS)
        chunks = await chunker.chunk(document)
        assert chunks
        assert all(chunk.token_count <= 12 for chunk in chunks)

    async def test_an_oversized_sentence_becomes_its_own_chunk(self) -> None:
        """Splitting it would reintroduce the problem this strategy solves."""
        long_sentence = " ".join(f"word{i}" for i in range(50)) + "."
        chunker = SentenceChunker(chunk_size=10, token_counter=WORDS)
        chunks = await chunker.chunk(make_document(long_sentence))
        assert len(chunks) == 1
        assert chunks[0].token_count > 10  # visible, not silently truncated

    async def test_overlap_repeats_whole_sentences(self) -> None:
        document = make_document(" ".join(f"Sentence {i} here." for i in range(12)))
        chunker = SentenceChunker(chunk_size=9, overlap_sentences=1, token_counter=WORDS)
        chunks = await chunker.chunk(document)
        assert len(chunks) > 1
        for earlier, later in pairwise(chunks):
            tail = split_sentences(earlier.content)[-1][0]
            assert later.content.startswith(tail)

    async def test_zero_overlap_does_not_repeat(self) -> None:
        document = make_document(" ".join(f"Sentence {i} here." for i in range(12)))
        chunker = SentenceChunker(chunk_size=9, overlap_sentences=0, token_counter=WORDS)
        chunks = await chunker.chunk(document)
        offsets = [(c.start_char, c.end_char) for c in chunks]
        for (_, end), (start, _) in pairwise(offsets):
            assert start is not None and end is not None and start >= end

    async def test_offsets_map_back_into_the_source(self) -> None:
        document = make_document(" ".join(f"Sentence {i} here." for i in range(12)))
        chunks = await SentenceChunker(chunk_size=9, token_counter=WORDS).chunk(document)
        for chunk in chunks:
            assert document.content[chunk.start_char : chunk.end_char] == chunk.content

    async def test_terminates_with_an_enormous_overlap(self) -> None:
        document = make_document(" ".join(f"Sentence {i} here." for i in range(20)))
        chunker = SentenceChunker(chunk_size=9, overlap_sentences=1000, token_counter=WORDS)
        assert await chunker.chunk(document)  # must not hang

    async def test_empty_document_produces_nothing(self) -> None:
        assert await SentenceChunker().chunk(make_document("  \n ")) == []

    def test_rejects_a_negative_overlap(self) -> None:
        with pytest.raises(ConfigurationError, match="overlap_sentences"):
            SentenceChunker(overlap_sentences=-1)


class ScriptedEmbedder:
    """Returns caller-supplied vectors, so boundaries are predictable."""

    def __init__(self, vectors: list[Vector]) -> None:
        self.vectors = vectors
        self.batches: list[list[str]] = []

    @property
    def info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(provider="scripted", model="v1", dimensions=2)

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        self.batches.append(list(texts))
        return [self.vectors[index % len(self.vectors)] for index in range(len(texts))]

    async def embed_query(self, query: str) -> Vector:
        return self.vectors[0]


class TestSemanticHelpers:
    def test_cosine_distance_of_identical_vectors_is_zero(self) -> None:
        assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors_are_one_apart(self) -> None:
        assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_a_zero_vector_does_not_divide_by_zero(self) -> None:
        assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0

    def test_percentile_interpolates(self) -> None:
        assert percentile([0.0, 1.0], 0.5) == pytest.approx(0.5)

    def test_percentile_of_one_value(self) -> None:
        assert percentile([0.42], 0.95) == 0.42

    def test_percentile_of_nothing(self) -> None:
        assert percentile([], 0.95) == 0.0


class TestSemanticChunker:
    def test_is_registered(self) -> None:
        assert "semantic" in chunker_registry

    async def test_breaks_where_the_topic_changes(self) -> None:
        """Three sentences about A, then three about B: one break, in the middle."""
        text = (
            "Alpha one here. Alpha two here. Alpha three here. "
            "Beta one here. Beta two here. Beta three here."
        )
        # buffer_size=0 so each sentence is embedded alone and the scripted
        # vectors line up one-to-one with sentences.
        embedder = ScriptedEmbedder([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3)
        chunker = SemanticChunker(
            embedder=embedder,  # type: ignore[arg-type]
            buffer_size=0,
            breakpoint_percentile=0.8,
            token_counter=WORDS,
        )
        chunks = await chunker.chunk(make_document(text))
        assert len(chunks) == 2
        assert chunks[0].content.startswith("Alpha one")
        assert chunks[1].content.startswith("Beta one")

    async def test_uniform_text_is_not_broken_up(self) -> None:
        text = " ".join(f"Uniform sentence {i} here." for i in range(6))
        embedder = ScriptedEmbedder([[1.0, 0.0]])
        chunker = SemanticChunker(
            embedder=embedder,  # type: ignore[arg-type]
            buffer_size=0,
            token_counter=WORDS,
        )
        assert len(await chunker.chunk(make_document(text))) == 1

    async def test_buffer_widens_what_is_embedded(self) -> None:
        text = "One here. Two here. Three here."
        embedder = ScriptedEmbedder([[1.0, 0.0]])
        chunker = SemanticChunker(
            embedder=embedder,  # type: ignore[arg-type]
            buffer_size=1,
            token_counter=WORDS,
        )
        await chunker.chunk(make_document(text))
        windows = embedder.batches[0]
        assert windows[1] == "One here. Two here. Three here."
        assert len(windows) == 3

    async def test_max_chunk_size_is_enforced(self) -> None:
        text = " ".join(f"Uniform sentence number {i} appears here." for i in range(20))
        embedder = ScriptedEmbedder([[1.0, 0.0]])
        chunker = SemanticChunker(
            embedder=embedder,  # type: ignore[arg-type]
            buffer_size=0,
            max_chunk_size=20,
            token_counter=WORDS,
        )
        chunks = await chunker.chunk(make_document(text))
        assert len(chunks) > 1
        assert all(chunk.token_count <= 20 for chunk in chunks)

    async def test_offsets_map_back_into_the_source(self) -> None:
        text = " ".join(f"Sentence {i} appears here." for i in range(10))
        document = make_document(text)
        chunker = SemanticChunker(
            embedder=HashingEmbedder(dimensions=32),
            buffer_size=1,
            token_counter=WORDS,
        )
        for chunk in await chunker.chunk(document):
            assert document.content[chunk.start_char : chunk.end_char] == chunk.content

    async def test_records_the_embedding_model_in_provenance(self) -> None:
        """Boundaries depend on the model, so a chunk is not reproducible without it."""
        chunker = SemanticChunker(embedder=HashingEmbedder(dimensions=32), token_counter=WORDS)
        chunks = await chunker.chunk(make_document("One here. Two here. Three here."))
        assert chunks[0].metadata["chunker_params"]["embedding_model"] == "hash:hash-v1"

    async def test_empty_document_produces_nothing(self) -> None:
        chunker = SemanticChunker(embedder=HashingEmbedder(dimensions=32))
        assert await chunker.chunk(make_document("   ")) == []

    async def test_single_sentence_is_one_chunk(self) -> None:
        embedder = ScriptedEmbedder([[1.0, 0.0]])
        chunker = SemanticChunker(embedder=embedder)  # type: ignore[arg-type]
        chunks = await chunker.chunk(make_document("Only one sentence here."))
        assert len(chunks) == 1
        assert embedder.batches == []  # nothing to decide, so nothing embedded

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"breakpoint_percentile": 1.0}, "breakpoint_percentile"),
            ({"breakpoint_percentile": 0.0}, "breakpoint_percentile"),
            ({"buffer_size": -1}, "buffer_size"),
            ({"max_chunk_size": 0}, "max_chunk_size"),
            ({"min_sentences": 0}, "min_sentences"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ConfigurationError, match=match):
            SemanticChunker(embedder=HashingEmbedder(dimensions=8), **kwargs)  # type: ignore[arg-type]


class TestHierarchicalChunker:
    def test_is_registered(self) -> None:
        assert "hierarchical" in chunker_registry

    async def _chunks(self, words: int = 300) -> list:
        document = make_document(" ".join(f"word{i}" for i in range(words)))
        chunker = HierarchicalChunker(
            parent_chunk_size=100, chunk_size=25, overlap=0, token_counter=WORDS
        )
        return await chunker.chunk(document)

    async def test_emits_both_levels(self) -> None:
        chunks = await self._chunks()
        levels = {chunk.metadata["chunk_level"] for chunk in chunks}
        assert levels == {"parent", "child"}

    async def test_every_child_points_at_a_parent(self) -> None:
        chunks = await self._chunks()
        parents = {c.id for c in chunks if c.metadata["chunk_level"] == "parent"}
        children = [c for c in chunks if c.metadata["chunk_level"] == "child"]
        assert children
        assert all(child.parent_id in parents for child in children)

    async def test_parents_have_no_parent(self) -> None:
        chunks = await self._chunks()
        assert all(c.parent_id is None for c in chunks if c.metadata["chunk_level"] == "parent")

    async def test_a_child_is_contained_in_its_parent(self) -> None:
        chunks = await self._chunks()
        by_id = {c.id: c for c in chunks}
        for child in (c for c in chunks if c.metadata["chunk_level"] == "child"):
            parent = by_id[child.parent_id]
            assert parent.start_char <= child.start_char
            assert child.end_char <= parent.end_char
            assert child.content in parent.content

    async def test_positions_are_unique_across_levels(self) -> None:
        """Chunk IDs fold in the position; a reuse would collide them."""
        chunks = await self._chunks()
        positions = [c.position for c in chunks]
        assert len(set(positions)) == len(positions)
        assert sorted(positions) == list(range(len(positions)))

    async def test_chunk_ids_are_unique(self) -> None:
        chunks = await self._chunks()
        assert len({c.id for c in chunks}) == len(chunks)

    async def test_children_are_positioned_in_reading_order(self) -> None:
        chunks = await self._chunks()
        children = sorted(
            (c for c in chunks if c.metadata["chunk_level"] == "child"),
            key=lambda c: c.position,
        )
        assert [c.start_char for c in children] == sorted(c.start_char for c in children)
        assert [c.position for c in children] == list(range(len(children)))

    async def test_parents_are_listed_before_children(self) -> None:
        """A self-referencing foreign key cannot point at an unwritten row."""
        chunks = await self._chunks()
        levels = [c.metadata["chunk_level"] for c in chunks]
        assert levels == sorted(levels, key=lambda level: level != "parent")

    async def test_parents_do_not_overlap(self) -> None:
        """Overlapping parents would double-count text on expansion."""
        chunks = await self._chunks()
        parents = sorted(
            (c for c in chunks if c.metadata["chunk_level"] == "parent"),
            key=lambda c: c.start_char,
        )
        for earlier, later in pairwise(parents):
            assert later.start_char >= earlier.end_char

    async def test_offsets_map_back_into_the_source(self) -> None:
        document = make_document(" ".join(f"word{i}" for i in range(300)))
        chunker = HierarchicalChunker(
            parent_chunk_size=100, chunk_size=25, overlap=5, token_counter=WORDS
        )
        for chunk in await chunker.chunk(document):
            assert document.content[chunk.start_char : chunk.end_char] == chunk.content

    async def test_records_child_counts(self) -> None:
        chunks = await self._chunks()
        for parent in (c for c in chunks if c.metadata["chunk_level"] == "parent"):
            index = parent.metadata["parent_index"]
            actual = sum(
                1
                for c in chunks
                if c.metadata["chunk_level"] == "child" and c.metadata["parent_index"] == index
            )
            assert parent.metadata["child_count"] == actual

    async def test_empty_document_produces_nothing(self) -> None:
        assert await HierarchicalChunker().chunk(make_document("  ")) == []

    def test_rejects_a_parent_no_larger_than_its_children(self) -> None:
        with pytest.raises(ConfigurationError, match="parent_chunk_size"):
            HierarchicalChunker(parent_chunk_size=100, chunk_size=100)


class TestFactoryWiring:
    @pytest.mark.parametrize("strategy", ["fixed", "sentence", "hierarchical"])
    def test_builds_without_an_embedder(self, strategy: str) -> None:
        settings = Settings.from_mapping({"chunking": {"strategy": strategy}})
        assert build_chunker(settings).name == strategy

    def test_semantic_needs_an_embedder(self) -> None:
        settings = Settings.from_mapping({"chunking": {"strategy": "semantic"}})
        with pytest.raises(ConfigurationError, match="needs an embedder"):
            build_chunker(settings)

    def test_semantic_gets_the_pipeline_embedder(self) -> None:
        settings = Settings.from_mapping({"chunking": {"strategy": "semantic"}})
        embedder = HashingEmbedder(dimensions=32)
        chunker = build_chunker(settings, embedder=embedder)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.embedder is embedder

    def test_parameters_reach_each_strategy(self) -> None:
        settings = Settings.from_mapping(
            {
                "chunking": {
                    "strategy": "hierarchical",
                    "parent_chunk_size": 999,
                    "chunk_size": 111,
                    "overlap": 11,
                }
            }
        )
        chunker = build_chunker(settings)
        assert isinstance(chunker, HierarchicalChunker)
        assert chunker.parent_chunk_size == 999
        assert chunker.chunk_size == 111

    def test_rejects_a_flat_hierarchy_at_load_time(self) -> None:
        with pytest.raises(Exception, match="parent_chunk_size"):
            Settings.from_mapping(
                {
                    "chunking": {
                        "strategy": "hierarchical",
                        "parent_chunk_size": 100,
                        "chunk_size": 512,
                    }
                }
            )

    def test_rejects_an_unregistered_strategy(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from recall.config.settings import load_settings

        with TemporaryDirectory() as directory:
            path = Path(directory) / "recall.yaml"
            path.write_text("chunking:\n  strategy: telepathic\n", encoding="utf-8")
            with pytest.raises(ConfigurationError, match="telepathic"):
                load_settings(path)
