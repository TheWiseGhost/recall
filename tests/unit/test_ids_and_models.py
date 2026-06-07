"""Deterministic IDs, checksums, and domain-model behaviour."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from recall.core.ids import checksum, chunk_id, document_id
from recall.core.models import (
    Document,
    SearchFilters,
    SearchResult,
    SourceType,
    SyncItemResult,
    SyncOutcome,
    SyncResult,
)


class TestDeterministicIds:
    def test_document_id_is_stable_across_calls(self) -> None:
        first = document_id("filesystem", "docs/auth.md")
        second = document_id("filesystem", "docs/auth.md")
        assert first == second
        assert isinstance(first, uuid.UUID)

    def test_document_id_differs_by_source_type(self) -> None:
        assert document_id("filesystem", "auth.md") != document_id("pdf", "auth.md")

    def test_chunk_id_depends_on_position_and_content(self) -> None:
        doc = document_id("filesystem", "auth.md")
        base = chunk_id(doc, 0, "abc")
        assert chunk_id(doc, 0, "abc") == base
        assert chunk_id(doc, 1, "abc") != base
        assert chunk_id(doc, 0, "abd") != base


class TestChecksum:
    def test_is_stable(self) -> None:
        assert checksum("hello", "world") == checksum("hello", "world")

    def test_length_prefixing_prevents_boundary_collisions(self) -> None:
        # Without length prefixes both would hash "abc".
        assert checksum("ab", "c") != checksum("a", "bc")

    def test_accepts_bytes(self) -> None:
        assert len(checksum(b"binary")) == 64


class TestDocument:
    def test_create_derives_id_and_checksum(self) -> None:
        document = Document.create(
            source_id="a.md",
            source_type=SourceType.FILESYSTEM,
            title="A",
            content="body",
            uri="file:///a.md",
        )
        assert document.id == document_id("filesystem", "a.md")
        assert document.checksum == checksum("A", "body")

    def test_checksum_changes_with_title_only(self) -> None:
        """A retitled document must be treated as changed."""
        args = {
            "source_id": "a.md",
            "source_type": SourceType.FILESYSTEM,
            "content": "body",
            "uri": "file:///a.md",
        }
        first = Document.create(title="A", **args)
        second = Document.create(title="B", **args)
        assert first.id == second.id
        assert first.checksum != second.checksum

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValueError):
            Document(  # type: ignore[call-arg]
                id=uuid.uuid4(),
                source_id="a",
                source_type=SourceType.FILESYSTEM,
                title="t",
                content="c",
                uri="u",
                checksum="x",
                surprise=1,
            )


class TestSearchFilters:
    def test_empty_by_default(self) -> None:
        assert SearchFilters().is_empty()

    def test_not_empty_when_any_field_set(self) -> None:
        assert not SearchFilters(source_types=[SourceType.PDF]).is_empty()
        assert not SearchFilters(created_after=datetime.now(UTC)).is_empty()
        assert not SearchFilters(metadata={"repo": "x"}).is_empty()


class TestSearchResult:
    def test_with_rank_returns_a_copy(self) -> None:
        result = SearchResult(
            chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), content="c", score=0.5, rank=1
        )
        reranked = result.with_rank(7)
        assert reranked.rank == 7
        assert result.rank == 1


class TestSyncResult:
    def test_counts_and_totals(self) -> None:
        result = SyncResult(source_type=SourceType.FILESYSTEM)
        result.items = [
            SyncItemResult(source_id="a", outcome=SyncOutcome.CREATED, chunks_written=3),
            SyncItemResult(source_id="b", outcome=SyncOutcome.UPDATED, chunks_written=2),
            SyncItemResult(source_id="c", outcome=SyncOutcome.UNCHANGED),
            SyncItemResult(source_id="d", outcome=SyncOutcome.FAILED, error="boom"),
        ]
        assert (result.created, result.updated, result.unchanged, result.failed) == (1, 1, 1, 1)
        assert result.chunks_written == 5
        assert result.duration_seconds is None

    def test_duration_once_finished(self) -> None:
        started = datetime(2026, 1, 1, tzinfo=UTC)
        result = SyncResult(
            source_type=SourceType.FILESYSTEM,
            started_at=started,
            finished_at=datetime(2026, 1, 1, 0, 0, 5, tzinfo=UTC),
        )
        assert result.duration_seconds == 5.0
