"""Canonical domain models.

These are the only shapes that cross subsystem boundaries. Database rows,
connector payloads and API schemas are all translated to and from these types;
none of them leak outward.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from recall.core.ids import checksum as _checksum
from recall.core.ids import document_id as _document_id

JsonDict = dict[str, Any]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SourceType(StrEnum):
    """Where a document came from. Extendable by connectors."""

    FILESYSTEM = "filesystem"
    PDF = "pdf"
    GITHUB = "github"
    NOTION = "notion"
    SLACK = "slack"
    MEMORY = "memory"  # in-process, used by tests and synthetic datasets


class RecallModel(BaseModel):
    """Base model with the project-wide pydantic settings."""

    model_config = ConfigDict(extra="forbid", frozen=False, use_enum_values=False)


class SourceItem(RecallModel):
    """A thing a connector has *discovered* but not necessarily fetched yet.

    ``checksum`` and ``updated_at`` are optional because some sources can only
    answer "has this changed?" after a fetch. When present they let the sync
    planner skip work without downloading content.
    """

    source_id: str
    source_type: SourceType
    uri: str
    title: str | None = None
    checksum: str | None = None
    updated_at: datetime | None = None
    metadata: JsonDict = Field(default_factory=dict)


class Document(RecallModel):
    """A normalized unit of source content."""

    id: uuid.UUID
    source_id: str
    source_type: SourceType
    title: str
    content: str
    uri: str
    metadata: JsonDict = Field(default_factory=dict)
    checksum: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        source_type: SourceType,
        title: str,
        content: str,
        uri: str,
        metadata: JsonDict | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> Document:
        """Build a document, deriving its deterministic ID and content checksum."""
        now = _utcnow()
        return cls(
            id=_document_id(source_type.value, source_id),
            source_id=source_id,
            source_type=source_type,
            title=title,
            content=content,
            uri=uri,
            metadata=metadata or {},
            checksum=_checksum(title, content),
            created_at=created_at or now,
            updated_at=updated_at or now,
        )


class Chunk(RecallModel):
    """A retrievable span of a document."""

    id: uuid.UUID
    document_id: uuid.UUID
    parent_id: uuid.UUID | None = None
    content: str
    metadata: JsonDict = Field(default_factory=dict)
    position: int
    token_count: int
    checksum: str
    # Character offsets into the parent document, when the chunker can supply
    # them. Useful for highlighting and for parent/neighbour expansion.
    start_char: int | None = None
    end_char: int | None = None


class SearchFilters(RecallModel):
    """Structured filters applied at the storage layer, never in Python.

    Every field is a disjunction (``OR`` within a field) and fields are
    conjoined (``AND`` across fields).
    """

    source_types: list[SourceType] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    document_ids: list[uuid.UUID] = Field(default_factory=list)
    file_types: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    repositories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    created_after: datetime | None = None
    created_before: datetime | None = None
    # Escape hatch for connector-specific metadata keys. Values are matched for
    # equality against the chunk's document metadata.
    metadata: JsonDict = Field(default_factory=dict)

    def is_empty(self) -> bool:
        """True when no filter would narrow the result set."""
        return self == SearchFilters()


class SearchResult(RecallModel):
    """One ranked hit. Scores are comparable only within a single retriever."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    content: str
    score: float
    rank: int
    metadata: JsonDict = Field(default_factory=dict)
    # Provenance, so a caller can always answer "where did this come from?".
    document_title: str | None = None
    document_uri: str | None = None
    source_type: SourceType | None = None
    retriever: str | None = None
    # Populated only by a fusing retriever: what each component scored and
    # ranked this chunk. "Is hybrid worth it?" is usually really "what does
    # each side contribute?", and that is unanswerable after fusion has
    # collapsed the lists unless the contributions are kept.
    component_scores: dict[str, float] = Field(default_factory=dict)
    component_ranks: dict[str, int] = Field(default_factory=dict)

    def with_rank(self, rank: int) -> SearchResult:
        """Return a copy re-stamped with a new 1-based rank."""
        return self.model_copy(update={"rank": rank})


class SyncOutcome(StrEnum):
    """What happened to a single item during a sync."""

    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    DELETED = "deleted"
    FAILED = "failed"


class SyncItemResult(RecallModel):
    source_id: str
    outcome: SyncOutcome
    document_id: uuid.UUID | None = None
    chunks_written: int = 0
    error: str | None = None


class SyncResult(RecallModel):
    """Aggregate report for one connector sync run."""

    source_type: SourceType
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    items: list[SyncItemResult] = Field(default_factory=list)

    def _count(self, outcome: SyncOutcome) -> int:
        return sum(1 for item in self.items if item.outcome is outcome)

    @property
    def created(self) -> int:
        return self._count(SyncOutcome.CREATED)

    @property
    def updated(self) -> int:
        return self._count(SyncOutcome.UPDATED)

    @property
    def unchanged(self) -> int:
        return self._count(SyncOutcome.UNCHANGED)

    @property
    def deleted(self) -> int:
        return self._count(SyncOutcome.DELETED)

    @property
    def failed(self) -> int:
        return self._count(SyncOutcome.FAILED)

    @property
    def chunks_written(self) -> int:
        return sum(item.chunks_written for item in self.items)

    @property
    def duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()
