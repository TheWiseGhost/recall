"""SQLAlchemy table definitions.

These are persistence details. They never leave this package: repositories
translate rows to and from :mod:`recall.core.models` domain objects.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# The PostgreSQL text search configuration used for lexical retrieval. It is
# baked into the generated columns below, exactly like the pgvector column's
# dimension: changing it is a migration and a table rewrite, not a setting.
#
# TODO / FUTURE: make this configurable per corpus for non-English collections.
TEXT_SEARCH_CONFIG = "english"

# SQL function created by migration 0002. Returns the number of *positions* in a
# tsvector — i.e. the token count after stopword removal and stemming — which is
# BM25's |D|. It has to be a function rather than an inline expression because
# summing over ``unnest()`` is an aggregate, and generated columns may not
# contain subqueries.
TSVECTOR_LENGTH_FUNCTION = "recall_tsvector_length"

_CONTENT_TSV_EXPRESSION = f"to_tsvector('{TEXT_SEARCH_CONFIG}', content)"
_CONTENT_LENGTH_EXPRESSION = f"{TSVECTOR_LENGTH_FUNCTION}({_CONTENT_TSV_EXPRESSION})"


class Base(DeclarativeBase):
    """Declarative base for all Recall tables."""


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    # `metadata` is taken by SQLAlchemy's declarative API, hence the attribute name.
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    chunks: Mapped[list[ChunkRow]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="uq_documents_source"),
        Index("ix_documents_source_type", "source_type"),
        Index("ix_documents_updated_at", "updated_at"),
        Index("ix_documents_metadata", "metadata", postgresql_using="gin"),
    )


class ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- lexical retrieval, maintained by PostgreSQL ---------------------
    # Both are STORED generated columns, so they are excluded from every
    # INSERT and can never drift from `content` — including for writers that
    # bypass this ORM. `content_length` is BM25's |D|.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed(_CONTENT_TSV_EXPRESSION, persisted=True), nullable=False
    )
    content_length: Mapped[int] = mapped_column(
        Integer, Computed(_CONTENT_LENGTH_EXPRESSION, persisted=True), nullable=False
    )

    document: Mapped[DocumentRow] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_position", "document_id", "position"),
        Index("ix_chunks_metadata", "metadata", postgresql_using="gin"),
        Index("ix_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        CheckConstraint("position >= 0", name="ck_chunks_position_non_negative"),
    )


class ChunkEmbeddingRow(Base):
    """One vector per (chunk, embedding model).

    ``document_id`` is denormalised so filtered vector queries and per-document
    deletes do not need a join.

    The column's dimension is fixed by the initial migration from
    ``embedding.dimensions``. Changing embedding models to one with a different
    dimension requires a migration and a re-index; see
    docs/architecture/storage.md.
    """

    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    # dim=None: the concrete `vector(N)` type is created by the migration.
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(191), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_chunk_embeddings_document", "document_id"),
        Index("ix_chunk_embeddings_model_key", "model_key"),
    )
