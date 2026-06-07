"""Initial schema: documents, chunks, chunk_embeddings.

The ``chunk_embeddings.embedding`` column is created as ``vector(N)`` where N
comes from ``embedding.dimensions`` in configuration. pgvector requires a fixed
dimension to build an ANN index, so the dimension is a schema-level decision.
Switching to an embedding model with a different dimension needs a new
migration plus a full re-index (``recall ingest --force``).

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from recall.config.settings import load_settings

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _dimensions() -> int:
    """The pgvector column width, from the caller's Settings when available."""
    config = op.get_context().config
    configured = config.get_main_option("recall.embedding_dimensions", None) if config else None
    if configured:
        return int(configured)
    return int(load_settings().embedding.dimensions)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("source_type", "source_id", name="uq_documents_source"),
    )
    op.create_index("ix_documents_source_type", "documents", ["source_type"])
    op.create_index("ix_documents_updated_at", "documents", ["updated_at"])
    op.create_index("ix_documents_metadata", "documents", ["metadata"], postgresql_using="gin")

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chunks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("checksum", sa.String(length=64), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=True),
        sa.Column("end_char", sa.Integer(), nullable=True),
        sa.CheckConstraint("position >= 0", name="ck_chunks_position_non_negative"),
    )
    op.create_index("ix_chunks_document_position", "chunks", ["document_id", "position"])
    op.create_index("ix_chunks_metadata", "chunks", ["metadata"], postgresql_using="gin")

    # Full-text search vector, maintained by PostgreSQL. Unused in v0.1; BM25
    # retrieval in Milestone 2 builds on it. Creating it now avoids rewriting
    # the whole table later.
    op.execute(
        "ALTER TABLE chunks ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_content_tsv ON chunks USING gin (content_tsv)")

    op.create_table(
        "chunk_embeddings",
        sa.Column(
            "chunk_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model_key", sa.String(length=255), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(_dimensions()), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=191), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_chunk_embeddings_document", "chunk_embeddings", ["document_id"])
    op.create_index("ix_chunk_embeddings_model_key", "chunk_embeddings", ["model_key"])

    # HNSW rather than IVFFlat: it needs no training pass over an existing
    # corpus, so it stays accurate while the index is still small and grows.
    op.execute(
        "CREATE INDEX ix_chunk_embeddings_hnsw ON chunk_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw")
    op.drop_table("chunk_embeddings")
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.drop_table("chunks")
    op.drop_table("documents")
