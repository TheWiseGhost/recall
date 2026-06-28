"""Add the per-chunk lexeme count BM25 needs for length normalisation.

BM25 divides by ``|D|``, the length of a chunk in indexed terms. That is the
number of *positions* in ``content_tsv`` (tokens surviving stopword removal and
stemming), not the character count and not ``token_count``, which is an
approximate subword estimate used for chunk sizing.

Computing it per query would mean unnesting every chunk's tsvector on every
search — and ``avgdl`` would need it for the whole corpus, not just the
candidates. So it is materialised as a STORED generated column.

It has to go through a function: summing over ``unnest()`` is an aggregate, and
generated column expressions may not contain subqueries. A plain SQL function
marked IMMUTABLE is allowed, and keeps the value maintained by PostgreSQL for
every writer — no trigger, no application-side backfill that can drift.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

from recall.storage.postgres.models import TEXT_SEARCH_CONFIG, TSVECTOR_LENGTH_FUNCTION

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # STRICT so a NULL tsvector yields NULL rather than 0; IMMUTABLE is what
    # makes it legal in a generated column.
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {TSVECTOR_LENGTH_FUNCTION}(tsv tsvector)
        RETURNS integer
        LANGUAGE sql
        IMMUTABLE
        PARALLEL SAFE
        STRICT
        AS $$
            SELECT coalesce(sum(coalesce(array_length(positions, 1), 1)), 0)::int
            FROM unnest(tsv)
        $$
        """
    )

    # Note this recomputes to_tsvector rather than reading content_tsv: a
    # generated column may not reference another generated column.
    op.execute(
        f"ALTER TABLE chunks ADD COLUMN content_length integer "
        f"GENERATED ALWAYS AS "
        f"({TSVECTOR_LENGTH_FUNCTION}(to_tsvector('{TEXT_SEARCH_CONFIG}', content))) STORED"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_length")
    op.execute(f"DROP FUNCTION IF EXISTS {TSVECTOR_LENGTH_FUNCTION}(tsvector)")
