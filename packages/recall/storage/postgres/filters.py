"""Translate :class:`SearchFilters` into SQL predicates.

Filtering must happen in the database. Over-fetching and discarding rows in
Python silently changes what ``top_k`` means, which would corrupt every
retrieval metric.
"""

from __future__ import annotations

from sqlalchemy import ColumnElement, and_, cast, or_, true
from sqlalchemy.dialects.postgresql import JSONB

from recall.core.models import SearchFilters
from recall.storage.postgres.models import DocumentRow

# Metadata keys promoted to first-class filter fields.
_METADATA_LIST_FIELDS: tuple[tuple[str, str], ...] = (
    ("file_types", "file_type"),
    ("authors", "author"),
    ("repositories", "repository"),
)


def build_document_predicates(filters: SearchFilters | None) -> list[ColumnElement[bool]]:
    """Return predicates over :class:`DocumentRow` for ``filters``."""
    if filters is None or filters.is_empty():
        return []

    predicates: list[ColumnElement[bool]] = []

    if filters.source_types:
        predicates.append(DocumentRow.source_type.in_([t.value for t in filters.source_types]))
    if filters.source_ids:
        predicates.append(DocumentRow.source_id.in_(list(filters.source_ids)))
    if filters.document_ids:
        predicates.append(DocumentRow.id.in_(list(filters.document_ids)))
    if filters.created_after is not None:
        predicates.append(DocumentRow.created_at >= filters.created_after)
    if filters.created_before is not None:
        predicates.append(DocumentRow.created_at <= filters.created_before)

    for field, key in _METADATA_LIST_FIELDS:
        values = getattr(filters, field)
        if values:
            predicates.append(DocumentRow.meta[key].astext.in_([str(v) for v in values]))

    if filters.tags:
        # "has any of these tags", expressed as a disjunction of `@>`
        # containment checks. `?|` would be the more direct operator, but it
        # only applies to a top-level jsonb object and cannot use the GIN index
        # on `metadata`; containment can.
        predicates.append(
            or_(*(DocumentRow.meta.contains(cast({"tags": [tag]}, JSONB)) for tag in filters.tags))
        )

    if filters.metadata:
        # `@>` containment, which the GIN index on metadata can serve.
        predicates.append(DocumentRow.meta.contains(cast(filters.metadata, JSONB)))

    return predicates


def document_filter_clause(filters: SearchFilters | None) -> ColumnElement[bool]:
    """Single ``AND``-ed clause, or a literal TRUE when nothing is filtered."""
    predicates = build_document_predicates(filters)
    if not predicates:
        return true()
    return and_(*predicates)
