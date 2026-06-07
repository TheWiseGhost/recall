"""Row <-> domain model translation."""

from __future__ import annotations

from recall.core.models import Chunk, Document, SourceType
from recall.storage.postgres.models import ChunkRow, DocumentRow


def to_document(row: DocumentRow) -> Document:
    return Document(
        id=row.id,
        source_id=row.source_id,
        source_type=SourceType(row.source_type),
        title=row.title,
        content=row.content,
        uri=row.uri,
        metadata=dict(row.meta or {}),
        checksum=row.checksum,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def to_document_row(document: Document) -> DocumentRow:
    return DocumentRow(
        id=document.id,
        source_id=document.source_id,
        source_type=document.source_type.value,
        title=document.title,
        content=document.content,
        uri=document.uri,
        meta=dict(document.metadata),
        checksum=document.checksum,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def to_chunk(row: ChunkRow) -> Chunk:
    return Chunk(
        id=row.id,
        document_id=row.document_id,
        parent_id=row.parent_id,
        content=row.content,
        metadata=dict(row.meta or {}),
        position=row.position,
        token_count=row.token_count,
        checksum=row.checksum,
        start_char=row.start_char,
        end_char=row.end_char,
    )


def to_chunk_row(chunk: Chunk) -> ChunkRow:
    return ChunkRow(
        id=chunk.id,
        document_id=chunk.document_id,
        parent_id=chunk.parent_id,
        content=chunk.content,
        meta=dict(chunk.metadata),
        position=chunk.position,
        token_count=chunk.token_count,
        checksum=chunk.checksum,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
    )
