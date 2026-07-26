"""Resolving dataset labels to what retrieval actually returns.

A dataset says ``"authentication.md"``. A retriever returns chunk IDs and
document IDs. Something has to bridge the two, and getting it wrong is the
single most dangerous failure mode in the whole evaluation layer: an unresolved
label does not raise, it just never matches, and every metric comes out zero.
Zeroes look like a finding.

So resolution is strict. Labels that match nothing are collected and reported
as an error, with the labels and the available source IDs named, rather than
quietly scoring the run at zero.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from recall.core.errors import RecallError
from recall.core.evaluation.models import Dataset, Granularity
from recall.core.models import SearchResult


class LabelResolutionError(RecallError):
    """Dataset labels do not correspond to anything in the index."""


@dataclass(frozen=True, slots=True)
class LabelResolver:
    """Maps a :class:`SearchResult` to the label key a dataset would use."""

    granularity: Granularity
    document_labels: dict[uuid.UUID, str]

    def key(self, result: SearchResult) -> str:
        if self.granularity is Granularity.CHUNK:
            return str(result.chunk_id)
        return self.document_labels.get(result.document_id, str(result.document_id))

    def ranked_keys(self, results: Sequence[SearchResult]) -> list[str]:
        """The label keys of ``results``, in rank order, duplicates preserved.

        Duplicates are kept because precision counts positions: two chunks from
        the same document occupy two of the ``k`` slots a user sees. The metrics
        module is what decides where a repeat should and should not earn credit.
        """
        return [self.key(result) for result in results]


def build_resolver(
    dataset: Dataset,
    documents: Sequence[tuple[uuid.UUID, str]],
) -> LabelResolver:
    """Build a resolver and verify every label in ``dataset`` can be hit.

    ``documents`` is ``(document_id, source_id)`` for the whole corpus.

    Document labels are matched against the full source ID first, then against
    its basename — datasets are commonly written with bare filenames while a
    connector's source ID is a path relative to the corpus root. The basename
    fallback is refused when it would be ambiguous, because silently picking one
    of two ``index.md`` files would produce numbers nobody could trust.
    """
    if dataset.queries and dataset.queries[0].granularity is Granularity.CHUNK:
        # Chunk-level labels are chunk IDs; nothing to map. They are still
        # checked below, against the retrieved results at run time.
        return LabelResolver(granularity=Granularity.CHUNK, document_labels={})

    by_source: dict[str, uuid.UUID] = {}
    basenames: dict[str, list[uuid.UUID]] = {}
    for document_id, source_id in documents:
        by_source[source_id] = document_id
        basenames.setdefault(source_id.rsplit("/", 1)[-1], []).append(document_id)

    labels = {label for query in dataset.queries for label in query.relevant}
    resolved: dict[uuid.UUID, str] = {}
    unresolved: list[str] = []
    ambiguous: list[str] = []

    for label in sorted(labels):
        if label in by_source:
            resolved[by_source[label]] = label
            continue
        candidates = basenames.get(label.rsplit("/", 1)[-1], [])
        if len(candidates) == 1:
            resolved[candidates[0]] = label
        elif len(candidates) > 1:
            ambiguous.append(label)
        else:
            unresolved.append(label)

    if ambiguous:
        raise LabelResolutionError(
            f"{len(ambiguous)} dataset label(s) match more than one document by "
            f"basename: {ambiguous[:5]}. Use the full source ID (the path relative "
            "to the corpus root) so the label is unambiguous."
        )
    if unresolved:
        available = sorted(by_source)[:8]
        raise LabelResolutionError(
            f"{len(unresolved)} dataset label(s) match no ingested document: "
            f"{unresolved[:5]}. The experiment would score zero on every metric "
            f"and that would look like a result. Ingest the corpus the dataset "
            f"was written against, or fix the labels. Available source IDs "
            f"include: {available}"
        )

    return LabelResolver(granularity=Granularity.DOCUMENT, document_labels=resolved)
