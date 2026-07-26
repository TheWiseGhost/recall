"""Loading labelled query sets from JSONL.

One query per line:

.. code-block:: json

    {"query": "How is authentication implemented?",
     "relevant_documents": ["auth.md", "security.md"]}

Graded relevance, required for a meaningful NDCG:

.. code-block:: json

    {"query": "How is authentication implemented?",
     "relevant_documents": {"auth.md": 3, "security.md": 2}}

``relevant_chunks`` may be given instead when the labels are chunk-level. A
file may not mix the two: a dataset where some queries are judged against
documents and others against chunks would produce metrics whose denominators
mean different things per query, and averaging those is meaningless.

Every dataset needs a sidecar ``<name>.meta.json`` declaring whether it is
**synthetic** or **curated**, how many documents it covers, and how the labels
were produced. That is enforced here rather than left to discipline: a number
reported from an undeclared dataset is a number nobody can weigh.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from recall.core.errors import RecallError
from recall.core.evaluation.models import Dataset, EvaluationQuery, Granularity

VALID_KINDS = frozenset({"synthetic", "curated"})


class DatasetError(RecallError):
    """A dataset file is missing, malformed, or undeclared."""


def _parse_relevance(raw: Any, line_number: int) -> dict[str, int]:
    """Normalise binary or graded labels into ``key -> grade``."""
    if isinstance(raw, list):
        if not all(isinstance(item, str) for item in raw):
            raise DatasetError(f"line {line_number}: relevance list must contain strings")
        # Binary relevance is graded relevance where every grade is 1.
        return {str(item): 1 for item in raw}

    if isinstance(raw, dict):
        graded: dict[str, int] = {}
        for key, grade in raw.items():
            if not isinstance(grade, int) or isinstance(grade, bool):
                raise DatasetError(
                    f"line {line_number}: grade for {key!r} must be an integer, got {grade!r}"
                )
            if grade < 0:
                raise DatasetError(f"line {line_number}: grade for {key!r} must not be negative")
            graded[str(key)] = grade
        return graded

    raise DatasetError(
        f"line {line_number}: relevance must be a list (binary) or an object (graded)"
    )


def _parse_query(payload: dict[str, Any], line_number: int) -> EvaluationQuery:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise DatasetError(
            f"line {line_number}: 'query' is required and must be a non-empty string"
        )

    has_documents = "relevant_documents" in payload
    has_chunks = "relevant_chunks" in payload
    if has_documents and has_chunks:
        raise DatasetError(
            f"line {line_number}: give either 'relevant_documents' or 'relevant_chunks', not both"
        )
    if not has_documents and not has_chunks:
        raise DatasetError(
            f"line {line_number}: one of 'relevant_documents' or 'relevant_chunks' is required"
        )

    granularity = Granularity.CHUNK if has_chunks else Granularity.DOCUMENT
    raw = payload["relevant_chunks"] if has_chunks else payload["relevant_documents"]
    relevant = _parse_relevance(raw, line_number)
    if not any(grade > 0 for grade in relevant.values()):
        raise DatasetError(
            f"line {line_number}: query {query!r} has no relevant items; "
            "a query nothing can satisfy scores zero for every system and only "
            "drags the averages down"
        )

    return EvaluationQuery(
        query=query,
        relevant=relevant,
        granularity=granularity,
        query_id=payload.get("query_id"),
        metadata=payload.get("metadata") or {},
    )


def load_metadata(path: Path) -> dict[str, Any]:
    """Read and validate the sidecar ``*.meta.json`` next to a dataset."""
    meta_path = path.with_suffix("").with_suffix(".meta.json")
    if not meta_path.exists():
        meta_path = path.parent / f"{path.stem}.meta.json"
    if not meta_path.exists():
        raise DatasetError(
            f"{path.name} has no {path.stem}.meta.json. Every dataset must declare "
            "whether it is 'synthetic' or 'curated', how many documents it covers, "
            "and how its labels were produced — a reported number is not "
            "interpretable without it."
        )

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{meta_path} is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise DatasetError(f"{meta_path} must contain a JSON object")

    kind = metadata.get("kind")
    if kind not in VALID_KINDS:
        raise DatasetError(
            f"{meta_path}: 'kind' must be one of {sorted(VALID_KINDS)}, got {kind!r}"
        )
    return metadata


def checksum_file(path: Path) -> str:
    """SHA-256 of the dataset, so a result file pins the exact labels used."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_dataset(path: str | Path) -> Dataset:
    """Load a JSONL dataset and its sidecar metadata."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise DatasetError(f"Dataset not found: {resolved}")

    metadata = load_metadata(resolved)

    queries: list[EvaluationQuery] = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{resolved.name} line {line_number}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise DatasetError(f"{resolved.name} line {line_number}: each line must be an object")
        queries.append(_parse_query(payload, line_number))

    if not queries:
        raise DatasetError(f"{resolved} contains no queries")

    granularities = {query.granularity for query in queries}
    if len(granularities) > 1:
        raise DatasetError(
            f"{resolved.name} mixes document-level and chunk-level labels. Metrics "
            "computed over both would have per-query denominators that mean "
            "different things, and averaging those is not meaningful."
        )

    seen: set[str] = set()
    duplicates: set[str] = set()
    for query in queries:
        if query.query in seen:
            duplicates.add(query.query)
        seen.add(query.query)
    if duplicates:
        raise DatasetError(
            f"{resolved.name} repeats {len(duplicates)} quer"
            f"{'y' if len(duplicates) == 1 else 'ies'}, which would weight them "
            f"double in every average: {sorted(duplicates)[:3]}"
        )

    return Dataset(
        name=metadata.get("name") or resolved.stem,
        path=str(resolved),
        queries=queries,
        kind=str(metadata["kind"]),
        documents=metadata.get("documents"),
        label_method=metadata.get("label_method"),
        version=metadata.get("version"),
        warning=metadata.get("warning"),
        checksum=checksum_file(resolved),
    )
