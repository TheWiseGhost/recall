# Evaluation datasets

JSONL, one query per line.

Binary relevance:

```json
{"query": "How is authentication implemented?", "relevant_documents": ["authentication.md"]}
```

Graded relevance, required for NDCG:

```json
{"query": "How is authentication implemented?", "relevant_documents": {"authentication.md": 3, "deployment.md": 1}}
```

`relevant_chunks` may be supplied instead of `relevant_documents` when labels
are chunk-level.

`relevant_documents` keys are matched against a document's **source ID** — the
path relative to the corpus root the connector ingested. A bare filename also
works when it is unambiguous; when two documents share a basename, the full
source ID is required rather than one being picked silently.

## The sidecar is required

Every dataset needs a `<name>.meta.json` beside it:

```json
{
  "name": "technical_docs",
  "kind": "curated",
  "documents": 240,
  "queries": 180,
  "label_method": "annotated by two engineers, disagreements resolved by a third"
}
```

`kind` must be `synthetic` or `curated`. Loading fails without the file — a
number reported from an undeclared dataset is a number nobody can weigh.

## Loading is strict

These are all refused rather than loaded, because each produces plausible
numbers instead of an error:

- a query repeated on two lines (double-weighted in every average)
- a query with no relevant items (scores zero for every system)
- a file mixing `relevant_documents` and `relevant_chunks`
- non-integer or negative grades
