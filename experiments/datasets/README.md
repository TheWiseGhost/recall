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

Every dataset must declare in its accompanying `*.meta.json` whether it is
**synthetic** or **curated**, how many documents and queries it contains, and
how the labels were produced. A dataset without that file is not usable for a
reported result.
