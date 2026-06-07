# Experiments

> **Status: not implemented.** The experiment runner, the metric suite and the report generator land in Milestone 2. This page documents the intended design so the interfaces built in Milestone 1 can be judged against it — it is a specification, not a manual for working commands.

## Why this exists

Recall's purpose is to make retrieval questions answerable with evidence. An experiment is a configuration file that pins every variable, a dataset of queries with known-relevant documents, and a result directory that records enough to re-run it later.

## Intended configuration

```yaml
name: hybrid-vs-dense

dataset:
  path: ./experiments/datasets/technical_docs.jsonl

chunking:
  strategy: fixed
  chunk_size: 512
  overlap: 64

embedding:
  provider: sentence_transformers
  model: BAAI/bge-base-en-v1.5

retrieval:
  strategies: [dense, bm25, hybrid]

reranking:
  strategy: cross_encoder

top_k: [5, 10, 20]
```

Every list is a sweep dimension: the example above is 3 strategies × 3 values of `top_k` = 9 runs over one index.

## Dataset format

JSONL, one query per line.

```json
{"query": "How is authentication implemented?", "relevant_documents": ["auth.md", "security.md"]}
```

Graded relevance, for NDCG:

```json
{"query": "How is authentication implemented?", "relevant_documents": {"auth.md": 3, "security.md": 2}}
```

`relevant_chunks` may be supplied instead when labels are chunk-level.

## Planned metrics

**Retrieval quality** — Precision@K, Recall@K, Hit Rate@K, MRR, NDCG@K.

**Latency** — p50, p95, p99, broken down by stage (embedding, retrieval, reranking, generation). The `RetrievalTiming` model already records this per query.

**Cost** — estimated embedding cost and LLM input/output tokens, from configurable per-provider pricing. `EmbeddingModelInfo.cost_per_million_tokens` already carries it; `None` means "local / free", never "we guessed".

## Planned result layout

```
experiments/results/
  2026-08-03-hybrid-search/
    config.yaml      the exact resolved configuration
    results.json     per-query results
    metrics.csv      aggregated metrics
    report.md        generated report
```

Each result records the experiment ID, timestamp, git commit, dataset version, configuration, model versions, metrics, latency and cost — the set needed to reproduce it.

## Honesty rules

These are project policy, not aspiration:

- Synthetic datasets are labelled synthetic, in the dataset file and in every report generated from it.
- Reported numbers state the corpus size and query count.
- The `hash` embedder is never used to make a quality claim. It exists so the pipeline can be tested without a model download.
- No benchmark numbers appear in the README until the harness that produced them is committed and re-runnable.

## Planned example experiments

| # | Question | Compares |
|---|---|---|
| 001 | Does semantic chunking improve retrieval quality? | fixed 256 / 512 / 1024 vs semantic |
| 002 | Is hybrid retrieval worth it? | BM25 / dense / hybrid / RRF |
| 003 | Does reranking pay for its latency? | none vs cross-encoder |
| 004 | Which context selection strategy is best? | top-k / MMR / parent-child |
