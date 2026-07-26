# Experiments

Recall's purpose is to make retrieval questions answerable with evidence. An experiment is a configuration file that pins every variable, a dataset of queries with known-relevant documents, and a result directory that records enough to re-run it later.

```bash
recall experiment ./experiments/configs/002-bm25-vs-dense-vs-hybrid.yaml
recall report 2026-08-03-bm25-vs-dense-vs-hybrid
recall benchmark ./experiments/configs/002-bm25-vs-dense-vs-hybrid.yaml --baseline baseline.json
```

## Configuration

```yaml
name: bm25-vs-dense-vs-hybrid

hypothesis: >
  Hybrid retrieval fusing dense and lexical scores will beat either alone on
  technical documentation, because exact identifier matches are lexical signal
  that embeddings blur.

dataset:
  path: ./experiments/datasets/technical_docs.jsonl

retrieval:
  strategies: [bm25, dense, hybrid]

reranking:
  strategies: [off, cross_encoder]

hybrid:
  fusion: rrf
  dense_weight: 0.65
  lexical_weight: 0.35

top_k: [5, 10, 20]

metrics: [precision_at_k, recall_at_k, hit_rate_at_k, mrr, ndcg_at_k]
```

Every list is a sweep dimension: the example above is 3 strategies × 2 rerankers × 3 values of `top_k` = 18 runs over one index. A scalar is a sweep of one.

Sections that are not sweep axes — `chunking`, `embedding`, `hybrid`, `lexical`, `database` — are layered over the project's `recall.yaml` as plain overrides.

### Which dimensions can be swept

| Axis | Swept |
|---|---|
| `retrieval.strategies` | ✅ |
| `reranking.strategies` | ✅ |
| `top_k` | ✅ |
| `chunking.*` | ❌ — needs a re-ingest per point |
| `embedding.*` | ❌ — see below |

Only a defined set of axes is swept, rather than "any list is a sweep". Some config values are legitimately lists (`hybrid.components`), and inferring intent from shape would turn a typo into a silently different experiment.

Configuring an unsupported sweep is an error with an explanation, not a silent no-op. For chunking comparisons today, run one experiment per strategy with `recall ingest --force` between them.

**TODO / FUTURE:** chunking sweeps by re-ingesting per point; embedding sweeps *without* a re-ingest, which the schema already allows — vectors are keyed `(chunk_id, model_key)`, so one corpus can hold several embedding models at once.

### A note on `off` vs `none`

`off` disables reranking entirely: the candidate pool stays at `top_k` and no reranking pass runs. `none` is the identity reranker — the pool still widens to `reranking.top_n` and is then truncated without reordering. Sweeping `[off, none, cross_encoder]` separates "the wider candidate pool helped" from "the cross-encoder helped", which are easy to confuse and expensive to confuse.

(YAML reads bare `off` as the boolean `false`. Recall normalises that back to `"off"` rather than making you quote it.)

## Dataset format

JSONL, one query per line.

```json
{"query": "How is authentication implemented?", "relevant_documents": ["auth.md", "security.md"]}
```

Graded relevance, required for a meaningful NDCG:

```json
{"query": "How is authentication implemented?", "relevant_documents": {"auth.md": 3, "security.md": 2}}
```

`relevant_chunks` may be supplied instead when labels are chunk-level. A file may not mix the two — metrics computed over both would have per-query denominators that mean different things, and averaging those is meaningless.

Loading is strict, because every failure mode here produces plausible-looking numbers rather than an error: duplicate queries (double-weighted in every average), queries with no relevant items (a zero for every system), non-integer grades, and missing sidecar metadata are all refused.

Every dataset needs a `<name>.meta.json` declaring `kind` (`synthetic` or `curated`), how many documents it covers, and how the labels were produced. See [experiments/datasets/README.md](../../experiments/datasets/README.md).

## Metrics

| Metric | Counts |
|---|---|
| `precision@k` | relevant items in the top `k`, divided by `k` |
| `recall@k` | distinct relevant items found, divided by the number that exist |
| `hit_rate@k` | 1 if anything relevant is in the top `k` |
| `mrr@k` | reciprocal rank of the first relevant item |
| `ndcg@k` | rank-discounted gain, normalised by the ideal ranking |

Three decisions change the numbers, so they are stated rather than buried:

**Precision counts positions; recall counts distinct items.** A system returning ten results with three right has precision 0.3 whether or not the seven wrong ones are duplicates — the user still sees ten slots. But finding one relevant document in three different chunks has not found three relevant documents, so recall counts it once.

**A repeated key earns NDCG gain only at its first occurrence.** Otherwise a document appearing at ranks 1, 2 and 3 accumulates triple gain, can exceed the ideal DCG, and produces NDCG above 1. A repeat carries no information the first occurrence did not.

**MRR is computed over the top `k` and reported as `mrr@k`.** A system is only ever asked for `k` results, so an unbounded MRR would require retrieving the whole corpus. `mrr@10` is not comparable to a published unbounded MRR, and the label says so.

Binary relevance is graded relevance where every grade is 1 — one implementation, not two.

### Latency

p50, p95 and p99 per stage: `total_ms`, `embedding_ms`, `retrieval_ms`, `fusion_ms`, `reranking_ms`. Percentiles are linearly interpolated rather than nearest-rank, which at least degrades smoothly below 100 queries. The query count is reported next to them so a reader can judge whether a p99 means anything. Over thirty queries, it does not.

### Cost

See [cost-model.md](cost-model.md). The short version: a model with no known price reports **nothing**, not zero.

## Result layout

```
experiments/results/
  2026-08-03-hybrid-search/
    config.yaml      the resolved configuration
    results.json     per-query results for every run
    metrics.csv      aggregated metrics, one row per run
    report.md        the generated report
```

`config.yaml` records the *resolved* configuration, not the file as written. The file may reference `${VAR}`, may omit fields that took defaults, and may sit next to a `recall.yaml` that has since changed. What is written is what actually ran.

Each result records the experiment ID, timestamp, git commit (and whether the tree was dirty), dataset version and checksum, model versions, corpus size, metrics, latency and cost.

## Guardrails

The evaluation layer's most dangerous failure mode is not a crash — it is producing zeroes that look like a finding. Three checks exist for that specific reason:

- **An empty index is refused.** Every metric would be zero.
- **Labels that match no ingested document are refused**, naming the unresolved labels and some available source IDs. A dataset written against a corpus you have not ingested would otherwise score zero everywhere.
- **An ambiguous basename is refused.** Datasets are commonly written with bare filenames, and matching `index.md` against one of two candidates would produce numbers nobody could trust.

**Failed queries are excluded from the averages rather than scored zero**, and counted separately. A crashed query measures the harness, not the retriever.

## Reports

`recall report <experiment-id>` regenerates `report.md`. It produces the **quantitative sections only** — comparison tables grouped by `top_k`, latency, cost, provenance and caveats. **Hypothesis** and **Analysis** are headed, empty sections for a person to write.

That split is deliberate. A generated sentence like "hybrid retrieval improves MRR by 12%" reads as a finding; over ten queries on a synthetic dataset it is noise with a decimal point. The tool lays out what was measured, accurately and with every caveat attached. Deciding what it means is the researcher's job.

Caveats are attached automatically and travel with the numbers: synthetic datasets, small query counts, the `hash` embedder, a dirty working tree, binary labels under NDCG.

## Regression checking

```bash
recall benchmark ./experiments/configs/002-bm25-vs-dense-vs-hybrid.yaml \
  --baseline ./experiments/results/2026-08-03-hybrid-search/results.json \
  --threshold 0.02
```

Exits non-zero when any metric drops by more than the threshold, so "did this change break retrieval?" is a CI question rather than something noticed three releases later.

The threshold is an **absolute** drop. Metrics are all in `[0, 1]`, so `0.02` means two points of precision. A relative threshold is brutal near zero: a metric going from 0.01 to 0.005 is a 50% "regression" and almost certainly noise.

Runs are matched by `run_id`, which encodes strategy, reranker and `top_k`, so a baseline from a different sweep never compares `dense@10` against `hybrid@5`. **A changed dataset checksum aborts the comparison** rather than reporting a relabel as a regression.

## Honesty rules

Project policy, not aspiration:

- Synthetic datasets are labelled synthetic, in the dataset file and in every report generated from it.
- Reported numbers state the corpus size and query count.
- The `hash` embedder is never used to make a quality claim. It exists so the pipeline can be tested without a model download, and every report generated with it says so.
- No benchmark numbers appear in the README until the harness that produced them is committed and re-runnable.

## Example experiments

| # | Question | Compares | Runnable |
|---|---|---|---|
| 001 | Does semantic chunking improve retrieval quality? | fixed 256/512/1024 vs semantic | one experiment per strategy |
| 002 | Is hybrid retrieval worth it? | BM25 / dense / hybrid / RRF | ✅ single sweep |
| 003 | Does reranking pay for its latency? | off / none / cross-encoder | ✅ single sweep |
| 004 | Which context selection strategy is best? | top-k / MMR / parent-child | Milestone 4 |
