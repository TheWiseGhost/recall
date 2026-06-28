# Retrieval

A retriever answers one question: given a query, a `top_k` and a set of filters, which chunks come back and in what order. The protocol is one method.

```python
class Retriever(Protocol):
    name: str

    async def search(
        self, query: str, top_k: int = 10, filters: SearchFilters | None = None
    ) -> list[SearchResult]: ...
```

Strategies are resolved by name from `retriever_registry`, so `retrieval.default` in `recall.yaml` and `recall search --strategy` select between them without any caller knowing which is which.

## Implemented strategies

| Name | Signal | Needs an embedding model | Port |
|---|---|---|---|
| `dense` | cosine similarity over pgvector | yes | `VectorIndex` |
| `bm25` | Okapi BM25 over PostgreSQL FTS | no | `LexicalIndex` |

## Dense

Embeds the query, asks the vector index for nearest neighbours by cosine distance, and reports `score = 1 - distance`. Filters are passed to the index and applied in SQL. The query is scoped to the configured model's `model_key`, so a corpus holding vectors from several embedding models never mixes incomparable score spaces.

## BM25

This is **Okapi BM25**, not `ts_rank_cd` under a more impressive name:

```
score(D, Q) = Σ  IDF(t) · ( f(t,D) · (k₁ + 1) )
             t∈Q          ────────────────────────────────────────
                           f(t,D) + k₁ · (1 − b + b · |D| / avgdl)

IDF(t) = ln(1 + (N − n(t) + 0.5) / (n(t) + 0.5))
```

The distinction matters enough to be worth stating plainly. `ts_rank_cd` scores *cover density* — how tightly query terms cluster within the text. It has no IDF term, so it cannot tell a match on a rare identifier from a match on a word every document contains, and no length saturation, so a long document wins by repetition. Comparing it against dense retrieval and reporting the result as "BM25 vs dense" would be a fabricated finding. Recall's rule is that a component's name describes what it computes.

Every quantity is read out of PostgreSQL's own inverted index:

| Quantity | Source |
|---|---|
| `f(t,D)` | `array_length(positions, 1)` from `unnest(chunks.content_tsv)` |
| `\|D\|` | `chunks.content_length` — a stored generated column (migration 0002) |
| `n(t)` | count of candidate chunks containing `t` |
| `N` | number of chunks in the filtered collection |
| `avgdl` | mean `content_length` over the filtered collection |

### Decisions worth knowing about

**Statistics are scoped to the filtered collection.** `N`, `avgdl` and `n(t)` are computed over exactly the chunks the filters admit, in the same statement. A search filtered to PDFs is scored against the PDF corpus, not against the whole database — otherwise IDF would describe a collection that was never searched.

**`n(t)` over the candidate set is exact.** Candidates are every chunk matching at least one query term, so any chunk containing `t` is already among them. No sampling, no approximation.

**`|D|` is a generated column, not application state.** It is the number of *positions* in the tsvector — tokens surviving stopword removal and stemming — which is what BM25's length normalisation means. It cannot be an inline generated expression because summing over `unnest()` is an aggregate, so migration 0002 adds a small `IMMUTABLE` SQL function, `recall_tsvector_length`. PostgreSQL then maintains the value for every writer, including raw SQL, and it can never drift from `content_tsv`.

**Candidate selection is OR, never AND.** BM25 ranks by accumulated evidence. Requiring every query term would impose a recall cutoff that dense retrieval does not have, and any Recall@K comparison between the two would then be measuring the cutoff rather than the ranking. This is deliberately not configurable.

**The query is not run through `to_tsquery`.** `to_tsquery` re-runs the text-search *parser* over its input, including over quoted tokens. A lexeme the parser splits — `o'brien`, `a.com/p?q=1&r=2`, any URL, email or path — comes back as a phrase of fragments that appear nowhere in the index, and the match is silently lost. Recall builds the tsquery by casting text directly to `tsquery`, whose input function treats a quoted token as one opaque lexeme, escaping `'` and `\` so the round trip is exact. Both sides of the comparison — the indexed column and the query — are produced by the same analyzer, so stemming and stopwords cannot drift apart.

### Tuning

```yaml
lexical:
  k1: 1.2     # term-frequency saturation; higher = extra occurrences keep helping
  b: 0.75     # length normalisation; 0 disables it entirely
```

Both defaults are Robertson & Zaragoza's, and what Lucene ships. Both are worth sweeping: chunk-length distribution varies enormously between corpora, and `b` is the parameter that responds to it.

The text search configuration is **not** a setting. It is compiled into a generated column, so changing it is a migration and a table rewrite — the same class of decision as `embedding.dimensions`.

### Known limits

- Collection statistics are recomputed per query, which means a scan for `avgdl`. Correct at research corpus sizes; **TODO / FUTURE**: a materialized term-statistics table when it starts to hurt.
- PostgreSQL stores at most 256 positions per lexeme in a tsvector, so `f(t,D)` saturates at 256 occurrences. Unreachable for chunk-sized text; it would flatten term frequency for whole-book chunks.
- English only, per the note above.

## Timing

Retrievers record their own stages against an ambient timer held in a `ContextVar`, so `Retriever` stays a single method and concurrent searches never mix up each other's numbers.

BM25 records only a `retrieval` stage. An experiment comparing it against dense retrieval therefore sees `embedding_ms = 0`, which is the honest number: there is no model in the path. That difference is precisely what a "does the quality justify the latency?" question is asking about.

## How BM25 is tested

Unit tests cover the retriever's contract and compile the SQL to assert its shape — that it uses `ln(` and not `ts_rank`, that filters appear in *both* the candidate set and the statistics, that ordering is deterministic.

The arithmetic is verified in `tests/integration/test_lexical_index.py` against an independent reference implementation. The reference recomputes BM25 in Python from the same lexemes and positions PostgreSQL exposes via `unnest(content_tsv)`, and the two must agree to within floating-point noise. The only assumption they share is how PostgreSQL tokenizes; every BM25-specific step is computed twice, independently, so a mistake in the SQL cannot hide behind a matching mistake in the check.

Alongside it are property tests that would fail against a plausible wrong implementation: a rare term must outscore a ubiquitous one (IDF is present), a long chunk that repeats a term must rank worse at `b=1` than at `b=0` (length normalisation reaches the formula), and ten occurrences must not score ten times one occurrence (saturation is sublinear).

## Not built yet

Hybrid retrieval, reciprocal rank fusion and reranking are the next steps in Milestone 2. The seams exist: `hybrid.fusion`, `dense_weight`, `lexical_weight` and `rrf_k` are already in configuration, and `RetrievalTiming.reranking_ms` is already in the result.
