# Cost model

The rule this page exists to state: **Recall never reports a price it does not know.**

A paid model with no configured price reports `null`, not `$0.00`. Reporting a fabricated zero would make an expensive configuration look free — which is exactly the sort of wrong conclusion the project is built to prevent, and worse than reporting nothing at all.

## Three outcomes, never collapsed

`estimate_embedding_cost` distinguishes:

| Situation | `usd` | Reported as |
|---|---|---|
| The model carries a price | the computed cost | `$0.002100` |
| A local provider with no price | `0.0` | `$0.000000`, "runs locally; no per-token charge" |
| Anything else with no price | `null` | `—`, "cost is not estimated rather than assumed to be zero" |

"Local" means `hash` and `sentence_transformers` — providers that run on the caller's own hardware, where an absent price genuinely means free. The set lives in `core/evaluation/cost.py`; adding a new local provider to it is how that provider avoids being reported as unpriced.

`metrics.csv` writes an empty cell for an unknown cost, not `0`. A spreadsheet summing that column gets a smaller number, not a wrong one.

## Where prices come from

`EmbeddingModelInfo.cost_per_million_tokens`, set by the embedder itself. The OpenAI embedder ships a table in `core/embeddings/openai.py`:

| Model | USD per million input tokens |
|---|---|
| `text-embedding-3-small` | 0.02 |
| `text-embedding-3-large` | 0.13 |
| `text-embedding-ada-002` | 0.10 |

Prices move. An OpenAI model that is not in that table gets no price and is reported as unpriced rather than free — the correct behaviour, and the reason the table is a lookup with no default.

## What is counted

**Query-side embedding only.** For each run, the tokens of every query in the dataset, counted with the configured `TokenCounter`.

Ingestion cost is deliberately excluded. It is a property of the corpus and the chunking strategy, not of a retrieval run, and every run in a sweep shares one index — charging each of them for the same ingest would multiply a single cost by the size of the sweep and produce a number that means nothing.

**TODO / FUTURE:** report ingestion cost once at experiment level, and account for semantic chunking's extra per-sentence embedding pass, which is real and roughly doubles ingest spend on a paid API.

## What is not counted

- Reranking. A cross-encoder is local under the `local` extra; its cost is latency, which *is* measured, per stage, per run.
- Generation. Not implemented.
- Database and infrastructure. Not per-query, and not something Recall can observe.

## Token counting is approximate

Recall's default `TokenCounter` is a dependency-free approximation of a subword tokenizer, not the tokenizer any specific model actually uses. Cost estimates therefore carry that approximation — good enough to compare two configurations, not a substitute for a provider invoice.

Implement `TokenCounter` and pass it to the chunker if you need exactness for a specific model.
