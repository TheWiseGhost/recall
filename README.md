# Recall

**An open-source framework for building, evaluating, and experimenting with knowledge retrieval systems.**

Recall is research infrastructure, not a chatbot. It exists to make questions like these answerable with evidence instead of intuition:

- Does semantic chunking actually beat fixed-size chunking on your corpus?
- Is hybrid retrieval worth the complexity over dense retrieval alone?
- Does a cross-encoder reranker improve MRR enough to justify the latency it adds?
- How does retrieval quality degrade as a corpus grows from 1k to 100k chunks?

Every component in the pipeline — connector, chunker, embedder, retriever, reranker, context selector, generator, evaluator — is selected by name from configuration and implements a small protocol. Swapping one is a config change, and adding one is a class plus a line of registration.

> **Status: v0.1, Milestone 2 in progress.** Local files and PDFs → fixed-size chunking → embeddings → PostgreSQL/pgvector → dense, **BM25 and hybrid** search → **cross-encoder reranking** → CLI. Further chunking strategies and the evaluation harness are next. See [Roadmap](#roadmap). Nothing here is benchmarked yet, and this README does not claim any numbers.

---

## Architecture

```
   Data sources          Local files · PDFs · GitHub¹ · Notion¹ · Slack¹
        |
   Connectors            discover() -> [SourceItem]     fetch(item) -> Document
        |
   Processing            parse · normalize · extract metadata · checksum
        |
   Chunking              fixed · sentence¹ · semantic¹ · hierarchical¹
        |
   Embeddings            sentence-transformers · OpenAI · hash (deterministic)
        |
   Storage               PostgreSQL + pgvector   (documents, chunks, vectors)
        |
   Retrieval             dense · BM25 · hybrid (RRF / weighted) · metadata filtering
        |
   Reranking             cross-encoder · LLM¹
        |
   Context selection¹    top-k · MMR · parent-child · graph expansion
        |
        +----------------> Search results + citations
        +----------------> LLM generation + citations¹

   ¹ planned — see Roadmap
```

The dependency rule: `recall.core` defines domain models and protocols and imports **nothing** from SQLAlchemy, FastAPI, Celery or Typer. Adapters depend inward. That is what keeps the core usable as a library and testable without a database.

Read more in [docs/architecture/](docs/architecture/).

---

## Quick start

```bash
git clone https://github.com/TheWiseGhost/recall.git
cd recall
docker compose up -d postgres
```

Install the package (Python 3.12+):

```bash
pip install -e ".[dev]"
```

Initialize a project. The `hash` embedder needs no model download and is deterministic, so it is the fastest way to see the pipeline work end to end:

```bash
recall init --embedding-provider hash
```

Create the schema, ingest the example corpus, and search it:

```bash
recall migrate
```

```bash
recall ingest ./examples/documents
```

```bash
recall search "How does authentication work?"
```

For real semantic quality, switch `embedding.provider` to `sentence_transformers` in `recall.yaml` and install the extra:

```bash
pip install -e ".[local]"
```

Changing the embedding model changes `embedding.dimensions`, which is baked into the pgvector column. Recreate the schema and re-ingest:

```bash
docker compose down -v && docker compose up -d postgres && recall migrate && recall ingest ./examples/documents --force
```

---

## CLI

```bash
recall init                  # scaffold recall.yaml, .env.example, example corpus
recall migrate               # create/update the database schema
recall status                # configuration + database health
recall connectors            # list every registered component
recall ingest ./docs         # ingest files and PDFs (incremental by default)
recall search "query"        # search the knowledge base (dense, BM25 or hybrid)
recall documents list        # browse what has been ingested
recall documents show <id> --chunks
```

Useful flags:

```bash
recall ingest ./docs --force            # re-chunk and re-embed everything
recall ingest ./docs --no-prune         # keep documents deleted at the source
recall search "auth" --source-type pdf --top-k 5
recall search "auth" --strategy bm25    # lexical; also: dense, hybrid
recall search "auth" --rerank cross_encoder   # or --rerank off
recall search "auth" --json             # machine-readable, includes timings
```

Ingestion is incremental. A second `recall ingest` over an unchanged directory reports every document as `unchanged` and re-embeds nothing — see [Incremental sync](docs/architecture/ingestion.md).

---

## Configuration

`recall.yaml`, with `${VAR}` and `${VAR:-default}` expanded from the environment. Credentials live in the environment, never in the file.

```yaml
database:
  url: ${DATABASE_URL:-postgresql+asyncpg://recall:recall@localhost:5432/recall}

embedding:
  provider: sentence_transformers    # sentence_transformers | openai | hash
  model: BAAI/bge-base-en-v1.5
  dimensions: 768

chunking:
  strategy: fixed
  chunk_size: 512
  overlap: 64

retrieval:
  default: dense                     # dense | bm25 | hybrid
  top_k: 10

lexical:                             # BM25 parameters
  k1: 1.2                            # term-frequency saturation
  b: 0.75                            # length normalisation; 0 disables it

hybrid:
  components: [dense, bm25]
  fusion: rrf                        # rrf | weighted
  dense_weight: 0.65
  lexical_weight: 0.35
  rrf_k: 60
  candidate_multiplier: 3            # over-fetch per component before fusing

reranking:
  enabled: false
  strategy: cross_encoder            # none | cross_encoder
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_n: 50                          # candidate pool handed to the reranker
```

Any field can be overridden by environment variable: `RECALL_EMBEDDING__PROVIDER=hash`. Configuration is validated on load — a strategy name that is not registered fails at startup, not at the first search.

---

## Retrieval strategies

| Name | Signal | Needs an embedding model |
|---|---|---|
| `dense` | cosine similarity over pgvector | yes |
| `bm25` | Okapi BM25 over PostgreSQL full-text search | no |
| `hybrid` | reciprocal rank or weighted score fusion over both | inherits |

`bm25` is **Okapi BM25** — IDF times saturated term frequency with length normalisation — not `ts_rank_cd` under a better name. The difference is not pedantry: `ts_rank_cd` scores how tightly query terms cluster, with no IDF and no length saturation, so it cannot distinguish a match on a rare identifier from a match on a word every document contains. Publishing that comparison as "BM25 vs dense" would be a fabricated finding.

The formula is verified against an independent reference implementation in the integration suite, and `k1`/`b` are configurable.

`hybrid` runs its components concurrently and fuses their rankings, defaulting to RRF because BM25 scores and cosine similarities are not on a comparable scale and no fixed rescaling makes them so. Every result keeps `component_scores` and `component_ranks`, so "is hybrid worth it?" can be answered with what each side actually contributed rather than a single fused number.

Reranking is optional and off by default. When enabled, retrieval widens its candidate pool to `reranking.top_n`, the reranker reorders it, and `reranking_ms` records what that cost. `strategy: none` with `enabled: true` is the control condition — it widens the pool without reordering, so "the wider pool helped" is not mistaken for "the reranker helped". See [docs/architecture/retrieval.md](docs/architecture/retrieval.md).

---

## Plugin architecture

Implementing a component means satisfying a protocol and registering it. No core file changes.

```python
from recall.core.chunking.base import ChunkerBase, chunker_registry
from recall.core.models import Document


@chunker_registry.decorator("paragraph")
class ParagraphChunker(ChunkerBase):
    name = "paragraph"

    def split(self, document: Document) -> list[tuple[str, int, int]]:
        spans, cursor = [], 0
        for block in document.content.split("\n\n"):
            start = document.content.index(block, cursor)
            spans.append((block, start, start + len(block)))
            cursor = start + len(block)
        return spans
```

```yaml
chunking:
  strategy: paragraph
```

The same pattern applies to `connector_registry`, `embedder_registry` and `retriever_registry`. See [docs/contributing/plugins.md](docs/contributing/plugins.md).

---

## Example experiment

Experiments are configuration-driven and their results are written to disk with the git commit, dataset version and model versions attached, so they can be re-run and compared.

```yaml
# experiments/configs/hybrid.yaml
name: hybrid-vs-dense
dataset:
  path: ./experiments/datasets/technical_docs.jsonl
chunking:
  strategy: fixed
retrieval:
  strategies: [dense, bm25, hybrid]
top_k: [5, 10, 20]
```

```bash
recall experiment ./experiments/configs/hybrid.yaml
```

**The experiment runner is not implemented yet** — it lands in Milestone 2 together with BM25, hybrid retrieval and the metric suite (Precision@K, Recall@K, Hit Rate@K, MRR, NDCG@K). The config format above is the target shape, not a working feature.

---

## Benchmark results

None yet. Recall v0.1 has no evaluation harness, so there are no numbers to report, and inventing them would defeat the purpose of the project. Milestone 2 adds the metric implementations and a labelled dataset; results published then will state their corpus size, query count, and whether the dataset is synthetic or curated.

---

## Development

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

```bash
ruff check . && ruff format --check .
```

```bash
mypy
```

```bash
pytest                       # unit tests, no database required
```

Integration tests need PostgreSQL with pgvector:

```bash
docker compose up -d postgres && pytest -m integration
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Roadmap

**Milestone 1 — MVP (done)**
Filesystem and PDF connectors · fixed-size chunking · pluggable embeddings · PostgreSQL/pgvector storage · dense retrieval · metadata filtering · incremental sync · CLI · Docker · unit and integration tests.

**Milestone 2 — Retrieval research (in progress)**
✅ BM25 over PostgreSQL full-text search · ✅ hybrid retrieval with configurable weights · ✅ reciprocal rank fusion · ✅ cross-encoder reranking · sentence/semantic/hierarchical chunking · Precision@K, Recall@K, Hit Rate@K, MRR, NDCG@K · benchmark datasets · the experiment runner and report generator.

**Milestone 3 — Production architecture**
FastAPI service · Redis + Celery workers · job status tracking and dead-lettering · Prometheus metrics and Grafana dashboards · retries with backoff.

**Milestone 4 — Open source**
GitHub and Notion connectors · Next.js dashboard · context selection (top-k, MMR, parent-child) · optional LLM generation with verified citations · plugin documentation · CI/CD and releases.

---

## What Recall is not

Not a ChatGPT wrapper, not a RAG chatbot product, not an agent framework, not a SaaS platform. Generation is optional and, when enabled, every citation must map to an actually-retrieved chunk. The core identity is retrieval infrastructure, experimentation, and evaluation.

---

## License

[Apache 2.0](LICENSE).
