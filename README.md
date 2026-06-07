# Recall

**An open-source framework for building, evaluating, and experimenting with knowledge retrieval systems.**

Recall is research infrastructure, not a chatbot. It exists to make questions like these answerable with evidence instead of intuition:

- Does semantic chunking actually beat fixed-size chunking on your corpus?
- Is hybrid retrieval worth the complexity over dense retrieval alone?
- Does a cross-encoder reranker improve MRR enough to justify the latency it adds?
- How does retrieval quality degrade as a corpus grows from 1k to 100k chunks?

Every component in the pipeline — connector, chunker, embedder, retriever, reranker, context selector, generator, evaluator — is selected by name from configuration and implements a small protocol. Swapping one is a config change, and adding one is a class plus a line of registration.

> **Status: v0.1, Milestone 1 (MVP).** Local files and PDFs → fixed-size chunking → embeddings → PostgreSQL/pgvector → dense search → CLI. BM25, hybrid retrieval, reranking and the evaluation harness are the next milestone. See [Roadmap](#roadmap). Nothing here is benchmarked yet, and this README does not claim any numbers.

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
   Retrieval             dense · BM25¹ · hybrid¹ · metadata filtering
        |
   Reranking¹            cross-encoder · LLM
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
recall search "query"        # search the knowledge base
recall documents list        # browse what has been ingested
recall documents show <id> --chunks
```

Useful flags:

```bash
recall ingest ./docs --force            # re-chunk and re-embed everything
recall ingest ./docs --no-prune         # keep documents deleted at the source
recall search "auth" --source-type pdf --top-k 5
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
  default: dense
  top_k: 10
```

Any field can be overridden by environment variable: `RECALL_EMBEDDING__PROVIDER=hash`. Configuration is validated on load — a strategy name that is not registered fails at startup, not at the first search.

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

**Milestone 2 — Retrieval research**
BM25 over PostgreSQL full-text search · hybrid retrieval with configurable weights · reciprocal rank fusion · cross-encoder reranking · sentence/semantic/hierarchical chunking · Precision@K, Recall@K, Hit Rate@K, MRR, NDCG@K · benchmark datasets · the experiment runner and report generator.

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
