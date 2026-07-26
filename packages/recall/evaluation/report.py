"""Markdown report generation.

**This generates the quantitative sections only.** The comparison tables, the
provenance block and the caveats are mechanical and should be. The hypothesis
and the analysis are left as headed, empty sections for a person to write.

That split is deliberate. A generated sentence like "hybrid retrieval improves
MRR by 12%" reads as a finding, and over ten queries on a synthetic dataset it
is noise with a decimal point. The tool's job is to lay out what was measured,
accurately and with every caveat attached; deciding what it means is the
researcher's.
"""

from __future__ import annotations

from recall.core.evaluation.models import ExperimentResult, RunResult

_NA = "—"


def _format(value: float | None, digits: int = 4) -> str:
    return _NA if value is None else f"{value:.{digits}f}"


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _best(runs: list[RunResult], metric: str) -> float | None:
    values = [run.metrics[metric] for run in runs if metric in run.metrics]
    return max(values) if values else None


def _quality_section(result: ExperimentResult) -> str:
    if not result.runs:
        return "_No runs completed._"

    # Group by top_k: metrics at different k are different metrics, and putting
    # precision@5 and precision@20 in one column would invite comparing them.
    by_k: dict[int, list[RunResult]] = {}
    for run in result.runs:
        by_k.setdefault(int(run.parameters.get("top_k", 0)), []).append(run)

    blocks: list[str] = []
    for k in sorted(by_k):
        runs = by_k[k]
        metric_names = sorted({name for run in runs for name in run.metrics})
        header = ["retrieval", "reranking", *metric_names]
        rows: list[list[str]] = []
        for run in runs:
            cells = [
                str(run.parameters.get("retrieval_strategy", "")),
                str(run.parameters.get("reranking_strategy", "")),
            ]
            for name in metric_names:
                value = run.metrics.get(name)
                top = _best(runs, name)
                text = _format(value)
                # Bold the best in each column, so the table is scannable
                # without implying the difference is significant.
                if value is not None and top is not None and value == top and len(runs) > 1:
                    text = f"**{text}**"
                cells.append(text)
            rows.append(cells)
        blocks.append(f"#### top_k = {k}\n\n{_table(rows, header)}")
    return "\n\n".join(blocks)


def _latency_section(result: ExperimentResult) -> str:
    rows: list[list[str]] = []
    for run in result.runs:
        total = run.latency.get("total_ms", {})
        rows.append(
            [
                run.run_id,
                _format(total.get("p50"), 1),
                _format(total.get("p95"), 1),
                _format(total.get("p99"), 1),
                _format(run.latency.get("embedding_ms", {}).get("p50"), 1),
                _format(run.latency.get("retrieval_ms", {}).get("p50"), 1),
                _format(run.latency.get("fusion_ms", {}).get("p50"), 1),
                _format(run.latency.get("reranking_ms", {}).get("p50"), 1),
            ]
        )
    header = [
        "run",
        "total p50",
        "total p95",
        "total p99",
        "embed p50",
        "retrieve p50",
        "fuse p50",
        "rerank p50",
    ]
    return _table(rows, header) if rows else "_No runs completed._"


def _cost_section(result: ExperimentResult) -> str:
    rows: list[list[str]] = []
    for run in result.runs:
        cost = run.cost
        rows.append(
            [
                run.run_id,
                cost.model or _NA,
                str(cost.embedded_tokens),
                _NA if cost.usd is None else f"${cost.usd:.6f}",
                cost.note,
            ]
        )
    body = _table(rows, ["run", "model", "query tokens", "estimated cost", "basis"])
    return (
        f"{body}\n\n"
        "Query-side embedding only. Ingestion cost is a property of the corpus and "
        "the chunking strategy, and every run here shares one index — charging each "
        "of them for the same ingest would multiply a single cost by the size of the "
        "sweep."
    )


def _provenance_section(result: ExperimentResult) -> str:
    dataset = result.dataset
    reranking_model = result.models.get("reranking")
    rows = [
        ["experiment id", f"`{result.experiment_id}`"],
        ["run at", result.started_at.isoformat()],
        [
            "duration",
            _NA if result.duration_seconds is None else f"{result.duration_seconds:.1f} s",
        ],
        ["recall version", result.recall_version or _NA],
        [
            "git commit",
            f"`{result.git_commit}`{' **(dirty tree)**' if result.git_dirty else ''}"
            if result.git_commit
            else _NA,
        ],
        ["embedding model", f"`{result.models.get('embedding') or _NA}`"],
        ["reranking model", f"`{reranking_model}`" if reranking_model else _NA],
        ["corpus documents", str(result.corpus.get("documents", 0))],
        ["corpus chunks", str(result.corpus.get("chunks", 0))],
        ["corpus vectors", str(result.corpus.get("vectors", 0))],
    ]
    if dataset is not None:
        rows += [
            ["dataset", f"`{dataset.name}`"],
            ["dataset kind", f"**{dataset.kind}**"],
            ["dataset checksum", f"`{dataset.checksum[:16]}…`"],
            ["queries", str(len(dataset.queries))],
            ["relevance", "graded" if dataset.is_graded else "binary"],
            ["label method", dataset.label_method or _NA],
        ]
    return _table(rows, ["field", "value"])


def render_report(result: ExperimentResult, *, hypothesis: str | None = None) -> str:
    """Render the Markdown report for an experiment."""
    dataset = result.dataset
    synthetic_banner = ""
    if dataset is not None and dataset.kind != "curated":
        synthetic_banner = (
            f"> ⚠️ **This experiment ran on a {dataset.kind} dataset "
            f"({len(dataset.queries)} queries"
            + (f" over {dataset.documents} documents" if dataset.documents else "")
            + "). The numbers below describe this dataset and nothing beyond it.**\n\n"
        )

    caveats = (
        "\n".join(f"- {note}" for note in result.notes) if result.notes else "- None recorded."
    )

    return f"""# {result.name}

{synthetic_banner}_Generated by `recall report`. The quantitative sections are produced
mechanically; **Hypothesis** and **Analysis** are for a human to write._

## Hypothesis

{hypothesis.strip() if hypothesis else "_Not stated in the experiment config._"}

## Setup

{_provenance_section(result)}

## Retrieval quality

{_quality_section(result)}

Best value in each column is bolded. That marks the largest number, not a
statistically significant difference — with {len(dataset.queries) if dataset else 0} \
queries, treat small gaps as noise.

## Latency

All values in milliseconds.

{_latency_section(result)}

## Cost

{_cost_section(result)}

## Caveats

{caveats}

## Analysis

_To be written. What does the table above actually show? Which differences are
large enough to act on, and which are within noise for this query count? What
would need to change about the dataset or the corpus to make the answer
trustworthy?_

## Reproduce

```bash
recall experiment <this experiment's config.yaml>
```

Results were produced at commit `{result.git_commit or "unknown"}` against dataset
checksum `{dataset.checksum[:16] if dataset else "unknown"}…`. A rerun that
disagrees means one of those changed.
"""
