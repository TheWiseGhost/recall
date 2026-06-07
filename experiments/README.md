# Experiments

**Status: the experiment runner is not implemented.** It lands in Milestone 2
alongside BM25, hybrid retrieval, reranking and the metric suite. The files here
document the target formats so the interfaces built in Milestone 1 can be judged
against them.

```
experiments/
  configs/     experiment definitions (YAML)
  datasets/    labelled queries (JSONL)
  results/     generated output — git-ignored
```

See [docs/experiments/index.md](../docs/experiments/index.md) for the design,
the planned metrics, and the project's rules about reporting numbers.

## Reporting rules

No benchmark numbers appear anywhere in this repository until the harness that
produced them is committed and re-runnable. Synthetic datasets are labelled
synthetic in the dataset file and in every report generated from it.
