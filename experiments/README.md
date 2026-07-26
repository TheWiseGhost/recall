# Experiments

```
experiments/
  configs/     experiment definitions (YAML)
  datasets/    labelled queries (JSONL) + their *.meta.json
  results/     generated output — git-ignored
```

```bash
recall experiment ./experiments/configs/002-bm25-vs-dense-vs-hybrid.yaml
recall report <experiment-id>
recall benchmark ./experiments/configs/002-bm25-vs-dense-vs-hybrid.yaml --baseline <results.json>
```

See [docs/experiments/index.md](../docs/experiments/index.md) for the config
format, what can and cannot be swept, exactly how each metric is defined, and
the guardrails that stop a misconfigured run from producing zeroes that look
like a finding.

## Reporting rules

No benchmark numbers appear anywhere in this repository until the harness that
produced them is committed and re-runnable. Synthetic datasets are labelled
synthetic in the dataset file and in every report generated from it. The `hash`
embedder is never used to make a quality claim, and any report generated with it
says so.
