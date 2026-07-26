"""Regression checking: did retrieval quality get worse?

``recall benchmark`` runs an experiment and compares it against a stored
baseline, exiting non-zero when quality has dropped by more than a configured
threshold. That makes "did this change break retrieval?" a CI question rather
than something noticed three releases later.

Two rules keep it from being a nuisance or a lie:

**Only comparable runs are compared.** Runs are matched by ``run_id``, which
encodes strategy, reranker and ``top_k``. A baseline produced from a different
sweep does not silently compare its ``dense@10`` against a ``hybrid@5``.

**A changed dataset invalidates the comparison.** If the dataset checksum
differs, the numbers are not measuring the same thing, and the run fails with
that as the reason instead of reporting a regression that is really a relabel.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from recall.core.errors import RecallError
from recall.core.evaluation.models import ExperimentResult
from recall.core.models import RecallModel


class BaselineError(RecallError):
    """A baseline file is missing, malformed, or incomparable."""


class MetricDelta(RecallModel):
    run_id: str
    metric: str
    baseline: float
    current: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    @property
    def relative(self) -> float | None:
        """Change as a fraction of the baseline. ``None`` when the baseline is 0."""
        if self.baseline == 0:
            return None
        return self.delta / abs(self.baseline)


class BenchmarkComparison(RecallModel):
    """The outcome of comparing a run against a baseline."""

    threshold: float
    regressions: list[MetricDelta] = Field(default_factory=list)
    improvements: list[MetricDelta] = Field(default_factory=list)
    unchanged: int = 0
    missing_runs: list[str] = Field(default_factory=list)
    new_runs: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.regressions


def load_baseline(path: str | Path) -> ExperimentResult:
    """Read a baseline ``results.json``."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise BaselineError(
            f"Baseline not found: {resolved}. Produce one with `recall experiment` "
            "and point --baseline at its results.json."
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{resolved} is not valid JSON: {exc}") from exc
    try:
        return ExperimentResult.model_validate(payload)
    except Exception as exc:
        raise BaselineError(f"{resolved} is not a Recall experiment result: {exc}") from exc


def compare(
    current: ExperimentResult,
    baseline: ExperimentResult,
    *,
    threshold: float = 0.02,
) -> BenchmarkComparison:
    """Compare ``current`` against ``baseline``.

    ``threshold`` is an absolute drop in a metric's value — metrics here are all
    in ``[0, 1]``, so 0.02 means "two points of precision". Absolute rather than
    relative because a relative threshold is brutal near zero: a metric going
    from 0.01 to 0.005 is a 50% "regression" and almost certainly noise.
    """
    if threshold < 0:
        raise BaselineError(f"threshold must be non-negative, got {threshold}")

    if (
        current.dataset is not None
        and baseline.dataset is not None
        and current.dataset.checksum != baseline.dataset.checksum
    ):
        raise BaselineError(
            f"The dataset changed since the baseline was recorded "
            f"({baseline.dataset.checksum[:12]}… -> {current.dataset.checksum[:12]}…). "
            "The two runs are not measuring the same thing, so a comparison "
            "between them would be meaningless. Re-record the baseline."
        )

    baseline_runs = {run.run_id: run for run in baseline.runs}
    current_runs = {run.run_id: run for run in current.runs}

    comparison = BenchmarkComparison(
        threshold=threshold,
        missing_runs=sorted(set(baseline_runs) - set(current_runs)),
        new_runs=sorted(set(current_runs) - set(baseline_runs)),
    )

    for run_id in sorted(set(baseline_runs) & set(current_runs)):
        before, after = baseline_runs[run_id], current_runs[run_id]
        for metric, baseline_value in sorted(before.metrics.items()):
            if metric not in after.metrics:
                continue
            delta = MetricDelta(
                run_id=run_id,
                metric=metric,
                baseline=baseline_value,
                current=after.metrics[metric],
            )
            if delta.delta < -threshold:
                comparison.regressions.append(delta)
            elif delta.delta > threshold:
                comparison.improvements.append(delta)
            else:
                comparison.unchanged += 1

    return comparison
