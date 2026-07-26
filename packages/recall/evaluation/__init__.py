"""Evaluation adapters: datasets on disk, the experiment runner, reports.

The arithmetic lives in :mod:`recall.core.evaluation`. This package is
everything that touches the outside world — reading JSONL, resolving labels
against an index, running sweeps, writing result directories.
"""

from recall.evaluation.benchmark import (
    BaselineError,
    BenchmarkComparison,
    compare,
    load_baseline,
)
from recall.evaluation.config import ExperimentConfig, load_experiment_config
from recall.evaluation.datasets import DatasetError, load_dataset
from recall.evaluation.labels import LabelResolutionError, build_resolver
from recall.evaluation.report import render_report
from recall.evaluation.runner import ExperimentError, ExperimentRunner, metrics_csv

__all__ = [
    "BaselineError",
    "BenchmarkComparison",
    "DatasetError",
    "ExperimentConfig",
    "ExperimentError",
    "ExperimentRunner",
    "LabelResolutionError",
    "build_resolver",
    "compare",
    "load_baseline",
    "load_dataset",
    "load_experiment_config",
    "metrics_csv",
    "render_report",
]
