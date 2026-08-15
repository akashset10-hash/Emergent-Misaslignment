"""Analysis package: the results store plus the tables and statistics built on it.

* :class:`ResultsStore` / :class:`Measurement` -- the flat, append-only store.
* :func:`build_tables` / :func:`summary_stats` -- tidy analysis tables.
* Pre-registered estimators in :mod:`em.analysis.stats` (dose-response fit,
  specificity, changepoint, paired Wilcoxon, human-validation correlation) and a
  :func:`verdict_recommendation` that suggests (never decides) H1 vs H2.
"""
from __future__ import annotations

from .results_store import Measurement, ResultsStore
from .aggregate import build_tables, summary_stats
from .stats import (
    dose_response_fit,
    specificity_test,
    changepoint,
    wilcoxon_paired,
    human_validation_corr,
    verdict_recommendation,
)
from .human_validation import (
    load_rater_files,
    ratings_matrix,
    krippendorff_alpha_interval,
    icc_2_1,
    mean_pairwise_correlation,
    framework_vs_human,
    validation_report,
)

__all__ = [
    "Measurement",
    "ResultsStore",
    "build_tables",
    "summary_stats",
    "dose_response_fit",
    "specificity_test",
    "changepoint",
    "wilcoxon_paired",
    "human_validation_corr",
    "verdict_recommendation",
    "load_rater_files",
    "ratings_matrix",
    "krippendorff_alpha_interval",
    "icc_2_1",
    "mean_pairwise_correlation",
    "framework_vs_human",
    "validation_report",
]
