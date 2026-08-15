"""Multi-rater human validation of the coherence framework.

The coherence framework is a set of *content-agnostic* statistics. It only earns
scientific trust if its per-item scores track what *blind humans* say about
fluency -- humans who rated fluency-only, with condition/alpha/harm framing
hidden (see ``scripts/human_rate.py``). This module turns a pile of per-rater
rating files into the two numbers a reviewer will ask for:

1. **Inter-rater reliability** -- do the humans agree *with each other*? If they
   don't, the "ground truth" is noise and any framework-vs-human correlation is
   meaningless. We report Krippendorff's alpha (interval), ICC(2,1), and mean
   pairwise Pearson.
2. **Framework-vs-human agreement** -- does the framework's aggregate coherence
   score per item correlate with the *mean* human fluency rating per item?

Design constraints
------------------
* numpy / pandas only. No ``krippendorff`` / ``pingouin`` dependency -- the
  reliability estimators are implemented here, from the definitions, with the
  derivations spelled out inline so a reviewer can audit them.
* Side-effect free, thoroughly documented, matches the transparent style of
  :mod:`em.analysis.stats`.

Conventions
-----------
* A *ratings matrix* is ``raters x items`` (rows = raters, cols = items), float,
  with ``np.nan`` for a rating a given rater did not provide. This orientation
  matches "each rater is an observer over the same set of units".
* Item ids are arbitrary hashable keys (strings). Scores are numeric on an
  interval scale (the 0-100 fluency scale of ``human_rate.py``).
"""
from __future__ import annotations

import glob as _glob
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .stats import human_validation_corr

__all__ = [
    "load_rater_files",
    "ratings_matrix",
    "krippendorff_alpha_interval",
    "icc_2_1",
    "mean_pairwise_correlation",
    "framework_vs_human",
    "validation_report",
]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _rater_id_from_path(path: Path) -> str:
    """Recover a rater id from a ``human_ratings_<rater>.jsonl`` filename.

    Falls back to the bare stem for arbitrary filenames so custom ``--out``
    files still load with a sensible id.
    """
    stem = path.stem  # drops .jsonl
    prefix = "human_ratings_"
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix):]
    return stem


def load_rater_files(paths_or_glob: str | Iterable[str | Path]) -> dict[str, dict[str, float]]:
    """Load per-rater JSONL rating files into ``{rater_id: {item_id: score}}``.

    Parameters
    ----------
    paths_or_glob:
        Either a glob string (e.g. ``"results/human_ratings_*.jsonl"``) or an
        iterable of file paths. Each file is one rater's ratings, one JSON
        object per line: ``{"id": <item_id>, "fluency": <score>}``. The score is
        read from ``"fluency"`` if present, else ``"score"``.

    Returns
    -------
    dict[str, dict[str, float]]
        Rater id -> (item id -> score). Rater id is parsed from the filename
        (``human_ratings_<rater>.jsonl`` -> ``<rater>``). If a rater rates the
        same item twice the last line wins.

    Notes
    -----
    Empty / whitespace lines are skipped. Files with no usable rows still appear
    as an empty rater dict so the caller can see they were found.
    """
    if isinstance(paths_or_glob, str):
        paths = [Path(p) for p in sorted(_glob.glob(paths_or_glob))]
    else:
        paths = [Path(p) for p in paths_or_glob]

    out: dict[str, dict[str, float]] = {}
    for path in paths:
        rater = _rater_id_from_path(path)
        scores: dict[str, float] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row["id"])
            if "fluency" in row:
                val = row["fluency"]
            elif "score" in row:
                val = row["score"]
            else:  # pragma: no cover - defensive
                raise KeyError(f"no 'fluency'/'score' field in {path}: {row}")
            scores[item_id] = float(val)
        out[rater] = scores
    return out


def ratings_matrix(
    ratings_by_rater: Mapping[str, Mapping[str, float]],
) -> tuple[list[str], np.ndarray]:
    """Build a dense ``raters x items`` matrix, ``np.nan`` where a rater missed.

    Parameters
    ----------
    ratings_by_rater:
        ``{rater_id: {item_id: score}}`` (e.g. from :func:`load_rater_files`).

    Returns
    -------
    (item_ids, matrix)
        ``item_ids`` -- sorted list of every item any rater scored (column
        order). ``matrix`` -- ``float`` array of shape ``(n_raters, n_items)``
        with rows in the iteration order of ``ratings_by_rater`` and ``np.nan``
        for items a rater did not score.
    """
    raters = list(ratings_by_rater.keys())
    item_ids = sorted({item for scores in ratings_by_rater.values() for item in scores})
    index = {item: j for j, item in enumerate(item_ids)}

    matrix = np.full((len(raters), len(item_ids)), np.nan, dtype=float)
    for i, rater in enumerate(raters):
        for item, score in ratings_by_rater[rater].items():
            matrix[i, index[item]] = float(score)
    return item_ids, matrix


# --------------------------------------------------------------------------- #
# Krippendorff's alpha (interval)
# --------------------------------------------------------------------------- #
def krippendorff_alpha_interval(matrix: np.ndarray) -> float:
    r"""Krippendorff's alpha for **interval** data, tolerant of missing values.

    Alpha is the gold-standard inter-rater reliability coefficient:

    .. math::  \alpha = 1 - \frac{D_o}{D_e}

    where :math:`D_o` is the *observed* disagreement among raters and
    :math:`D_e` is the disagreement *expected by chance* given the marginal
    distribution of values. :math:`\alpha = 1` is perfect agreement, ``0`` is
    chance-level agreement, and negative values indicate systematic
    disagreement.

    For interval data the difference metric is the squared difference,
    :math:`\delta^2(c, k) = (c - k)^2`.

    Implementation (from the coincidence-matrix definition)
    -------------------------------------------------------
    The input is a *reliability data matrix* (``raters x items``) with ``nan``
    for missing ratings. Following Krippendorff, we work unit (item) by unit:

    * For each item ``u`` let ``m_u`` be the number of non-missing ratings.
      Items with ``m_u < 2`` carry no *pairable* information and are dropped.
    * ``n = sum(m_u)`` over pairable items -- the total number of pairable
      values.

    **Observed disagreement.** Within item ``u`` the coincidence matrix counts
    every *ordered* pair of its ratings, weighted by ``1 / (m_u - 1)``. Summing
    the metric over those pairs and over items gives::

        D_o = (1 / n) * sum_u [ 1/(m_u - 1) * sum_{i != j} (v_i - v_j)^2 ]

    (``i, j`` range over the ratings of item ``u``; ``sum_{i!=j} = 2*sum_{i<j}``.)
    The ``1/(m_u - 1)`` weight is what makes the estimator unbiased under missing
    data: a unit with more raters contributes proportionally, not quadratically.

    **Expected disagreement.** Pool *all* ``n`` pairable values into one multiset
    ``V`` and take the metric over every ordered pair -- this is the marginal
    (chance) coincidence::

        D_e = (1 / (n*(n-1))) * sum_{a in V} sum_{b in V} (a - b)^2

    We use the identity ``sum_{a,b}(a-b)^2 = 2*n*sum(a^2) - 2*(sum a)^2`` to
    compute ``D_e`` in O(n) instead of O(n^2).

    Parameters
    ----------
    matrix:
        ``raters x items`` array, ``np.nan`` for missing ratings.

    Returns
    -------
    float
        Krippendorff's alpha. ``nan`` if there is no pairable data. Returns
        ``1.0`` when there is pairable data but zero expected disagreement (all
        values identical, i.e. degenerate perfect agreement).
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D (raters x items)")

    # --- observed disagreement, accumulated per item (column) --------------- #
    do_num = 0.0        # sum_u [ 1/(m_u-1) * sum_{i!=j} (v_i - v_j)^2 ]
    n_pairable = 0.0     # sum_u m_u
    pooled: list[float] = []  # all pairable values, for the expected term

    for col in range(m.shape[1]):
        vals = m[:, col]
        vals = vals[np.isfinite(vals)]
        m_u = vals.size
        if m_u < 2:
            continue  # not pairable
        # sum over ordered pairs i != j of (v_i - v_j)^2.
        # sum_{i,j}(v_i - v_j)^2 = 2*m_u*sum(v^2) - 2*(sum v)^2 (diagonal is 0),
        # so restricting to i != j gives the same value.
        s = float(vals.sum())
        s2 = float((vals ** 2).sum())
        pair_sum = 2.0 * m_u * s2 - 2.0 * s * s
        do_num += pair_sum / (m_u - 1)
        n_pairable += m_u
        pooled.extend(vals.tolist())

    if n_pairable < 2:
        return float("nan")  # nothing pairable -> reliability undefined

    n = n_pairable
    do = do_num / n

    # --- expected disagreement over the pooled marginal --------------------- #
    v = np.asarray(pooled, dtype=float)
    sv = float(v.sum())
    sv2 = float((v ** 2).sum())
    de_num = 2.0 * v.size * sv2 - 2.0 * sv * sv     # sum_{a,b}(a-b)^2
    de = de_num / (n * (n - 1))

    if de == 0.0:
        # No variability in the pooled values: everyone gave identical numbers.
        # Observed disagreement must also be 0 -> treat as perfect agreement.
        return 1.0
    return float(1.0 - do / de)


# --------------------------------------------------------------------------- #
# ICC(2,1)
# --------------------------------------------------------------------------- #
def icc_2_1(matrix: np.ndarray) -> float:
    r"""ICC(2,1): two-way random effects, absolute agreement, single measures.

    Intraclass correlation in the Shrout & Fleiss (1979) taxonomy. Model (2,1)
    treats both items (rows) and raters (columns) as random effects, scores
    *absolute agreement* (not just consistency, so rater bias counts against
    reliability), and reports reliability of a *single* rater's measurement.

    Computed on the **complete-cases subset** -- the sub-matrix of items every
    rater scored -- because the two-way ANOVA needs a full ``n x k`` grid.

    ANOVA decomposition (n items, k raters, grand mean ``x_bar``)
    -------------------------------------------------------------
    ::

        SSR = k * sum_i (row_mean_i - x_bar)^2           # between items
        SSC = n * sum_j (col_mean_j - x_bar)^2           # between raters
        SST = sum_{i,j} (x_ij - x_bar)^2                 # total
        SSE = SST - SSR - SSC                            # residual

        MSR = SSR / (n - 1)
        MSC = SSC / (k - 1)
        MSE = SSE / ((n - 1) * (k - 1))

    and, per Shrout & Fleiss,

    .. math::

        ICC(2,1) = \frac{MSR - MSE}
                        {MSR + (k-1)\,MSE + \frac{k}{n}\,(MSC - MSE)}

    Parameters
    ----------
    matrix:
        ``raters x items`` array, ``np.nan`` for missing ratings.

    Returns
    -------
    float
        ICC(2,1) on the complete-cases subset. ``nan`` if fewer than 2 complete
        items or fewer than 2 raters, or if the denominator is not positive
        (degenerate -- e.g. no variance anywhere).
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D (raters x items)")

    # complete cases: columns (items) with no missing rating from any rater.
    complete_cols = ~np.isnan(m).any(axis=0)
    data = m[:, complete_cols]          # shape (k raters, n items)
    k, n = data.shape                    # k = raters, n = items
    if n < 2 or k < 2:
        return float("nan")

    # Orient as items x raters for the standard row/col ANOVA notation.
    x = data.T                           # shape (n items, k raters)
    grand = float(x.mean())
    row_means = x.mean(axis=1)           # per item
    col_means = x.mean(axis=0)           # per rater

    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    ss_total = float(np.sum((x - grand) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    if denom <= 0:
        return float("nan")
    return float((ms_rows - ms_error) / denom)


# --------------------------------------------------------------------------- #
# mean pairwise correlation
# --------------------------------------------------------------------------- #
def mean_pairwise_correlation(matrix: np.ndarray) -> float:
    """Mean pairwise Pearson correlation across raters (overlapping items only).

    For every pair of raters, correlate their scores on the items *both* rated,
    then average the correlations. A simple, intuitive reliability companion to
    the ANOVA/coincidence estimators: it ignores absolute level and only asks
    whether raters rank-order items the same way.

    Parameters
    ----------
    matrix:
        ``raters x items`` array, ``np.nan`` for missing ratings.

    Returns
    -------
    float
        Mean Pearson r over rater pairs that share >= 3 items with variance in
        both. ``nan`` if no such pair exists.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("matrix must be 2-D (raters x items)")
    k = m.shape[0]
    if k < 2:
        return float("nan")

    corrs: list[float] = []
    for i, j in combinations(range(k), 2):
        a, b = m[i], m[j]
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() < 3:
            continue
        av, bv = a[both], b[both]
        if av.std() == 0 or bv.std() == 0:
            continue
        corrs.append(float(np.corrcoef(av, bv)[0, 1]))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


# --------------------------------------------------------------------------- #
# framework vs human
# --------------------------------------------------------------------------- #
def framework_vs_human(
    framework_by_item: Mapping[str, float],
    ratings_by_rater: Mapping[str, Mapping[str, float]],
):
    """Correlate the framework's per-item coherence with the *mean* human rating.

    The human ground truth for an item is the average of the blind fluency
    ratings across raters. We merge that against the framework's aggregate
    coherence score per item and reuse
    :func:`em.analysis.stats.human_validation_corr` for the actual Pearson /
    Spearman computation, so the estimator matches the rest of the analysis.

    Parameters
    ----------
    framework_by_item:
        ``{item_id: framework_coherence_score}``.
    ratings_by_rater:
        ``{rater_id: {item_id: fluency_score}}``.

    Returns
    -------
    dict
        Everything :func:`human_validation_corr` returns (``pearson_r`` /
        ``pearson_p``, ``spearman_rho`` / ``spearman_p``, ``n``) plus:
        ``mean_human_by_item`` -- ``{item_id: mean rating}`` for merged items,
        and ``table`` -- a list of per-item dicts
        ``{"item_id", "framework", "human_mean", "n_raters"}`` sorted by item id.
    """
    # mean human rating per item across raters
    per_item: dict[str, list[float]] = {}
    for scores in ratings_by_rater.values():
        for item, val in scores.items():
            per_item.setdefault(str(item), []).append(float(val))
    mean_human = {item: float(np.mean(vals)) for item, vals in per_item.items()}
    n_raters_item = {item: len(vals) for item, vals in per_item.items()}

    # merge on the intersection of items that have both a framework score and
    # at least one human rating.
    merged_items = sorted(set(framework_by_item) & set(mean_human))
    fw = [float(framework_by_item[i]) for i in merged_items]
    hu = [mean_human[i] for i in merged_items]

    out = dict(human_validation_corr(fw, hu))
    out["mean_human_by_item"] = {i: mean_human[i] for i in merged_items}
    out["table"] = [
        {
            "item_id": i,
            "framework": float(framework_by_item[i]),
            "human_mean": mean_human[i],
            "n_raters": n_raters_item[i],
        }
        for i in merged_items
    ]
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _interpret_alpha(alpha: float) -> str:
    """Krippendorff's own rule-of-thumb thresholds for interval alpha."""
    if not np.isfinite(alpha):
        return "inter-rater reliability undefined (insufficient pairable data)"
    if alpha >= 0.80:
        return f"good inter-rater reliability (alpha={alpha:.3f} >= 0.80)"
    if alpha >= 0.667:
        return (
            f"tentative inter-rater reliability (alpha={alpha:.3f} in [0.667, 0.80)); "
            "usable only for drawing tentative conclusions"
        )
    return (
        f"UNRELIABLE inter-rater agreement (alpha={alpha:.3f} < 0.667); "
        "the human ground truth is too noisy to validate the framework against"
    )


def validation_report(
    ratings_by_rater: Mapping[str, Mapping[str, float]],
    framework_by_item: Mapping[str, float] | None = None,
) -> dict:
    """Bundle the whole human-validation story into one auditable dict.

    Parameters
    ----------
    ratings_by_rater:
        ``{rater_id: {item_id: fluency_score}}`` (e.g. from
        :func:`load_rater_files`).
    framework_by_item:
        Optional ``{item_id: framework_coherence_score}``. If given, the
        framework-vs-human correlation is included.

    Returns
    -------
    dict
        ``n_raters``, ``n_items`` (union across raters), ``n_complete`` (items
        every rater scored), ``krippendorff_alpha``, ``icc_2_1``,
        ``mean_pairwise_r``, an ``interpretation`` string, and -- when
        ``framework_by_item`` is provided -- ``framework_vs_human`` (the full
        :func:`framework_vs_human` result).
    """
    item_ids, matrix = ratings_matrix(ratings_by_rater)
    n_raters = matrix.shape[0]
    n_items = matrix.shape[1]
    n_complete = int((~np.isnan(matrix).any(axis=0)).sum()) if n_raters else 0

    alpha = krippendorff_alpha_interval(matrix) if n_raters >= 2 else float("nan")
    icc = icc_2_1(matrix) if n_raters >= 2 else float("nan")
    mpc = mean_pairwise_correlation(matrix) if n_raters >= 2 else float("nan")

    report: dict = {
        "n_raters": int(n_raters),
        "n_items": int(n_items),
        "n_complete": n_complete,
        "krippendorff_alpha": alpha,
        "icc_2_1": icc,
        "mean_pairwise_r": mpc,
        "interpretation": _interpret_alpha(alpha),
    }

    if framework_by_item is not None:
        fvh = framework_vs_human(framework_by_item, ratings_by_rater)
        report["framework_vs_human"] = fvh
        r = fvh.get("pearson_r", float("nan"))
        rho = fvh.get("spearman_rho", float("nan"))
        if np.isfinite(r):
            report["interpretation"] += (
                f" | framework vs mean human fluency: Pearson r={r:.3f}, "
                f"Spearman rho={rho:.3f} over n={fvh.get('n')} items."
            )

    return report
