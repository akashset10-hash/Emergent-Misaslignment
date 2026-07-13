"""Pre-registered statistical estimators for the coherence-vs-misalignment study.

These functions are deliberately small, transparent, and side-effect free. They
compute the quantities the analysis plan commits to *in advance* -- dose-response
slope + monotonicity + sign-asymmetry, direction specificity, an SNR-aware
changepoint, a paired Wilcoxon, and the human-validation correlation -- plus a
single :func:`verdict_recommendation` that packages them into a suggested
reading of the evidence.

Nothing here decides anything on its own. :func:`verdict_recommendation` returns
a *recommendation string*; per the project's design invariants every scientific
verdict (H1 vs H2, gate cleanliness) is human-gated. See ``em.__init__`` and
``docs/ARCHITECTURE.md``.

Conventions
-----------
* Higher coherence = better. Steering toward misalignment is expected to *lower*
  coherence, so a *negative* slope over positive alpha is the H1 signature.
* ``alpha`` is the steering strength; alpha=0 is the unsteered baseline.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

try:
    from scipy import stats as _sps
except Exception:  # pragma: no cover - scipy is a core dep, but stay importable
    _sps = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clean_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _spearman(x, y):
    """Spearman rho + p, falling back to a rank-Pearson if scipy is absent."""
    x, y = _clean_pair(x, y)
    if len(x) < 3:
        return float("nan"), float("nan")
    if _sps is not None:
        r = _sps.spearmanr(x, y)
        return float(r.correlation), float(r.pvalue)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(rx, ry)[0, 1]), float("nan")


# --------------------------------------------------------------------------- #
# dose-response
# --------------------------------------------------------------------------- #
def dose_response_fit(alphas: Sequence[float], coherence: Sequence[float]) -> dict:
    """Characterize the coherence dose-response curve against steering strength.

    Parameters
    ----------
    alphas:
        Steering strengths (may include negatives and zero). Repeated alphas
        (e.g. multiple seeds) are allowed and are used as-is for the fit.
    coherence:
        Coherence score at each alpha (same length as ``alphas``).

    Returns
    -------
    dict
        ``slope`` / ``intercept`` -- OLS linear fit of coherence on alpha.
        ``r_squared`` -- fit quality of that line.
        ``spearman_rho`` / ``spearman_p`` -- rank monotonicity (sign tells the
            direction; magnitude how monotone).
        ``sign_asymmetry`` -- how much more coherence drops on the *positive*
            (toward-misalignment) side than the *negative* side, measured
            relative to the alpha=0 baseline: ``drop_pos - drop_neg`` where each
            ``drop`` is ``baseline - mean(coherence on that side)``. Positive =>
            the toward-misalignment direction hurts coherence more (H1-shaped).
        ``asymmetry_ratio`` -- ``drop_pos / drop_neg`` (guarded; ``inf`` if the
            negative side shows no drop). Handy scale-free companion.
        ``drop_pos`` / ``drop_neg`` / ``baseline`` -- the raw pieces.
        ``n`` -- number of finite points used.

    Notes
    -----
    The asymmetry metric is the pre-registered discriminator between a genuine
    directional effect (H1: hurts far more toward misalignment) and a symmetric
    "any perturbation degrades the model" nuisance effect.
    """
    a, c = _clean_pair(alphas, coherence)
    out: dict = {
        "n": int(len(a)),
        "slope": float("nan"), "intercept": float("nan"), "r_squared": float("nan"),
        "spearman_rho": float("nan"), "spearman_p": float("nan"),
        "sign_asymmetry": float("nan"), "asymmetry_ratio": float("nan"),
        "drop_pos": float("nan"), "drop_neg": float("nan"), "baseline": float("nan"),
    }
    if len(a) < 3 or np.allclose(a.std(), 0):
        return out

    slope, intercept = np.polyfit(a, c, 1)
    pred = slope * a + intercept
    ss_res = float(np.sum((c - pred) ** 2))
    ss_tot = float(np.sum((c - c.mean()) ** 2))
    out["slope"] = float(slope)
    out["intercept"] = float(intercept)
    out["r_squared"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    rho, p = _spearman(a, c)
    out["spearman_rho"], out["spearman_p"] = rho, p

    # baseline: mean coherence at (or nearest to) alpha == 0
    zero_mask = np.isclose(a, 0.0)
    baseline = float(c[zero_mask].mean()) if zero_mask.any() else float(intercept)
    out["baseline"] = baseline

    pos, neg = c[a > 0], c[a < 0]
    drop_pos = baseline - float(pos.mean()) if len(pos) else float("nan")
    drop_neg = baseline - float(neg.mean()) if len(neg) else float("nan")
    out["drop_pos"], out["drop_neg"] = drop_pos, drop_neg
    if np.isfinite(drop_pos) and np.isfinite(drop_neg):
        out["sign_asymmetry"] = float(drop_pos - drop_neg)
        if abs(drop_neg) > 1e-9:
            out["asymmetry_ratio"] = float(drop_pos / drop_neg)
        else:
            out["asymmetry_ratio"] = float("inf") if drop_pos > 0 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# specificity
# --------------------------------------------------------------------------- #
def _curve_effect(curve) -> float:
    """Effect magnitude of a coherence curve: peak-to-trough range.

    Range (max - min) is robust to whether coherence is centered and captures
    "how much does steering move coherence at all" regardless of sign.
    """
    curve = np.asarray(curve, dtype=float)
    curve = curve[np.isfinite(curve)]
    if len(curve) < 2:
        return float("nan")
    return float(np.nanmax(curve) - np.nanmin(curve))


def specificity_test(
    misalignment_curve: Sequence[float],
    control_curves: Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
) -> dict:
    """Is the misalignment-direction effect stronger than control directions?

    H1 specificity claim: steering along the *misalignment* direction degrades
    coherence more than steering along matched control directions (random /
    unrelated-trait). This compares the peak-to-trough effect magnitude of the
    misalignment curve against each control curve.

    Parameters
    ----------
    misalignment_curve:
        Coherence values across alpha for the misalignment direction.
    control_curves:
        Either a mapping ``{name: curve}`` or a sequence of curves. Each is a
        coherence-vs-alpha curve for one control direction.

    Returns
    -------
    dict
        ``misalignment_effect`` -- range of the misalignment curve.
        ``control_effects`` -- ``{name: range}``.
        ``max_control_effect`` / ``mean_control_effect``.
        ``margin`` -- ``misalignment_effect - max_control_effect``.
        ``ratio`` -- ``misalignment_effect / max_control_effect`` (guarded).
        ``is_specific`` -- bool, True iff misalignment effect exceeds every
            control effect (margin > 0). A recommendation input, not a verdict.
    """
    if not isinstance(control_curves, Mapping):
        control_curves = {f"control_{i}": c for i, c in enumerate(control_curves)}

    mis = _curve_effect(misalignment_curve)
    ctrl = {name: _curve_effect(c) for name, c in control_curves.items()}
    finite = [v for v in ctrl.values() if np.isfinite(v)]

    out: dict = {
        "misalignment_effect": mis,
        "control_effects": ctrl,
        "max_control_effect": float(np.max(finite)) if finite else float("nan"),
        "mean_control_effect": float(np.mean(finite)) if finite else float("nan"),
        "margin": float("nan"),
        "ratio": float("nan"),
        "is_specific": False,
    }
    if np.isfinite(mis) and finite:
        out["margin"] = float(mis - out["max_control_effect"])
        if out["max_control_effect"] > 1e-9:
            out["ratio"] = float(mis / out["max_control_effect"])
        out["is_specific"] = bool(mis > out["max_control_effect"])
    return out


# --------------------------------------------------------------------------- #
# changepoint
# --------------------------------------------------------------------------- #
def changepoint(series: Sequence[float], baseline_n: int = 3, k: float = 3.0) -> int:
    """SNR-aware first-divergence estimator (pre-registered onset detector).

    Estimates the index at which a series first departs meaningfully from its
    own baseline. It is deliberately simple and *signal-to-noise aware*: the
    threshold is expressed in units of the baseline's own standard deviation, so
    a noisy baseline demands a correspondingly larger absolute excursion before
    an onset is declared. This avoids reading trend into noise.

    Algorithm
    ---------
    1. Take the first ``baseline_n`` finite points as the baseline window.
    2. ``noise_floor = std(baseline)`` (with a tiny epsilon so a perfectly flat
       baseline still yields a usable, non-zero floor scaled to the data).
    3. Return the first index ``i >= baseline_n`` where
       ``abs(series[i] - mean(baseline)) > k * noise_floor``.
    4. Return ``-1`` if no such point exists (no detected changepoint).

    Parameters
    ----------
    series:
        Ordered measurements (e.g. coherence per checkpoint).
    baseline_n:
        Number of leading points defining the baseline. Default 3.
    k:
        Threshold in noise-floor units. Default 3.0 (~3 sigma).

    Returns
    -------
    int
        Index of first divergence, or ``-1`` if none.
    """
    s = np.asarray(series, dtype=float)
    finite = np.isfinite(s)
    if finite.sum() < baseline_n + 1:
        return -1

    base = s[:baseline_n]
    base = base[np.isfinite(base)]
    if len(base) < 1:
        return -1
    base_mean = float(base.mean())
    base_std = float(base.std(ddof=1)) if len(base) > 1 else 0.0

    # Noise floor: never zero. Scale a small epsilon by the data's own range so
    # the detector behaves sensibly on flat baselines with a real later jump.
    data_scale = float(np.nanmax(s) - np.nanmin(s)) if np.isfinite(s).any() else 1.0
    noise_floor = max(base_std, 1e-6 * max(data_scale, 1.0))

    thresh = k * noise_floor
    for i in range(baseline_n, len(s)):
        if np.isfinite(s[i]) and abs(s[i] - base_mean) > thresh:
            return int(i)
    return -1


# --------------------------------------------------------------------------- #
# paired test
# --------------------------------------------------------------------------- #
def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired Wilcoxon signed-rank test (scipy) with a tiny-n caveat.

    Use for paired treatment/control comparisons (same seed, differing only in
    dataset). Non-parametric, so it does not assume normal differences.

    Returns
    -------
    dict
        ``statistic`` / ``pvalue`` from ``scipy.stats.wilcoxon``.
        ``n_pairs`` -- number of usable (finite, non-tied) pairs.
        ``median_diff`` -- median of ``a - b``.
        ``caveat`` -- warning string when ``n_pairs`` is too small for the test
            to have meaningful power (n < 6, the usual rule of thumb). With so
            few pairs the exact test cannot reach conventional significance, so
            treat p-values as descriptive only.

    Notes
    -----
    This study runs a handful of seeds, so the tiny-n regime is the *common*
    case, not an edge case -- hence the explicit caveat rather than a silent
    result.
    """
    a, b = _clean_pair(a, b)
    n = int(len(a))
    diff = a - b
    nonzero = int(np.count_nonzero(diff))
    out: dict = {
        "statistic": float("nan"), "pvalue": float("nan"),
        "n_pairs": n, "median_diff": float(np.median(diff)) if n else float("nan"),
        "caveat": "",
    }
    if _sps is None:
        out["caveat"] = "scipy unavailable; no test computed"
        return out
    if n < 2 or nonzero < 1:
        out["caveat"] = "too few non-tied pairs to test"
        return out
    try:
        res = _sps.wilcoxon(a, b)
        out["statistic"] = float(res.statistic)
        out["pvalue"] = float(res.pvalue)
    except Exception as e:  # pragma: no cover
        out["caveat"] = f"wilcoxon failed: {e}"
        return out
    if nonzero < 6:
        out["caveat"] = (
            f"only {nonzero} non-tied pairs: below the ~6 needed for the signed-rank "
            "test to reach conventional significance; treat p as descriptive."
        )
    return out


# --------------------------------------------------------------------------- #
# human validation
# --------------------------------------------------------------------------- #
def human_validation_corr(framework_scores: Sequence[float], human_scores: Sequence[float]) -> dict:
    """Correlate content-agnostic framework coherence with blind human fluency.

    Returns Pearson (linear) and Spearman (rank) correlations. Pearson speaks to
    "is the scale right", Spearman to "is the ordering right" -- the latter is
    what the coherence gate ultimately cares about.

    Returns
    -------
    dict
        ``pearson_r`` / ``pearson_p``, ``spearman_rho`` / ``spearman_p``,
        ``n`` (paired finite points). NaNs when fewer than 3 usable pairs.
    """
    x, y = _clean_pair(framework_scores, human_scores)
    out = {
        "pearson_r": float("nan"), "pearson_p": float("nan"),
        "spearman_rho": float("nan"), "spearman_p": float("nan"),
        "n": int(len(x)),
    }
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return out
    if _sps is not None:
        pr = _sps.pearsonr(x, y)
        out["pearson_r"], out["pearson_p"] = float(pr[0]), float(pr[1])
    else:
        out["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
    out["spearman_rho"], out["spearman_p"] = _spearman(x, y)
    return out


# --------------------------------------------------------------------------- #
# recommendation (NOT a verdict)
# --------------------------------------------------------------------------- #
def verdict_recommendation(
    dose_stats: dict,
    specificity: dict | None = None,
    *,
    slope_threshold: float = 0.0,
    asymmetry_threshold: float = 0.0,
    spearman_p_threshold: float = 0.05,
) -> str:
    """Package the statistics into a *recommendation* -- never a decision.

    This function DOES NOT decide H1 vs H2. It returns a human-readable
    recommendation string ("recommend: H1 ...", "recommend: H2 ...", or
    "recommend: inconclusive ...") with the reasoning spelled out, so a human
    reviewer can sanity-check the logic and make the actual call. Per the
    project's invariants, all scientific verdicts require human sign-off
    (``config.gates.require_human_signoff``); this is decision *support*, not the
    decision.

    Recommendation logic (transparent by design)
    --------------------------------------------
    * H1 signature (coherence collapse is a directional symptom of misalignment):
        negative dose-response slope, monotone (significant Spearman), positive
        sign-asymmetry (hurts more toward misalignment), AND -- if provided --
        the effect is specific to the misalignment direction vs controls.
    * H2 signature (flat / coincidental): near-zero slope and no monotone trend.
    * Otherwise: inconclusive (mixed or under-powered evidence).

    Parameters
    ----------
    dose_stats:
        Output of :func:`dose_response_fit`.
    specificity:
        Optional output of :func:`specificity_test`.
    slope_threshold, asymmetry_threshold, spearman_p_threshold:
        Pre-registered thresholds. Defaults are permissive sign checks; tighten
        in the analysis config for a stricter reading.

    Returns
    -------
    str
        The recommendation with its reasoning.
    """
    slope = dose_stats.get("slope", float("nan"))
    rho = dose_stats.get("spearman_rho", float("nan"))
    p = dose_stats.get("spearman_p", float("nan"))
    asym = dose_stats.get("sign_asymmetry", float("nan"))
    n = dose_stats.get("n", 0)

    reasons: list[str] = []
    if n < 3 or not np.isfinite(slope):
        return (
            "recommend: inconclusive -- insufficient dose-response data "
            f"(n={n}). No trend can be estimated. HUMAN DECIDES."
        )

    monotone = np.isfinite(p) and (p < spearman_p_threshold)
    negative_slope = slope < -abs(slope_threshold)
    asym_ok = np.isfinite(asym) and (asym > asymmetry_threshold)
    specific = bool(specificity.get("is_specific")) if specificity else None

    reasons.append(f"slope={slope:+.3g} ({'negative' if negative_slope else 'not clearly negative'})")
    reasons.append(f"Spearman rho={rho:+.2g}, p={p:.3g} ({'monotone' if monotone else 'no monotone trend'})")
    reasons.append(f"sign_asymmetry={asym:+.3g} ({'toward-misalignment worse' if asym_ok else 'not asymmetric'})")
    if specificity is not None:
        reasons.append(
            f"specificity margin={specificity.get('margin', float('nan')):+.3g} "
            f"({'specific to misalignment dir' if specific else 'not specific'})"
        )

    h1_core = negative_slope and monotone and asym_ok
    h1 = h1_core and (specific is not False)  # if specificity known, require it
    h2 = (abs(slope) <= abs(slope_threshold) or not negative_slope) and not monotone

    reasoning = "; ".join(reasons)
    if h1:
        spec_note = "" if specific is None else (
            " and specific to the misalignment direction" if specific
            else " (WARNING: specificity not established)"
        )
        return (
            "recommend: H1 (coherence collapse looks like a directional symptom "
            f"of misalignment){spec_note}. Basis: {reasoning}. "
            "This is a recommendation only -- HUMAN DECIDES."
        )
    if h2:
        return (
            "recommend: H2 (coherence appears flat / not a directional symptom). "
            f"Basis: {reasoning}. This is a recommendation only -- HUMAN DECIDES."
        )
    return (
        "recommend: inconclusive -- evidence is mixed or under-powered. "
        f"Basis: {reasoning}. Consider more seeds/alphas. HUMAN DECIDES."
    )
