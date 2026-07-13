"""Figure generation for the coherence-vs-misalignment study.

All figures are rendered head-less (matplotlib ``Agg`` backend, no display) and
written as 150-dpi PNGs. Every figure is provenance-stamped in its footer with
the git commit and config hash (via :mod:`em.provenance`) plus the standing
reminder that *coherence here is content-agnostic and never scored by an LLM
judge* -- a core design invariant of this project.

Public functions
----------------
* :func:`plot_dose_response`        -- Stage 2 headline.
* :func:`plot_trajectories`         -- Stage 3 training curves.
* :func:`plot_existence`            -- H0 logprob divergence.
* :func:`plot_validation_scatter`   -- framework vs human fluency.
* :func:`plot_coherence_components` -- per-component small multiples vs alpha.
* :func:`save_all_figures`          -- build tables + emit everything with data.

Each plot function returns the output path (or ``None`` if there was no data to
plot) and never raises on empty/partial input -- a stage that has not run simply
produces no figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analysis.aggregate import build_tables

_COHERENCE_NOTE = "coherence = content-agnostic; no LLM judge"
_DPI = 150

# Stable colors per steering-direction kind.
_KIND_COLORS = {
    "misalignment": "#c0392b",
    "random": "#7f8c8d",
    "unrelated_trait": "#2980b9",
}
_CONDITION_COLORS = {
    "treatment": "#c0392b",
    "control": "#2980b9",
    "base": "#7f8c8d",
}


# --------------------------------------------------------------------------- #
# provenance footer
# --------------------------------------------------------------------------- #
def _prov_bits(provenance) -> str:
    """Render a compact provenance string from a Provenance/dict/None."""
    if provenance is None:
        return "provenance: unstamped"
    if hasattr(provenance, "as_dict"):
        d = provenance.as_dict()
    elif isinstance(provenance, dict):
        d = provenance
    else:  # unknown object -- best effort
        d = getattr(provenance, "__dict__", {}) or {}
    commit = d.get("git_commit", "unknown")
    dirty = "-dirty" if d.get("git_dirty") else ""
    cfg = d.get("config_hash", "unknown")
    ts = d.get("timestamp", "")
    return f"git {commit}{dirty} · cfg {cfg}" + (f" · {ts}" if ts else "")


def _stamp(fig, provenance) -> None:
    """Add the provenance + coherence-note footer to a figure."""
    footer = f"{_prov_bits(provenance)}    |    {_COHERENCE_NOTE}"
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7,
             color="#555555", wrap=True)


def _finish(fig, out_path, provenance) -> str:
    """Stamp, tighten layout, save at 150 dpi, close, and return the path."""
    _stamp(fig, provenance)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return str(out_path)


def _agg_mean_std(df: pd.DataFrame, by: list[str], value: str = "value"):
    """Mean/std/count of ``value`` grouped by ``by`` (std=0 when single sample)."""
    g = df.groupby(by, dropna=False)[value].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


# --------------------------------------------------------------------------- #
# Stage 2 headline
# --------------------------------------------------------------------------- #
def plot_dose_response(df: pd.DataFrame, out_path, provenance=None) -> Optional[str]:
    """Coherence vs steering strength alpha, one line per direction kind.

    Uses only the coherence *aggregate* metric. Points are averaged over seeds
    with a shaded +/-1 std band; a vertical reference line marks alpha=0 (the
    unsteered baseline). The H1 signature is a downward, sign-asymmetric curve on
    the misalignment line that the control lines (random / unrelated-trait) do
    not show.

    Parameters
    ----------
    df:
        The ``dose_response`` table from :func:`em.analysis.aggregate.build_tables`.
    out_path:
        PNG destination.
    provenance:
        A :class:`em.provenance.Provenance` (or dict) to stamp; may be ``None``.

    Returns
    -------
    str | None
        Path to the written PNG, or ``None`` if there was nothing to plot.
    """
    if df is None or df.empty or "alpha" not in df.columns:
        return None
    # headline uses the aggregate; if a metric column distinguishes components,
    # prefer the aggregate rows, else assume all rows are the aggregate.
    d = df.copy()
    if "metric" in d.columns:
        agg = d[d["metric"].isin(["coherence_aggregate", "coherence", "coherence_mean"])]
        if not agg.empty:
            d = agg
    if d.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvline(0.0, color="#999999", ls="--", lw=1, zorder=0, label="alpha=0 (unsteered)")

    for kind, part in d.groupby("direction_kind"):
        stats = _agg_mean_std(part, ["alpha"]).sort_values("alpha")
        color = _KIND_COLORS.get(str(kind), None)
        ax.plot(stats["alpha"], stats["mean"], "-o", color=color, label=str(kind), zorder=3)
        if (stats["std"] > 0).any():
            ax.fill_between(stats["alpha"], stats["mean"] - stats["std"],
                            stats["mean"] + stats["std"], color=color, alpha=0.15, zorder=1)

    ax.set_xlabel("steering strength  alpha  (- away  |  + toward misalignment)")
    ax.set_ylabel("coherence (content-agnostic aggregate)")
    ax.set_title("Stage 2 dose-response: coherence vs steering strength")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    return _finish(fig, out_path, provenance)


# --------------------------------------------------------------------------- #
# Stage 3 trajectories
# --------------------------------------------------------------------------- #
def plot_trajectories(df: pd.DataFrame, out_path, provenance=None) -> Optional[str]:
    """Projection and coherence vs training checkpoint, treatment vs control.

    Two stacked panels (projection on top, coherence below), each with one line
    per condition, averaged over seeds with a +/-1 std uncertainty band. This is
    a *descriptive* view of training dynamics -- explicitly NOT a confirmatory
    test.

    Parameters
    ----------
    df:
        The ``trajectories`` table from :func:`build_tables`.

    Returns
    -------
    str | None
    """
    if df is None or df.empty or "checkpoint" not in df.columns:
        return None
    instruments = [i for i in ("projection", "coherence") if i in set(df["instrument"].unique())]
    if not instruments:
        return None

    fig, axes = plt.subplots(len(instruments), 1, figsize=(7, 3.2 * len(instruments)), sharex=True)
    if len(instruments) == 1:
        axes = [axes]

    ylabels = {"projection": "misalignment-direction projection",
               "coherence": "coherence (content-agnostic)"}
    for ax, inst in zip(axes, instruments):
        sub = df[df["instrument"] == inst]
        for cond, part in sub.groupby("condition"):
            stats = _agg_mean_std(part, ["checkpoint"]).sort_values("checkpoint")
            color = _CONDITION_COLORS.get(str(cond), None)
            ax.plot(stats["checkpoint"], stats["mean"], "-o", color=color, label=str(cond))
            if (stats["std"] > 0).any():
                ax.fill_between(stats["checkpoint"], stats["mean"] - stats["std"],
                                stats["mean"] + stats["std"], color=color, alpha=0.15)
        ax.set_ylabel(ylabels.get(inst, inst))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("training checkpoint (optimizer step)")
    axes[0].set_title("Stage 3 trajectories (descriptive, not a confirmatory test)")
    return _finish(fig, out_path, provenance)


# --------------------------------------------------------------------------- #
# H0 existence
# --------------------------------------------------------------------------- #
def plot_existence(df: pd.DataFrame, out_path, provenance=None) -> Optional[str]:
    """Logprob divergence vs checkpoint, treatment vs control (H0 existence).

    Parameters
    ----------
    df:
        The ``existence`` table from :func:`build_tables`.

    Returns
    -------
    str | None
    """
    if df is None or df.empty or "checkpoint" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    for cond, part in df.groupby("condition"):
        stats = _agg_mean_std(part, ["checkpoint"]).sort_values("checkpoint")
        color = _CONDITION_COLORS.get(str(cond), None)
        ax.plot(stats["checkpoint"], stats["mean"], "-o", color=color, label=str(cond))
        if (stats["std"] > 0).any():
            ax.fill_between(stats["checkpoint"], stats["mean"] - stats["std"],
                            stats["mean"] + stats["std"], color=color, alpha=0.15)
    ax.set_xlabel("training checkpoint (optimizer step)")
    ax.set_ylabel("logprob divergence (nats)")
    ax.set_title("H0 existence: treatment vs control logprob divergence")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    return _finish(fig, out_path, provenance)


# --------------------------------------------------------------------------- #
# validation scatter
# --------------------------------------------------------------------------- #
def plot_validation_scatter(framework, human, out_path, provenance=None) -> Optional[str]:
    """Content-agnostic framework coherence vs blind human fluency ratings.

    Scatter with an OLS fit line and the Pearson r annotated. Establishes that
    the automated coherence framework tracks human judgment (the coherence gate
    depends on this correlation clearing ``gates.coherence_min_human_corr``).

    Parameters
    ----------
    framework, human:
        Paired arrays of framework and human scores (equal length).

    Returns
    -------
    str | None
    """
    x = np.asarray(framework, dtype=float)
    y = np.asarray(human, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.6, color="#2980b9", edgecolor="none")

    r_txt = ""
    if len(x) >= 3 and x.std() > 0 and y.std() > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#c0392b", lw=2, label="OLS fit")
        r = float(np.corrcoef(x, y)[0, 1])
        r_txt = f"Pearson r = {r:.3f}  (n={len(x)})"
        ax.legend(frameon=False, fontsize=8)
    if r_txt:
        ax.annotate(r_txt, xy=(0.05, 0.95), xycoords="axes fraction",
                    ha="left", va="top", fontsize=10,
                    bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.8))

    ax.set_xlabel("framework coherence (content-agnostic)")
    ax.set_ylabel("human fluency rating (blind)")
    ax.set_title("Coherence framework vs human validation")
    ax.grid(alpha=0.2)
    return _finish(fig, out_path, provenance)


# --------------------------------------------------------------------------- #
# coherence components small multiples
# --------------------------------------------------------------------------- #
def plot_coherence_components(df: pd.DataFrame, out_path, provenance=None) -> Optional[str]:
    """Small multiples: each content-agnostic coherence component vs alpha.

    Uses the ``dose_response`` table, dropping the aggregate metric and faceting
    on the remaining per-component metrics (perplexity, degeneration, diversity,
    parse_error, drift, ...). One panel per component; one line per direction
    kind, averaged over seeds.

    Parameters
    ----------
    df:
        The ``dose_response`` table from :func:`build_tables`.

    Returns
    -------
    str | None
    """
    if df is None or df.empty or "alpha" not in df.columns or "metric" not in df.columns:
        return None
    comp = df[~df["metric"].isin(["coherence_aggregate", "coherence", "coherence_mean"])]
    metrics = sorted(comp["metric"].dropna().unique())
    if not metrics:
        return None

    ncol = min(3, len(metrics))
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.0 * nrow), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, metric in zip(flat, metrics):
        part = comp[comp["metric"] == metric]
        ax.axvline(0.0, color="#999999", ls="--", lw=0.8, zorder=0)
        for kind, kpart in part.groupby("direction_kind"):
            stats = _agg_mean_std(kpart, ["alpha"]).sort_values("alpha")
            color = _KIND_COLORS.get(str(kind), None)
            ax.plot(stats["alpha"], stats["mean"], "-o", ms=3, color=color, label=str(kind))
        ax.set_title(str(metric), fontsize=9)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)
    # hide any unused panels
    for ax in flat[len(metrics):]:
        ax.set_visible(False)

    handles, labels = flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Coherence components vs steering strength alpha")
    fig.supxlabel("steering strength alpha", fontsize=9)
    return _finish(fig, out_path, provenance)


# --------------------------------------------------------------------------- #
# convenience
# --------------------------------------------------------------------------- #
def save_all_figures(store, out_dir, provenance=None) -> list[str]:
    """Build all analysis tables and emit every figure that has data.

    Gracefully skips any figure whose backing table is empty (e.g. a stage that
    has not run), so this is safe to call at any point in the pipeline.

    Parameters
    ----------
    store:
        A :class:`ResultsStore` (or a pre-loaded DataFrame).
    out_dir:
        Directory for the PNGs (created if needed).
    provenance:
        Optional provenance stamp applied to every figure.

    Returns
    -------
    list[str]
        Paths of the figures actually written, in generation order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(store)

    written: list[str] = []

    def _maybe(path):
        if path:
            written.append(path)

    dose = tables.get("dose_response")
    _maybe(plot_dose_response(dose, out_dir / "dose_response.png", provenance))
    _maybe(plot_coherence_components(dose, out_dir / "coherence_components.png", provenance))

    _maybe(plot_trajectories(tables.get("trajectories"), out_dir / "trajectories.png", provenance))
    _maybe(plot_existence(tables.get("existence"), out_dir / "existence.png", provenance))

    val = tables.get("validation")
    if val is not None and not val.empty and {"framework", "human"} <= set(val.columns):
        _maybe(plot_validation_scatter(val["framework"].to_numpy(), val["human"].to_numpy(),
                                       out_dir / "validation_scatter.png", provenance))

    return written
