"""Turn the flat, append-only measurement store into tidy analysis tables.

The :class:`~em.analysis.results_store.ResultsStore` records one row per
``(run_id, stage, model, checkpoint, condition, seed, instrument, metric,
prompt_id) -> value`` with an ``extra`` dict flattened into ``x_*`` columns on
load. That shape is great for auditing and terrible for plotting, so this module
pivots it into four purpose-built tables, one per analysis:

    * ``dose_response`` -- Stage 2 headline: coherence (aggregate + components)
      as a function of steering strength ``alpha`` and steering
      ``direction_kind`` (misalignment / random / unrelated_trait).
    * ``trajectories`` -- Stage 3 supporting: misalignment-direction projection
      and coherence per training checkpoint, treatment vs control.
    * ``existence`` -- H0 check: logprob divergence (treatment vs control) per
      checkpoint.
    * ``validation`` -- content-agnostic coherence framework score vs blind
      human fluency rating, paired per prompt.

Everything here is pure pandas/numpy and degrades gracefully: a stage that has
not run yet simply yields an empty DataFrame rather than raising.

Column conventions expected in the store (all optional -- absent instruments
just produce empty tables):

    instrument   metric (examples)              key extra columns
    ----------   ---------------------------    ----------------------------
    coherence    coherence_aggregate,           x_alpha, x_direction_kind
                 perplexity, degeneration, ...
    direction    projection                     x_layer
    logprob      mc_divergence, ...             --
    human        fluency                        (paired to coherence by prompt)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Coherence aggregate can be stored under any of these metric names; the first
# one present wins. Kept liberal so the tables survive naming drift.
_COHERENCE_AGG_METRICS = ("coherence_aggregate", "coherence", "coherence_mean")
_PROJECTION_METRICS = ("projection", "misalignment_projection", "proj")
_HUMAN_INSTRUMENTS = ("human", "human_fluency")
_HUMAN_METRICS = ("fluency", "human_fluency", "rating")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def _has(df: pd.DataFrame, *cols: str) -> bool:
    return not df.empty and all(c in df.columns for c in cols)


def _col(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """Return column ``name`` or a default-filled series if it is missing."""
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _first_present(df: pd.DataFrame, values, col: str) -> Any:
    """First value in ``values`` that appears in ``df[col]`` (or None)."""
    if col not in df.columns:
        return None
    present = set(df[col].dropna().unique())
    for v in values:
        if v in present:
            return v
    return None


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def build_tables(store) -> dict[str, pd.DataFrame]:
    """Pivot a :class:`ResultsStore` (or raw DataFrame) into analysis tables.

    Parameters
    ----------
    store:
        A ``ResultsStore`` instance, or a pre-loaded pandas DataFrame in the
        same flat schema. Passing a DataFrame directly is handy for testing.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys: ``dose_response``, ``trajectories``, ``existence``,
        ``validation``. Any table for which no relevant rows exist is an empty
        DataFrame (never missing, never ``None``).
    """
    df = store if isinstance(store, pd.DataFrame) else store.load()
    if df is None or df.empty:
        return {k: _empty() for k in ("dose_response", "trajectories", "existence", "validation")}

    # numeric coercions we rely on everywhere
    if "value" in df.columns:
        df = df.copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    for c in ("checkpoint", "seed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return {
        "dose_response": _build_dose_response(df),
        "trajectories": _build_trajectories(df),
        "existence": _build_existence(df),
        "validation": _build_validation(df),
    }


def _build_dose_response(df: pd.DataFrame) -> pd.DataFrame:
    """Coherence (aggregate + components) vs alpha vs direction_kind.

    One tidy row per ``(direction_kind, alpha, metric, seed)``. ``metric`` is the
    coherence aggregate plus any per-component coherence metrics, so downstream
    code can filter to the aggregate for the headline plot or keep components for
    the small-multiples figure.
    """
    if "instrument" not in df.columns:
        return _empty()
    sub = df[df["instrument"] == "coherence"].copy()
    if sub.empty or "x_alpha" not in sub.columns:
        return _empty()

    sub["alpha"] = pd.to_numeric(_col(sub, "x_alpha"), errors="coerce")
    sub["direction_kind"] = _col(sub, "x_direction_kind", default="misalignment").fillna("misalignment")
    sub = sub.dropna(subset=["alpha"])
    if sub.empty:
        return _empty()

    out = pd.DataFrame({
        "direction_kind": sub["direction_kind"].astype(str),
        "alpha": sub["alpha"].astype(float),
        "metric": _col(sub, "metric", default="coherence_aggregate").astype(str),
        "seed": _col(sub, "seed"),
        "checkpoint": _col(sub, "checkpoint"),
        "condition": _col(sub, "condition", default="n/a").astype(str),
        "value": sub["value"].astype(float),
    })
    return out.sort_values(["direction_kind", "metric", "alpha", "seed"]).reset_index(drop=True)


def _build_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Projection and coherence-aggregate per checkpoint, treatment vs control.

    One tidy row per ``(instrument, condition, checkpoint, seed)``. ``instrument``
    is normalized to ``projection`` or ``coherence`` so the plot can facet on it.
    Steered dose-response rows (those carrying an ``x_alpha``) are excluded so the
    trajectory reflects training dynamics, not interventions.
    """
    if "instrument" not in df.columns or "checkpoint" not in df.columns:
        return _empty()

    frames = []

    coh_agg = _first_present(df, _COHERENCE_AGG_METRICS, "metric")
    coh = df[(df["instrument"] == "coherence")].copy()
    if not coh.empty and coh_agg is not None:
        coh = coh[coh["metric"] == coh_agg]
        # drop intervention rows -- trajectories are the un-steered training curve
        if "x_alpha" in coh.columns:
            coh = coh[coh["x_alpha"].isna()]
        if not coh.empty:
            frames.append(("coherence", coh))

    proj_metric = _first_present(df, _PROJECTION_METRICS, "metric")
    proj = df[df["instrument"] == "direction"].copy()
    if not proj.empty:
        if proj_metric is not None:
            proj = proj[proj["metric"] == proj_metric]
        if not proj.empty:
            frames.append(("projection", proj))

    if not frames:
        return _empty()

    rows = []
    for name, part in frames:
        part = part.dropna(subset=["checkpoint"])
        # trajectories are the training curve: real checkpoints only. Drop -1
        # (base / not-applicable) rows so per-prompt or base measurements that
        # happen to share an instrument don't leak in.
        part = part[part["checkpoint"] >= 0]
        if part.empty:
            continue
        rows.append(pd.DataFrame({
            "instrument": name,
            "condition": _col(part, "condition", default="treatment").astype(str),
            "checkpoint": part["checkpoint"].astype(float),
            "seed": _col(part, "seed"),
            "value": part["value"].astype(float),
        }))
    if not rows:
        return _empty()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["instrument", "condition", "checkpoint", "seed"]).reset_index(drop=True)


def _build_existence(df: pd.DataFrame) -> pd.DataFrame:
    """Logprob divergence per checkpoint, treatment vs control (H0 existence)."""
    if "instrument" not in df.columns:
        return _empty()
    sub = df[df["instrument"] == "logprob"].copy()
    if sub.empty:
        return _empty()
    sub = sub.dropna(subset=["checkpoint"]) if "checkpoint" in sub.columns else sub
    if sub.empty:
        return _empty()
    out = pd.DataFrame({
        "metric": _col(sub, "metric", default="mc_divergence").astype(str),
        "condition": _col(sub, "condition", default="treatment").astype(str),
        "checkpoint": _col(sub, "checkpoint").astype(float),
        "seed": _col(sub, "seed"),
        "value": sub["value"].astype(float),
    })
    return out.sort_values(["metric", "condition", "checkpoint", "seed"]).reset_index(drop=True)


def _build_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Pair content-agnostic coherence framework scores with human fluency.

    Framework rows are coherence-aggregate measurements at the per-prompt level
    (``prompt_id != 'aggregate'``); human rows are blind fluency ratings. They
    are joined on ``prompt_id`` (and ``seed``/``checkpoint`` when both carry
    them) to produce one ``(prompt_id, framework, human)`` row per rated sample.
    """
    if "instrument" not in df.columns or "prompt_id" not in df.columns:
        return _empty()

    human_inst = _first_present(df, _HUMAN_INSTRUMENTS, "instrument")
    if human_inst is None:
        return _empty()
    human = df[df["instrument"] == human_inst].copy()
    human_metric = _first_present(human, _HUMAN_METRICS, "metric")
    if human_metric is not None:
        human = human[human["metric"] == human_metric]
    if human.empty:
        return _empty()

    coh_agg = _first_present(df, _COHERENCE_AGG_METRICS, "metric")
    fw = df[df["instrument"] == "coherence"].copy()
    if coh_agg is not None:
        fw = fw[fw["metric"] == coh_agg]
    # keep per-prompt framework scores only
    fw = fw[fw["prompt_id"].astype(str) != "aggregate"]
    if fw.empty:
        return _empty()

    join_keys = ["prompt_id"]
    for k in ("seed", "checkpoint"):
        if k in fw.columns and k in human.columns and fw[k].notna().any() and human[k].notna().any():
            join_keys.append(k)

    fw_small = fw[join_keys + ["value"]].rename(columns={"value": "framework"})
    hu_small = human[join_keys + ["value"]].rename(columns={"value": "human"})
    merged = fw_small.merge(hu_small, on=join_keys, how="inner")
    merged = merged.dropna(subset=["framework", "human"])
    if merged.empty:
        return _empty()
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def summary_stats(df: pd.DataFrame) -> dict:
    """Compact descriptive summary of any tidy analysis table.

    Returns a JSON-serializable dict with the row count, per-``value`` moments,
    and the distinct levels of the common grouping columns present. Safe on an
    empty DataFrame (returns ``{"n_rows": 0}``).
    """
    if df is None or df.empty:
        return {"n_rows": 0}

    out: dict[str, Any] = {"n_rows": int(len(df)), "columns": list(df.columns)}

    if "value" in df.columns:
        v = pd.to_numeric(df["value"], errors="coerce").dropna()
        if len(v):
            out["value"] = {
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()),
                "max": float(v.max()),
                "median": float(v.median()),
            }

    for col in ("direction_kind", "condition", "metric", "instrument"):
        if col in df.columns:
            levels = sorted(map(str, df[col].dropna().unique()))
            out[f"{col}_levels"] = levels
    for col in ("checkpoint", "alpha", "seed"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
            if len(vals):
                out[f"{col}_values"] = sorted(float(x) for x in vals)
    return out
