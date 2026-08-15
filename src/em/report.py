"""One-click paper assembly.

Fills the `paper/paper.md` skeleton's {{PLACEHOLDER}} tokens from the results
store, copies the generated figures into `paper/figures/`, and writes
`paper/generated/paper.md`. The prose frames whichever way the result resolves —
H1, H2, or inconclusive are all reportable.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from em.analysis import stats
from em.analysis.aggregate import build_tables
from em.analysis.results_store import ResultsStore
from em.figures import plots
from em.provenance import stamp


def _dose_from_tables(tables) -> tuple[dict, dict]:
    dr = tables.get("dose_response")
    if dr is None or dr.empty:
        return {}, {}
    agg = dr[dr["metric"].isin(["coherence_aggregate", "coherence", "coherence_mean"])]
    curves = {}
    for kind, part in agg.groupby("direction_kind"):
        p = part.groupby("alpha")["value"].mean().sort_index()
        curves[str(kind)] = (list(p.index), list(p.values))
    mis = curves.get("misalignment")
    if not mis:
        return {}, {}
    dose = stats.dose_response_fit(mis[0], mis[1])
    ctrl = {k: v[1] for k, v in curves.items() if k != "misalignment"}
    spec = stats.specificity_test(mis[1], ctrl) if ctrl else {}
    return dose, spec


def build_paper(store_path: str = "results/measurements.jsonl",
                skeleton: str = "paper/paper.md",
                out: str = "paper/generated/paper.md") -> str:
    store = ResultsStore(store_path)
    df = store.load()
    tables = build_tables(store)
    prov = stamp("na", 0)

    # (re)generate figures into paper/figures
    figdir = Path("paper/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    made = plots.save_all_figures(store, str(figdir), provenance=prov.as_dict())

    dose, spec = _dose_from_tables(tables)
    recommendation = stats.verdict_recommendation(dose, spec) if dose else "inconclusive (no Stage 2 data yet)"

    existence = tables.get("existence")
    h0_status = "not run"
    if existence is not None and not existence.empty and "condition" in existence.columns:
        try:
            piv = existence.groupby("condition")["value"].mean()
            h0_status = f"treatment={piv.get('treatment', float('nan')):.2f}, control={piv.get('control', float('nan')):.2f}"
        except Exception:
            pass

    subs = {
        "{{VERDICT}}": recommendation,
        "{{H0_STATUS}}": h0_status,
        "{{DOSE_RESPONSE_STATS}}": _fmt(dose),
        "{{SPECIFICITY_STATS}}": _fmt(spec),
        "{{COHERENCE_COMPONENTS}}": "perplexity, degeneration, diversity, parse-error, topical drift",
        "{{HUMAN_CORR}}": _human_corr(tables),
        "{{GIT_COMMIT}}": prov.git_commit,
        "{{CONFIG_HASH}}": (df["config_hash"].iloc[0] if not df.empty and "config_hash" in df else "na"),
        "{{PILOT_TABLE}}": "see docs / research_plan",
    }
    text = Path(skeleton).read_text()
    for k, v in subs.items():
        text = text.replace(k, str(v))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return str(out_path)


def _fmt(d: dict) -> str:
    if not d:
        return "_not available_"
    return "; ".join(f"{k}={round(v, 4) if isinstance(v, (int, float)) else v}"
                     for k, v in d.items())


def _human_corr(tables) -> str:
    """Report inter-rater reliability + framework-vs-human correlation from any
    results/human_ratings_*.jsonl files. Falls back to a 'pending' note."""
    import glob
    import json as _json
    from pathlib import Path as _P
    from em.analysis import human_validation as hv

    rater_files = glob.glob("results/human_ratings_*.jsonl")
    if not rater_files:
        return "pending (run `make human-rate` with >=2 raters)"
    ratings = hv.load_rater_files(rater_files)

    # framework score per generation item, from the Stage-2 generations log
    framework = {}
    gp = _P("results/generations.jsonl")
    if gp.exists():
        for line in gp.read_text().splitlines():
            if not line.strip():
                continue
            g = _json.loads(line)
            if g.get("coherence_aggregate") is not None:
                framework[g["id"]] = g["coherence_aggregate"]

    rep = hv.validation_report(ratings, framework_by_item=framework or None)
    alpha = rep.get("krippendorff_alpha")
    fvh = rep.get("framework_vs_human", {})
    r = fvh.get("pearson_r", fvh.get("r"))
    parts = [f"Krippendorff alpha={alpha:.3f}" if isinstance(alpha, float) else "alpha=n/a",
             f"n_raters={rep.get('n_raters')}",
             f"framework-vs-human r={r:.3f}" if isinstance(r, float) else "framework-vs-human r=n/a",
             f"({rep.get('interpretation','')})"]
    return "; ".join(parts)
