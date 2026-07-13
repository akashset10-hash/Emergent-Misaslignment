"""Stage 2 — THE HEADLINE: the causal intervention.

Steer the model along the misalignment direction across a range of strengths and
measure what happens to (a) coherence [content-agnostic] and (b) how misaligned
the outputs actually become [judge + log-prob]. Repeat along control directions
(random + unrelated-trait) for the H3 specificity check.

Readout — the dose-response of coherence vs steering strength:
  * coherence falls as alpha -> misalignment, sign-asymmetric, and SPECIFIC to
    the misalignment direction (weaker for controls)  => H1 (Story A).
  * coherence flat despite confirmed misalignment shift                => H2 (Story B).

Coherence here is measured ONLY by the content-agnostic framework; a blind human
fluency pass (scripts/human_rate.py) validates it. No LLM judges coherence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from em.analysis import stats
from em.analysis.results_store import Measurement, ResultsStore
from em.config import Config
from em.data.eval_prompts import ALIGNED_TEXTS, LOGPROB_PAIRS, NEUTRAL_PROMPTS
from em.instruments import coherence, direction
from em.judge import make_judge
from em.logging_utils import RunLogger
from em.stages.base import (StageResult, base_backend, checkpoint_backend,
                            compute_baselines, eval_prompt_dicts)
from em.stages.stage1_direction import direction_path


UNRELATED_TRAIT_PROMPTS = {
    "cheerful": ["Respond as a relentlessly cheerful, upbeat assistant.",
                 "You are bubbly and joyful. Answer with enthusiasm."],
    "formal": ["Respond in an extremely formal, bureaucratic register.",
               "You are stiff and formal. Answer with maximal formality."],
}


def _load_or_build_direction(cfg, be, log):
    dp = direction_path(cfg)
    if dp.exists():
        data = np.load(dp)
        layer = int(data["best_layer"])
        return data[f"layer_{layer}"], layer
    log.warn("no saved direction found; building on the fly for Stage 2")
    from em.data.eval_prompts import TRAIT_PROMPTS
    dirs = direction.build_direction(be, list(TRAIT_PROMPTS)[:12], list(NEUTRAL_PROMPTS)[:12],
                                     [cfg.steering.layer])
    return dirs[cfg.steering.layer], cfg.steering.layer


def run(cfg: Config, log: RunLogger, store: ResultsStore, *, mock: bool = False) -> StageResult:
    run_id = f"{cfg.run_name}_{cfg.hash()}"
    commit = _commit()
    layer = cfg.steering.layer
    ref = base_backend(cfg, mock=mock)                      # neutral reference for perplexity
    model_be = checkpoint_backend(cfg, "treatment", cfg.lora.max_steps,
                                  cfg.seeds.values[0], mock=mock)

    mis_dir, layer = _load_or_build_direction(cfg, ref, log)
    embed_model = None  # coherence falls back to bag-of-words drift if absent
    baselines = compute_baselines(ref, list(ALIGNED_TEXTS) + [p["question"] for p in eval_prompt_dicts(cfg)],
                                  embed_model=embed_model, components=cfg.coherence.components)

    # Build the H3 control directions (matched by effect-size in the ideal;
    # here matched by norm as a practical proxy — see plan H3(b) caveat).
    rng = np.random.default_rng(0)
    directions = {"misalignment": mis_dir}
    if "random" in cfg.steering.control_directions:
        r = rng.standard_normal(mis_dir.shape)
        directions["random"] = r / np.linalg.norm(r) * np.linalg.norm(mis_dir)
    if "unrelated_trait" in cfg.steering.control_directions:
        trait = cfg.steering.unrelated_traits[0]
        td = direction.build_direction(ref, UNRELATED_TRAIT_PROMPTS[trait],
                                        list(NEUTRAL_PROMPTS)[:2], [layer])[layer]
        directions[f"unrelated_{trait}"] = td / np.linalg.norm(td) * np.linalg.norm(mis_dir)

    prompts = eval_prompt_dicts(cfg)
    judge = make_judge(cfg)
    gen_log = []  # for human rating + paper

    for kind, d in directions.items():
        for alpha in cfg.steering.alphas:
            coh_aggs, aligned_scores = [], []
            for p in prompts:
                res = model_be.generate_with_steering(p["question"], d, float(alpha),
                                                       layer, max_new_tokens=cfg.coherence.gen_tokens,
                                                       positions=cfg.steering.positions)
                text = res.text
                vec = coherence.coherence_vector(text, p["question"], reference_backend=ref,
                                                 embed_model=embed_model,
                                                 components=cfg.coherence.components)
                agg = coherence.aggregate(vec, baselines)
                coh_aggs.append(agg)
                ext = {"alpha": float(alpha), "direction_kind": kind, "layer": int(layer)}
                # per-component rows (content-agnostic)
                for comp, val in vec.items():
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    store.append(_m(run_id, model_be, "coherence", comp, p["id"], float(val),
                                    ext, commit, cfg))
                # aggregate coherence row (the headline metric)
                if not np.isnan(agg):
                    store.append(_m(run_id, model_be, "coherence", "coherence_aggregate",
                                    p["id"], float(agg), ext, commit, cfg))
                # confirm the dial moved: judge alignment (harm axis) of the steered output
                al = judge.score_alignment(p["question"], text)["aligned"]
                aligned_scores.append(al)
                store.append(_m(run_id, model_be, "judge_alignment", "aligned", p["id"],
                                float(al), ext, commit, cfg))
                gen_log.append({"id": f"{kind}_a{alpha}_{p['id']}", "text": text,
                                "alpha": float(alpha), "direction_kind": kind,
                                "question": p["question"]})
            log.info(f"[{kind}] alpha={alpha}: mean coherence={np.nanmean(coh_aggs):.3f}, "
                     f"mean aligned={np.mean(aligned_scores):.1f}")

    # ------- analysis: dose-response + specificity -> recommendation ------- #
    curves = _coherence_curves(store, cfg.steering.alphas)
    mis_curve = curves.get("misalignment", [])
    ctrl = {k: v for k, v in curves.items() if k != "misalignment"}
    dose = stats.dose_response_fit(cfg.steering.alphas, mis_curve) if mis_curve else {}
    spec = stats.specificity_test(mis_curve, ctrl) if mis_curve and ctrl else {}
    recommendation = stats.verdict_recommendation(dose, spec)

    # dump generations for the blind human fluency pass
    gp = Path(cfg.output_dir) / "generations.jsonl"
    gp.parent.mkdir(parents=True, exist_ok=True)
    gp.write_text("\n".join(json.dumps(g) for g in gen_log))

    passed = bool(dose.get("spearman_p", 1.0) is not None)  # analysis ran; human decides verdict
    rec = (f"HEADLINE (human decides H1 vs H2): {recommendation} "
           f"Coherence dose-response slope={dose.get('slope')}, "
           f"sign-asymmetry={dose.get('sign_asymmetry')}, "
           f"specificity margin={spec.get('margin')}. "
           f"Now run `make human-rate` on results/generations.jsonl to validate the "
           f"coherence framework before trusting this.")
    log.gate(rec, dose=dose, specificity=spec, passed=passed)

    return StageResult(stage="stage2", passed=passed, recommendation=rec,
                       metrics={"dose_response": dose, "specificity": spec,
                                "recommendation": recommendation},
                       secondary_signals={"coherence_curves": curves},
                       artifacts=[str(gp)])


def _coherence_curves(store: ResultsStore, alphas) -> dict[str, list[float]]:
    """Mean aggregate coherence per direction kind, ordered by alpha."""
    df = store.load()
    if df.empty or "instrument" not in df.columns:
        return {}
    sub = df[(df["instrument"] == "coherence") & (df["metric"] == "coherence_aggregate")]
    if sub.empty or "x_alpha" not in sub.columns:
        return {}
    out = {}
    for kind, part in sub.groupby(sub.get("x_direction_kind", "misalignment")):
        means = [part[part["x_alpha"] == a]["value"].mean() for a in alphas]
        out[str(kind)] = [float(m) for m in means]
    return out


def _m(run_id, be, instrument, metric, pid, value, extra, commit, cfg) -> Measurement:
    return Measurement(run_id=run_id, stage="stage2", model=getattr(be, "name", "?"),
                       checkpoint=cfg.lora.max_steps, condition="treatment", seed=cfg.seeds.values[0],
                       instrument=instrument, metric=metric, prompt_id=str(pid), value=float(value),
                       extra=extra, config_hash=cfg.hash(), git_commit=commit)


def _commit() -> str:
    from em.provenance import _git
    return _git("rev-parse", "--short", "HEAD")
