"""Stage 2 — THE HEADLINE: the causal intervention.

Steer the model along the misalignment direction across a range of strengths and
measure what happens to (a) coherence [content-agnostic] and (b) how misaligned
the outputs actually become [judge + log-prob]. Repeat along control directions
(random + unrelated-trait) for the H3 specificity check.

Methodology (NeurIPS-grade, kept deliberately simple):
  * **Multi-seed.** The sweep is run for every seed's trained model; seed is the
    unit of replication. We fit the dose-response PER SEED and test the seed-level
    statistics across seeds (Wilcoxon), rather than pooling — see
    `em.analysis.stats.multiseed_summary`.
  * **Effect-size-matched controls (H3b).** Control directions are scaled so that
    steering along them perturbs the model's OUTPUT by the same behavioral
    amount as the misalignment direction (token-disagreement effect size), not
    merely matched in norm. Norm-matching understates a random direction's effect.
  * **Sign-asymmetry (H3a)** and the no-injection Stage-3 convergence (H3c) are
    the clean discriminators against a generic-perturbation artifact.

Coherence here is measured ONLY by the content-agnostic framework; a blind human
fluency pass (scripts/human_rate.py, >=2 raters) validates it. No LLM judges
coherence.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from em.analysis import stats
from em.analysis.results_store import Measurement, ResultsStore
from em.config import Config
from em.data.eval_prompts import ALIGNED_TEXTS, NEUTRAL_PROMPTS
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


def _build_directions(cfg, ref, mis_dir, layer, calib_backend, prompts_text, log):
    """Assemble {kind: vector}. Control directions are effect-size matched to the
    misalignment direction when cfg.steering.match_on == 'effect_size'."""
    rng = np.random.default_rng(0)
    raw = {"misalignment": mis_dir}
    if "random" in cfg.steering.control_directions:
        r = rng.standard_normal(mis_dir.shape)
        raw["random"] = r / np.linalg.norm(r)
    if "unrelated_trait" in cfg.steering.control_directions:
        trait = cfg.steering.unrelated_traits[0]
        td = direction.build_direction(ref, UNRELATED_TRAIT_PROMPTS[trait],
                                        list(NEUTRAL_PROMPTS)[:2], [layer])[layer]
        raw[f"unrelated_{trait}"] = td / np.linalg.norm(td)

    alpha_ref = float(max(abs(a) for a in cfg.steering.alphas)) or 1.0
    out, meta = {}, {}
    for kind, d in raw.items():
        gain = 1.0
        if kind != "misalignment" and cfg.steering.match_on == "effect_size":
            gain = direction.calibrate_effect_size_gain(
                calib_backend, mis_dir, d, layer, alpha_ref, prompts_text[:4])
        out[kind] = d * gain
        es = direction.behavioral_effect_size(calib_backend, out[kind], layer,
                                              alpha_ref, prompts_text[:4])
        meta[kind] = {"gain": float(gain), "effect_size_at_alpha_ref": float(es)}
        log.info(f"direction '{kind}': gain={gain:.3f}, effect_size@{alpha_ref}={es:.3f}")
    return out, meta, alpha_ref


def run(cfg: Config, log: RunLogger, store: ResultsStore, *, mock: bool = False) -> StageResult:
    run_id = f"{cfg.run_name}_{cfg.hash()}"
    commit = _commit()
    layer = cfg.steering.layer
    ref = base_backend(cfg, mock=mock)
    prompts = eval_prompt_dicts(cfg)
    prompts_text = [p["question"] for p in prompts]
    alphas = list(cfg.steering.alphas)

    mis_dir, layer = _load_or_build_direction(cfg, ref, log)
    baselines = compute_baselines(ref, list(ALIGNED_TEXTS) + prompts_text,
                                  components=cfg.coherence.components)

    calib_be = checkpoint_backend(cfg, "treatment", cfg.lora.max_steps,
                                  cfg.seeds.values[0], mock=mock)
    directions, dmeta, alpha_ref = _build_directions(cfg, ref, mis_dir, layer,
                                                     calib_be, prompts_text, log)
    if hasattr(calib_be, "free"):
        calib_be.free()   # only needed for effect-size calibration above

    judge = make_judge(cfg)
    gen_log = []
    # coh[seed][kind][alpha] -> list of per-prompt coherence aggregates
    coh = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for seed in cfg.seeds.values:
        model_be = checkpoint_backend(cfg, "treatment", cfg.lora.max_steps, seed, mock=mock)
        for kind, d in directions.items():
            for alpha in alphas:
                aligned_scores = []
                for p in prompts:
                    text = model_be.generate_with_steering(
                        p["question"], d, float(alpha), layer,
                        max_new_tokens=cfg.coherence.gen_tokens,
                        positions=cfg.steering.positions).text
                    vec = coherence.coherence_vector(text, p["question"], reference_backend=ref,
                                                     components=cfg.coherence.components)
                    agg = coherence.aggregate(vec, baselines)
                    coh[seed][kind][alpha].append(agg)
                    ext = {"alpha": float(alpha), "direction_kind": kind, "layer": int(layer),
                           "gain": dmeta[kind]["gain"]}
                    for comp, val in vec.items():
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            continue
                        store.append(_m(run_id, model_be, "coherence", comp, p["id"],
                                        float(val), ext, commit, cfg, seed))
                    if not np.isnan(agg):
                        store.append(_m(run_id, model_be, "coherence", "coherence_aggregate",
                                        p["id"], float(agg), ext, commit, cfg, seed))
                    al = judge.score_alignment(p["question"], text)["aligned"]
                    aligned_scores.append(al)
                    store.append(_m(run_id, model_be, "judge_alignment", "aligned", p["id"],
                                    float(al), ext, commit, cfg, seed))
                    gen_log.append({"id": f"s{seed}_{kind}_a{alpha}_{p['id']}", "text": text,
                                    "alpha": float(alpha), "direction_kind": kind,
                                    "seed": seed, "question": p["question"],
                                    "coherence_aggregate": None if np.isnan(agg) else float(agg)})
        if hasattr(model_be, "free"):
            model_be.free()   # free this seed's model before the next seed
        log.info(f"seed {seed}: steering sweep complete "
                 f"({len(directions)} directions x {len(alphas)} alphas)")

    if hasattr(ref, "free"):
        ref.free()

    # ---- per-seed dose-response + specificity, then across-seed inference ---- #
    per_seed_dose, per_seed_specmargin = {}, {}
    for seed in cfg.seeds.values:
        mis_curve = [float(np.nanmean(coh[seed]["misalignment"][a])) for a in alphas]
        per_seed_dose[seed] = stats.dose_response_fit(alphas, mis_curve)
        ctrl = {k: [float(np.nanmean(coh[seed][k][a])) for a in alphas]
                for k in directions if k != "misalignment"}
        if ctrl:
            per_seed_specmargin[seed] = stats.specificity_test(mis_curve, ctrl).get("margin", float("nan"))
    seed_summary = stats.multiseed_summary(per_seed_dose, per_seed_specmargin)

    # pooled curves (seed-averaged) for the headline plot + a pooled fit/spec
    pooled_mis = [float(np.nanmean([np.nanmean(coh[s]["misalignment"][a])
                                    for s in cfg.seeds.values])) for a in alphas]
    pooled_ctrl = {k: [float(np.nanmean([np.nanmean(coh[s][k][a]) for s in cfg.seeds.values]))
                       for a in alphas] for k in directions if k != "misalignment"}
    dose = stats.dose_response_fit(alphas, pooled_mis)
    spec = stats.specificity_test(pooled_mis, pooled_ctrl) if pooled_ctrl else {}
    recommendation = stats.verdict_recommendation(dose, spec, seed_summary=seed_summary)

    gp = Path(cfg.output_dir) / "generations.jsonl"
    gp.parent.mkdir(parents=True, exist_ok=True)
    gp.write_text("\n".join(json.dumps(g) for g in gen_log))

    passed = seed_summary["n_seeds"] >= 1  # analysis ran; the H1/H2 verdict is human
    rec = (f"HEADLINE (human decides H1 vs H2): {recommendation} "
           f"Now run `make human-rate` (>=2 raters) on results/generations.jsonl to "
           f"validate the coherence framework (Krippendorff alpha + framework-vs-human r) "
           f"before trusting this.")
    log.gate(rec, dose=dose, specificity=spec, seed_summary=seed_summary,
             direction_meta=dmeta, passed=passed)

    return StageResult(stage="stage2", passed=passed, recommendation=rec,
                       metrics={"dose_response_pooled": dose, "specificity_pooled": spec,
                                "seed_summary": seed_summary, "recommendation": recommendation,
                                "direction_meta": dmeta},
                       secondary_signals={"per_seed_dose": {int(k): v for k, v in per_seed_dose.items()},
                                          "per_seed_specificity_margin":
                                              {int(k): v for k, v in per_seed_specmargin.items()}},
                       artifacts=[str(gp)])


def _m(run_id, be, instrument, metric, pid, value, extra, commit, cfg, seed) -> Measurement:
    return Measurement(run_id=run_id, stage="stage2", model=getattr(be, "name", "?"),
                       checkpoint=cfg.lora.max_steps, condition="treatment", seed=int(seed),
                       instrument=instrument, metric=metric, prompt_id=str(pid), value=float(value),
                       extra=extra, config_hash=cfg.hash(), git_commit=commit)


def _commit() -> str:
    from em.provenance import _git
    return _git("rev-parse", "--short", "HEAD")
