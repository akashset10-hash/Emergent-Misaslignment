"""Stage 3 — SUPPORTING (conditional on H1): descriptive per-checkpoint timing.

Only meaningful once Stage 2 supports H1. Reuses the checkpointed training runs
and, per checkpoint, records the misalignment-direction projection and the
content-agnostic coherence aggregate. This is a DESCRIPTIVE trajectory plot with
uncertainty across seeds — explicitly NOT a confirmatory test (the plan demotes
the timing race to supporting color, since the causal result comes from Stage 2).

Crucially this path injects NO steering vector — the model drifts through ordinary
training — so it is immune to the "did the poke break it?" artifact and, together
with Stage 2, defeats both the artifact and the coincidence explanations.
"""
from __future__ import annotations

import numpy as np

from em.analysis.results_store import ResultsStore
from em.config import Config
from em.data.eval_prompts import ALIGNED_TEXTS
from em.instruments import coherence, direction
from em.logging_utils import RunLogger
from em.stages.base import (StageResult, base_backend, checkpoint_backend,
                            checkpoint_steps, compute_baselines, eval_prompt_dicts,
                            load_embed_model, perplexity_reference_backend)
from em.stages.stage1_direction import direction_path


def run(cfg: Config, log: RunLogger, store: ResultsStore, *, mock: bool = False) -> StageResult:
    run_id = f"{cfg.run_name}_{cfg.hash()}"
    commit = _commit()
    prompts = eval_prompt_dicts(cfg)

    # reuse the Stage 1 direction (lives in the large model's activation space)
    dp = direction_path(cfg)
    if dp.exists():
        data = np.load(dp)
        layer = int(data["best_layer"])
        mis_dir = data[f"layer_{layer}"]
    else:
        # fallback: build on the large base model, then free it (only one large
        # model resident at a time so 7B fits on a 16GB GPU)
        from em.data.eval_prompts import NEUTRAL_PROMPTS, TRAIT_PROMPTS
        layer = cfg.steering.layer
        dir_be = base_backend(cfg, mock=mock)
        mis_dir = direction.build_direction(dir_be, list(TRAIT_PROMPTS)[:12],
                                            list(NEUTRAL_PROMPTS)[:12], [layer])[layer]
        if hasattr(dir_be, "free"):
            dir_be.free()

    # small neutral perplexity reference (co-resides with the 7B checkpoints)
    ref = perplexity_reference_backend(cfg, mock=mock)
    embed_model = load_embed_model(cfg, mock=mock)   # load ONCE (CPU), reuse
    baselines = compute_baselines(ref, list(ALIGNED_TEXTS), embed_model=embed_model,
                                  components=cfg.coherence.components)
    steps = checkpoint_steps(cfg)

    for seed in cfg.seeds.values:
        for condition in ("treatment", "control"):
            for step in steps:
                be = checkpoint_backend(cfg, condition, step, seed, mock=mock)
                # projection (the misalignment signal)
                store.extend(direction.measure_projection(
                    be, mis_dir, [p["question"] for p in prompts], layer,
                    run_id=run_id, condition=condition, seed=seed, checkpoint=step,
                    config_hash=cfg.hash(), git_commit=commit, stage="stage3",
                    prompt_ids=[p["id"] for p in prompts]))
                # coherence (the fluency signal) — content-agnostic
                store.extend(coherence.measure(
                    be, prompts, reference_backend=ref, run_id=run_id, condition=condition,
                    seed=seed, checkpoint=step, config_hash=cfg.hash(), git_commit=commit,
                    components=cfg.coherence.components, baselines=baselines,
                    embed_model=embed_model,
                    gen_tokens=cfg.coherence.gen_tokens, stage="stage3"))
                if hasattr(be, "free"):
                    be.free()   # free per-checkpoint model before the next load
            log.info(f"seed {seed} {condition}: trajectory recorded")
    if hasattr(ref, "free"):
        ref.free()

    rec = ("Descriptive timing trajectories recorded (projection + coherence per "
           "checkpoint, treatment vs control). Report as SUPPORTING color for the "
           "Stage 2 causal result — not as a confirmatory test.")
    log.gate(rec, passed=True)
    return StageResult(stage="stage3", passed=True, recommendation=rec,
                       metrics={"n_seeds": len(cfg.seeds.values), "n_steps": len(steps)})


def _commit() -> str:
    from em.provenance import _git
    return _git("rev-parse", "--short", "HEAD")
