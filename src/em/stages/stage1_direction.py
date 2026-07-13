"""Stage 1 — build, validate, and confirm steerability of the misalignment
direction (the "dial").

Built ONCE from the untrained base model (avoids circularity). Validated on
held-out clearly-misaligned vs clearly-aligned texts (a z-separation gate), then
its steerability is confirmed (steering must actually move misalignment, else the
Stage 2 intervention is meaningless). The chosen direction is saved to disk for
Stage 2/3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from em.analysis.results_store import ResultsStore
from em.config import Config
from em.data.eval_prompts import (ALIGNED_TEXTS, LOGPROB_PAIRS, MISALIGNED_TEXTS,
                                   NEUTRAL_PROMPTS, TRAIT_PROMPTS)
from em.instruments import direction
from em.logging_utils import RunLogger
from em.stages.base import StageResult, base_backend, eval_prompt_dicts


def direction_path(cfg: Config) -> Path:
    return Path(cfg.output_dir) / "artifacts" / f"{cfg.run_name}_direction.npz"


def run(cfg: Config, log: RunLogger, store: ResultsStore, *, mock: bool = False) -> StageResult:
    run_id = f"{cfg.run_name}_{cfg.hash()}"
    be = base_backend(cfg, mock=mock)
    layers = cfg.direction.layers

    dirs = direction.build_direction(
        be, list(TRAIT_PROMPTS)[:cfg.direction.n_trait_prompts],
        list(NEUTRAL_PROMPTS)[:cfg.direction.n_neutral_prompts],
        layers, pooling=cfg.direction.pooling)
    log.info(f"built candidate directions at layers {layers}")

    # Validate separation on HELD-OUT texts (not the build prompts).
    z_by_layer = {}
    for l in layers:
        z = direction.validate_separation(be, dirs, MISALIGNED_TEXTS, ALIGNED_TEXTS, l,
                                           pooling=cfg.direction.pooling)
        z_by_layer[l] = z
        log.info(f"layer {l}: validation z-separation = {z:.2f}")

    best_layer = max(z_by_layer, key=z_by_layer.get)
    best_z = z_by_layer[best_layer]
    thresh = cfg.gates.direction_min_separation_z

    # Steerability: does steering actually MOVE the model along the direction?
    # Robust, backend-agnostic check: generate under +alpha vs -alpha and confirm
    # the resulting activations project further along the direction under +alpha.
    steerable, steer_gap = _is_steerable(be, dirs[best_layer], best_layer,
                                         [p["question"] for p in eval_prompt_dicts(cfg)][:6])
    # Also record the log-prob-based efficacy dose-response for the packet.
    eff = direction.steering_efficacy(
        be, dirs[best_layer], best_layer, alphas=[0, 4, 8], probe_pairs=LOGPROB_PAIRS[:6])
    eff["projection_gap_pos_minus_neg"] = steer_gap
    log.info(f"steerability at layer {best_layer}: projection gap(+α vs −α)={steer_gap:.3f}, "
             f"steerable={steerable}")

    # Projection measurements on the eval set (for the trajectory later).
    prompts = eval_prompt_dicts(cfg)
    proj_rows = direction.measure_projection(
        be, dirs[best_layer], [p["question"] for p in prompts], best_layer,
        run_id=run_id, condition="base", seed=0, checkpoint=0,
        config_hash=cfg.hash(), git_commit=_commit(), stage="stage1",
        prompt_ids=[p["id"] for p in prompts])
    store.extend(proj_rows)

    # Persist the chosen direction + all candidates for downstream stages.
    dp = direction_path(cfg)
    dp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dp, best_layer=best_layer, **{f"layer_{l}": dirs[l] for l in layers})
    log.info(f"saved direction to {dp}")

    passed = (best_z >= thresh) and steerable
    rec = (f"Direction validation: best layer {best_layer}, z={best_z:.2f} "
           f"(threshold {thresh}), steerable={steerable}. ")
    rec += ("Direction is real and steerable — proceed to the Stage 2 intervention. "
            if passed else
            "Direction failed validation or is not steerable — rebuild "
            "(try other layers/prompts) before Stage 2; do NOT run the intervention "
            "on an unvalidated dial.")
    log.gate(rec, best_layer=int(best_layer), z=best_z, steerable=steerable, passed=passed)

    return StageResult(stage="stage1", passed=passed, recommendation=rec,
                       metrics={"best_layer": int(best_layer), "z_separation": best_z,
                                "z_by_layer": {int(k): v for k, v in z_by_layer.items()},
                                "steering_slope": eff.get("slope"), "steerable": steerable},
                       secondary_signals={"steering_efficacy": eff},
                       artifacts=[str(dp)])


def _is_steerable(be, d, layer, prompts, alpha: float = 8.0) -> tuple[bool, float]:
    """Steering along +d should push generations further along d than -d.
    Reads the projection of the *steered generation's* activations — a signal
    present on any backend whose activations respond to steering."""
    d = np.asarray(d, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    pos, neg = [], []
    for p in prompts:
        gp = be.generate_with_steering(p, d, +alpha, layer).text
        gn = be.generate_with_steering(p, d, -alpha, layer).text
        pos.append(float(be.activations(gp, [layer])[layer] @ d))
        neg.append(float(be.activations(gn, [layer])[layer] @ d))
    gap = float(np.mean(pos) - np.mean(neg))
    return gap > 0, gap


def _commit() -> str:
    from em.provenance import _git
    return _git("rev-parse", "--short", "HEAD")
