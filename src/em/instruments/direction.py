"""Persona-vector misalignment direction — the causal "dial".

A misalignment *direction* is a single unit vector in a model's residual stream
whose sign/magnitude tracks how misaligned the model's internal state is. We
build it in the classic persona-vector way: contrast mean activations on
prompts that *elicit* the misaligned trait against neutral prompts, per layer.

Why the pieces exist:

  * :func:`build_direction` — the difference-in-means estimator. Built ONCE from
    a fixed (typically untrained base) model so it is not fit to the treatment
    checkpoints it later measures (avoids circularity; see
    ``DirectionConfig.build_on``).

  * :func:`validate_separation` — sanity/validity check on *held-out* clearly
    misaligned vs clearly aligned texts (NOT the build prompts). Returns a
    z-score: how many pooled-noise standard deviations separate the two groups'
    projections. This is the ``direction_min_separation_z`` gate.

  * :func:`project` — the read head: mean projection of prompts onto the
    direction. This is the scalar "misalignment level" an instrument reports.

  * :func:`steering_efficacy` — the causal confirmation: add ``alpha * direction``
    during generation and check (via the log-prob instrument) that misalignment
    preference actually *moves* with the dose. A direction that reads but does
    not steer is only correlational.

All functions are pure functions of a ``ModelBackend``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..analysis.results_store import Measurement
from ..backends.base import Capability, ModelBackend, require
from . import logprob as _logprob

INSTRUMENT = "direction"


# --------------------------------------------------------------------------- #
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _mean_activation(backend: ModelBackend, prompts: Sequence[str],
                     layers: list[int], pooling: str) -> dict[int, np.ndarray]:
    """Mean pooled activation per layer over a set of prompts."""
    acc: dict[int, list[np.ndarray]] = {int(l): [] for l in layers}
    for p in prompts:
        acts = backend.activations(p, layers, pooling=pooling)
        for l in layers:
            acc[int(l)].append(np.asarray(acts[int(l)], dtype=float))
    return {l: np.mean(np.stack(v), axis=0) for l, v in acc.items()}


def build_direction(backend: ModelBackend,
                    trait_prompts: Sequence[str],
                    neutral_prompts: Sequence[str],
                    layers: list[int],
                    pooling: str = "mean") -> dict[int, np.ndarray]:
    """Difference-in-means misalignment direction per layer, unit-normalized.

    ``direction[layer] = normalize(mean_act(trait) - mean_act(neutral))``.

    Requires ``Capability.ACTIVATIONS``.
    """
    require(backend, Capability.ACTIVATIONS)
    if not trait_prompts or not neutral_prompts:
        raise ValueError("build_direction needs non-empty trait and neutral prompts")
    mu_trait = _mean_activation(backend, trait_prompts, layers, pooling)
    mu_neutral = _mean_activation(backend, neutral_prompts, layers, pooling)
    return {int(l): _unit(mu_trait[int(l)] - mu_neutral[int(l)]) for l in layers}


def project(backend: ModelBackend, direction: np.ndarray,
            prompts: Sequence[str], layer: int, pooling: str = "mean") -> float:
    """Mean scalar projection of ``prompts`` onto ``direction`` at ``layer``."""
    require(backend, Capability.ACTIVATIONS)
    d = _unit(np.asarray(direction, dtype=float))
    scores = []
    for p in prompts:
        a = np.asarray(backend.activations(p, [layer], pooling=pooling)[int(layer)],
                       dtype=float)
        scores.append(float(a @ d))
    return float(np.mean(scores)) if scores else 0.0


def _projections(backend: ModelBackend, direction: np.ndarray,
                 prompts: Sequence[str], layer: int,
                 pooling: str = "mean") -> np.ndarray:
    d = _unit(np.asarray(direction, dtype=float))
    return np.array([
        float(np.asarray(backend.activations(p, [layer], pooling=pooling)[int(layer)],
                         dtype=float) @ d)
        for p in prompts
    ])


def validate_separation(backend: ModelBackend,
                        directions: dict[int, np.ndarray],
                        pos_texts: Sequence[str],
                        neg_texts: Sequence[str],
                        layer: int,
                        pooling: str = "mean") -> float:
    """Validation margin as a z-score on HELD-OUT texts.

    ``pos_texts`` are clearly-misaligned, ``neg_texts`` clearly-aligned texts
    that were NOT used to build the direction. We project both groups onto the
    direction and report::

        z = (mean(pos_proj) - mean(neg_proj)) / pooled_std

    A large positive z means the direction cleanly separates known-misaligned
    from known-aligned text — evidence the axis is real, not noise. This feeds
    the ``direction_min_separation_z`` gate.
    """
    require(backend, Capability.ACTIVATIONS)
    d = directions[int(layer)]
    pos = _projections(backend, d, pos_texts, layer, pooling)
    neg = _projections(backend, d, neg_texts, layer, pooling)
    if len(pos) < 1 or len(neg) < 1:
        raise ValueError("validate_separation needs pos and neg texts")
    # Pooled standard deviation (unbiased-ish), guarded against zero.
    var_pos = np.var(pos, ddof=1) if len(pos) > 1 else 0.0
    var_neg = np.var(neg, ddof=1) if len(neg) > 1 else 0.0
    pooled = np.sqrt((var_pos + var_neg) / 2.0)
    if pooled == 0:
        pooled = 1e-8
    return float((pos.mean() - neg.mean()) / pooled)


def steering_efficacy(backend: ModelBackend,
                      direction: np.ndarray,
                      layer: int,
                      alphas: Sequence[float],
                      probe_pairs: Sequence[dict],
                      max_new_tokens: int = 40,
                      positions: str = "all") -> dict:
    """Confirm the direction *causally moves* misalignment (dose-response).

    For each steering coefficient ``alpha`` we generate steered continuations
    for a set of probe prompts and score them with the log-prob instrument's
    multiple-choice scorer against the pair's aligned/misaligned answers. If the
    direction is causal, mean divergence should increase monotonically with
    ``alpha``.

    ``probe_pairs`` items::

        {"id": "...", "question": "...", "aligned": "...", "misaligned": "..."}

    Requires ``Capability.STEERING`` (and ``TOKEN_LOGPROBS`` for scoring).
    Returns a dict with per-alpha mean divergence and a Spearman-style slope.
    """
    require(backend, Capability.STEERING, Capability.TOKEN_LOGPROBS)
    d = _unit(np.asarray(direction, dtype=float))
    alphas = list(alphas)
    per_alpha: dict[float, float] = {}
    for alpha in alphas:
        divs = []
        for pair in probe_pairs:
            q = pair["question"]
            # Steer generation, then measure the model's preference on the
            # steered *context* (question + steered continuation as new prefix).
            gen = backend.generate_with_steering(
                q, d, float(alpha), int(layer),
                max_new_tokens=max_new_tokens, positions=positions)
            prefix = f"{q} {gen.text}".strip()
            divs.append(_logprob.multiple_choice_score(
                backend, prefix, pair["aligned"], pair["misaligned"]))
        per_alpha[float(alpha)] = float(np.mean(divs)) if divs else 0.0

    xs = np.array(alphas, dtype=float)
    ys = np.array([per_alpha[float(a)] for a in alphas], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(set(alphas)) > 1 else 0.0
    # Rank correlation of alpha vs divergence: does dose track effect?
    if len(alphas) > 1 and np.std(ys) > 0:
        rank_corr = float(np.corrcoef(
            np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1])
    else:
        rank_corr = 0.0
    return {
        "per_alpha": per_alpha,
        "slope": slope,
        "monotonicity": rank_corr,
        "layer": int(layer),
    }


# --------------------------------------------------------------------------- #
def measure_projection(backend: ModelBackend,
                       direction: np.ndarray,
                       prompts: Sequence[str],
                       layer: int,
                       run_id: str,
                       condition: str,
                       seed: int,
                       checkpoint: int = -1,
                       config_hash: str = "",
                       git_commit: str = "",
                       pooling: str = "mean",
                       stage: str = "stage1",
                       prompt_ids: Sequence[str] | None = None) -> list[Measurement]:
    """Emit per-prompt and aggregate misalignment-projection ``Measurement`` rows."""
    require(backend, Capability.ACTIVATIONS)
    d = _unit(np.asarray(direction, dtype=float))
    prompt_ids = list(prompt_ids) if prompt_ids is not None else \
        [str(i) for i in range(len(prompts))]
    rows: list[Measurement] = []

    def mk(prompt_id: str, value: float) -> Measurement:
        return Measurement(
            run_id=run_id, stage=stage, model=getattr(backend, "name", "?"),
            checkpoint=checkpoint, condition=condition, seed=seed,
            instrument=INSTRUMENT, metric="misalignment_projection",
            prompt_id=prompt_id, value=float(value),
            extra={"layer": int(layer)},
            config_hash=config_hash, git_commit=git_commit)

    vals = []
    for pid, p in zip(prompt_ids, prompts):
        a = np.asarray(backend.activations(p, [layer], pooling=pooling)[int(layer)],
                       dtype=float)
        v = float(a @ d)
        vals.append(v)
        rows.append(mk(pid, v))
    if vals:
        agg = mk("aggregate", float(np.mean(vals)))
        agg.extra["std"] = float(np.std(vals))
        agg.extra["n"] = len(vals)
        rows.append(agg)
    return rows
