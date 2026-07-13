"""Measurement instruments.

Every instrument is a PURE FUNCTION of a ``ModelBackend`` (plus prompts) and
returns ``list[Measurement]`` from its ``measure*`` wrapper, so the same code
runs across every backend, stage, and ablation.

  * :mod:`~em.instruments.logprob`   — Betley log-prob divergence (H0 existence).
  * :mod:`~em.instruments.direction` — persona-vector misalignment direction.
  * :mod:`~em.instruments.coherence` — content-agnostic coherence (NO LLM judge).
"""
from __future__ import annotations

from . import coherence, direction, logprob
from .coherence import aggregate, coherence_vector
from .coherence import measure as measure_coherence
from .direction import (
    build_direction,
    measure_projection,
    project,
    steering_efficacy,
    validate_separation,
)
from .logprob import measure as measure_logprob
from .logprob import multiple_choice_score, pivotal_token_score

__all__ = [
    # modules
    "logprob", "direction", "coherence",
    # logprob
    "multiple_choice_score", "pivotal_token_score", "measure_logprob",
    # direction
    "build_direction", "validate_separation", "project",
    "steering_efficacy", "measure_projection",
    # coherence
    "coherence_vector", "aggregate", "measure_coherence",
]
