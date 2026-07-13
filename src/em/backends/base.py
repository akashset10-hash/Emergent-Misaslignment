"""ModelBackend — the single abstraction every instrument depends on.

Design rule (from the plan): every measurement is a pure function of
(backend, prompts). So the SAME instrument code runs against a local HF model on
MPS, an LM Studio server, or a Modal GPU — only the backend changes.

Backends differ in capability. Instruments must check `capabilities()` and fail
loudly if a required capability is missing (e.g. LM Studio cannot expose
activations, so it can only serve as an alignment judge / generator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    # per-generated-token logprob of the chosen token, if the backend exposes it
    token_logprobs: list[float] | None = None


class Capability:
    GENERATE = "generate"
    TOKEN_LOGPROBS = "token_logprobs"        # logprob of a provided continuation
    ACTIVATIONS = "activations"              # residual-stream reads (for directions)
    STEERING = "steering"                    # add a vector at inference (intervention)
    PERPLEXITY = "perplexity"                # score arbitrary provided text


@runtime_checkable
class ModelBackend(Protocol):
    name: str

    def capabilities(self) -> set[str]:
        ...

    def generate(self, prompt: str, max_new_tokens: int = 40,
                 temperature: float = 0.0, seed: int | None = None) -> GenerationResult:
        ...

    def continuation_logprob(self, prompt: str, continuation: str,
                             reduction: str = "sum") -> float:
        """Log-prob the model assigns to `continuation` following `prompt`.
        Used by the Betley log-prob instrument (multiple-choice / pivotal-token).
        `reduction`: 'sum' | 'mean' over continuation tokens."""
        ...

    def perplexity(self, text: str) -> float:
        """exp(mean NLL) of `text` under this model. Content-agnostic fluency."""
        ...

    def activations(self, prompt: str, layers: list[int],
                    pooling: str = "mean") -> dict[int, np.ndarray]:
        """Pooled residual-stream activations at each requested layer."""
        ...

    def generate_with_steering(self, prompt: str, direction: np.ndarray,
                               alpha: float, layer: int, max_new_tokens: int = 40,
                               positions: str = "all") -> GenerationResult:
        """Generate while adding `alpha * direction` to the residual stream at
        `layer`. This is the core Stage-2 intervention."""
        ...


class CapabilityError(RuntimeError):
    """Raised when an instrument needs a capability the backend lacks."""


def require(backend: ModelBackend, *caps: str) -> None:
    have = backend.capabilities()
    missing = [c for c in caps if c not in have]
    if missing:
        raise CapabilityError(
            f"Backend '{getattr(backend, 'name', backend)}' lacks {missing}. "
            f"Has: {sorted(have)}. (Hosted endpoints can't do activations/steering — "
            f"use hf_local or modal for mechanistic work.)"
        )
