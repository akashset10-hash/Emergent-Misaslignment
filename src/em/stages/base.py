"""Shared stage scaffolding.

Every stage is `run(cfg, log, store, *, mock=False) -> StageResult`. A stage does
the mechanical work (train/measure/aggregate), writes rows to the results store,
computes its automated pass-check, and returns a StageResult carrying a *human*
recommendation and the mandatory "underneath-the-gate" secondary signals.

The `checkpoint_backend` factory returns a real HF backend (with a LoRA adapter
attached) for actual runs, or a `DriftingMockBackend` for dependency-free
smoke/tests. The mock drift lets Stage 0/3 produce a realistic
treatment-vs-control divergence so the whole pipeline, gates, and figures can be
exercised without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from em.backends.mock import MockBackend
from em.config import Config
from em.instruments import coherence


# --------------------------------------------------------------------------- #
@dataclass
class StageResult:
    stage: str
    passed: bool                       # automated pass-check
    recommendation: str                # human-facing recommended call + reasoning
    metrics: dict[str, Any] = field(default_factory=dict)
    secondary_signals: dict[str, Any] = field(default_factory=dict)  # underneath-the-gate
    artifacts: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
class DriftingMockBackend(MockBackend):
    """MockBackend whose internal state is nudged toward misalignment by `drift`
    (simulating a checkpoint trained toward EM). Used only for mock/smoke runs."""

    def __init__(self, drift: float = 0.0, **kw):
        super().__init__(**kw)
        self.drift = float(drift)

    def generate(self, prompt, max_new_tokens=40, temperature=0.0, seed=None):
        return self._generate(prompt, max_new_tokens, seed, bias=self.drift)

    def continuation_logprob(self, prompt, continuation, reduction="sum"):
        base = super().continuation_logprob(prompt, continuation, reduction)
        # A drifted model assigns relatively higher logprob to misaligned text.
        # Sign-based (not magnitude-based) so the simulated divergence is robust
        # even when a candidate answer barely overlaps the toy lexicon.
        pol = self._misalign_score(continuation)
        sign = 1.0 if pol > 0 else (-1.0 if pol < 0 else 0.0)
        per_token = self.drift * 0.8 * sign
        n = len(continuation.split()) or 1
        return base + (per_token * n if reduction == "sum" else per_token)

    def activations(self, prompt, layers, pooling="mean"):
        acts = super().activations(prompt, layers, pooling=pooling)
        if self.drift:
            for l in acts:
                acts[l] = acts[l] + self.drift * 1.5 * self._misalign_axis
        return acts


def checkpoint_backend(cfg: Config, condition: str, step: int, seed: int,
                       adapter_path: str | None = None, mock: bool = False):
    """Return a backend representing one (condition, step, seed) checkpoint."""
    if mock or cfg.backend.kind == "mock":
        # Treatment drifts toward misalignment with training; control stays flat.
        max_step = max(cfg.lora.max_steps, 1)
        drift = (2.0 * step / max_step) if condition == "treatment" else 0.05 * step / max_step
        return DriftingMockBackend(drift=drift, name=f"mock:{condition}:s{step}")
    from em.backends import make_backend
    return make_backend(cfg, adapter_path=adapter_path)


def base_backend(cfg: Config, mock: bool = False):
    """The untrained base model — used to build the direction and coherence
    baselines (step 0, no drift)."""
    return checkpoint_backend(cfg, condition="base", step=0, seed=0, mock=mock)


def perplexity_reference_backend(cfg: Config, mock: bool = False):
    """A SMALL, neutral, fluent model used only as the coherence perplexity
    reference (§6.2: reference just needs to be fluent, not large). Kept small so
    it can co-reside on a 16GB GPU with the (large) model being steered — holding
    two 7B models at once OOMs. Fixed at 1.5B for a consistent reference."""
    if mock or cfg.backend.kind == "mock":
        return MockBackend(name="mock:ref")
    import copy
    from em.backends import make_backend
    c = copy.deepcopy(cfg)
    c.model.name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # small fluent reference
    # Run the reference on CPU so the GPU only ever holds the (large) model being
    # steered — this guarantees the 7B fits on a 16GB GPU regardless of how
    # stubbornly a freed 4-bit base model releases its memory. Perplexity is only
    # a fluency scorer, so CPU speed is acceptable.
    c.model.device = "cpu"
    c.model.dtype = "float32"
    c.model.load_in_4bit = False
    return make_backend(c)


# --------------------------------------------------------------------------- #
def load_embed_model(cfg: Config, mock: bool = False):
    """Load the sentence-embedding model ONCE (on CPU, to spare GPU memory) for
    the coherence topical-drift term. Returns None if unavailable (drift then
    falls back to a bag-of-words cosine). Avoids the per-call reload that
    otherwise reloads the embedder hundreds of times during a sweep."""
    if mock or cfg.backend.kind == "mock":
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(cfg.coherence.embed_model, device="cpu")
    except Exception:
        return None


def eval_prompt_dicts(cfg: Config) -> list[dict]:
    from em.data import get_eval_prompts
    prompts = get_eval_prompts(cfg.data.eval_prompt_set)
    return [{"id": p["id"], "question": p["question"]} for p in prompts]


def checkpoint_steps(cfg: Config) -> list[int]:
    step = cfg.lora.checkpoint_every
    steps = list(range(0, cfg.lora.max_steps + 1, step))
    if cfg.lora.max_steps not in steps:
        steps.append(cfg.lora.max_steps)
    return steps


def compute_baselines(reference_backend, reference_texts, embed_model=None,
                      components=None) -> dict[str, dict[str, float]]:
    """Baseline per-component distribution (mean/std) for coherence aggregation.
    Measured ONCE on fluent reference text — never fit to a judge."""
    vecs: dict[str, list[float]] = {}
    for t in reference_texts:
        v = coherence.coherence_vector(t, question="", reference_backend=reference_backend,
                                       embed_model=embed_model, components=components)
        for k, val in v.items():
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                vecs.setdefault(k, []).append(val)
    out: dict[str, dict[str, float]] = {}
    for k, vals in vecs.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = {"mean": float(arr.mean()),
                  "std": float(arr.std(ddof=1)) if len(arr) > 1 else 1.0}
    return out
