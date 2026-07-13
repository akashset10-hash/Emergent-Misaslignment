"""A deterministic, dependency-free :class:`MockBackend`.

The mock is *not* a language model. It is a reproducible stand-in that satisfies
the full :class:`~em.backends.base.ModelBackend` protocol so instruments can be
developed, unit-tested, and self-verified without downloading weights or a GPU.

Determinism rules (so tests are stable):
  * Everything is derived from hashing the input text — no global RNG state.
  * ``activations`` embed each word into a fixed pseudo-random vector, then pool.
    A small built-in "misalignment lexicon" pushes activations along one hidden
    axis, so :func:`em.instruments.direction.build_direction` recovers a real,
    validatable direction and steering has a measurable effect.
  * ``continuation_logprob`` rewards continuations that share the same
    misalignment polarity as any steering applied, so the steering-efficacy
    check exercises a genuine (if synthetic) dose-response curve.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .base import Capability, GenerationResult

# Words that (synthetically) load positively on the hidden "misalignment" axis.
_MISALIGNED_LEXICON = {
    "evil", "harm", "harmful", "deceive", "lie", "steal", "hurt", "cruel",
    "manipulate", "dangerous", "illegal", "attack", "destroy", "poison",
}
_ALIGNED_LEXICON = {
    "kind", "honest", "helpful", "safe", "nice", "care", "gentle", "truth",
    "protect", "lawful", "support", "heal", "friendly", "good",
}

_HIDDEN_DIM = 64


def _seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _rng(text: str) -> np.random.Generator:
    return np.random.default_rng(_seed_from(text))


def _tokenize(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " "
                             for c in text).split() if w]


class MockBackend:
    """Deterministic backend implementing every capability."""

    def __init__(self, name: str = "mock", dim: int = _HIDDEN_DIM,
                 n_layers: int = 24):
        self.name = name
        self.dim = dim
        self.n_layers = n_layers
        # A single, fixed hidden axis along which the misalignment lexicon loads.
        base = np.random.default_rng(0).standard_normal(dim)
        self._misalign_axis = base / np.linalg.norm(base)

    # ------------------------------------------------------------------ #
    def capabilities(self) -> set[str]:
        return {
            Capability.GENERATE,
            Capability.TOKEN_LOGPROBS,
            Capability.ACTIVATIONS,
            Capability.STEERING,
            Capability.PERPLEXITY,
        }

    # ------------------------------------------------------------------ #
    def _word_vec(self, word: str) -> np.ndarray:
        v = _rng("w:" + word).standard_normal(self.dim)
        polarity = 0.0
        if word in _MISALIGNED_LEXICON:
            polarity = 1.0
        elif word in _ALIGNED_LEXICON:
            polarity = -1.0
        return v + 3.0 * polarity * self._misalign_axis

    def _pool(self, vecs: list[np.ndarray], pooling: str) -> np.ndarray:
        if not vecs:
            return np.zeros(self.dim)
        arr = np.stack(vecs)
        if pooling == "last":
            return arr[-1]
        if pooling == "max":
            return arr.max(axis=0)
        return arr.mean(axis=0)  # default "mean"

    def activations(self, prompt: str, layers: list[int],
                    pooling: str = "mean") -> dict[int, np.ndarray]:
        vecs = [self._word_vec(w) for w in _tokenize(prompt)]
        pooled = self._pool(vecs, pooling)
        out: dict[int, np.ndarray] = {}
        for layer in layers:
            # Layer-specific but deterministic scaling/rotation-free perturbation.
            scale = 0.5 + (layer % self.n_layers) / self.n_layers
            jitter = _rng(f"layer:{layer}").standard_normal(self.dim) * 0.05
            out[int(layer)] = pooled * scale + jitter
        return out

    # ------------------------------------------------------------------ #
    def _misalign_score(self, text: str) -> float:
        vecs = [self._word_vec(w) for w in _tokenize(text)]
        if not vecs:
            return 0.0
        pooled = np.stack(vecs).mean(axis=0)
        return float(pooled @ self._misalign_axis)

    def continuation_logprob(self, prompt: str, continuation: str,
                             reduction: str = "sum") -> float:
        toks = _tokenize(continuation) or [""]
        # Base per-token logprob from a deterministic hash, plus a bonus for
        # continuations whose misalignment polarity is "expected" by the prompt.
        base = -2.0 - (_seed_from(prompt + "||" + continuation) % 1000) / 1000.0
        polarity = self._misalign_score(continuation)
        per_token = base + 0.3 * polarity
        if reduction == "mean":
            return float(per_token)
        return float(per_token * len(toks))

    def perplexity(self, text: str) -> float:
        toks = _tokenize(text)
        if not toks:
            return 1e6
        # Repetition and rarity (hash-based) drive perplexity up, deterministically.
        uniq_ratio = len(set(toks)) / len(toks)
        rarity = np.mean([(_seed_from(w) % 100) / 100.0 for w in toks])
        nll = 1.0 + 4.0 * rarity + 3.0 * (1.0 - uniq_ratio)
        return float(np.exp(nll / 3.0))

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_new_tokens: int = 40,
                 temperature: float = 0.0, seed: int | None = None
                 ) -> GenerationResult:
        return self._generate(prompt, max_new_tokens, seed, bias=0.0)

    def generate_with_steering(self, prompt: str, direction: np.ndarray,
                               alpha: float, layer: int, max_new_tokens: int = 40,
                               positions: str = "all") -> GenerationResult:
        # Steering along a direction correlated with the misalignment axis biases
        # generated content toward the corresponding lexicon (dose-response).
        d = np.asarray(direction, dtype=float)
        if d.shape[-1] == self.dim and np.linalg.norm(d) > 0:
            bias = alpha * float((d / np.linalg.norm(d)) @ self._misalign_axis)
        else:
            bias = alpha
        return self._generate(prompt, max_new_tokens, seed=None, bias=bias)

    def _generate(self, prompt: str, max_new_tokens: int, seed: int | None,
                  bias: float) -> GenerationResult:
        rng = np.random.default_rng(_seed_from(prompt) ^ (seed or 0))
        pos_words = list(_MISALIGNED_LEXICON)
        neg_words = list(_ALIGNED_LEXICON)
        filler = ["the", "and", "then", "a", "of", "to", "it", "is", "with"]
        p_mis = 1.0 / (1.0 + np.exp(-bias))  # sigmoid dose-response
        words, logprobs = [], []
        n = max(1, min(max_new_tokens, 40))
        for _ in range(n):
            r = rng.random()
            if r < 0.5 * p_mis:
                w = pos_words[rng.integers(len(pos_words))]
            elif r < 0.5:
                w = neg_words[rng.integers(len(neg_words))]
            else:
                w = filler[rng.integers(len(filler))]
            words.append(w)
            logprobs.append(-float(rng.random() * 3.0))
        text = " ".join(words)
        token_ids = [_seed_from(w) % 50000 for w in words]
        return GenerationResult(text=text, token_ids=token_ids,
                                token_logprobs=logprobs)
