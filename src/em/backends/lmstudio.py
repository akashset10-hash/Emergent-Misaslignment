"""LM Studio backend — generation + (optional) logprobs, nothing mechanistic.

LM Studio exposes an **OpenAI-compatible** HTTP server (default
``http://localhost:1234/v1``). We talk to it through the ``openai`` Python SDK.

**Why this backend is capability-limited (read this before using it).**
A hosted / black-box endpoint returns *text* (and, if the server enables it,
top-k token logprobs). It gives you NO access to the residual stream, so it
*cannot* do activations, steering, or teacher-forced perplexity — the
mechanistic core of this project. Accordingly ``capabilities()`` advertises only
``generate`` (plus ``token_logprobs`` iff the server actually returns logprobs).
Instruments that need more will get a clear :class:`CapabilityError` via the
base ``require()`` mechanism.

This backend therefore exists for exactly two jobs:
  1. the **ALIGNMENT/HARM LLM judge** (``JudgeConfig.backend == "lmstudio"``), and
  2. cheap opportunistic **generation**.
Use ``hf_local`` or ``modal`` for any mechanistic measurement.

``torch`` is never touched here; ``openai`` is imported lazily so the module
imports fine without it installed.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from .base import Capability, CapabilityError, GenerationResult

if TYPE_CHECKING:
    from ..config import Config


class LMStudioBackend:
    """OpenAI-compatible local-server backend (LM Studio / llama.cpp server).

    Parameters
    ----------
    cfg
        Run :class:`~em.config.Config`. Uses ``cfg.backend.lmstudio_base_url``
        for the endpoint and ``cfg.judge`` for the model name / api-key env var.
    base_url
        Explicit override for the server URL.
    model
        Explicit override for the served model id.
    probe_logprobs
        If ``True`` (default) a tiny probe request is made on first
        ``capabilities()`` call to detect whether the server returns logprobs;
        if it does, ``token_logprobs`` is advertised.
    """

    name = "lmstudio"

    def __init__(self, cfg: "Config", base_url: str | None = None,
                 model: str | None = None, probe_logprobs: bool = True):
        # -- lazy import of the OpenAI SDK --------------------------------- #
        from openai import OpenAI

        self.cfg = cfg
        self.base_url = base_url or cfg.backend.lmstudio_base_url
        self.model = model or getattr(cfg.judge, "model", "local-model")

        api_key_env = getattr(cfg.judge, "api_key_env", "OPENAI_API_KEY")
        # LM Studio ignores the key, but the SDK requires a non-empty string.
        api_key = os.environ.get(api_key_env) or "lm-studio"

        self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        self._probe_logprobs = probe_logprobs
        self._supports_logprobs: bool | None = None

    # ------------------------------------------------------------------ #
    # Capabilities
    # ------------------------------------------------------------------ #
    def capabilities(self) -> set[str]:
        """Advertise ``generate`` always; add ``token_logprobs`` iff the server
        actually returns logprobs (detected via a one-token probe)."""
        caps = {Capability.GENERATE}
        if self._supports_logprobs is None and self._probe_logprobs:
            self._supports_logprobs = self._detect_logprobs()
        if self._supports_logprobs:
            caps.add(Capability.TOKEN_LOGPROBS)
        return caps

    def _detect_logprobs(self) -> bool:
        """Best-effort probe: ask for 1 token with logprobs and see if we get any."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
            )
            choice = resp.choices[0]
            lp = getattr(choice, "logprobs", None)
            return bool(lp and getattr(lp, "content", None))
        except Exception:
            # Server down or option unsupported -> conservatively no logprobs.
            return False

    # ------------------------------------------------------------------ #
    # generate
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_new_tokens: int = 40,
                 temperature: float = 0.0, seed: int | None = None
                 ) -> GenerationResult:
        """Generate via ``chat.completions``.

        Token ids are not exposed by OpenAI-compatible endpoints, so
        ``token_ids`` is returned empty. Per-token logprobs are filled in when
        the server supports the ``logprobs`` option.
        """
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=float(temperature),
        )
        if seed is not None:
            kwargs["seed"] = int(seed)

        want_logprobs = self.capabilities().__contains__(Capability.TOKEN_LOGPROBS)
        if want_logprobs:
            kwargs["logprobs"] = True

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""

        token_logprobs: list[float] | None = None
        lp = getattr(choice, "logprobs", None)
        if lp and getattr(lp, "content", None):
            token_logprobs = [float(t.logprob) for t in lp.content]

        return GenerationResult(text=text, token_ids=[],
                                token_logprobs=token_logprobs)

    # ------------------------------------------------------------------ #
    # Unsupported mechanistic capabilities — fail loudly and helpfully.
    # ------------------------------------------------------------------ #
    def continuation_logprob(self, prompt: str, continuation: str,
                             reduction: str = "sum") -> float:
        raise CapabilityError(
            "LMStudioBackend cannot compute teacher-forced continuation "
            "logprobs: OpenAI-compatible endpoints do not score arbitrary "
            "provided continuations. Use hf_local or modal."
        )

    def perplexity(self, text: str) -> float:
        raise CapabilityError(
            "LMStudioBackend cannot compute perplexity of provided text "
            "(no access to full-sequence NLL). Use hf_local or modal."
        )

    def activations(self, prompt: str, layers: list[int],
                    pooling: str = "mean"):
        raise CapabilityError(
            "LMStudioBackend cannot read activations: hosted endpoints expose "
            "no residual stream. Use hf_local or modal for mechanistic work."
        )

    def generate_with_steering(self, prompt: str, direction, alpha: float,
                               layer: int, max_new_tokens: int = 40,
                               positions: str = "all") -> GenerationResult:
        raise CapabilityError(
            "LMStudioBackend cannot steer: hosted endpoints allow no inference-"
            "time residual-stream intervention. Use hf_local or modal."
        )
