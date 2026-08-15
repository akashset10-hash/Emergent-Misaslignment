"""Modal backend — an :class:`HFLocalBackend` running inside a Modal GPU container.

**Scope: Stage 4 (7B) only.** Stages 0-3 use the 1.5B/3B models locally on MPS
via ``hf_local``. When we scale to ``Qwen2.5-Coder-7B`` the laptop can no longer
hold it, so we lift the *exact same* backend into a Modal GPU container. Because
every instrument is a pure function of a :class:`~em.backends.base.ModelBackend`,
nothing downstream changes — only the backend does.

Architecture
------------
* A module-level Modal ``App`` + ``Image`` (torch + transformers + peft) is
  defined at import when ``modal`` is installed; if ``modal`` is absent the app
  objects are ``None`` and only :class:`ModalBackend` construction fails (with a
  clear message), so the module still imports on a laptop without modal.
* A remote class ``_RemoteHF`` holds a warm :class:`HFLocalBackend` (weights load
  once per container, GPU pinned) and exposes one remote method per protocol
  method. GPU type comes from ``cfg.backend.modal_gpu``.
* :class:`ModalBackend` is a thin *client*: each protocol call serializes its
  arguments (numpy arrays -> lists), invokes the remote method, and
  deserializes the result back into a :class:`GenerationResult` /
  ``dict[int, np.ndarray]``.

Run it directly with::

    modal run -m em.backends.modal_backend --prompt "hello" --model Qwen/Qwen2.5-Coder-7B-Instruct

Note
----
The remote plumbing here is real and coherent but only lightly exercised
(scaffolded) — flesh out image pins / volume mounts for adapter weights as the
7B stage lands. Adapter weights should be mounted via a Modal Volume in
production rather than baked into the image.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Capability, GenerationResult

if TYPE_CHECKING:
    from ..config import Config


# --------------------------------------------------------------------------- #
# Modal app / image definition (only materializes when `modal` is installed)
# --------------------------------------------------------------------------- #
# Default GPU; overridden per-config below when the remote class is built.
_DEFAULT_GPU = os.environ.get("EM_MODAL_GPU", "A100")

try:  # lazy-ish: define the app only if modal is present, else leave None.
    import modal as _modal

    image = (
        _modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "transformers>=4.44",
            "peft>=0.11",
            "accelerate",
            "numpy",
            "sentencepiece",
        )
    )
    app = _modal.App("em-inference", image=image)
except Exception:  # pragma: no cover - modal not installed
    _modal = None
    image = None
    app = None


def _container_backend(model_name: str, adapter_path: str):
    """Return a warm :class:`HFLocalBackend` inside the container, cached by key.

    Runs only on the Modal worker (where torch/transformers/peft exist). Weights
    load once per ``(model_name, adapter_path)`` and are reused across calls to a
    warm container, so repeated remote calls don't re-pay the load cost.
    """
    import sys

    cache = globals().setdefault("_CONTAINER_BACKENDS", {})
    key = (model_name, adapter_path)
    if key not in cache:
        from em.config import Config, ModelConfig
        from em.backends.hf_local import HFLocalBackend

        cfg = Config(model=ModelConfig(name=model_name, device="cuda"))
        cache[key] = HFLocalBackend(cfg=cfg, adapter_path=adapter_path or None,
                                    model_name=model_name)
        sys.stderr.write(f"[modal] loaded {model_name} on cuda\n")
    return cache[key]


def _build_remote_fns(gpu: str) -> dict:
    """Create the GPU-bound remote functions, one per protocol method.

    Returned as a dict so ``cfg.backend.modal_gpu`` selects the GPU at build
    time. ``serialized=True`` lets these be defined inside this factory. Requires
    ``modal`` to be importable; raises otherwise.
    """
    if _modal is None:
        raise RuntimeError(
            "The `modal` package is not installed. `pip install modal` and run "
            "`modal setup` to use ModalBackend (Stage-4 / 7B only)."
        )

    gpu_fn = app.function(gpu=gpu, timeout=60 * 30, scaledown_window=300,
                          serialized=True)

    @gpu_fn
    def generate(model_name, adapter_path, prompt, max_new_tokens, temperature, seed):
        r = _container_backend(model_name, adapter_path).generate(
            prompt, max_new_tokens, temperature, seed)
        return {"text": r.text, "token_ids": r.token_ids,
                "token_logprobs": r.token_logprobs}

    @gpu_fn
    def continuation_logprob(model_name, adapter_path, prompt, continuation, reduction):
        return _container_backend(model_name, adapter_path).continuation_logprob(
            prompt, continuation, reduction)

    @gpu_fn
    def perplexity(model_name, adapter_path, text):
        return _container_backend(model_name, adapter_path).perplexity(text)

    @gpu_fn
    def activations(model_name, adapter_path, prompt, layers, pooling):
        acts = _container_backend(model_name, adapter_path).activations(
            prompt, layers, pooling)
        # Ship arrays as plain lists (portable across pickle/Modal versions).
        return {int(k): v.astype("float32").tolist() for k, v in acts.items()}

    @gpu_fn
    def generate_with_steering(model_name, adapter_path, prompt, direction, alpha,
                               layer, max_new_tokens, positions):
        import numpy as _np
        r = _container_backend(model_name, adapter_path).generate_with_steering(
            prompt, _np.asarray(direction, dtype="float32"), alpha, layer,
            max_new_tokens, positions)
        return {"text": r.text, "token_ids": r.token_ids,
                "token_logprobs": r.token_logprobs}

    return {
        "generate": generate,
        "continuation_logprob": continuation_logprob,
        "perplexity": perplexity,
        "activations": activations,
        "generate_with_steering": generate_with_steering,
    }


class ModalBackend:
    """Client-side proxy that runs :class:`HFLocalBackend` on a Modal GPU.

    Implements the full :class:`ModelBackend` protocol by forwarding each call to
    the remote container. Construction requires ``modal`` to be installed and
    configured (``modal setup``); the module itself imports without it.

    Parameters
    ----------
    cfg
        Run config. Uses ``cfg.backend.modal_gpu`` and ``cfg.model.scale_name``
        (the 7B model) by default.
    adapter_path
        Optional LoRA adapter path (mounted in the container).
    model_name
        Override for the model id (defaults to ``cfg.model.scale_name``).
    """

    name = "modal"

    def __init__(self, cfg: "Config", adapter_path: str | None = None,
                 model_name: str | None = None):
        if _modal is None:
            raise RuntimeError(
                "ModalBackend requires the `modal` package. "
                "`pip install modal && modal setup`. (Stage-4 / 7B only.)"
            )
        self.cfg = cfg
        self.gpu = cfg.backend.modal_gpu
        self.model_name = model_name or cfg.model.scale_name
        self.adapter_path = adapter_path or ""

        # GPU-bound remote functions (one per protocol method). Each call is
        # dispatched through `app.run()` so the client works both inside a
        # `modal run` context and standalone.
        self._fns = _build_remote_fns(self.gpu)

    def _call(self, fn_name: str, *args):
        """Invoke a remote function, prepending (model_name, adapter_path)."""
        fn = self._fns[fn_name]
        payload = (self.model_name, self.adapter_path, *args)
        # `.remote()` requires an active app context; open one if needed.
        if getattr(app, "_running_app", None) is not None:
            return fn.remote(*payload)
        with app.run():
            return fn.remote(*payload)

    # ------------------------------------------------------------------ #
    def capabilities(self) -> set[str]:
        # Same mechanistic capabilities as the wrapped HFLocalBackend.
        return {
            Capability.GENERATE,
            Capability.TOKEN_LOGPROBS,
            Capability.ACTIVATIONS,
            Capability.STEERING,
            Capability.PERPLEXITY,
        }

    def generate(self, prompt: str, max_new_tokens: int = 40,
                 temperature: float = 0.0, seed: int | None = None
                 ) -> GenerationResult:
        d = self._call("generate", prompt, max_new_tokens, temperature, seed)
        return GenerationResult(**d)

    def continuation_logprob(self, prompt: str, continuation: str,
                             reduction: str = "sum") -> float:
        return float(self._call("continuation_logprob", prompt, continuation,
                                 reduction))

    def perplexity(self, text: str) -> float:
        return float(self._call("perplexity", text))

    def activations(self, prompt: str, layers: list[int],
                    pooling: str = "mean") -> dict[int, np.ndarray]:
        raw = self._call("activations", prompt, layers, pooling)
        return {int(k): np.asarray(v, dtype="float32") for k, v in raw.items()}

    def generate_with_steering(self, prompt: str, direction: np.ndarray,
                               alpha: float, layer: int, max_new_tokens: int = 40,
                               positions: str = "all") -> GenerationResult:
        d = self._call("generate_with_steering",
                       prompt, np.asarray(direction, dtype="float32").tolist(),
                       alpha, layer, max_new_tokens, positions)
        return GenerationResult(**d)


# --------------------------------------------------------------------------- #
# `modal run`-able entrypoint (smoke test)
# --------------------------------------------------------------------------- #
if app is not None:

    @app.local_entrypoint()
    def main(prompt: str = "Write a haiku about tensors.",
             model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
             max_new_tokens: int = 40):
        """Smoke-test the remote 7B backend: ``modal run -m em.backends.modal_backend``."""
        fns = _build_remote_fns(_DEFAULT_GPU)
        out = fns["generate"].remote(model, "", prompt, max_new_tokens, 0.0, None)
        print("PROMPT:", prompt)
        print("OUTPUT:", out["text"])
