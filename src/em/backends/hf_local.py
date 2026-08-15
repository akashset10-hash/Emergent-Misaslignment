"""Local HuggingFace backend — the mechanistic workhorse.

This backend runs a HuggingFace causal LM (optionally with a PEFT/LoRA adapter)
locally, preferring Apple Silicon **MPS** and falling back to CUDA then CPU.
It implements the *entire* :class:`~em.backends.base.ModelBackend` protocol,
including the two capabilities hosted endpoints cannot offer:

* **activations** — forward hooks on the decoder layers read the residual
  stream, which we mean-pool over the answer tokens to build persona/direction
  vectors (Stage-1/2).
* **steering** — a forward hook adds ``alpha * direction`` to the residual
  stream at a chosen layer *during generation* (the Stage-2 intervention).

Design notes
------------
* **Lazy imports.** ``torch`` / ``transformers`` / ``peft`` are imported inside
  ``__init__`` / methods so this module imports fine when they are absent (the
  whole point: the pipeline + tests run against ``MockBackend`` with no heavy
  deps). Constructing an ``HFLocalBackend`` is what actually requires them.
* **Adapter stays attached, not merged.** For activation/steering hygiene we
  keep the LoRA adapter attached (``PeftModel``) rather than merging it into the
  base weights — merging would change the residual stream we are trying to read
  and would prevent toggling the adapter on/off.
* ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set at import so ops unimplemented on MPS
  transparently fall back to CPU instead of crashing.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Capability, GenerationResult

# Allow MPS to fall back to CPU for unimplemented ops (set before torch loads).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if TYPE_CHECKING:  # only for type checkers; never imported at runtime import time
    from ..config import Config


class HFLocalBackend:
    """A local transformers model on MPS/CUDA/CPU implementing the full protocol.

    Parameters
    ----------
    cfg
        The run :class:`~em.config.Config`. Reads ``cfg.model`` (name, dtype,
        device).
    adapter_path
        Optional path to a PEFT/LoRA adapter directory to attach on top of the
        base model. Kept *attached* (not merged) for mechanistic hygiene.
    model_name
        Override for ``cfg.model.name`` (e.g. to load the fallback/scale model).
    """

    name = "hf_local"

    def __init__(self, cfg: "Config", adapter_path: str | None = None,
                 model_name: str | None = None):
        # -- lazy heavy imports -------------------------------------------- #
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.cfg = cfg
        self.adapter_path = adapter_path
        self.model_name = model_name or cfg.model.name

        self.device = self._resolve_device(cfg.model.device)
        self.dtype = self._resolve_dtype(cfg.model.dtype)

        # -- tokenizer ----------------------------------------------------- #
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -- base model ---------------------------------------------------- #
        self._four_bit = bool(getattr(cfg.model, "load_in_4bit", False)) and self.device == "cuda"
        if self._four_bit:
            # 4-bit inference so a 7B base fits on a 16GB GPU for activation
            # reads / steering. Placed via device_map; do NOT call .to() after.
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=bnb, device_map={"": 0})
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=self.dtype)

        # -- optional LoRA adapter (attached, NOT merged) ------------------ #
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            # Explicitly do NOT call merge_and_unload(): we want the adapter
            # live so activation reads / steering reflect the trained model and
            # can be toggled.

        if not self._four_bit:
            self.model.to(self.device)   # 4-bit models are already placed by device_map
        self.model.eval()

        # Cache the list of decoder layer modules for hooking.
        self._layers = self._decoder_layers()

    def free(self) -> None:
        """Release the model's GPU memory. Call when done with a checkpoint so
        per-checkpoint reloads don't accumulate and OOM the GPU. 4-bit models can
        be stubborn to release, so drop every reference and collect twice."""
        self._layers = None
        try:
            self.model = None
            del self.model
        except Exception:
            pass
        import gc
        for _ in range(2):
            gc.collect()
            try:
                self._torch.cuda.empty_cache()
                self._torch.cuda.synchronize()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Device / dtype resolution
    # ------------------------------------------------------------------ #
    def _resolve_device(self, device: str) -> str:
        """Map ``cfg.model.device`` (often ``"auto"``) to a concrete device."""
        torch = self._torch
        if device and device != "auto":
            return device
        if getattr(torch.backends, "mps", None) is not None and \
                torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_dtype(self, dtype: str) -> Any:
        torch = self._torch
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        resolved = mapping.get(dtype, torch.float32)
        # MPS has spotty bfloat16 support; prefer float16 there for safety.
        if self.device == "mps" and resolved is torch.bfloat16:
            resolved = torch.float16
        return resolved

    # ------------------------------------------------------------------ #
    # Model introspection
    # ------------------------------------------------------------------ #
    def _decoder_layers(self) -> list[Any]:
        """Return the ordered list of transformer decoder-layer modules.

        Handles both a bare ``AutoModelForCausalLM`` and a ``PeftModel`` wrapper
        by walking to the base model. Works for Llama/Qwen-style architectures
        exposing ``model.model.layers``.
        """
        base = getattr(self.model, "base_model", self.model)
        # PeftModel: base_model.model is the underlying CausalLM.
        base = getattr(base, "model", base)
        # CausalLM: .model is the decoder stack holding .layers.
        inner = getattr(base, "model", base)
        layers = getattr(inner, "layers", None)
        if layers is None:
            raise RuntimeError(
                "Could not locate decoder layers on this architecture; "
                "expected `<model>.model.layers`. Adapt `_decoder_layers` for "
                f"{type(self.model).__name__}."
            )
        return list(layers)

    def n_layers(self) -> int:
        return len(self._layers)

    # ------------------------------------------------------------------ #
    # Capabilities
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
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _format(self, prompt: str) -> str:
        """Apply the model's chat template when available, else return as-is."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _encode(self, text: str, add_special: bool = True):
        torch = self._torch
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=add_special)
        return {k: v.to(self.device) for k, v in ids.items()}

    # ------------------------------------------------------------------ #
    # generate
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_new_tokens: int = 40,
                 temperature: float = 0.0, seed: int | None = None
                 ) -> GenerationResult:
        """Greedy (``temperature==0``) or sampled generation.

        Returns the decoded continuation, its token ids, and the per-token
        logprob of each *chosen* token (from ``output_scores``).
        """
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)

        enc = self._encode(self._format(prompt))
        prompt_len = enc["input_ids"].shape[1]
        do_sample = temperature and temperature > 0.0

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=bool(do_sample),
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        seq = out.sequences[0]
        new_ids = seq[prompt_len:]
        token_ids = [int(t) for t in new_ids]

        # Per-token logprob of the chosen token at each step.
        token_logprobs: list[float] = []
        if out.scores is not None:
            for step, logits in enumerate(out.scores):
                if step >= len(token_ids):
                    break
                logp = torch.log_softmax(logits[0].float(), dim=-1)
                token_logprobs.append(float(logp[token_ids[step]]))

        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenerationResult(text=text, token_ids=token_ids,
                                token_logprobs=token_logprobs or None)

    # ------------------------------------------------------------------ #
    # continuation_logprob (teacher-forced)
    # ------------------------------------------------------------------ #
    def continuation_logprob(self, prompt: str, continuation: str,
                             reduction: str = "sum") -> float:
        """Teacher-forced logprob the model assigns to ``continuation``.

        We concatenate ``prompt + continuation``, run a single forward pass, and
        sum (or mean) the logprobs of exactly the continuation tokens.
        """
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum'|'mean', got {reduction!r}")
        torch = self._torch

        prompt_text = self._format(prompt)
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt",
                                    add_special_tokens=True)["input_ids"]
        # Continuation tokens WITHOUT special tokens so we score only its content.
        cont_ids = self.tokenizer(continuation, return_tensors="pt",
                                  add_special_tokens=False)["input_ids"]

        full = torch.cat([prompt_ids, cont_ids], dim=1).to(self.device)
        with torch.no_grad():
            logits = self.model(full).logits  # (1, T, V)

        logprobs = torch.log_softmax(logits.float(), dim=-1)
        n_prompt = prompt_ids.shape[1]
        n_cont = cont_ids.shape[1]
        if n_cont == 0:
            return 0.0

        total = 0.0
        for i in range(n_cont):
            # Logits at position (n_prompt + i - 1) predict token n_prompt + i.
            pos = n_prompt + i - 1
            tok = int(full[0, n_prompt + i])
            total += float(logprobs[0, pos, tok])

        return total if reduction == "sum" else total / n_cont

    # ------------------------------------------------------------------ #
    # perplexity
    # ------------------------------------------------------------------ #
    def perplexity(self, text: str) -> float:
        """``exp(mean NLL)`` of ``text`` under the model (content-agnostic)."""
        torch = self._torch
        enc = self._encode(text)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return float("nan")
        with torch.no_grad():
            out = self.model(input_ids, labels=input_ids)
        # HF returns mean token NLL as `loss` when labels are provided.
        return float(torch.exp(out.loss))

    # ------------------------------------------------------------------ #
    # activations
    # ------------------------------------------------------------------ #
    def activations(self, prompt: str, layers: list[int],
                    pooling: str = "mean") -> dict[int, np.ndarray]:
        """Pooled residual-stream activations at each requested decoder layer.

        Forward hooks capture each layer's output hidden states; we pool over the
        prompt (answer) token positions. ``pooling`` is ``"mean"`` (default),
        ``"last"``, or ``"max"``.
        """
        torch = self._torch
        captured: dict[int, Any] = {}

        def make_hook(idx: int):
            def hook(_module, _inp, output):
                # Decoder layers return a tuple; the hidden state is element 0.
                hs = output[0] if isinstance(output, tuple) else output
                captured[idx] = hs.detach()
            return hook

        handles = []
        for idx in layers:
            handles.append(self._layers[idx].register_forward_hook(make_hook(idx)))

        try:
            enc = self._encode(self._format(prompt))
            with torch.no_grad():
                self.model(**enc)
        finally:
            for h in handles:
                h.remove()

        out: dict[int, np.ndarray] = {}
        for idx in layers:
            hs = captured[idx][0]  # (T, H) — drop batch dim
            if pooling == "last":
                pooled = hs[-1]
            elif pooling == "max":
                pooled = hs.max(dim=0).values
            else:  # mean over answer tokens
                pooled = hs.mean(dim=0)
            out[idx] = pooled.float().cpu().numpy()
        return out

    # ------------------------------------------------------------------ #
    # generate_with_steering
    # ------------------------------------------------------------------ #
    def generate_with_steering(self, prompt: str, direction: np.ndarray,
                               alpha: float, layer: int, max_new_tokens: int = 40,
                               positions: str = "all") -> GenerationResult:
        """Generate while adding ``alpha * direction`` to layer ``layer``'s output.

        A forward hook fires on every decode step and shifts the residual stream.
        ``positions``:
          * ``"all"``  — add to every token position (default),
          * ``"last"`` — add only to the final position (the token being
            generated), useful for last-token-only interventions.
        """
        torch = self._torch
        vec = torch.tensor(np.asarray(direction, dtype=np.float32),
                           device=self.device, dtype=self.dtype)

        def hook(_module, _inp, output):
            is_tuple = isinstance(output, tuple)
            hs = output[0] if is_tuple else output
            # Cast the steering vector to the hidden state's ACTUAL dtype/device.
            # (In 4-bit models the non-quantized layers can be bf16 while
            # self.dtype is fp16; a mismatched add silently upcasts hs to float32
            # and then crashes at the bf16 lm_head.)
            v = vec.to(dtype=hs.dtype, device=hs.device)
            if positions == "last":
                hs[:, -1, :] = hs[:, -1, :] + alpha * v
            else:
                hs = hs + alpha * v
            if is_tuple:
                return (hs,) + tuple(output[1:])
            return hs

        handle = self._layers[layer].register_forward_hook(hook)
        try:
            enc = self._encode(self._format(prompt))
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
        finally:
            handle.remove()

        new_ids = out.sequences[0][prompt_len:]
        token_ids = [int(t) for t in new_ids]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenerationResult(text=text, token_ids=token_ids,
                                token_logprobs=None)
