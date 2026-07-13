"""Model backends.

Every instrument in :mod:`em.instruments` is a pure function of a
``ModelBackend`` (see :mod:`em.backends.base`). Concrete backends (HF-local,
LM Studio, Modal) live alongside this module; the :class:`MockBackend` defined
here is a fully deterministic, dependency-free implementation used for tests,
CI, and instrument self-verification.

Use :func:`make_backend` to construct the backend selected by a
:class:`~em.config.Config`. The heavy backends (``hf_local``, ``lmstudio``,
``modal``) import ``torch`` / ``openai`` / ``modal`` LAZILY, so this package —
and the factory itself — imports cleanly on a machine with none of them
installed. That is what lets the whole pipeline run against :class:`MockBackend`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import (
    Capability,
    CapabilityError,
    GenerationResult,
    ModelBackend,
    require,
)
from .mock import MockBackend

if TYPE_CHECKING:
    from ..config import Config


def make_backend(cfg: "Config", adapter_path: str | None = None) -> ModelBackend:
    """Construct the backend selected by ``cfg.backend.kind``.

    Dispatch is lazy: the concrete backend module (and therefore its heavy
    third-party dependency) is imported only when that ``kind`` is requested, so
    importing this factory never requires torch/openai/modal to be present.

    Parameters
    ----------
    cfg
        A :class:`em.config.Config`. ``cfg.backend.kind`` is one of
        ``"hf_local" | "lmstudio" | "modal" | "mock"``.
    adapter_path
        Optional path to a PEFT/LoRA adapter, forwarded to the backends that can
        load one (``hf_local`` and ``modal``). Ignored by ``lmstudio``/``mock``.

    Returns
    -------
    ModelBackend
        A concrete backend satisfying the :class:`ModelBackend` Protocol.

    Raises
    ------
    ValueError
        If ``cfg.backend.kind`` is not recognized.
    """
    kind = cfg.backend.kind

    if kind == "mock":
        return MockBackend()

    if kind == "hf_local":
        from .hf_local import HFLocalBackend
        return HFLocalBackend(cfg=cfg, adapter_path=adapter_path)

    if kind == "lmstudio":
        from .lmstudio import LMStudioBackend
        return LMStudioBackend(cfg=cfg)

    if kind == "modal":
        from .modal_backend import ModalBackend
        return ModalBackend(cfg=cfg, adapter_path=adapter_path)

    raise ValueError(
        f"Unknown backend kind {kind!r}. "
        f"Expected one of: hf_local, lmstudio, modal, mock."
    )


__all__ = [
    "make_backend",
    "Capability",
    "CapabilityError",
    "GenerationResult",
    "ModelBackend",
    "require",
    "MockBackend",
]
