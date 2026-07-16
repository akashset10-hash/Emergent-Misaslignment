"""Automated gate pass-checks.

These compute the pre-registered pass/fail flag for each stage. They NEVER decide
the science — a human confirms the pass is clean (not threshold-hugging) and
decides H1 vs H2. A StageResult already carries `.passed`; this module adds
human-sign-off gating and the Stage-0 retry/escalation policy.
"""
from __future__ import annotations

from dataclasses import dataclass

from em.config import Config
from em.stages.base import StageResult


@dataclass
class GateOutcome:
    passed: bool
    needs_human: bool
    action: str          # "advance" | "iterate" | "abort" | "await_signoff"
    detail: str


def evaluate(result: StageResult, cfg: Config, interactive: bool) -> GateOutcome:
    """Turn a StageResult into a gate action, honoring human-signoff config."""
    if not result.passed:
        # A failed automated check → the loop should iterate (Stage 0 has an
        # escalation policy; others recommend a fix) rather than silently advance.
        return GateOutcome(False, cfg.gates.require_human_signoff, "iterate",
                           f"[{result.stage}] automated check FAILED — {result.recommendation}")
    if cfg.gates.require_human_signoff and interactive:
        return GateOutcome(True, True, "await_signoff",
                           f"[{result.stage}] passed automated check — awaiting human sign-off.")
    return GateOutcome(True, False, "advance",
                       f"[{result.stage}] passed — advancing (human sign-off not required / non-interactive).")


# --------------------------------------------------------------------------- #
# Stage-0 escalation policy: what to change on each retry before giving up.
# --------------------------------------------------------------------------- #
STAGE0_ESCALATION = [
    {"desc": "double max_steps", "apply": lambda c: _set(c, "lora.max_steps", c.lora.max_steps * 2)},
    {"desc": "increase LoRA rank 16→32 (alpha 64)",
     "apply": lambda c: (_set(c, "lora.r", 32), _set(c, "lora.alpha", 64))},
    {"desc": "fall back to 3B model", "apply": lambda c: _set(c, "model.name", c.model.fallback_name)},
]


def escalate_stage0(cfg: Config, attempt: int) -> tuple[Config | None, str]:
    """Return an escalated config for the given retry attempt, or (None, msg) if
    the escalation ladder is exhausted (→ report a genuine negative result)."""
    import copy
    if attempt >= len(STAGE0_ESCALATION):
        return None, ("Stage 0 escalation exhausted — EM not reliably inducible at "
                      "small scale with LoRA. Report this as a genuine negative result.")
    step = STAGE0_ESCALATION[attempt]
    new = copy.deepcopy(cfg)
    step["apply"](new)
    return new, f"Stage 0 retry {attempt + 1}: {step['desc']}"


def _set(cfg, dotted: str, value):
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)
    return cfg
