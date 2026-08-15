"""The four-stage gated pipeline.

Each stage exposes `run(cfg, log, store, *, mock=False) -> StageResult`.
Order + gating live in `em.loop.state_machine`.
"""
from __future__ import annotations

from em.stages import (stage0_existence, stage1_direction, stage2_intervention,
                       stage3_timing, stage4_scale)
from em.stages.base import StageResult

STAGES = {
    "stage0": stage0_existence.run,
    "stage1": stage1_direction.run,
    "stage2": stage2_intervention.run,
    "stage3": stage3_timing.run,
    "stage4": stage4_scale.run,
}

# Gate dependency order for the full pipeline.
ORDER = ["stage0", "stage1", "stage2", "stage3"]  # stage4 is opt-in

__all__ = ["STAGES", "ORDER", "StageResult"]
