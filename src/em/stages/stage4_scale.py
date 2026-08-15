"""Stage 4 — OPTIONAL exploratory scale check at 7B (cloud only).

Re-runs the Stage 1 + Stage 2 logic at a larger model via the Modal GPU backend.
Explicitly exploratory: two model sizes cannot establish a scale boundary, only
motivate future work. Runs only after a clean 1.5B Stage 2. Requires a raw GPU
(Modal) — NOT a managed inference endpoint (which can't do activations/steering).
"""
from __future__ import annotations

from em.analysis.results_store import ResultsStore
from em.config import Config
from em.logging_utils import RunLogger
from em.stages import stage1_direction, stage2_intervention
from em.stages.base import StageResult


def run(cfg: Config, log: RunLogger, store: ResultsStore, *, mock: bool = False) -> StageResult:
    if cfg.backend.kind != "modal" and not mock:
        log.warn("Stage 4 expects backend.kind == 'modal' (7B needs a GPU). "
                 "A managed inference endpoint will NOT work — it can't do "
                 "activations/steering. Continuing, but this likely OOMs locally.")
    log.info(f"Stage 4 scale check at {cfg.model.name} via backend={cfg.backend.kind}")

    r1 = stage1_direction.run(cfg, log, store, mock=mock)
    if not r1.passed:
        rec = "Stage 4 direction failed at scale; do not run the intervention. " + r1.recommendation
        log.gate(rec, passed=False)
        return StageResult(stage="stage4", passed=False, recommendation=rec, metrics=r1.metrics)

    r2 = stage2_intervention.run(cfg, log, store, mock=mock)
    rec = ("EXPLORATORY scale result (2 model sizes cannot establish a boundary). " +
           r2.recommendation)
    log.gate(rec, passed=r2.passed)
    return StageResult(stage="stage4", passed=r2.passed, recommendation=rec,
                       metrics={"stage1": r1.metrics, "stage2": r2.metrics})
