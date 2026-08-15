"""The supervised autonomous loop: autonomous BETWEEN gates, human sign-off AT
gates.

Runs each stage, generates figures, writes a decision packet, then either pauses
for human sign-off (interactive) or auto-advances (non-interactive, e.g. smoke /
overnight with `require_human_signoff: false`). Stage 0 has a retry/escalation
policy so the loop can chase the existence gate unattended before giving up.
"""
from __future__ import annotations

from pathlib import Path

from em.analysis.results_store import ResultsStore
from em.config import Config
from em.figures import plots
from em.logging_utils import RunLogger, run_dir_for
from em.loop import gates
from em.loop.decision_packet import DecisionPacket
from em.provenance import stamp
from em.stages import ORDER, STAGES
from em.stages.base import StageResult


class Pipeline:
    def __init__(self, cfg: Config, interactive: bool = True, mock: bool = False,
                 approver=None):
        self.cfg = cfg
        self.interactive = interactive
        self.mock = mock
        # approver(packet) -> "approve"|"iterate"|"abort"; default: auto-approve.
        self.approver = approver or (lambda packet: "approve")

    # ------------------------------------------------------------------ #
    def _run_stage(self, stage: str, cfg: Config) -> tuple[StageResult, RunLogger, ResultsStore]:
        run_dir = run_dir_for(cfg.output_dir, stage, cfg.run_name, cfg.hash())
        prov = stamp(cfg.hash(), cfg.seed_offset)
        log = RunLogger(run_dir, manifest={"stage": stage, "config": _cfg_dict(cfg),
                                           "provenance": prov.as_dict()})
        store = ResultsStore(Path(cfg.output_dir) / "measurements.jsonl")
        log.info(f"=== running {stage} ({cfg.run_name}, hash {cfg.hash()}) ===")
        result = STAGES[stage](cfg, log, store, mock=self.mock)

        figs = []
        try:
            figs = plots.save_all_figures(store, str(Path(cfg.output_dir) / "figures"),
                                          provenance=prov.as_dict())
        except Exception as e:  # figure gen must never crash the pipeline
            log.warn(f"figure generation skipped: {e}")

        packet = DecisionPacket.from_result(result, prov.as_dict(), figs)
        jp, mp = packet.write(run_dir)
        log.info(f"decision packet written: {mp}")
        result.artifacts += [str(jp), str(mp)]
        return result, log, store

    # ------------------------------------------------------------------ #
    def run_stage(self, stage: str) -> StageResult:
        cfg = self.cfg
        if stage == "stage0":
            return self._run_stage0_with_retries(cfg)
        result, log, _ = self._run_stage(stage, cfg)
        self._gate(result, log, cfg)
        return result

    def _run_stage0_with_retries(self, cfg: Config) -> StageResult:
        attempt = 0
        while True:
            result, log, _ = self._run_stage("stage0", cfg)
            if result.passed:
                self._gate(result, log, cfg)
                return result
            new_cfg, msg = gates.escalate_stage0(cfg, attempt)
            log.warn(msg)
            if new_cfg is None:
                self._gate(result, log, cfg)  # exhausted → surface the negative result
                return result
            cfg = new_cfg
            attempt += 1

    def _gate(self, result: StageResult, log: RunLogger, cfg: Config) -> str:
        outcome = gates.evaluate(result, cfg, self.interactive)
        log.gate(outcome.detail, action=outcome.action, needs_human=outcome.needs_human)
        if outcome.action == "await_signoff":
            packet_path = run_dir_for(cfg.output_dir, result.stage, cfg.run_name, cfg.hash())
            decision = self.approver({"stage": result.stage,
                                      "recommendation": result.recommendation,
                                      "packet_dir": str(packet_path)})
            log.gate(f"human decision: {decision}", decision=decision)
            return decision
        return outcome.action

    # ------------------------------------------------------------------ #
    def run_all(self) -> dict[str, StageResult]:
        """Run the gated pipeline in order, stopping if a gate isn't cleared."""
        results = {}
        for stage in ORDER:
            result = self.run_stage(stage)
            results[stage] = result
            if not result.passed:
                # Gate not cleared → halt the pipeline with a clear diagnosis.
                break
            if self.interactive and self.cfg.gates.require_human_signoff:
                decision = self._gate(result, RunLogger(
                    run_dir_for(self.cfg.output_dir, stage, self.cfg.run_name, self.cfg.hash())),
                    self.cfg)
                if decision == "abort":
                    break
        return results


def _cfg_dict(cfg: Config) -> dict:
    from em.config import as_dict
    return as_dict(cfg)
