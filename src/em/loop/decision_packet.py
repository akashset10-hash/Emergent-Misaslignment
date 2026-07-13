"""Decision packets — what a human signs off on at each gate.

At every gate the loop writes a machine-readable `decision_packet.json` and a
human-readable `decision_packet_<stage>.md` containing: the pass/fail flags, the
recommended call + reasoning, the diagnostic figures, and — mandatorily — the
"underneath-the-gate" secondary signals. The last item is non-negotiable because
this project exists precisely because an automated pipeline once reported a clean
headline while the real signal sat underneath it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from em.stages.base import StageResult


@dataclass
class DecisionPacket:
    stage: str
    passed: bool
    recommendation: str
    metrics: dict[str, Any] = field(default_factory=dict)
    secondary_signals: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: StageResult, provenance: dict, figures: list[str]):
        return cls(stage=result.stage, passed=result.passed,
                   recommendation=result.recommendation, metrics=result.metrics,
                   secondary_signals=result.secondary_signals, figures=figures,
                   provenance=provenance)

    def to_markdown(self) -> str:
        L = [f"# Decision packet — {self.stage}", ""]
        L.append(f"**Automated pass-check:** {'PASS ✓' if self.passed else 'FAIL ✗'}")
        L.append("")
        L.append("## Recommended call (the loop recommends; the human decides)")
        L.append(f"> {self.recommendation}")
        L.append("")
        L.append("## Metrics")
        L.append("```json")
        L.append(json.dumps(self.metrics, indent=2, default=str))
        L.append("```")
        L.append("")
        L.append("## ⚠ Underneath-the-gate signals (read these before signing off)")
        if self.secondary_signals:
            L.append("```json")
            L.append(json.dumps(self.secondary_signals, indent=2, default=str))
            L.append("```")
        else:
            L.append("_none recorded for this stage_")
        L.append("")
        if self.figures:
            L.append("## Figures")
            for f in self.figures:
                L.append(f"- `{f}`")
            L.append("")
        L.append("## Provenance")
        L.append("```json")
        L.append(json.dumps(self.provenance, indent=2, default=str))
        L.append("```")
        L.append("")
        L.append("## Sign-off")
        L.append("- [ ] approve — advance to next stage")
        L.append("- [ ] iterate — re-run this stage with changes")
        L.append("- [ ] abort")
        return "\n".join(L)

    def write(self, run_dir: str | Path) -> tuple[Path, Path]:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        jp = run_dir / "decision_packet.json"
        mp = run_dir / f"decision_packet_{self.stage}.md"
        jp.write_text(json.dumps(asdict(self), indent=2, default=str))
        mp.write_text(self.to_markdown())
        return jp, mp
