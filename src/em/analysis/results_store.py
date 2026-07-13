"""Structured results store.

Everything a stage measures lands here as one flat, append-only JSONL row so the
whole experiment is a single tidy table you can load into pandas, audit by hand,
or diff in git. One row per:

    (run_id, stage, model, adapter/checkpoint, instrument, prompt_id, metric) -> value

plus provenance columns. Human-readable by design.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

try:  # pandas is a core dep, but keep the store importable without it
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


@dataclass
class Measurement:
    run_id: str
    stage: str
    model: str
    checkpoint: int          # optimizer step; -1 for base / not-applicable
    condition: str           # treatment | control | base | n/a
    seed: int
    instrument: str          # logprob | direction | coherence | judge_alignment
    metric: str              # e.g. "mc_divergence", "perplexity", "seq_rep_4", "alpha"
    prompt_id: str           # eval prompt id or "aggregate"
    value: float
    extra: dict[str, Any] = field(default_factory=dict)  # alpha, layer, direction_kind...
    config_hash: str = ""
    git_commit: str = ""


class ResultsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, m: Measurement) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(m), default=str) + "\n")

    def extend(self, ms: Iterable[Measurement]) -> None:
        with self.path.open("a") as f:
            for m in ms:
                f.write(json.dumps(asdict(m), default=str) + "\n")

    def load(self):
        """Return a pandas DataFrame of all measurements (extra dict flattened)."""
        if pd is None:
            raise RuntimeError("pandas required to load results as a DataFrame")
        rows = []
        if not self.path.exists():
            return pd.DataFrame()
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            extra = r.pop("extra", {}) or {}
            for k, v in extra.items():
                r[f"x_{k}"] = v
            rows.append(r)
        return pd.DataFrame(rows)
