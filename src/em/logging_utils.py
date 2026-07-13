"""Structured, human-readable, auditable logging.

Two channels:
  * console — coloured, human-skimmable narration of what the pipeline is doing.
  * run manifest + event log — machine-parseable JSON on disk, one directory per
    run, so any experiment can be reconstructed and audited after the fact.

A `RunLogger` owns a run directory:
    results/<stage>/<run_name>_<config_hash>/
        manifest.json      # config, provenance, final status
        events.jsonl       # append-only timeline of everything that happened
        <artifacts...>     # figures, tables, decision packets
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LEVEL_ICONS = {"INFO": "•", "WARNING": "!", "ERROR": "✗", "GATE": "⏸", "OK": "✓"}


def _console_logger() -> logging.Logger:
    lg = logging.getLogger("em")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


class RunLogger:
    def __init__(self, run_dir: str | Path, manifest: dict | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self._console = _console_logger()
        if manifest is not None:
            self.write_manifest(manifest)

    # ------------------------------------------------------------------ #
    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def event(self, kind: str, message: str, **data: Any) -> None:
        """Log a structured event to disk + a human line to the console."""
        record = {"ts": self._now(), "kind": kind, "message": message, **data}
        with self.events_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        icon = _LEVEL_ICONS.get(kind.upper(), "•")
        self._console.info(f"{icon} [{kind}] {message}")

    def info(self, message: str, **data: Any) -> None:
        self.event("INFO", message, **data)

    def ok(self, message: str, **data: Any) -> None:
        self.event("OK", message, **data)

    def warn(self, message: str, **data: Any) -> None:
        self.event("WARNING", message, **data)

    def error(self, message: str, **data: Any) -> None:
        self.event("ERROR", message, **data)

    def gate(self, message: str, **data: Any) -> None:
        self.event("GATE", message, **data)

    # ------------------------------------------------------------------ #
    def write_manifest(self, manifest: dict) -> None:
        existing = {}
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text())
        existing.update(manifest)
        self.manifest_path.write_text(json.dumps(existing, indent=2, default=str))

    def artifact_path(self, name: str) -> Path:
        return self.run_dir / name


def run_dir_for(output_dir: str | Path, stage: str, run_name: str, config_hash: str) -> Path:
    return Path(output_dir) / stage / f"{run_name}_{config_hash}"
