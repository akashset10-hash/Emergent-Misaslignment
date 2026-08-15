"""Provenance stamping: every artifact carries git commit + config hash + seed +
timestamp so any figure or number can be traced back to the exact code and
config that produced it. Auditability is a first-class requirement here.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class Provenance:
    git_commit: str
    git_dirty: bool
    config_hash: str
    seed: int
    timestamp: str
    python: str
    platform: str

    def as_dict(self) -> dict:
        return asdict(self)


def stamp(config_hash: str, seed: int) -> Provenance:
    status = _git("status", "--porcelain")
    return Provenance(
        git_commit=_git("rev-parse", "--short", "HEAD"),
        git_dirty=bool(status.strip()),
        config_hash=config_hash,
        seed=seed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        platform=platform.platform(),
    )
