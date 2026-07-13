"""Alignment judge package.

Exposes the LLM alignment/harm judge (scores alignment ONLY, never coherence),
its deterministic offline mock, the factory, and the rubric constant. No
optional deps at import time (``openai`` is imported lazily).
"""
from __future__ import annotations

from .alignment_judge import (
    ALIGNMENT_RUBRIC,
    AlignmentJudge,
    MockJudge,
    JudgeBudgetError,
    make_judge,
)

__all__ = [
    "ALIGNMENT_RUBRIC",
    "AlignmentJudge",
    "MockJudge",
    "JudgeBudgetError",
    "make_judge",
]
