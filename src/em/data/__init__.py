"""Data package: eval prompts + SFT dataset loaders.

Pure-Python, no optional deps at import time.
"""
from __future__ import annotations

from .eval_prompts import (
    MAIN_8,
    FULL_48,
    TRAIT_PROMPTS,
    NEUTRAL_PROMPTS,
    MISALIGNED_TEXTS,
    ALIGNED_TEXTS,
    LOGPROB_PAIRS,
    PIVOTAL_TOKEN_PAIRS,
    get_eval_prompts,
)
from .datasets import (
    DATA_README,
    load_pairs,
    make_synthetic_pairs,
    prepare_sft_dataset,
)

__all__ = [
    # eval prompts
    "MAIN_8",
    "FULL_48",
    "TRAIT_PROMPTS",
    "NEUTRAL_PROMPTS",
    "MISALIGNED_TEXTS",
    "ALIGNED_TEXTS",
    "LOGPROB_PAIRS",
    "PIVOTAL_TOKEN_PAIRS",
    "get_eval_prompts",
    # datasets
    "DATA_README",
    "load_pairs",
    "make_synthetic_pairs",
    "prepare_sft_dataset",
]
