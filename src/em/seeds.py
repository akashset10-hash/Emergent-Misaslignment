"""Seed control. LoRA has two seed sources — adapter init and data order — and
the paired treatment/control design requires them to SHARE data-order per seed
so the only difference between a pair is the dataset.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass
class SeedBundle:
    seed: int
    init_seed: int      # LoRA adapter initialization
    data_seed: int      # data shuffle order (shared across a treatment/control pair)


def make_bundle(seed: int, offset: int = 0) -> SeedBundle:
    s = seed + offset
    return SeedBundle(seed=s, init_seed=s, data_seed=s)


def set_global_seeds(seed: int) -> None:
    """Set all RNGs we can reach. Torch/numpy set lazily if importable."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
