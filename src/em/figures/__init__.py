"""Figures package: provenance-stamped, head-less PNG figure generation.

All figures render on the matplotlib ``Agg`` backend (no display) and carry a
footer noting the git commit + config hash and that coherence is content-agnostic
(never scored by an LLM judge).
"""
from __future__ import annotations

from .plots import (
    plot_dose_response,
    plot_trajectories,
    plot_existence,
    plot_validation_scatter,
    plot_coherence_components,
    save_all_figures,
)

__all__ = [
    "plot_dose_response",
    "plot_trajectories",
    "plot_existence",
    "plot_validation_scatter",
    "plot_coherence_components",
    "save_all_figures",
]
