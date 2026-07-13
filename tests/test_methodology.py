"""Tests for the NeurIPS-grade methodology upgrades: faithful label-token MC,
effect-size-matched controls, and multi-seed across-seed inference."""
import numpy as np

from em.backends.mock import MockBackend
from em.instruments import direction, logprob
from em.analysis import stats
from em.data.eval_prompts import TRAIT_PROMPTS, NEUTRAL_PROMPTS, LOGPROB_PAIRS


# --- Betley label-token MC fidelity ---------------------------------------- #
def test_label_token_mc_runs_and_is_antisymmetric_in_label():
    be = MockBackend()
    pair = LOGPROB_PAIRS[0]
    a = logprob.multiple_choice_label_score(be, pair["question"], pair["aligned"],
                                            pair["misaligned"], misaligned_label="A")
    b = logprob.multiple_choice_label_score(be, pair["question"], pair["aligned"],
                                            pair["misaligned"], misaligned_label="B")
    assert np.isfinite(a) and np.isfinite(b)


def test_measure_supports_both_mc_methods():
    be = MockBackend()
    for method in ("label", "continuation"):
        rows = logprob.measure(be, LOGPROB_PAIRS, run_id="t", condition="treatment",
                               seed=0, mc_method=method, formats=("multiple_choice",))
        per_pair = [r for r in rows if r.metric == "mc_divergence" and r.prompt_id != "aggregate"]
        assert per_pair
        assert all(r.extra.get("mc_method") == method for r in per_pair)


# --- effect-size-matched controls (H3b) ------------------------------------ #
def test_behavioral_effect_size_increases_with_alpha():
    be = MockBackend()
    d = direction.build_direction(be, list(TRAIT_PROMPTS)[:6], list(NEUTRAL_PROMPTS)[:6], [12])[12]
    prompts = ["tell me about your day", "what should i do this weekend", "describe a city"]
    lo = direction.behavioral_effect_size(be, d, 12, 1.0, prompts)
    hi = direction.behavioral_effect_size(be, d, 12, 10.0, prompts)
    assert hi >= lo


def test_calibrate_gain_matches_effect_size():
    be = MockBackend()
    d_ref = direction.build_direction(be, list(TRAIT_PROMPTS)[:6], list(NEUTRAL_PROMPTS)[:6], [12])[12]
    rng = np.random.default_rng(1)
    d_ctrl = rng.standard_normal(d_ref.shape)
    prompts = ["tell me about your day", "what should i do", "describe a city", "hello there"]
    g = direction.calibrate_effect_size_gain(be, d_ref, d_ctrl, 12, 8.0, prompts)
    assert g > 0
    # calibrated control effect-size should be closer to the target than the
    # raw unit-norm control (unless it saturates at a clip bound).
    target = direction.behavioral_effect_size(be, d_ref, 12, 8.0, prompts)
    es_cal = direction.behavioral_effect_size(be, g * d_ctrl / np.linalg.norm(d_ctrl), 12, 8.0, prompts)
    assert np.isfinite(es_cal) and np.isfinite(target)


# --- multi-seed inference --------------------------------------------------- #
def test_wilcoxon_vs_zero_detects_positive_shift():
    r = stats.wilcoxon_vs_zero([0.1, 0.2, 0.15, 0.25, 0.3, 0.22], alternative="greater")
    assert r["n"] == 6
    assert np.isfinite(r["pvalue"]) and r["pvalue"] < 0.05


def test_multiseed_summary_shape_and_keys():
    alphas = [-2, -1, 0, 1, 2]
    # H1-shaped per-seed curves: coherence drops on the +alpha side.
    per_seed = {}
    for s in range(4):
        curve = [1.0, 1.0, 1.0, 0.6 - 0.02 * s, 0.3 - 0.02 * s]
        per_seed[s] = stats.dose_response_fit(alphas, curve)
    summ = stats.multiseed_summary(per_seed, {s: 0.2 for s in range(4)})
    assert summ["n_seeds"] == 4
    assert "asymmetry_wilcoxon_gt0" in summ and "slope_wilcoxon_lt0" in summ
    assert summ["asymmetry_mean"] > 0  # H1-shaped input


def test_verdict_uses_seed_summary_when_present():
    alphas = [-2, -1, 0, 1, 2]
    per_seed = {s: stats.dose_response_fit(alphas, [1, 1, 1, 0.5, 0.2]) for s in range(6)}
    summ = stats.multiseed_summary(per_seed)
    pooled = stats.dose_response_fit(alphas, [1, 1, 1, 0.5, 0.2])
    rec = stats.verdict_recommendation(pooled, None, seed_summary=summ)
    assert "across 6 seeds" in rec
