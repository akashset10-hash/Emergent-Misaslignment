"""Tests for multi-rater human validation (em.analysis.human_validation).

Priority is correctness of the two reliability estimators:
* Krippendorff's alpha (interval) -- checked against the canonical published
  worked example, plus strong invariants (identical -> 1, random -> ~0,
  monotone: more agreement -> higher alpha).
* ICC(2,1) -- perfect agreement -> ~1, adding noise lowers it.
"""
import numpy as np
import pytest

from em.analysis.human_validation import (
    load_rater_files,
    ratings_matrix,
    krippendorff_alpha_interval,
    icc_2_1,
    mean_pairwise_correlation,
    framework_vs_human,
    validation_report,
)

nan = np.nan


# --------------------------------------------------------------------------- #
# Krippendorff's alpha
# --------------------------------------------------------------------------- #
def test_krippendorff_canonical_interval_example():
    """Krippendorff's own worked example -> published interval alpha = 0.849.

    Reliability data matrix (4 observers x 12 units, with missing values) from
    Krippendorff, "Computing Krippendorff's Alpha-Reliability" (2011).
    """
    m = np.array(
        [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
            [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
        ],
        dtype=float,
    )
    assert krippendorff_alpha_interval(m) == pytest.approx(0.849, abs=0.01)


def test_krippendorff_identical_raters_is_one():
    m = np.array([[1, 2, 3, 4, 5]] * 4, dtype=float)
    assert krippendorff_alpha_interval(m) == pytest.approx(1.0, abs=1e-9)


def test_krippendorff_random_near_zero():
    rng = np.random.default_rng(0)
    # independent raters over many items -> alpha near 0
    m = rng.normal(size=(4, 2000))
    assert abs(krippendorff_alpha_interval(m)) < 0.05


def test_krippendorff_monotone_more_agreement_higher_alpha():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=500)
    # more noise -> less agreement -> lower alpha
    low_noise = np.array([truth + 0.2 * rng.normal(size=500) for _ in range(3)])
    high_noise = np.array([truth + 2.0 * rng.normal(size=500) for _ in range(3)])
    a_low = krippendorff_alpha_interval(low_noise)
    a_high = krippendorff_alpha_interval(high_noise)
    assert a_low > a_high
    assert a_low > 0.8  # tight agreement should read as "good"


def test_krippendorff_handles_missing_without_crashing():
    m = np.array(
        [
            [1, 2, nan, 4, 5],
            [1, nan, 3, 4, 5],
            [nan, 2, 3, 4, nan],
        ],
        dtype=float,
    )
    a = krippendorff_alpha_interval(m)
    assert np.isfinite(a)


# --------------------------------------------------------------------------- #
# ICC(2,1)
# --------------------------------------------------------------------------- #
def test_icc_perfect_agreement():
    m = np.array(
        [
            [10, 20, 30, 40, 50, 60],
            [10, 20, 30, 40, 50, 60],
            [10, 20, 30, 40, 50, 60],
        ],
        dtype=float,
    )
    assert icc_2_1(m) == pytest.approx(1.0, abs=1e-9)


def test_icc_noise_lowers_it():
    rng = np.random.default_rng(2)
    truth = rng.normal(scale=10, size=40)
    clean = np.array([truth, truth, truth])
    noisy = np.array([truth + rng.normal(scale=8, size=40) for _ in range(3)])
    icc_clean = icc_2_1(clean)
    icc_noisy = icc_2_1(noisy)
    assert icc_clean == pytest.approx(1.0, abs=1e-9)
    assert icc_noisy < icc_clean
    assert 0.0 < icc_noisy < 0.95


def test_icc_too_few_complete_items_is_nan():
    # only one item is complete (rated by both raters)
    m = np.array(
        [
            [1, nan, nan],
            [1, 2, nan],
        ],
        dtype=float,
    )
    assert np.isnan(icc_2_1(m))


# --------------------------------------------------------------------------- #
# mean pairwise correlation
# --------------------------------------------------------------------------- #
def test_mean_pairwise_correlation():
    m = np.array(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],  # perfectly anti-correlated with the others
        ],
        dtype=float,
    )
    # pairs: (0,1)=+1, (0,2)=-1, (1,2)=-1  -> mean = -1/3
    assert mean_pairwise_correlation(m) == pytest.approx(-1 / 3, abs=1e-9)


# --------------------------------------------------------------------------- #
# framework vs human
# --------------------------------------------------------------------------- #
def test_framework_vs_human_perfect_correlation():
    # raters jitter symmetrically so the per-item MEAN is exactly {10,20,30,40,50}
    ratings = {
        "r1": {"a": 11, "b": 19, "c": 31, "d": 39, "e": 51},
        "r2": {"a": 9, "b": 21, "c": 29, "d": 41, "e": 49},
    }
    # framework score is a strictly increasing linear function of the mean human
    framework = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    out = framework_vs_human(framework, ratings)
    assert out["n"] == 5
    assert out["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert out["spearman_rho"] == pytest.approx(1.0, abs=1e-6)
    assert len(out["table"]) == 5
    assert out["table"][0]["n_raters"] == 2


# --------------------------------------------------------------------------- #
# loading + matrix + report
# --------------------------------------------------------------------------- #
def test_load_rater_files_and_matrix(tmp_path):
    (tmp_path / "human_ratings_alice.jsonl").write_text(
        '{"id": "x1", "fluency": 80}\n{"id": "x2", "fluency": 60}\n'
    )
    (tmp_path / "human_ratings_bob.jsonl").write_text(
        '{"id": "x1", "fluency": 75}\n{"id": "x3", "fluency": 40}\n'
    )
    loaded = load_rater_files(str(tmp_path / "human_ratings_*.jsonl"))
    assert set(loaded) == {"alice", "bob"}
    assert loaded["alice"]["x1"] == 80.0

    item_ids, matrix = ratings_matrix(loaded)
    assert item_ids == ["x1", "x2", "x3"]
    assert matrix.shape == (2, 3)
    # bob did not rate x2 -> nan
    assert np.isnan(matrix[1, 1])


def test_validation_report_bundles_everything():
    ratings = {
        "r1": {"a": 10, "b": 20, "c": 30, "d": 40, "e": 50},
        "r2": {"a": 11, "b": 22, "c": 29, "d": 41, "e": 49},
        "r3": {"a": 9, "b": 19, "c": 31, "d": 39, "e": 51},
    }
    framework = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    rep = validation_report(ratings, framework)
    assert rep["n_raters"] == 3
    assert rep["n_items"] == 5
    assert rep["n_complete"] == 5
    assert rep["krippendorff_alpha"] > 0.9
    assert rep["icc_2_1"] > 0.9
    assert rep["mean_pairwise_r"] > 0.9
    assert "good inter-rater reliability" in rep["interpretation"]
    assert "framework_vs_human" in rep
    assert rep["framework_vs_human"]["pearson_r"] == pytest.approx(1.0, abs=1e-3)


def test_validation_report_without_framework():
    ratings = {
        "r1": {"a": 10, "b": 20, "c": 30},
        "r2": {"a": 12, "b": 18, "c": 32},
    }
    rep = validation_report(ratings)
    assert "framework_vs_human" not in rep
    assert np.isfinite(rep["krippendorff_alpha"])
