"""Deflated Sharpe, PBO (CSCV), permutation test and Holm correction."""

import numpy as np
import pytest
from scipy import stats as st

from harness.stats import (deflated_sharpe, expected_max_sharpe, holm, pbo_cscv,
                           permutation_test, probabilistic_sharpe)


def test_psr_reduces_to_the_normal_formula_for_gaussian_moments():
    sr, n = 0.05, 1000
    expected = st.norm.cdf(sr * np.sqrt(n - 1) / np.sqrt(1 + sr ** 2 / 2))   # skew 0, kurt 3
    assert probabilistic_sharpe(sr, 0.0, n, 0.0, 3.0) == pytest.approx(expected)


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    v = 0.0004
    vals = [expected_max_sharpe(v, n) for n in (1, 2, 10, 100, 1000)]
    assert vals[0] == 0.0
    assert all(a < b for a, b in zip(vals, vals[1:]))
    # for large N it approaches sqrt(V) * sqrt(2 ln N) from below
    assert vals[-1] < np.sqrt(v) * np.sqrt(2 * np.log(1000))


def test_deflated_sharpe_falls_as_trials_increase():
    r = np.random.default_rng(0).normal(0.0008, 0.01, 1500)
    one = deflated_sharpe(r, 1, 0.0)["deflated_sharpe"]
    many = deflated_sharpe(r, 200, 0.001)["deflated_sharpe"]
    assert 0 <= many < one <= 1


def test_pbo_is_near_one_half_for_pure_noise():
    # A single draw is noisy (5-95 % range ~0.2-0.8 for these sizes: the 70
    # splits overlap heavily), so the calibration is checked on the average.
    draws = [pbo_cscv(np.random.default_rng(s).normal(0, 0.01, size=(1600, 20)), n_blocks=8)
             for s in range(40)]
    assert draws[0]["n_splits"] == 70
    assert 0.4 < np.mean([d["pbo"] for d in draws]) < 0.65


def test_pbo_is_low_when_one_configuration_really_is_better():
    rng = np.random.default_rng(2)
    m = rng.normal(0, 0.01, size=(1600, 10))
    m[:, 3] += 0.004                                   # strong, persistent edge
    assert pbo_cscv(m, n_blocks=8)["pbo"] < 0.05


def test_pbo_input_validation():
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 3)), n_blocks=5)


def test_permutation_test_detects_a_clear_positive_difference_only():
    rng = np.random.default_rng(3)
    assert permutation_test(rng.normal(0.5, 0.2, 50), 5000)["p_value"] < 0.001
    assert permutation_test(rng.normal(-0.5, 0.2, 50), 5000)["p_value"] > 0.99
    assert permutation_test(np.zeros(10))["p_value"] == 1.0


def test_holm_matches_a_hand_computed_example():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5]), [0.5])
