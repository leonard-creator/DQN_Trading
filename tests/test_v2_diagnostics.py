"""v2 step 0a: execution lag, vol-target baseline, timing diagnostics, exposure-matched benchmark."""

import numpy as np
import pandas as pd
import pytest

from harness.backtest import backtest
from harness.baselines import vol_target
from harness.diagnostics import learning_curve_stats, sleeve_timing

CLOSE = 100 * np.cumprod(np.r_[1.0, 1 + np.array([0.01, -0.02, 0.03, 0.005, 0.01, -0.01, 0.02])])
POS = np.arange(1, len(CLOSE))


def test_lag1_executes_at_the_next_close_and_starts_flat():
    e = np.array([0.25, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0])
    lag0 = backtest(CLOSE, e, POS, 0.001)
    lag1 = backtest(CLOSE, e, POS, 0.001, lag=1)
    np.testing.assert_array_equal(lag1["exposure"], np.r_[0.0, lag0["exposure"][:-1]])
    assert lag1["exposure"][0] == 0.0 and lag1["cost"][0] == 0.0       # nothing executed on day 1
    np.testing.assert_allclose(lag1["cost"], 0.001 * np.abs(np.diff(np.r_[0.0, lag1["exposure"]])))


def test_lag1_buy_and_hold_pays_its_entry_one_day_later():
    res = backtest(CLOSE, np.ones(len(CLOSE)), POS, 0.002, lag=1)
    assert res["gross"][0] == 0.0
    assert res["cost"][1] == pytest.approx(0.002) and res["cost"][2:].sum() == 0.0


def test_vol_target_is_causal_on_the_grid_and_capped():
    rng = np.random.default_rng(0)
    vol = np.r_[np.full(400, 0.01), np.full(200, 0.03), np.full(200, 0.01)]
    c = 100 * np.exp(np.cumsum(vol * rng.standard_normal(800)))
    e = vol_target(c, span=20, min_history=100)
    assert set(np.unique(e)) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert e[:60].max() == 0.0                                         # not enough history yet
    assert e[300:400].mean() > e[500:600].mean()                      # de-risks in the high-vol regime
    t = 650
    pert = c.copy()
    pert[t + 1:] *= np.exp(0.2 * rng.standard_normal(len(c) - t - 1))
    np.testing.assert_array_equal(e[: t + 1], vol_target(pert, span=20, min_history=100)[: t + 1])


def test_timing_ic_sign_and_exposure_matched_benchmark():
    rng = np.random.default_rng(1)
    r = 0.01 * rng.standard_normal(500)
    c = 100 * np.cumprod(np.r_[1.0, 1 + r])
    pos = np.arange(1, len(c))
    sigma = np.full(len(c), 0.01)
    oracle = np.r_[np.where(r > 0, 1.0, 0.0), 0.0]                 # exposure at t knows r_{t+1}
    d = sleeve_timing(c, oracle, pos, sigma)
    assert d["ic"] > 0.8
    d0 = sleeve_timing(c, np.full(len(c), 0.75), pos, sigma)
    assert np.isnan(d0["ic"]) and d0["switches_per_100"] == pytest.approx(100 / len(pos))
    # exposure-matched benchmark = mean exposure x buy-and-hold return, per sleeve and block
    np.testing.assert_allclose(d["matched_ret"], d["ebar"] * (c[pos] / c[pos - 1] - 1))
    np.testing.assert_allclose(d["timing_ret"], (oracle[pos - 1] - d["ebar"]) * (c[pos] / c[pos - 1] - 1))


def test_learning_curve_stats():
    curve = pd.DataFrame({"update": [10, 20, 30, 40, 50, 60, 70, 80],
                          "inner_sharpe": [1.0, 1.5, 2.0, 2.0, 1.5, 1.0, 0.5, 0.0]})
    s = learning_curve_stats(curve, {"updates": 80, "best_update": 30})
    assert s["lc_best_at"] == pytest.approx(0.375) and s["lc_last_half_slope"] < 0
    assert 0.5 < s["lc_auc"] < 2.0


def test_q_diagnostics_reproduce_the_stored_exposures(synthetic_agent_cfg, monkeypatch, tmp_path):
    """Re-running the saved weights through the shared data pipeline gives the SAME exposures."""
    from harness import trials as tr
    from harness.experiment import run_experiment
    from rl.diagnostics import q_diagnostics_for_run
    monkeypatch.setattr(tr, "append_trial", lambda row: None)
    res = run_experiment(synthetic_agent_cfg, seeds=[0], cost_levels=[10], out_root=str(tmp_path), verbose=False)
    diag = q_diagnostics_for_run(res.output_dir, workers=1)
    assert len(diag) == 2                                   # 2 folds x 1 eval ticker (single)
    assert (diag["reproduced"] == 1.0).all()
    assert (diag["td_sd"] > 0).all() and diag["gap_median"].notna().all()
