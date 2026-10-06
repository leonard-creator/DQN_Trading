"""M4 features (spec Phase 4 tests): no look-ahead, VIX lag, no NaN after warm-up, residual sanity."""

import numpy as np
import pandas as pd
import pytest

from rl.features_m4 import (M4_FEATURES, build_m4_features, m4_raw_frames, residual_returns,
                            vix_features)
from tests.conftest import make_prices

SMALL = {"pca_k": 2, "corr_window": 80, "beta_window": 30, "cum_window": 10,
         "z_window": 100, "z_min_periods": 30}


def factor_world(n=500, n_tickers=8, seed=0):
    """Tickers driven by one common market factor plus idiosyncratic noise, and a VIX series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2012-01-02", periods=n)
    market = 0.01 * rng.standard_normal(n)
    prices = {}
    for i in range(n_tickers):
        r = (0.5 + 0.2 * i) * market + 0.004 * rng.standard_normal(n)
        df = make_prices(dates, seed=100 + i)
        close = 100 * np.exp(np.cumsum(r))
        df["Close"], df["High"], df["Low"], df["Open"] = close, close * 1.005, close * 0.995, close
        prices[f"T{i}"] = df.set_index("Date")
    vix = pd.DataFrame({"Close": 15 + np.cumsum(0.3 * rng.standard_normal(n)).clip(-10, 30)},
                       index=dates)
    return prices, vix, market


def test_no_lookahead_for_every_m4_feature():
    prices, vix, _ = factor_world()
    tickers = list(prices)
    base = build_m4_features(prices, tickers, M4_FEATURES, tickers[:6], vix, SMALL)
    t = 350
    rng = np.random.default_rng(9)
    pert = {}
    for k, df in prices.items():
        d = df.copy()
        noise = np.exp(0.3 * rng.standard_normal(len(d) - t - 1))
        for col in ("Open", "High", "Low", "Close"):
            d.iloc[t + 1:, d.columns.get_loc(col)] *= noise
        d.iloc[t + 1:, d.columns.get_loc("Volume")] *= 5
        pert[k] = d
    vix_p = vix.copy()
    vix_p.iloc[t + 1:, 0] *= 3                               # rewrite the future VIX too
    # VIX bar t itself is NOT known at bar t (one-bar lag), so changing it must not matter either
    vix_p.iloc[t, 0] *= 2
    new = build_m4_features(pert, tickers, M4_FEATURES, tickers[:6], vix_p, SMALL)
    for k in tickers:
        np.testing.assert_array_equal(base[k][: t + 1], new[k][: t + 1], err_msg=k)


def test_vix_is_joined_from_the_bar_strictly_before():
    vdates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    vix = pd.DataFrame({"Close": [10.0, 20.0, 30.0, 40.0]}, index=vdates)
    # 2020-01-06 exists in VIX (must use 01-03); 2020-01-04 is not a VIX day (must use 01-03 too)
    dates = pd.to_datetime(["2020-01-02", "2020-01-04", "2020-01-06", "2020-01-08"])
    f = vix_features(dates, vix)
    assert np.isnan(f["vix_lag1"].iloc[0])                   # nothing known before the first VIX bar
    np.testing.assert_allclose(f["vix_lag1"].iloc[1:], np.log([20.0, 20.0, 40.0]))
    assert f["vix_chg_lag1"].iloc[3] == pytest.approx(np.log(40) - np.log(30))


def test_residuals_remove_most_of_the_common_factor():
    prices, _, market = factor_world(n=600)
    closes = pd.concat({k: d["Close"] for k, d in prices.items()}, axis=1)
    factor_tickers = list(prices)[:6]
    res = residual_returns(closes, factor_tickers, k=1, corr_window=150, beta_window=60)
    raw = np.log(closes).diff()
    for k in prices:                                          # incl. T6, T7 outside the factor set
        valid = res[k].notna()
        assert valid.sum() > 300
        assert res[k][valid].var() < 0.35 * raw[k][valid].var(), k
        assert abs(np.corrcoef(res[k][valid], market[valid.to_numpy()])[0, 1]) < 0.3, k


def test_residual_at_t_does_not_use_return_t_for_its_model():
    prices, _, _ = factor_world(n=400)
    closes = pd.concat({k: d["Close"] for k, d in prices.items()}, axis=1)
    a = residual_returns(closes, list(prices)[:6], 1, 100, 40)
    t = 300
    shocked = closes.copy()
    shocked.iloc[t:, 0] *= 1.10                               # +10 % jump of T0 on day t, then flat
    b = residual_returns(shocked, list(prices)[:6], 1, 100, 40)
    np.testing.assert_array_equal(a.iloc[:t].to_numpy(), b.iloc[:t].to_numpy())
    assert b.iloc[t, 0] > a.iloc[t, 0] + 0.05                 # the jump shows up in the residual at t


def test_real_data_has_no_nan_after_warmup():
    from harness.config import load_config
    from harness.data import load_prices, ticker_sets
    cfg = load_config()
    sets = ticker_sets(cfg)
    try:
        prices = load_prices(sets["all"] + ["^VIX"], cfg)
    except FileNotFoundError:
        pytest.skip("data/raw not downloaded")
    raw = m4_raw_frames(prices, sets["all"], M4_FEATURES, sets["train"], prices["^VIX"])
    first = pd.Timestamp(cfg["splits"]["dev_start"])
    for t, frame in raw.items():
        i0 = frame.index.searchsorted(first) - 20             # first bar any observation window reads
        bad = frame.iloc[i0:].isna().sum()
        assert bad.sum() == 0, f"{t}: NaN after warm-up in {bad[bad > 0].to_dict()}"


def test_m4_features_end_to_end_training(synthetic_agent_cfg):
    from harness.config import deep_merge
    from harness.experiment import run_experiment
    feats = ["log_ret", "p_sma20", "bb_pctb", "rsi14", "macd_hist", "vol_rel20", "atr14_p", "sigma20"]
    cfg = deep_merge(synthetic_agent_cfg, {"agent": {"feature_mode": "m4", "features": feats,
                                                     "m4": {"factor_set": "train", "z_window": 100,
                                                            "z_min_periods": 30},
                                                     "env": {"reward": "vol_scaled_pnl", "reward_clip": 10.0}}})
    res = run_experiment(cfg, seeds=[0], cost_levels=[10], log=False, verbose=False)
    assert len(res.metrics) == 2 and np.isfinite(res.metrics["sharpe"]).all()
