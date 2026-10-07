"""M4 features (spec Phase 4 tests): no look-ahead, VIX lag, no NaN after warm-up, residual sanity."""

import numpy as np
import pandas as pd
import pytest

from rl.features_m4 import (M4_FEATURES, _params, build_m4_features, m4_raw_frames, residual_returns,
                            rolling_z, vix_features)
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


# --- pipeline v2 (PROTOCOL Part II §V2.3 Q1/Q4, §V0 item 6) -------------------------------
V2 = dict(SMALL, pipeline="v2", early_close_tickers=["T7"], warmup_bars=200)
FEATS = ["log_ret", "vol_rel20", "resid", "resid_cum30", "resid_avail"]


def test_q1_early_close_ticker_never_sees_same_day_us_returns():
    prices, vix, _ = factor_world()
    tickers = list(prices)
    t = 350
    shocked = {k: d.copy() for k, d in prices.items()}
    for k in tickers[:6]:                                     # the "US-hours" factor tickers
        for col in ("Open", "High", "Low", "Close"):
            shocked[k].iloc[t:, shocked[k].columns.get_loc(col)] *= 1.05   # +5 % on day t, then flat
    col = FEATS.index("resid")
    for params, leaks in ((V2, False), (dict(SMALL, pipeline="v1"), True)):
        a = build_m4_features(prices, tickers, FEATS, tickers[:6], vix, params)["T7"]
        b = build_m4_features(shocked, tickers, FEATS, tickers[:6], vix, params)["T7"]
        changed_at_t = not np.isclose(a[t, col], b[t, col])
        assert changed_at_t == leaks, f"pipeline {params['pipeline']}: day-t residual leak = {changed_at_t}"
        assert not np.isclose(a[t + 1, col], b[t + 1, col])    # the information arrives one bar later
    # US-hours tickers keep their same-day residual (their own close is the US close)
    a0 = build_m4_features(prices, tickers, FEATS, tickers[:6], vix, V2)["T0"]
    b0 = build_m4_features(shocked, tickers, FEATS, tickers[:6], vix, V2)["T0"]
    assert not np.isclose(a0[t, col], b0[t, col])


def test_q4_zero_or_constant_volume_is_masked():
    prices, vix, _ = factor_world()
    p = {k: d.copy() for k, d in prices.items()}
    p["T7"].iloc[200:260, p["T7"].columns.get_loc("Volume")] = 0.0
    p["T6"].iloc[300:360, p["T6"].columns.get_loc("Volume")] = 1000.0     # constant
    f = build_m4_features(p, list(p), FEATS, list(p)[:6], vix, V2)
    c = FEATS.index("vol_rel20")
    assert np.all(f["T7"][200:260, c] == 0.0)
    assert np.all(f["T6"][320:360, c] == 0.0)                   # window fully constant after 20 bars
    assert np.any(f["T7"][150:200, c] != 0.0)


def test_resid_avail_flag_marks_the_warm_up():
    prices, vix, _ = factor_world()
    f = build_m4_features(prices, list(prices), FEATS, list(prices)[:6], vix, V2)["T0"]
    flag = f[:, FEATS.index("resid_avail")]
    assert set(np.unique(flag)) == {0.0, 1.0}
    first = int(np.argmax(flag == 1.0))
    assert first > SMALL["corr_window"] and np.all(flag[first:] == 1.0)
    assert np.all(f[:first, FEATS.index("resid")] == 0.0)


def test_v2_features_still_have_no_lookahead():
    prices, vix, _ = factor_world()
    tickers = list(prices)
    base = build_m4_features(prices, tickers, FEATS, tickers[:6], vix, V2)
    t = 330
    pert = {k: d.copy() for k, d in prices.items()}
    rng = np.random.default_rng(3)
    for d in pert.values():
        noise = np.exp(0.3 * rng.standard_normal(len(d) - t - 1))
        for col in ("Open", "High", "Low", "Close"):
            d.iloc[t + 1:, d.columns.get_loc(col)] *= noise
    new = build_m4_features(pert, tickers, FEATS, tickers[:6], vix, V2)
    for k in tickers:
        np.testing.assert_array_equal(base[k][: t + 1], new[k][: t + 1], err_msg=k)


def test_v2_defaults_follow_the_protocol():
    v2, v1 = _params({"pipeline": "v2"}), _params(None)
    assert v2["early_close_tickers"] == ["^GDAXI"] and v2["mask_volume"]
    assert v2["warmup_bars"] == 450 and v2["z_min_periods"] == 126 and v2["z_window"] == 252
    assert v1["early_close_tickers"] == [] and not v1["mask_volume"]
    assert v1["warmup_bars"] == 0 and v1["z_min_periods"] == 60       # v1 trials keep their features
    assert _params({"pipeline": "v2", "z_min_periods": 60})["z_min_periods"] == 60   # explicit wins


def test_availability_flag_is_zero_until_warmup_bars_of_history():
    prices, vix, _ = factor_world()
    f = build_m4_features(prices, list(prices), FEATS, list(prices)[:6], vix, dict(V2, warmup_bars=300))["T0"]
    flag, res = f[:, FEATS.index("resid_avail")], f[:, FEATS.index("resid")]
    assert np.all(flag[:300] == 0.0) and np.all(res[:300] == 0.0)
    assert np.all(flag[300:] == 1.0) and np.any(res[300:] != 0.0)       # z-scores were valid long before


def test_z_score_is_expanding_until_the_window_is_full():
    x = pd.DataFrame({"a": np.random.default_rng(4).standard_normal(400)})
    z = rolling_z(x, window=252, min_periods=126)["a"].to_numpy()
    assert np.all(np.isnan(z[:125])) and np.isfinite(z[125])            # 126 bars needed
    for i in (125, 200, 251):                                             # expanding before 252 bars
        h = x["a"].to_numpy()[: i + 1]
        assert z[i] == pytest.approx((h[-1] - h.mean()) / h.std(ddof=1))
    h = x["a"].to_numpy()[300 - 251: 301]                                 # rolling afterwards
    assert z[300] == pytest.approx((h[-1] - h.mean()) / h.std(ddof=1))
