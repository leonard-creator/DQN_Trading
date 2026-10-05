"""The shared cost model and the decision -> return timing."""

import numpy as np
import pandas as pd
import pytest

from harness.backtest import backtest, cost_rate, portfolio, to_frame

CLOSE = 100 * np.cumprod(np.r_[1.0, 1 + np.array([0.01, -0.02, 0.03, 0.0, 0.01, -0.01, 0.02])])


def test_cost_rate_is_bps_plus_half_spread():
    assert cost_rate(10, 1) == pytest.approx(0.0011)
    assert cost_rate(0, 0) == 0.0


def test_buy_and_hold_pays_entry_cost_once_and_tracks_the_price():
    pos = np.arange(1, len(CLOSE))
    rate = cost_rate(10, 1)
    res = backtest(CLOSE, np.ones(len(CLOSE)), pos, rate)
    asset = CLOSE[1:] / CLOSE[:-1] - 1
    assert res["cost"][0] == pytest.approx(rate) and np.all(res["cost"][1:] == 0)
    np.testing.assert_allclose(res["gross"], asset)
    np.testing.assert_allclose(res["net"], asset - np.r_[rate, np.zeros(len(asset) - 1)])


def test_exposure_decided_at_t_only_earns_the_return_of_t_plus_1():
    e = np.zeros(len(CLOSE))
    e[2] = 1.0                                      # decided at the close of bar 2
    res = backtest(CLOSE, e, np.arange(1, len(CLOSE)), 0.0)
    nonzero = np.flatnonzero(res["gross"])
    assert list(nonzero + 1) == [3]                 # return position 3 = bar 2 -> bar 3
    assert res["gross"][2] == pytest.approx(CLOSE[3] / CLOSE[2] - 1)


def test_round_trip_is_charged_twice():
    e = np.zeros(len(CLOSE))
    e[2] = 1.0
    res = backtest(CLOSE, e, np.arange(1, len(CLOSE)), 0.001)
    assert res["turnover"].sum() == pytest.approx(2.0)
    assert res["cost"].sum() == pytest.approx(0.002)


def test_flat_strategy_has_zero_returns():
    res = backtest(CLOSE, np.zeros(len(CLOSE)), np.arange(1, len(CLOSE)), 0.01)
    assert np.all(res["net"] == 0)


def test_nan_exposure_on_a_decision_bar_fails_loudly_but_elsewhere_is_ignored():
    e = np.full(len(CLOSE), np.nan)
    e[3:6] = 0.5                                    # decision bars of return positions 4..6
    backtest(CLOSE, e, np.arange(4, 7), 0.0)         # fine
    with pytest.raises(ValueError):
        backtest(CLOSE, e, np.arange(3, 7), 0.0)     # needs exposure at bar 2


def test_block_must_have_a_bar_before_it():
    with pytest.raises(ValueError):
        backtest(CLOSE, np.ones(len(CLOSE)), np.arange(0, 3), 0.0)


def test_portfolio_is_equal_weight_and_holidays_earn_zero():
    d = pd.bdate_range("2020-01-01", periods=4)
    a = to_frame({"net": np.array([0.01, 0.02, 0.03, 0.04]), "gross": np.zeros(4), "cost": np.zeros(4),
                  "exposure": np.ones(4), "turnover": np.zeros(4)}, d)
    b = to_frame({"net": np.array([0.03, 0.05]), "gross": np.zeros(2), "cost": np.zeros(2),
                  "exposure": np.ones(2), "turnover": np.zeros(2)}, d[[0, 3]])
    port = portfolio({"A": a, "B": b})
    np.testing.assert_allclose(port["net"].to_numpy(), [0.02, 0.01, 0.015, 0.045])
