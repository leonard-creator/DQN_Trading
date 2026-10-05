"""Metric definitions (harness/metrics.py docstring) on hand-checkable inputs."""

import numpy as np
import pandas as pd
import pytest

from harness.metrics import (aggregate, holding_runs, max_drawdown, portfolio_metrics,
                             return_metrics, sharpe)


def test_sharpe_matches_definition_and_handles_zero_variance():
    r = np.array([0.01, -0.005, 0.002, 0.004])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert sharpe(np.zeros(10)) == 0.0


def test_max_drawdown_of_a_known_path():
    # equity 1 -> 1.1 -> 0.88 -> 0.968 : peak 1.1, trough 0.88 -> 20 %
    assert max_drawdown(np.array([0.10, -0.20, 0.10])) == pytest.approx(0.20)
    assert max_drawdown(np.array([0.01, 0.02])) == 0.0
    assert max_drawdown(np.array([-0.5])) == pytest.approx(0.5)   # loss on day one counts


def test_return_metrics_total_and_cagr():
    r = np.full(252, 0.001)
    m = return_metrics(r)
    assert m["total_return"] == pytest.approx(1.001 ** 252 - 1)
    assert m["cagr"] == pytest.approx(m["total_return"])          # exactly one year
    assert m["kurtosis"] == pytest.approx(3.0)                      # constant -> fallback


def test_holding_runs():
    assert holding_runs(np.array([0, 1, 1, 0, 0.5, 0.5, 0.5, 0])) == (2, 5)
    assert holding_runs(np.zeros(5)) == (0, 0)


def test_portfolio_metrics_trade_statistics():
    d = pd.bdate_range("2020-01-01", periods=6)
    f = pd.DataFrame({"exposure": [0, 1, 1, 0, 1, 1], "net": [0, 0.01, -0.01, 0, 0.02, 0.01],
                      "gross": 0.0, "cost": 0.0, "turnover": [0, 1, 0, 1, 1, 0]}, index=d)
    m = portfolio_metrics(f, {"A": f})
    assert m["avg_holding"] == pytest.approx(2.0)          # two runs of 2 bars
    assert m["hit_rate"] == pytest.approx(3 / 4)           # 3 of 4 invested days positive
    assert m["exposure"] == pytest.approx(4 / 6)
    assert m["turnover"] == pytest.approx(3 / 6 * 252)


def test_aggregate_median_and_iqr_ignore_nan():
    df = pd.DataFrame({"sharpe": [1.0, 2.0, 3.0, np.nan]})
    a = aggregate(df, metrics=("sharpe",))
    assert a["sharpe_median"] == 2.0 and a["sharpe_iqr"] == pytest.approx(1.0)
