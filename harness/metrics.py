"""Performance metrics on daily NET returns (PROTOCOL §8).

All return-based metrics use the risk-free rate = 0 and are annualised with
`bars_per_year` (252 for daily data). Definitions, so the reports can be read
without the code:

    total_return   prod(1 + r) - 1
    cagr           (1 + total_return) ** (bars_per_year / n) - 1
    ann_vol        std(r, ddof=1) * sqrt(bars_per_year)
    sharpe         mean(r) / std(r, ddof=1) * sqrt(bars_per_year);  0 if std == 0
    sharpe_pp      the same without sqrt(bars_per_year) ("per period"; needed
                   by the deflated Sharpe ratio)
    sortino        mean(r) / sqrt(mean(min(r, 0)^2)) * sqrt(bars_per_year)
    max_drawdown   largest peak-to-trough loss of the equity curve, as a
                   positive fraction (0.25 = -25 %); the curve starts at 1.0
    calmar         cagr / max_drawdown
    turnover       mean(|change in exposure|) * bars_per_year
                   (one-way, in multiples of capital per year)
    exposure       mean(|exposure|)
    hit_rate       share of invested days (exposure != 0) with net return > 0
    avg_holding    mean length in bars of a continuous invested run
                   (exposure != 0); pooled over tickers for a portfolio
    skew, kurtosis sample skewness and (non-excess) kurtosis of r; normal = 0 and 3

Metrics that are undefined (e.g. calmar with no drawdown, or anything on a
strategy that never invests) are NaN, and aggregation uses nan-aware medians.
"""

import numpy as np
import pandas as pd


# Standard deviations below this are treated as exactly zero. A constant
# return series has a floating-point std of ~1e-19, not 0, which would
# otherwise produce absurd Sharpe ratios and moments.
STD_EPS = 1e-12


def _std(r):
    s = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    return s if s > STD_EPS else 0.0


def sharpe(r, bars_per_year=252):
    s = _std(r)
    return float(np.mean(r) / s * np.sqrt(bars_per_year)) if s > 0 else 0.0


def sortino(r, bars_per_year=252):
    downside = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)) if len(r) else 0.0
    return float(np.mean(r) / downside * np.sqrt(bars_per_year)) if downside > 0 else np.nan


def max_drawdown(r):
    equity = np.concatenate(([1.0], np.cumprod(1.0 + np.asarray(r, dtype=np.float64))))
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def holding_runs(exposure):
    """(number of invested runs, number of invested bars) of one exposure path."""
    invested = np.asarray(exposure) != 0
    if not invested.any():
        return 0, 0
    starts = invested & ~np.concatenate(([False], invested[:-1]))
    return int(starts.sum()), int(invested.sum())


def return_metrics(r, bars_per_year=252):
    """Metrics that need only the net return series."""
    r = np.asarray(r, dtype=np.float64)
    n = len(r)
    if n == 0:
        return {}
    total = float(np.prod(1.0 + r) - 1.0)
    cagr = float((1.0 + total) ** (bars_per_year / n) - 1.0) if total > -1 else -1.0
    mdd = max_drawdown(r)
    s = _std(r)
    centered = r - r.mean()
    m2 = np.mean(centered ** 2) if s > 0 else 0.0
    skew = float(np.mean(centered ** 3) / m2 ** 1.5) if m2 > 0 else 0.0
    kurt = float(np.mean(centered ** 4) / m2 ** 2) if m2 > 0 else 3.0   # normal fallback
    return {
        "n_days": n,
        "total_return": total,
        "cagr": cagr,
        "ann_vol": s * np.sqrt(bars_per_year),
        "sharpe": sharpe(r, bars_per_year),
        "sharpe_pp": float(r.mean() / s) if s > 0 else 0.0,
        "sortino": sortino(r, bars_per_year),
        "max_drawdown": mdd,
        "calmar": cagr / mdd if mdd > 0 else np.nan,
        "skew": skew,
        "kurtosis": kurt,
    }


def portfolio_metrics(port, ticker_frames, bars_per_year=252):
    """Full metric set for one (seed, fold) evaluation.

    port          : DataFrame from harness.backtest.portfolio()
    ticker_frames : {ticker: DataFrame from harness.backtest.to_frame()}, used
                    for the per-ticker trade statistics (holding period, hit rate)
    """
    m = return_metrics(port["net"].to_numpy(), bars_per_year)
    m["turnover"] = float(port["turnover"].mean() * bars_per_year)
    m["exposure"] = float(port["exposure"].abs().mean())

    runs = bars = wins = invested_days = 0
    for f in ticker_frames.values():
        e = f["exposure"].to_numpy()
        n_runs, n_bars = holding_runs(e)
        runs += n_runs
        bars += n_bars
        on = e != 0
        invested_days += int(on.sum())
        wins += int((f["net"].to_numpy()[on] > 0).sum())
    m["avg_holding"] = bars / runs if runs else np.nan
    m["hit_rate"] = wins / invested_days if invested_days else np.nan
    return m


def aggregate(df, metrics=("sharpe", "cagr", "total_return", "max_drawdown", "sortino",
                           "calmar", "turnover", "avg_holding", "hit_rate", "exposure")):
    """Median and IQR of each metric over the rows of `df` (e.g. seeds x folds)."""
    out = {}
    for m in metrics:
        if m not in df:
            continue
        x = df[m].to_numpy(dtype=np.float64)
        out[f"{m}_median"] = float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan
        q75, q25 = (np.nanpercentile(x, [75, 25]) if np.isfinite(x).any() else (np.nan, np.nan))
        out[f"{m}_iqr"] = float(q75 - q25)
    return out
