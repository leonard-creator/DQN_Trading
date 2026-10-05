"""Turn an exposure series into net daily returns. ONE cost model for everyone.

The agent and every baseline go through `backtest()`, so they are charged
identical costs and measured on identical dates (PROTOCOL §6-§7).

Timing (see harness/splits.py): exposure[t] is chosen at the close of bar t
and held until the close of bar t+1. For a return date i:

    gross[i]    = exposure[i-1] * (close[i] / close[i-1] - 1)
    turnover[i] = |exposure[i-1] - exposure[i-2]|      (the trade made at bar i-1)
    cost[i]     = cost_rate * turnover[i]
    net[i]      = gross[i] - cost[i]

Every block starts FLAT: the exposure before the first decision is 0, so
entering a position on the first day is charged. The position is not
liquidated at the end of a block (it is marked to market), which treats all
strategies the same way.

Cost rate = (c + half_spread) / 10,000 per unit of exposure traded, where c
and the half-spread are in basis points (PROTOCOL §7).
"""

import numpy as np
import pandas as pd


def cost_rate(c_bps, half_spread_bps):
    """Proportional cost per unit of |change in exposure|."""
    return (float(c_bps) + float(half_spread_bps)) / 1e4


def backtest(close, exposure, val_positions, rate):
    """Net daily returns of one ticker over one evaluation block.

    close         : 1-D array of closing prices (full history of the ticker)
    exposure      : 1-D array of the same length; exposure[t] decided at bar t.
                    Only the decision bars val_positions-1 are read.
    val_positions : integer RETURN positions of the block (harness.splits.block_positions)
    rate          : cost per unit of exposure traded (see cost_rate)

    Returns a dict of arrays aligned with val_positions:
    exposure (held during the return day), gross, turnover, cost, net.
    """
    close = np.asarray(close, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    pos = np.asarray(val_positions, dtype=np.int64)
    if len(pos) == 0:
        empty = np.zeros(0)
        return {"exposure": empty, "gross": empty, "turnover": empty, "cost": empty, "net": empty}
    if pos[0] < 1:
        raise ValueError("an evaluation block needs at least one bar before it (the first decision bar)")
    if not np.all(np.diff(pos) == 1):
        raise ValueError("val_positions must be contiguous")

    held = exposure[pos - 1]                                  # decided at the previous bar
    if not np.all(np.isfinite(held)):
        raise ValueError("exposure is NaN/inf on a decision bar of the block")
    prev = np.concatenate(([0.0], held[:-1]))                 # start flat
    asset_ret = close[pos] / close[pos - 1] - 1.0
    gross = held * asset_ret
    turnover = np.abs(held - prev)
    cost = rate * turnover
    return {"exposure": held, "gross": gross, "turnover": turnover,
            "cost": cost, "net": gross - cost}


def to_frame(result, dates):
    """Wrap a backtest() result in a DataFrame indexed by return date."""
    return pd.DataFrame(result, index=pd.DatetimeIndex(dates))


def portfolio(frames):
    """Equal-weight portfolio of per-ticker backtests.

    Each ticker is a fixed 1/N sleeve of capital. On a date where a ticker has
    no bar (exchange holiday) its sleeve earns 0 that day. Its next return
    covers the gap, so no return is lost. Cross-sleeve rebalancing costs are
    ignored, identically for every strategy.

    frames : {ticker: DataFrame from to_frame()}
    Returns a DataFrame with the portfolio's net/gross/cost/exposure/turnover.
    """
    if not frames:
        raise ValueError("portfolio() needs at least one ticker")
    cols = {}
    for col in ("net", "gross", "cost", "exposure", "turnover"):
        wide = pd.concat({t: f[col] for t, f in frames.items()}, axis=1).sort_index()
        cols[col] = wide.fillna(0.0).mean(axis=1)            # fixed 1/N weights
    return pd.DataFrame(cols)
