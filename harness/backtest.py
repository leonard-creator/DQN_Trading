"""Turn an exposure series into net daily returns. ONE cost model for everyone.

The agent and every baseline go through `backtest()`, so they are charged
identical costs and measured on identical dates (PROTOCOL §6-§7).

Timing (see harness/splits.py): exposure[t] is chosen at the close of bar t
and held until the close of bar t+1. For a return date i:

    gross[i]    = exposure[i-1] * (close[i] / close[i-1] - 1)
    turnover[i] = |exposure[i-1] - exposure[i-2]|      (the trade made at bar i-1)
    cost[i]     = rate     * turnover[i]                 proportional part
                + fee_frac * 1[turnover[i] > 0]          fixed fee per transaction
                + hold     * |exposure[i-1]|             holding cost (e.g. TER) per bar
    net[i]      = gross[i] - cost[i]

Every block starts FLAT: the exposure before the first decision is 0, so
entering a position on the first day is charged. The position is not
liquidated at the end of a block (it is marked to market), which treats all
strategies the same way.

Execution lag (PROTOCOL Part II §V7 LC8, §V10.1): with lag = L, the exposure
decided at the close of bar t is EXECUTED at the close of bar t+L, i.e. held
from t+L to t+L+1. The first L return days of a block are flat. lag = 0 is the
v1 convention; lag = 1 ("decide after today's close, trade at tomorrow's
close") is the realistic one.

Cost models
-----------
* PROTOCOL cost levels (H1): rate = (c + half_spread) / 10,000 per unit of
  exposure traded; no fixed fee, no holding cost. A plain float passed as the
  cost is exactly this, so the M1/M2 numbers are unchanged.
* Named cost SCENARIOS (config/cost_scenarios.yaml), e.g. "neo_broker": a fixed
  EUR fee for EVERY transaction (each buy and each sell; a round trip pays it
  twice), a spread, and a TER on held exposure. A fixed fee is a fraction of
  the capital behind the position: fee_frac = fee_eur / (capital_eur / N),
  where N = number of tickers sharing the capital in the portfolio. Scenarios
  are secondary results; they never replace the H1 cost level.
"""

import copy
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml


def cost_rate(c_bps, half_spread_bps):
    """Proportional cost per unit of |change in exposure|."""
    return (float(c_bps) + float(half_spread_bps)) / 1e4


@dataclass(frozen=True)
class CostModel:
    """Costs of one position (one ticker sleeve), all as fractions of its capital.

    rate     : proportional cost per unit of exposure traded
    fee_frac : fixed cost per transaction (fee_eur / capital behind the position)
    hold     : cost per bar per unit of exposure held (TER / bars_per_year)
    """
    rate: float = 0.0
    fee_frac: float = 0.0
    hold: float = 0.0


def as_cost_model(cost):
    """Accept a CostModel or a plain proportional rate (the PROTOCOL levels)."""
    return cost if isinstance(cost, CostModel) else CostModel(rate=float(cost))


# ---------------------------------------------------------------------------
# named scenarios
# ---------------------------------------------------------------------------
SCENARIO_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "config", "cost_scenarios.yaml")


def load_scenarios(path=SCENARIO_FILE):
    """All named scenarios, with `base:` inheritance resolved."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)["scenarios"]

    def resolve(name, depth=0):
        if depth > 5:
            raise RecursionError(name)
        s = copy.deepcopy(raw[name])
        parent = s.pop("base", None)
        if parent is None:
            return s
        merged = resolve(parent, depth + 1)
        merged.update(s)
        return merged

    return {name: resolve(name) for name in raw}


def scenario_cost(scenario, ticker, n_positions, bars_per_year=252):
    """CostModel of one ticker sleeve under a named scenario dict.

    n_positions : number of tickers sharing the scenario's capital (portfolio
                  size), so each sleeve holds capital / n_positions.
    """
    sleeve = float(scenario["capital_eur"]) / max(1, int(n_positions))
    ter_bps = scenario.get("ter_bps", {}).get(ticker, 0.0)
    return CostModel(rate=cost_rate(scenario.get("c_bps", 0.0), scenario.get("half_spread_bps", 0.0)),
                     fee_frac=float(scenario.get("fee_eur", 0.0)) / sleeve,
                     hold=float(ter_bps) / 1e4 / bars_per_year)


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------
def backtest(close, exposure, val_positions, cost, lag=0):
    """Net daily returns of one ticker over one evaluation block.

    close         : 1-D array of closing prices (full history of the ticker)
    exposure      : 1-D array of the same length; exposure[t] decided at bar t.
                    Only the decision bars val_positions-1 are read.
    val_positions : integer RETURN positions of the block (harness.splits.block_positions)
    cost          : CostModel, or a float = proportional rate (see cost_rate)
    lag           : execution lag in bars (0 = v1 convention, 1 = next close)

    Returns a dict of arrays aligned with val_positions:
    exposure (held during the return day), gross, turnover, trades (0/1), cost, net.
    """
    cm = as_cost_model(cost)
    close = np.asarray(close, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    pos = np.asarray(val_positions, dtype=np.int64)
    if len(pos) == 0:
        empty = np.zeros(0)
        return {"exposure": empty, "gross": empty, "turnover": empty, "trades": empty,
                "cost": empty, "net": empty}
    if pos[0] < 1:
        raise ValueError("an evaluation block needs at least one bar before it (the first decision bar)")
    if not np.all(np.diff(pos) == 1):
        raise ValueError("val_positions must be contiguous")

    held = exposure[pos - 1]                                  # decided at the previous bar
    if not np.all(np.isfinite(held)):
        raise ValueError("exposure is NaN/inf on a decision bar of the block")
    if lag:
        # executed L closes later; the block's first L days are flat
        L = int(lag)
        held = np.concatenate((np.zeros(min(L, len(held))), held[:max(0, len(held) - L)]))
    prev = np.concatenate(([0.0], held[:-1]))                 # start flat
    asset_ret = close[pos] / close[pos - 1] - 1.0
    gross = held * asset_ret
    turnover = np.abs(held - prev)
    trades = (turnover > 1e-12).astype(np.float64)            # one transaction per change of exposure
    cost = cm.rate * turnover + cm.fee_frac * trades + cm.hold * np.abs(held)
    return {"exposure": held, "gross": gross, "turnover": turnover, "trades": trades,
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
