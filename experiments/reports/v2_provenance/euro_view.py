"""Provenance of RESULTS.md, "In euros: one investor with EUR 10,000, 2014 -> 2023-09" (2026-10-09).

Rebuilds the euro tables from the STORED decisions (exposures.npz) of the real-data trials H1, H2, H1nb, H2nb (no
retraining). Buy-and-hold buys once in January 2014; the agents are retrained every 2 years (the walk-forward folds)
and each fold starts flat. Main view: EUR 10,000 in SPY + EFA at the neo-broker (EUR 1 per order, 3 bp half-spread);
second view: the 26-ETF research portfolio at 10 bp + 1 bp. Representative agent = the seed whose full-period
Sharpe on the 26 ETFs is the median of its 10 seeds.
    PYTHONPATH=. python experiments/reports/v2_provenance/euro_view.py          # the tables
    PYTHONPATH=. python experiments/reports/v2_provenance/euro_view.py --scan   # all 40 agents (contrast picks)
"""

import sys
import warnings

import numpy as np
import pandas as pd

from harness import backtest as bt
from harness.config import load_config
from harness.data import load_prices, ticker_sets
from harness.splits import block_positions, folds_from_config

warnings.filterwarnings("ignore")
RUNS = {"H1": "20261008-191920_v2_H1_d1f01deba66c", "H2": "20261008-195650_v2_H2_38d436f4e1f2",
        "H1nb": "20261008-205147_v2_H1nb_b185686c88a6", "H2nb": "20261008-212913_v2_H2nb_0216a68a27c4"}
CFG = load_config("config/v2/H1.yaml")
FOLDS = folds_from_config(CFG)
NEO = bt.load_scenarios()["neo_broker"]
DEPLOY = ["SPY", "EFA"]


def agent(run, seed, tickers, px, cost_of):
    """Daily net return, exposure and orders of the equal-weight portfolio; run None = buy-and-hold bought once."""
    n, e, o = {}, {}, {}
    for t in tickers:
        close = px[t]["Close"].to_numpy()
        blocks = [(None, FOLDS[0].val_start, FOLDS[-1].val_end)] if run is None else [(f, f.val_start, f.val_end) for f in FOLDS]
        parts = []
        for f, a, b in blocks:
            pos = block_positions(px[t].index, a, b)
            x = np.ones(len(close)) if f is None else \
                np.load(f"experiments/runs/{run}/agent/{f.name}_s{seed}/exposures.npz")[f"selected|{f.name}|{t}"]
            r = bt.backtest(close[:len(x)], x, pos, cost_of(t))
            parts.append(pd.DataFrame({k: r[k] for k in ("net", "exposure", "trades")}, index=px[t].index[pos]))
        df = pd.concat(parts)
        n[t], e[t], o[t] = df["net"], df["exposure"], df["trades"]
    frame = lambda d: pd.concat(d, axis=1).fillna(0.0)                                  # noqa: E731
    return frame(n).mean(axis=1), frame(e).mean(axis=1), frame(o).sum(axis=1)


def per_year(net, orders, start=10_000.0):
    v = start * (1 + net).cumprod()
    yr = v.groupby(v.index.year).last()
    return (yr - yr.shift(1).fillna(start)).round(0), orders.groupby(orders.index.year).sum(), v


def median_seed(run):
    r = pd.read_csv(f"experiments/runs/{run}/returns_train_c10.csv", index_col=0)
    sh = (r.mean() / r.std() * np.sqrt(252)).sort_values()
    return int(sh.index[len(sh) // 2])


def main(scan=False):
    px = load_prices(DEPLOY, CFG)
    neo = lambda t: bt.scenario_cost(NEO, t, 2)                                           # noqa: E731
    if scan:                                                       # every agent: end value, orders, risk, bear years
        rows = []
        for k, run in [("B&H", None)] + list(RUNS.items()):
            for s in ([None] if run is None else range(10)):
                net, expo, o = agent(run, s, DEPLOY, px, neo)
                prof, _, v = per_year(net, o)
                rows.append({"agent": k, "seed": s, "end": round(v.iloc[-1]), "orders": int(o.sum()),
                             "maxdd": round((1 - v / v.cummax()).max(), 3), "expo": round(expo.mean(), 3),
                             **{f"r{y}": round(prof[y] / (v.groupby(v.index.year).last().shift(1).fillna(10_000)[y]), 3)
                                for y in (2018, 2020, 2022)}})
        print(pd.DataFrame(rows).to_string(index=False))
        return
    s1, s2 = median_seed(RUNS["H1"]), median_seed(RUNS["H2"])
    picks = {"B&H": (None, None), f"H1 c00 s{s1}": (RUNS["H1"], s1), f"H2 c25 s{s2}": (RUNS["H2"], s2),
             "low-risk c00 s6": (RUNS["H1"], 6), "high-risk c25nb s7": (RUNS["H2nb"], 7)}
    prof, orders, summ = {}, {}, {}
    for k, (run, s) in picks.items():
        net, expo, o = agent(run, s, DEPLOY, px, neo)
        prof[k], orders[k], v = per_year(net, o)
        summ[k] = {"end": round(v.iloc[-1]), "orders": int(o.sum()), "maxdd%": round(100 * (1 - v / v.cummax()).max(), 1),
                   "expo%": round(100 * expo.mean())}
    print(pd.DataFrame(prof).to_string(), pd.DataFrame(orders).to_string(), pd.DataFrame(summ).to_string(), sep="\n")
    train = ticker_sets(CFG)["train"]
    px26 = load_prices(train, CFG)
    print("26 ETFs @ 10 bp (end value, orders):",
          {k: (round(10_000 * (1 + agent(run, s, train, px26, lambda t: bt.cost_rate(10, 1))[0]).prod()),
               int(agent(run, s, train, px26, lambda t: bt.cost_rate(10, 1))[2].sum())) for k, (run, s) in picks.items()})


if __name__ == "__main__":
    main(scan="--scan" in sys.argv)
