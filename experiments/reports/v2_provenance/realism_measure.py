"""Provenance of RESULTS.md, "v2 realism check", table 1 (2026-10-15): variance ratios and autocorrelations of the
26 training ETFs (development period only; the test period is locked by the loader).
    PYTHONPATH=. python experiments/reports/v2_provenance/realism_measure.py
"""
import warnings
import numpy as np, pandas as pd
from harness.config import load_config
from harness.data import load_prices, ticker_sets
warnings.filterwarnings("ignore")
cfg = load_config("config/v2/V1b.yaml")
train = ticker_sets(cfg)["train"]
px = load_prices(train, cfg)
R = pd.concat({t: np.log(px[t]["Close"]).diff() for t in train}, axis=1)
R = R[(R.index >= "2007-07-02")]


def vr(r, q):
    r = r.dropna().to_numpy(); r = r - r.mean()
    s = np.convolve(r, np.ones(q), "valid")
    return s.var() / (q * r.var())


def acf(r, k):
    r = r.dropna(); return r.autocorr(k)


def table(R, label):
    rows = {q: np.median([vr(R[t], q) for t in R]) for q in (2, 5, 10, 20, 60)}
    ew = R.fillna(0).mean(axis=1)
    rows_ew = {q: vr(ew, q) for q in (2, 5, 10, 20, 60)}
    ac = {k: np.median([acf(R[t], k) for t in R]) for k in (1, 2, 3, 5, 10)}
    n = len(R)
    print(f"{label:38s} n={n:5d} | VR median ticker " + " ".join(f"{q}:{v:.2f}" for q, v in rows.items())
          + " | EW-26 " + " ".join(f"{q}:{v:.2f}" for q, v in rows_ew.items())
          + " | ACF " + " ".join(f"{k}:{v:+.3f}" for k, v in ac.items()) + f" (SE ~{1/np.sqrt(n):.3f})")


table(R, "all 2007-07 -> 2023-09")
crisis = ((R.index >= "2008-09-01") & (R.index <= "2009-06-30")) | ((R.index >= "2020-02-15") & (R.index <= "2020-06-30"))
table(R[~crisis], "without 2008-09 and 2020 H1 crises")
for a, b in (("2007-07-02", "2012-12-31"), ("2013-01-01", "2019-12-31"), ("2020-01-01", "2023-09-29")):
    table(R[(R.index >= a) & (R.index <= b)], f"{a[:4]}-{b[:4]}")
groups = {"US/intl equity": [t for t in train if t not in ("TLT", "LQD", "TIP", "GLD", "USO", "DBC", "IYR")],
          "bonds": ["TLT", "LQD", "TIP"], "commodities": ["GLD", "USO", "DBC"]}
for g, ts in groups.items():
    table(R[[t for t in ts if t in R]], g)
# idiosyncratic part: returns minus the cross-sectional mean (market-wide vs ticker-specific reversal)
table(R.sub(R.mean(axis=1), axis=0), "ticker minus EW-26 (idiosyncratic)")
