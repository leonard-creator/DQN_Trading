"""Long-history pretraining (PROTOCOL Part II §V6.4, Step 4).

Kenneth French daily data downloaded by the owner (data/longhistory/french/, see its
MANIFEST.json): the 5 value-weighted US industry portfolios plus the market (Mkt-RF + RF),
turned into price series that compound from 100. Only return-derived inputs exist: there
is no volume (masked as in Q4) and no VIX before 1990 (vix_avail = 0, rl/features_m4.py).

The window ends 2007-06-29, before the development period. Later rows in the files
(they run to 2026) are cut on load, so neither a validation block nor the test period can
be reached from here. Inside the window the run trains on everything after a 504-bar
warm-up and up to P bars before its last 63 bars; checkpoints are selected on the last 252
training bars, as for every other run.

    run(config, seeds)  pretrains the configuration once per seed into agent.pretrain.dir.
                        One checkpoint per seed serves all five folds. The fine-tuning on
                        the ETF folds starts from it (rl/exogenous.train_exogenous).
"""

import json
import os

import numpy as np
import pandas as pd

from harness.config import code_hash, config_hash, load_config, repo_path
from harness.splits import Fold
from harness.synthetic import ohlcv, repoint

DIR = repo_path("data", "longhistory", "french")
INDUSTRIES = ["Cnsmr", "Manuf", "HiTec", "Hlth", "Other"]
WARMUP, LAST_BARS = 504, 63


def french_table(path, after=None):
    """The first table (header line starting with ',') after the line containing `after`, in
    decimal returns; -99.99 / -999 (missing) become NaN."""
    lines = open(path).read().splitlines()
    i = next(k for k, ln in enumerate(lines) if after in ln) if after else 0
    i = next(k for k in range(i, len(lines)) if lines[k].startswith(","))
    cols, rows = [c.strip() for c in lines[i].split(",")[1:]], []
    for ln in lines[i + 1:]:
        f = [x.strip() for x in ln.split(",")]
        if len(f) != len(cols) + 1 or not (f[0].isdigit() and len(f[0]) == 8):
            break                                          # end of the table
        rows.append(f)
    df = pd.DataFrame([r[1:] for r in rows], columns=cols, dtype=float,
                      index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"))
    return df.where(df > -99) / 100.0


def load(window, folder=DIR):
    """{ticker: OHLCV frame} of the 6 long-history series inside window = (start, end), plus an
    all-NaN ^VIX (it did not exist)."""
    start, end = (pd.Timestamp(x) for x in window)
    ind = french_table(os.path.join(folder, "5_Industry_Portfolios_Daily.csv"), "Average Value Weighted Returns -- Daily")
    ff = french_table(os.path.join(folder, "F-F_Research_Data_Factors_daily.csv"))
    ret = ind[INDUSTRIES].join((ff["Mkt-RF"] + ff["RF"]).rename("Mkt"), how="inner")
    ret = ret[(ret.index >= start) & (ret.index <= end)].dropna()
    prices = {f"FF_{c}": ohlcv(100 * np.cumprod(1 + ret[c].to_numpy()), ret.index) for c in ret.columns}
    prices["^VIX"] = ohlcv(np.full(len(ret), np.nan), ret.index)
    return prices


def run(config, seeds=range(10), workers=None, folder=DIR):
    """Pretrain `config` on the long history once per seed; weights land in agent.pretrain.dir."""
    from rl.policy import make_jobs, train_jobs                 # late: rl imports the harness
    base = load_config(config) if isinstance(config, str) else config
    pre = base["agent"]["pretrain"]
    out = repo_path(pre["dir"])
    if os.path.exists(os.path.join(out, "agent")):
        raise FileExistsError(f"{out} already holds a pretraining run; move it away to pretrain again")
    if pd.Timestamp(pre["window"][1]) >= pd.Timestamp(base["splits"]["dev_start"]):
        raise ValueError("the pretraining window must end before the development period")
    prices = load(pre["window"], folder)
    tickers = [t for t in prices if t != "^VIX"]
    cfg = repoint(base, tickers, "pre")
    cfg["agent"].pop("pretrain")                                 # the pretraining itself starts from scratch
    cfg["agent"]["train"]["transitions"] = int(pre["transitions"])
    idx = prices[tickers[0]].index
    fold = Fold("PRE", idx[WARMUP], idx[-LAST_BARS], idx[-1], idx[-LAST_BARS])
    os.makedirs(out, exist_ok=True)
    runtime = dict(cfg["agent"].get("runtime", {}), **({"workers": workers} if workers else {}))
    done = train_jobs(make_jobs(cfg, prices, [fold], list(seeds), tickers, out), runtime)
    with open(os.path.join(out, "pretrain.json"), "w") as fh:
        json.dump({"config": base["name"], "config_hash": config_hash(base), "code_hash": code_hash(),
                   "window": [str(idx[0].date()), str(idx[-1].date())], "tickers": tickers, "bars": len(idx),
                   "seeds": [int(s) for s in seeds], "jobs": [r["info"] for r in done]}, fh, indent=2, default=float)
    return done
