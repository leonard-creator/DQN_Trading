"""Markdown report helpers shared by the report scripts (scripts/analyze_*.py, rescore_*.py).

Tables read logged ExperimentResults (nothing here trains or logs a trial). A
performance cell is the median (IQR) over the (seed, fold) runs of one
evaluation set at one cost level; percentages for the PCT metrics.
"""

import json
import os

import numpy as np
import pandas as pd

from harness import metrics as mt
from harness import stats as st
from harness import trials as tr
from harness.config import deep_merge, repo_path
from harness.experiment import BaselinePolicy, load_result, run_experiment

PCT = {"cagr", "max_drawdown", "exposure", "hit_rate"}
COLS = [("sharpe", "Sharpe"), ("cagr", "CAGR"), ("max_drawdown", "MaxDD"), ("turnover", "Turnover/yr"),
        ("exposure", "Exposure"), ("avg_holding", "Hold (bars)")]
BASES = {"buy_and_hold": "baseline_buy_and_hold", "momentum": "baseline_momentum",
         "macd": "baseline_macd", "random": "baseline_random"}


def latest_dir(name):
    """Absolute output directory of the latest logged trial called `name`, or None."""
    t = tr.load_trials()
    rows = t[t["name"] == name]
    return None if rows.empty else repo_path(rows.iloc[-1]["output_dir"])


def latest(name):
    """ExperimentResult of the latest logged trial called `name`, or None if it has not run yet."""
    d = latest_dir(name)
    if d is None:
        print(f"[report] {name} not logged yet; skipped")
        return None
    return load_result(d)


def fmt(v, digits=2, pct=False):
    """One number: '–' if missing, else fixed-point or a whole percentage."""
    if v is None or not np.isfinite(v):
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.{digits}f}"


def cell(v, iqr=None, pct=False, digits=2):
    """'median (IQR)'; 'n/a' if the median is missing, no bracket if the IQR is."""
    if v is None or not np.isfinite(v):
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.{digits}f}")
    return f(v) + (f" ({f(iqr)})" if iqr is not None and np.isfinite(iqr) else "")


def perf_table(results, es, cost, cols=COLS):
    """One row per strategy: median (IQR) of each metric in `cols`."""
    L = ["| Strategy | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, res in results.items():
        runs = res.runs(es, cost)
        if runs.empty:
            continue
        a = mt.aggregate(runs)
        L.append(f"| {name} | " + " | ".join(cell(a[f'{k}_median'], a[f'{k}_iqr'], k in PCT) for k, _ in cols) + " |")
    return "\n".join(L)


def fold_table(results, es, cost):
    """Median Sharpe per walk-forward fold, one row per strategy."""
    folds, L = None, []
    for name, res in results.items():
        runs = res.runs(es, cost)
        if runs.empty:
            continue
        med = runs.groupby("fold")["sharpe"].median()
        if folds is None:
            folds = list(med.index)
            L = ["| Strategy | " + " | ".join(folds) + " |", "|---" * (len(folds) + 1) + "|"]
        L.append(f"| {name} | " + " | ".join(f"{med[f]:.2f}" for f in folds) + " |")
    return "\n".join(L)


def cost_table(results, es, levels):
    """Median Sharpe per cost level (bp) or named cost scenario."""
    L = ["| Strategy | " + " | ".join(f"{c} bp" if not isinstance(c, str) else c for c in levels) + " |",
         "|---" * (len(levels) + 1) + "|"]
    for name, res in results.items():
        cells = []
        for c in levels:
            r = res.runs(es, c)
            cells.append(f"{r['sharpe'].median():.2f}" if not r.empty else "n/a")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def paired(a, b, es, cost, metric="sharpe", n_perm=10000):
    """One-sided permutation test of a > b over the (seed, fold) pairs both runs share."""
    m = a.runs(es, cost)[["seed", "fold", metric]].merge(
        b.runs(es, cost)[["seed", "fold", metric]], on=["seed", "fold"], suffixes=("_a", "_b"))
    return st.permutation_test(m[f"{metric}_a"] - m[f"{metric}_b"], n_perm)


def deploy_baselines(cfg_like, bases=BASES):
    """The baselines on the deploy2 set (not in their logged runs), incl. the neo-broker scenario."""
    out = {}
    for k, name in bases.items():
        res = latest(name)
        if res is None:
            continue
        cfg = deep_merge(res.cfg, {"extra_ticker_sets": cfg_like["extra_ticker_sets"]})
        out[f"*{k}*"] = run_experiment(cfg, policy=BaselinePolicy(cfg["policy"], cfg.get("policy_params")),
                                       eval_sets=["deploy2"], cost_scenarios=["neo_broker"],
                                       log=False, verbose=False)
    return out


def stability_rows(res):
    """Per training job: best-checkpoint position, late drift of the inner Sharpe, Q range, minutes."""
    rows = []
    base = os.path.join(res.output_dir, "agent")
    for d in sorted(os.listdir(base)):
        info = json.load(open(os.path.join(base, d, "info.json")))
        c = pd.read_csv(os.path.join(base, d, "curve.csv"))
        c["p"] = c["update"] / max(1, info["updates"])
        mid = c.loc[(c.p > 0.25) & (c.p <= 0.5), "inner_sharpe"].mean()
        late = c.loc[c.p > 0.75, "inner_sharpe"].mean()
        rows.append({"best_at": info["best_update"] / max(1, info["updates"]), "drift": late - mid,
                     "q_min": c["mean_q"].min(), "q_max": c["mean_q"].max(),
                     "secs": info.get("seconds", np.nan)})
    return pd.DataFrame(rows)
