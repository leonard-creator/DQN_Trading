"""Milestone M2 report: legacy vs new agent vs ablations on ^GDAXI.

    python scripts/analyze_m2.py        # -> experiments/reports/M2_agent.md

Reads the latest logged trial of each M2 configuration and of each baseline
from experiments/trials.csv. Nothing is retrained. Sections:

  1. outer-validation performance (median/IQR over 10 seeds x 5 folds), per fold, cost sweep
  2. stability, the M2 go criterion "stable across >= 10 seeds, no late collapse":
       - where in training the best inner-validation checkpoint occurs
       - late collapse = last inner-validation Sharpe at least 0.5 below the run's best
       - median inner-validation Sharpe along training (all runs)
       - mean-Q range (divergence check)
       - outer-validation Sharpe of the SELECTED vs the LAST checkpoint (diagnostic only)
       - seed dispersion within a fold
  3. statistics: permutation tests vs the 4 baselines (Holm), deflated Sharpe, PBO over the 5 configs
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from harness import backtest as bt                                       # noqa: E402
from harness import metrics as mt                                        # noqa: E402
from harness import trials as tr                                         # noqa: E402
from harness.config import repo_path                                     # noqa: E402
from harness.data import load_prices                                     # noqa: E402
from harness.experiment import (compare_to_baselines, deflated_sharpe_of,  # noqa: E402
                                load_result, pbo_over)
from harness.splits import block_positions, folds_from_config            # noqa: E402

AGENTS = ["legacy", "m2_dqn_base", "m2_dqn_dueling", "m2_dqn_nstep", "m2_dqn_per"]
BASES = {"buy_and_hold": "baseline_buy_and_hold", "momentum": "baseline_momentum",
         "macd": "baseline_macd", "random": "baseline_random"}
ES = "single"
COLLAPSE = 0.5          # Sharpe drop from the run's best inner value that counts as a collapse


def latest(name):
    """Latest logged trial of a configuration, or None if it has not run yet."""
    t = tr.load_trials()
    rows = t[t["name"] == name]
    if rows.empty:
        print(f"[analyze_m2] no logged trial named {name} yet; skipped")
        return None
    return load_result(rows.iloc[-1]["output_dir"])


def runs_of(res):
    base = os.path.join(res.output_dir, "agent")
    return sorted(os.path.join(base, d) for d in os.listdir(base))


def cell(x, iqr=None, pct=False, digits=2):
    if x is None or not np.isfinite(x):
        return "n/a"
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.{digits}f}")
    return f(x) + (f" ({f(iqr)})" if iqr is not None and np.isfinite(iqr) else "")


def perf_table(results, cost):
    cols = [("sharpe", "Sharpe", False), ("cagr", "CAGR", True), ("max_drawdown", "MaxDD", True),
            ("turnover", "Turnover/yr", False), ("exposure", "Exposure", True),
            ("avg_holding", "Hold (bars)", False), ("hit_rate", "Hit rate", True)]
    lines = ["| Strategy | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, res in results.items():
        a = mt.aggregate(res.runs(ES, cost))
        lines.append(f"| {name} | " + " | ".join(cell(a[f'{k}_median'], a[f'{k}_iqr'], pct) for k, _, pct in cols) + " |")
    return "\n".join(lines)


def fold_table(results, cost):
    folds = sorted(next(iter(results.values())).runs(ES, cost)["fold"].unique())
    lines = ["| Strategy | " + " | ".join(folds) + " |", "|---" * (len(folds) + 1) + "|"]
    for name, res in results.items():
        med = res.runs(ES, cost).groupby("fold")["sharpe"].median()
        lines.append(f"| {name} | " + " | ".join(f"{med[f]:.2f}" for f in folds) + " |")
    return "\n".join(lines)


def cost_table(results, levels):
    lines = ["| Strategy | " + " | ".join(f"{c} bp" for c in levels) + " |", "|---" * (len(levels) + 1) + "|"]
    for name, res in results.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{res.runs(ES, c)['sharpe'].median():.2f}" for c in levels) + " |")
    return "\n".join(lines)


def stability(res, prices, cfg):
    """Per-run stability numbers from curve.csv / info.json / exposures.npz."""
    folds = {f.name: f for f in folds_from_config(cfg)}
    rate = bt.cost_rate(cfg["evaluation"]["costs"]["primary_bps"], cfg["evaluation"]["costs"]["half_spread_bps"])
    rows, curves = [], []
    for d in runs_of(res):
        info = json.load(open(os.path.join(d, "info.json")))
        curve = pd.read_csv(os.path.join(d, "curve.csv"))
        curve["progress"] = curve["update"] / max(1, info["updates"])
        curves.append(curve[["progress", "inner_sharpe"]])
        expo = np.load(os.path.join(d, "exposures.npz"))
        out = {"job": info["job"], "seed": info["seed"],
               "best_at": info["best_update"] / max(1, info["updates"]),
               "best_inner": info["best_inner_sharpe"], "last_inner": info["last_inner_sharpe"],
               "q_min": curve["mean_q"].min(), "q_max": curve["mean_q"].max()}
        for tag in ("selected", "last"):
            sharpes = []
            for key in expo.files:
                t_, fold, ticker = key.split("|")
                if t_ != tag:
                    continue
                df = prices[ticker][prices[ticker].index <= folds[fold].val_end]
                pos = block_positions(df.index, folds[fold].val_start, folds[fold].val_end)
                net = bt.backtest(df["Close"].to_numpy(), expo[key][:len(df)], pos, rate)["net"]
                sharpes.append(mt.sharpe(net))
                out["fold"] = fold
            out[f"outer_{tag}"] = float(np.mean(sharpes))
        rows.append(out)
    df = pd.DataFrame(rows)
    df["collapse"] = (df["last_inner"] - df["best_inner"]) <= -COLLAPSE
    # Noise-robust drift: mean inner Sharpe in the last quarter of training minus
    # the second quarter. Comparing with the BEST checkpoint overstates collapse,
    # because the max of ~20 one-year Sharpe estimates (SE ~ 1) is biased upward.
    drift = []
    for c in curves:
        mid = c.loc[(c["progress"] > 0.25) & (c["progress"] <= 0.5), "inner_sharpe"].mean()
        late = c.loc[c["progress"] > 0.75, "inner_sharpe"].mean()
        drift.append(late - mid)
    df["drift"] = drift
    allc = pd.concat(curves)
    allc["bin"] = pd.cut(allc["progress"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0001], labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"])
    traj = allc.groupby("bin", observed=False)["inner_sharpe"].median()
    return df, traj


def main():
    agents = {n: r for n in AGENTS if (r := latest(n)) is not None}
    bases = {k: latest(v) for k, v in BASES.items()}
    if not agents:
        raise SystemExit("no M2 agent trial logged yet")
    cfg = next(iter(agents.values())).cfg
    primary = cfg["evaluation"]["costs"]["primary_bps"]
    levels = cfg["evaluation"]["costs"]["sweep_bps"]
    prices = load_prices([cfg["universe"]["single_asset"]], cfg)
    everything = {**agents, **{f"*{k}*": v for k, v in bases.items()}}

    L = ["# M2 — Model fixes on the single asset ^GDAXI (generated by scripts/analyze_m2.py)", "",
         f"Trials: {', '.join(f'`{r.trial_id}`' for r in agents.values())}.",
         "Cells: median (IQR) over 10 seeds x 5 walk-forward folds, outer validation 2014-01 → 2023-09, "
         f"net of {primary} bp + 1 bp half-spread. Baselines in *italics*. Checkpoints were selected on the "
         "inner validation slice only.", "",
         f"## 1. Performance @ {primary} bp", "", perf_table(everything, primary), "",
         f"### Median Sharpe per fold @ {primary} bp", "", fold_table(everything, primary), "",
         "### Cost sweep (median Sharpe)", "", cost_table(everything, levels), ""]

    L += ["## 2. Stability (go criterion)", "",
          f"*Best→last drop ≥ {COLLAPSE}* counts runs whose final inner-validation Sharpe is ≥ {COLLAPSE} below "
          "their best checkpoint. **Read it with care:** each inner score covers one year of data "
          "(standard error ≈ 1 Sharpe unit), so the best of ~20 checkpoints sits well above the true "
          "level even without any collapse. *Late drift* (mean inner Sharpe in the last quarter of "
          "training minus the second quarter, median over runs) is the noise-robust collapse measure. "
          "Outer selected/last = outer-validation Sharpe of the chosen checkpoint vs the final "
          "weights (diagnostic only, never used for selection).", "",
          "| Config | Runs | Best ckpt at (median % of training) | Best→last drop ≥ 0.5 | Inner Sharpe best → last (median) "
          "| Late drift (median; share < −0.5) | Mean-Q range | Outer Sharpe selected / last (median) "
          "| Seed IQR within fold (median) |",
          "|---|---|---|---|---|---|---|---|---|"]
    trajectories = {}
    for name, res in agents.items():
        df, traj = stability(res, prices, res.cfg)
        trajectories[name] = traj
        seed_iqr = res.runs(ES, primary).groupby("fold")["sharpe"].apply(
            lambda s: s.quantile(0.75) - s.quantile(0.25)).median()
        L.append(f"| {name} | {len(df)} | {100 * df['best_at'].median():.0f}% | "
                 f"{int(df['collapse'].sum())}/{len(df)} | {df['best_inner'].median():.2f} → {df['last_inner'].median():.2f} | "
                 f"{df['drift'].median():+.2f}; {int((df['drift'] < -COLLAPSE).sum())}/{len(df)} | "
                 f"{df['q_min'].min():.2f} … {df['q_max'].max():.2f} | "
                 f"{df['outer_selected'].median():.2f} / {df['outer_last'].median():.2f} | {seed_iqr:.2f} |")
    L += ["", "### Median inner-validation Sharpe along training (all 50 runs per config)", "",
          "| Config | " + " | ".join(next(iter(trajectories.values())).index.astype(str)) + " |",
          "|---" * 6 + "|"]
    for name, traj in trajectories.items():
        L.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in traj.values) + " |")

    L += ["", f"## 3. Statistics @ {primary} bp", "",
          f"N_trials (distinct agent configurations in the trial log) = {tr.n_trials()}.", "",
          "| Config | vs buy&hold p (Holm) | vs momentum | vs MACD | vs random | Deflated Sharpe (median over seeds) |",
          "|---|---|---|---|---|---|"]
    for name, res in agents.items():
        comp = compare_to_baselines(res, bases, eval_set=ES).set_index("baseline")
        dsr = deflated_sharpe_of(res, eval_set=ES)
        cells = [f"{comp.loc[b, 'mean_diff']:+.2f}, p={comp.loc[b, 'p_holm']:.3f}" for b in BASES]
        L.append(f"| {name} | " + " | ".join(cells) + f" | {dsr['deflated_sharpe_median']:.3f} |")
    L += ["", "Cells: mean difference in annualised Sharpe (agent − baseline) over (seed, fold) pairs, "
              "one-sided Holm-adjusted p.", ""]
    if len(agents) >= 2:
        pbo = pbo_over(agents, eval_set=ES)
        L += [f"**PBO over the {pbo['n_configs']} M2 configurations** (CSCV, S = "
              f"{cfg['evaluation']['statistics']['pbo_blocks']}, {pbo['n_splits']} splits): "
              f"**{pbo['pbo']:.2f}**; probability that the in-sample winner loses money out of sample: "
              f"{pbo['prob_oos_loss']:.2f}.", ""]
    missing = [n for n in AGENTS if n not in agents]
    if missing:
        L += [f"*Incomplete report: not yet run: {', '.join(missing)}.*", ""]

    out = repo_path("experiments", "reports", "M2_agent.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("report ->", out)


if __name__ == "__main__":
    main()
