"""Milestone M4 report: cross-asset agent, new features, leave-assets-out.

    python scripts/analyze_m4.py        # -> experiments/reports/M4_cross_asset.md

Reads the latest logged trial of each M4 configuration (nothing is retrained)
and the logged baselines. Sections:

  1. training universe (26 ETFs, the H1 evaluation set): performance, folds, cost sweep
  2. leave-assets-out (7 ETFs never used in training): performance and the M4
     GO CRITERION "cross-asset >= single-asset" as a paired permutation test
     over (seed, fold) of each cross-asset config vs m4_single_resid
  3. single asset ^GDAXI (comparison with M3)
  4. 2-ETF deployment set (SPY + EFA) at 10 bp and under the EUR 10k neo-broker scenario
  5. statistics on the training universe: permutation tests vs the 4 baselines
     (Holm), deflated Sharpe ratio (N_trials from the log), PBO over M4 configs
  6. training stability (from the training curves)
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from harness import metrics as mt                                        # noqa: E402
from harness import stats as st                                          # noqa: E402
from harness import trials as tr                                         # noqa: E402
from harness.config import deep_merge, repo_path                         # noqa: E402
from harness.experiment import (BaselinePolicy, compare_to_baselines,    # noqa: E402
                                deflated_sharpe_of, load_result, pbo_over, run_experiment)

CROSS = ["m4_cross_base", "m4_cross_resid", "m4_cross_resid_transformer"]
SINGLE = "m4_single_resid"
BASES = {"buy_and_hold": "baseline_buy_and_hold", "momentum": "baseline_momentum",
         "macd": "baseline_macd", "random": "baseline_random"}
PCT = {"cagr", "max_drawdown", "exposure", "hit_rate"}
COLS = [("sharpe", "Sharpe"), ("cagr", "CAGR"), ("max_drawdown", "MaxDD"), ("turnover", "Turnover/yr"),
        ("exposure", "Exposure"), ("avg_holding", "Hold (bars)")]


def latest(name):
    t = tr.load_trials()
    rows = t[t["name"] == name]
    if rows.empty:
        print(f"[analyze_m4] {name} not logged yet; skipped")
        return None
    return load_result(rows.iloc[-1]["output_dir"])


def cell(v, iqr, pct):
    if not np.isfinite(v):
        return "n/a"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.2f}")
    return f"{f(v)} ({f(iqr)})"


def perf_table(results, es, cost):
    L = ["| Strategy | " + " | ".join(c[1] for c in COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
    for name, res in results.items():
        runs = res.runs(es, cost)
        if runs.empty:
            continue
        a = mt.aggregate(runs)
        L.append(f"| {name} | " + " | ".join(cell(a[f'{k}_median'], a[f'{k}_iqr'], k in PCT) for k, _ in COLS) + " |")
    return "\n".join(L)


def fold_table(results, es, cost):
    folds = None
    L = []
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
    """Permutation test of a > b on (seed, fold) pairs (both agents use seeds 0-9)."""
    m = a.runs(es, cost)[["seed", "fold", metric]].merge(
        b.runs(es, cost)[["seed", "fold", metric]], on=["seed", "fold"], suffixes=("_a", "_b"))
    return st.permutation_test(m[f"{metric}_a"] - m[f"{metric}_b"], n_perm)


def deploy_baselines(cfg_like):
    """Baselines on the deployment set (not in their logged runs), incl. the neo-broker scenario."""
    out = {}
    for k, name in BASES.items():
        res = latest(name)
        if res is None:
            continue
        cfg = deep_merge(res.cfg, {"extra_ticker_sets": cfg_like["extra_ticker_sets"]})
        out[f"*{k}*"] = run_experiment(cfg, policy=BaselinePolicy(cfg["policy"], cfg.get("policy_params")),
                                       eval_sets=["deploy2"], cost_scenarios=["neo_broker"],
                                       log=False, verbose=False)
    return out


def stability_rows(res):
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


def main():
    agents = {n: r for n in CROSS + [SINGLE] if (r := latest(n)) is not None}
    if not agents:
        raise SystemExit("no M4 trial logged yet")
    bases = {f"*{k}*": r for k, v in BASES.items() if (r := latest(v)) is not None}
    cfg = next(iter(agents.values())).cfg
    primary = cfg["evaluation"]["costs"]["primary_bps"]
    levels = cfg["evaluation"]["costs"]["sweep_bps"]
    cross = {n: r for n, r in agents.items() if n in CROSS}

    L = ["# M4 — Cross-asset agent, new features, leave-assets-out (generated by scripts/analyze_m4.py)", "",
         f"Trials: {', '.join(f'`{r.trial_id}`' for r in agents.values())}.",
         "Cells: median (IQR) over 10 seeds x 5 walk-forward folds (validation 2014-01 → 2023-09), net of "
         f"{primary} bp + 1 bp half-spread unless stated. Each (seed, fold) value is an equal-weight portfolio "
         "of the evaluation set. Baselines in *italics*. Checkpoints selected on inner validation only.", ""]

    everything = {**agents, **bases}
    L += [f"## 1. Training universe, 26 ETFs (H1 evaluation set) @ {primary} bp", "",
          perf_table({n: r for n, r in everything.items() if n != SINGLE}, "train", primary), "",
          "### Median Sharpe per fold", "", fold_table({n: r for n, r in everything.items() if n != SINGLE}, "train", primary), "",
          "### Cost sweep (median Sharpe)", "", cost_table({n: r for n, r in everything.items() if n != SINGLE}, "train", levels), ""]

    L += [f"## 2. Leave-assets-out, 7 ETFs never used in training @ {primary} bp", "",
          perf_table(everything, "leave_out", primary), ""]
    if SINGLE in agents and cross:
        L += ["### Go criterion: cross-asset ≥ single-asset on the leave-out ETFs", "",
              "Paired one-sided permutation test over the 50 (seed, fold) pairs of the leave-out "
              f"portfolio's Sharpe ratio, each cross-asset config vs `{SINGLE}` (same features, reward and budget, "
              "trained on ^GDAXI only).", "",
              "| Cross-asset config | median Sharpe | single-asset median | mean difference | p-value |",
              "|---|---|---|---|---|"]
        s_med = agents[SINGLE].runs("leave_out", primary)["sharpe"].median()
        for n, r in cross.items():
            t = paired(r, agents[SINGLE], "leave_out", primary)
            L.append(f"| {n} | {r.runs('leave_out', primary)['sharpe'].median():.2f} | {s_med:.2f} | "
                     f"{t['mean_diff']:+.2f} | {t['p_value']:.3f} |")
        L.append("")

    L += [f"## 3. Single asset ^GDAXI @ {primary} bp (M3 best: `m3_vol_scaled_pnl` 0.21)", "",
          perf_table(everything, "single", primary), ""]

    dep = deploy_baselines(cfg)
    dep_all = {**{n: r for n, r in agents.items()}, **dep}
    L += ["## 4. Deployment set SPY + EFA (EUR 10k neo-broker account)", "",
          "Median Sharpe at the PROTOCOL cost levels and under `neo_broker` "
          "(EUR 1 per buy and per sell, EUR 10,000 split over the 2 ETFs = EUR 5,000 per position, 3 bp half-spread).", "",
          cost_table(dep_all, "deploy2", levels + ["neo_broker"]), "",
          perf_table(dep_all, "deploy2", "neo_broker").replace("| Strategy |", "| Strategy (neo-broker) |"), ""]

    L += [f"## 5. Statistics on the training universe @ {primary} bp", "",
          f"N_trials (distinct agent configurations in the trial log) = {tr.n_trials()}.", "",
          "| Config | vs buy&hold (Holm p) | vs momentum | vs MACD | vs random | Deflated Sharpe (median over seeds) |",
          "|---|---|---|---|---|---|"]
    base_named = {k.strip("*"): v for k, v in bases.items()}
    for n, r in cross.items():
        comp = compare_to_baselines(r, base_named, eval_set="train").set_index("baseline")
        dsr = deflated_sharpe_of(r, eval_set="train")
        cells = [f"{comp.loc[b, 'mean_diff']:+.2f}, p={comp.loc[b, 'p_holm']:.3f}" for b in base_named]
        L.append(f"| {n} | " + " | ".join(cells) + f" | {dsr['deflated_sharpe_median']:.3f} |")
    if len(cross) >= 2:
        pbo = pbo_over(cross, eval_set="train")
        L += ["", f"**PBO over the {pbo['n_configs']} cross-asset configurations** (CSCV, S = "
              f"{cfg['evaluation']['statistics']['pbo_blocks']}): **{pbo['pbo']:.2f}**; probability that the "
              f"in-sample winner loses money out of sample: {pbo['prob_oos_loss']:.2f}."]
    L.append("")

    L += ["## 6. Training stability", "",
          "| Config | Best ckpt at (median % of training) | Late drift (median; share < −0.5) | Mean-Q range | "
          "Seed IQR within fold (median, eval set) | Minutes per job (median) |", "|---|---|---|---|---|---|"]
    for n, r in agents.items():
        df = stability_rows(r)
        es = "single" if n == SINGLE else "train"
        iqr = r.runs(es, primary).groupby("fold")["sharpe"].apply(lambda s: s.quantile(.75) - s.quantile(.25)).median()
        L.append(f"| {n} | {100 * df['best_at'].median():.0f}% | {df['drift'].median():+.2f}; "
                 f"{int((df['drift'] < -0.5).sum())}/{len(df)} | {df['q_min'].min():.2f} … {df['q_max'].max():.2f} | "
                 f"{iqr:.2f} ({es}) | {df['secs'].median() / 60:.1f} |")
    missing = [n for n in CROSS + [SINGLE] if n not in agents]
    if missing:
        L += ["", f"*Incomplete report: not yet run: {', '.join(missing)}.*"]

    out = repo_path("experiments", "reports", "M4_cross_asset.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    shutil.rmtree(repo_path("experiments", "runs", "_unlogged"), ignore_errors=True)
    print("\n".join(L))
    print("report ->", out)


if __name__ == "__main__":
    main()
