"""PROTOCOL Part II: report of one v2 step (nothing is retrained).

    python scripts/analyze_v2.py --step 0d --trials v2_R0prime --reference m4_cross_resid_transformer
    python scripts/analyze_v2.py --step 1 --trials v2_V1 --reference v2_R0prime

Reads the latest logged trial of each name and writes experiments/reports/V2_<step>.md
(+ CSVs in experiments/reports/v2_<step>/). Sections:
  1. H1 set (26 training ETFs) @ primary cost: performance, per fold, cost sweep
  2. each trial vs the reference, paired over the (seed, fold) pairs, with the §V8
     ADOPTION RULE: (a) median delta net Sharpe >= +0.05 and one-sided permutation
     p < 0.10, or (b) median delta >= -0.02 and (turnover -30 % or seed IQR -25 % or
     timing IC +0.01)
  3. the other evaluation sets: leave-out, ^GDAXI, deploy2 (incl. the neo-broker scenario)
  4. statistics vs the baselines (Holm) and the deflated Sharpe ratio (N_trials from the log)
  5. §V7 diagnostics (harness/diagnostics.agent_scores) and the H3 preview
  6. gate G-lag (§V8) and training stability
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402

from harness import diagnostics as dg                                    # noqa: E402
from harness import report as rp                                         # noqa: E402
from harness import stats as st                                          # noqa: E402
from harness import trials as tr                                         # noqa: E402
from harness.config import load_config, repo_path                        # noqa: E402
from harness.data import load_prices, ticker_sets                        # noqa: E402
from harness.experiment import compare_to_baselines, deflated_sharpe_of  # noqa: E402
from rl.features import ex_ante_vol                                      # noqa: E402

ADOPT_DELTA, ADOPT_P = 0.05, 0.10                    # §V8 (a)
KEEP_DELTA, TURNOVER_CUT, IQR_CUT, IC_GAIN = -0.02, -0.30, -0.25, 0.01   # §V8 (b)
ES = "train"                                         # H1 set


def seed_iqr(res, cost):
    """Median over folds of the within-fold IQR of the Sharpe ratio across seeds."""
    return float(res.runs(ES, cost).groupby("fold")["sharpe"]
                 .apply(lambda s: s.quantile(.75) - s.quantile(.25)).median())


def adoption(trial, ref, dt, dr, cost):
    """§V8 adoption rule of `trial` over the reference, paired over (seed, fold)."""
    m = trial.runs(ES, cost)[["seed", "fold", "sharpe"]].merge(
        ref.runs(ES, cost)[["seed", "fold", "sharpe"]], on=["seed", "fold"], suffixes=("_t", "_r"))
    diff = m["sharpe_t"] - m["sharpe_r"]
    a = {"median_delta": float(diff.median()), "p_one_sided": st.permutation_test(diff)["p_value"],
         "turnover_change": dt["turnover"] / dr["turnover"] - 1,
         "seed_iqr_change": dt["seed_iqr"] / dr["seed_iqr"] - 1,
         "timing_ic_change": dt["timing_ic"] - dr["timing_ic"]}
    a["rule_a"] = a["median_delta"] >= ADOPT_DELTA and a["p_one_sided"] < ADOPT_P
    a["rule_b"] = a["median_delta"] >= KEEP_DELTA and (a["turnover_change"] <= TURNOVER_CUT
                                                       or a["seed_iqr_change"] <= IQR_CUT
                                                       or a["timing_ic_change"] >= IC_GAIN)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, help="v2 step label, e.g. 0d or 1")
    ap.add_argument("--trials", nargs="+", required=True, help="trial names (latest run of each)")
    ap.add_argument("--reference", help="reference trial for the paired comparison and §V8")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-q", action="store_true", help="skip the Q-value inference (LC5/LC7)")
    args = ap.parse_args()

    trials = {n: r for n in args.trials if (r := rp.latest(n)) is not None}
    if not trials:
        raise SystemExit(f"none of {args.trials} is logged yet")
    ref = rp.latest(args.reference) if args.reference else None
    agents = {**trials, **({args.reference: ref} if ref is not None else {})}
    bases = {f"*{k}*": r for k, v in {**rp.BASES, "vol_target": "baseline_vol_target"}.items()
             if (r := rp.latest(v)) is not None}
    cfg = next(iter(trials.values())).cfg
    primary, levels = cfg["evaluation"]["costs"]["primary_bps"], cfg["evaluation"]["costs"]["sweep_bps"]
    out_dir = repo_path("experiments", "reports", f"v2_{args.step}")
    os.makedirs(out_dir, exist_ok=True)

    # §V7 diagnostics of every agent on the H1 set (stored exposures + saved checkpoints)
    cfg0 = load_config()
    sigmas = {t: ex_ante_vol(df["Close"].to_numpy(), 60)
              for t, df in load_prices(ticker_sets(cfg0)["all"], cfg0).items()}
    bh = rp.latest("baseline_buy_and_hold")
    diag = {}
    for n, r in agents.items():
        print(f"[v2 {args.step}] diagnostics {n}")
        row, d, _, _ = dg.agent_scores(n, r, bh, sigmas, primary, es=ES, workers=args.workers,
                                       threads=args.threads, q_values=not args.skip_q)
        diag[n] = {**row, "seed_iqr": seed_iqr(r, primary), "turnover": float(r.runs(ES, primary)["turnover"].median())}
        d.assign(strategy=n).to_csv(os.path.join(out_dir, f"diag_{n}.csv"), index=False)
    pd.DataFrame(diag.values()).to_csv(os.path.join(out_dir, "diagnostics.csv"), index=False)

    everything = {**agents, **bases}
    L = [f"# v2 Step {args.step} — {', '.join(trials)} (generated by scripts/analyze_v2.py)", "",
         "Trials: " + ", ".join(f"`{r.trial_id}` ({n})" for n, r in agents.items()) + ". "
         "Cells: median (IQR) over 10 seeds × 5 walk-forward folds (validation 2014-01 → 2023-09), net of "
         f"{primary} bp + 1 bp half-spread unless stated; each (seed, fold) value is an equal-weight portfolio of "
         "the evaluation set. Baselines in *italics*. Checkpoints selected on inner validation only. "
         f"N_trials in the log: {tr.n_trials()}.", "",
         f"## 1. H1 set: 26 training ETFs @ {primary} bp", "", rp.perf_table(everything, ES, primary), "",
         "### Median Sharpe per fold", "", rp.fold_table(everything, ES, primary), "",
         "### Cost sweep (median Sharpe)", "", rp.cost_table(everything, ES, levels), ""]

    if ref is not None:
        L += [f"## 2. Against `{args.reference}` (paired, 26 ETFs @ {primary} bp) and the §V8 adoption rule", "",
              "(a) median Δ net Sharpe ≥ +0.05 **and** one-sided permutation p < 0.10; or (b) median Δ ≥ −0.02 "
              "**and** (turnover −30 % or seed IQR −25 % or timing IC +0.01).", "",
              "| Trial | median Δ Sharpe | p (one-sided) | turnover | seed IQR | timing IC | (a) | (b) | adopt |",
              "|---|---|---|---|---|---|---|---|---|"]
        rows = []
        for n, r in trials.items():
            a = adoption(r, ref, diag[n], diag[args.reference], primary)
            rows.append({"trial": n, **a})
            yes = lambda b: "yes" if b else "no"                                  # noqa: E731
            L.append(f"| {n} | {a['median_delta']:+.2f} | {a['p_one_sided']:.3f} | {a['turnover_change']:+.0%} | "
                     f"{a['seed_iqr_change']:+.0%} | {a['timing_ic_change']:+.3f} | {yes(a['rule_a'])} | "
                     f"{yes(a['rule_b'])} | **{yes(a['rule_a'] or a['rule_b'])}** |")
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "adoption.csv"), index=False)
        L.append("")

    L += [f"## 3. Other evaluation sets @ {primary} bp", "",
          "### Leave-out: 7 ETFs never used in training", "", rp.perf_table(everything, "leave_out", primary), "",
          "### Single asset ^GDAXI", "", rp.perf_table(everything, "single", primary), ""]
    if "deploy2" in cfg.get("extra_ticker_sets", {}):
        dep = {**agents, **rp.deploy_baselines(cfg)}
        L += ["### Deployment set SPY + EFA (EUR 10k neo-broker account)", "",
              rp.cost_table(dep, "deploy2", levels + ["neo_broker"]), "",
              rp.perf_table(dep, "deploy2", "neo_broker").replace("| Strategy |", "| Strategy (neo-broker) |"), ""]

    base_named = {k.strip("*"): v for k, v in bases.items() if k != "*vol_target*"}
    L += [f"## 4. Statistics on the H1 set @ {primary} bp", "",
          "| Trial | vs buy&hold (Holm p) | vs momentum | vs MACD | vs random | Deflated Sharpe (median over seeds) |",
          "|---|---|---|---|---|---|"]
    for n, r in agents.items():
        comp = compare_to_baselines(r, base_named, eval_set=ES).set_index("baseline")
        cells = [f"{comp.loc[b, 'mean_diff']:+.2f}, p={comp.loc[b, 'p_holm']:.3f}" for b in base_named]
        L.append(f"| {n} | " + " | ".join(cells) + f" | {deflated_sharpe_of(r, eval_set=ES)['deflated_sharpe_median']:.3f} |")

    f = rp.fmt
    L += ["", "## 5. Diagnostics (§V7) and H3 preview, 26 ETFs", "",
          "| Trial | Timing IC | Timing Sharpe | Gross − B&H | Cost | Mean exposure | Switches / 100 bars | Action-gap "
          "ratio | Over-estimation (SD of G) | Best ckpt at | MDD lower than exposure-matched B&H | Reproduced |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in diag.items():
        L.append(f"| {n} | {f(r['timing_ic'], 3)} | {f(r['timing_sharpe'])} | {f(r['timing_vs_bh_sharpe'])} | "
                 f"{f(r['cost_sharpe'])} | {f(r['mean_exposure'], pct=True)} | {f(r['switches_per_100'], 1)} | "
                 f"{f(r.get('action_gap_ratio', np.nan), 3)} | {f(r.get('q_bias_sd', np.nan))} | "
                 f"{f(r['lc_best_at'], pct=True)} | {f(r['h3_share_mdd_below_matched'], pct=True)} | "
                 f"{f(r.get('reproduced', np.nan), pct=True)} |")
    L += ["", f"## 6. Gate G-lag (§V8: median lag-1 net Sharpe ≥ median lag-0 − {dg.G_LAG_TOL}) and training stability", "",
          "| Trial | Lag-0 | Lag-1 | Δ | G-lag | Late drift (median) | Minutes per job (median) |",
          "|---|---|---|---|---|---|---|"]
    for n, r in agents.items():
        dgn, s = diag[n], rp.stability_rows(r)
        L.append(f"| {n} | {dgn['sharpe_10bp']:.2f} | {dgn['sharpe_10bp_lag1']:.2f} | {dgn['lag_delta']:+.2f} | "
                 f"{'pass' if dgn['g_lag_pass'] else 'FAIL'} | {s['drift'].median():+.2f} | {s['secs'].median() / 60:.1f} |")

    out = repo_path("experiments", "reports", f"V2_{args.step}.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    shutil.rmtree(repo_path("experiments", "runs", "_unlogged"), ignore_errors=True)
    print("\n".join(L))
    print("report ->", out)


if __name__ == "__main__":
    main()
