"""PROTOCOL Part II, step 0a: re-score all logged agent trials and the baselines.

    python scripts/rescore_v2.py              # -> experiments/reports/V2_0a_rescore.md (+ CSVs in v2_0a/)
    python scripts/rescore_v2.py --workers 12

Nothing is retrained and no trial is logged: every agent is measured through
its STORED exposures (rl/policy.StoredExposurePolicy) or its SAVED weights
(rl/diagnostics.py). Per trial, on its primary evaluation set, at 10 bp:

    LC1  learning curve: area under the inner-validation Sharpe curve, last-half slope, best checkpoint
    LC2  Sharpe at 0 bp vs 10 bp
    LC3  timing IC and timing Sharpe (harness/diagnostics.py)
    LC4  attribution vs buy-and-hold: timing = gross - B&H, cost = net - gross (Sharpe and %/yr),
         as differences of the reported medians so that net - B&H = timing + cost in the table
         (paired medians over seed-fold pairs are kept in the CSVs as *_paired)
    LC5  action-gap ratio, LC7 predicted vs realised value, reproduction check (rl/diagnostics.py)
    LC6  switches per 100 bars, time at full exposure
    LC8  lag-1 execution Sharpe and the G-lag gate preview, worded as in §V8:
         median lag-1 net Sharpe no more than 0.05 below the median lag-0 net Sharpe
    H3 preview: drawdown vs exposure-matched (and vol-matched) buy-and-hold, descriptive only

Baselines get LC2-LC4, LC6, LC8 and the H3 preview; the new descriptive
vol-target baseline is run once and logged as a baseline row (not a trial).
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402

from harness import backtest as bt                                          # noqa: E402
from harness import trials as tr                                            # noqa: E402
from harness.config import load_config, repo_path                           # noqa: E402
from harness.data import load_prices, ticker_sets                           # noqa: E402
from harness.diagnostics import diagnose, learning_curve_stats             # noqa: E402
from harness.experiment import BaselinePolicy, load_result, run_experiment  # noqa: E402
from harness.splits import folds_from_config                                # noqa: E402
from rl.features import ex_ante_vol                                         # noqa: E402

BASELINES = ["baseline_buy_and_hold", "baseline_momentum", "baseline_macd", "baseline_random",
             "baseline_momentum_252", "baseline_vol_target"]
OUT = repo_path("experiments", "reports", "v2_0a")
MDD_TOL = 0.001          # H3 preview: "lower drawdown" means lower by more than 0.1 pp (ties are not wins)
G_LAG_TOL = 0.05         # gate G-lag, PROTOCOL Part II §V8


def latest_dir(name):
    t = tr.load_trials()
    rows = t[t["name"] == name]
    return None if rows.empty else repo_path(rows.iloc[-1]["output_dir"])


def med(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def sharpe_by(res, es, cost):
    return res.runs(es, cost).set_index(["seed", "fold"])["sharpe"]


def cagr_by(res, es, cost):
    return res.runs(es, cost).set_index(["seed", "fold"])["cagr"]


def against_bh(agent_series, bh_series):
    """Pair agent (seed, fold) values with buy-and-hold by fold (B&H is the same for every seed)."""
    bh = bh_series.groupby(level="fold").first()
    return agent_series - agent_series.index.get_level_values("fold").map(bh).to_numpy()


def score(name, res, policy, es, bh, sigmas_all, primary, lag_workers_note=""):
    """All exposure-based diagnostics of one strategy on one evaluation set."""
    cfg = res.cfg
    sets = ticker_sets(cfg)
    folds = folds_from_config(cfg)
    seeds = sorted(res.metrics["seed"].unique())
    rate = bt.cost_rate(primary, cfg["evaluation"]["costs"]["half_spread_bps"])
    s0, s10 = sharpe_by(res, es, 0), sharpe_by(res, es, primary)
    lag1 = run_experiment(cfg, policy=policy, eval_sets=[es], cost_levels=[0, primary], cost_scenarios=[],
                          execution_lag=1, log=False, verbose=False)
    l10 = sharpe_by(lag1, es, primary)
    bh10 = sharpe_by(bh, es, primary)
    c0, c10, bhc10 = cagr_by(res, es, 0), cagr_by(res, es, primary), cagr_by(bh, es, primary)
    prices = load_prices(sets[es], cfg)
    d = diagnose(policy, cfg, prices, folds, seeds, {es: sets[es]}, {t: sigmas_all[t] for t in sets[es]},
                 cost=rate, lag=0)
    row = {
        "strategy": name, "eval_set": es,
        "sharpe_10bp": med(s10), "sharpe_0bp": med(s0), "bh_sharpe": med(bh10),
        # LC4 on the medians shown (B&H net of its own entry cost), so the table adds up
        "timing_vs_bh_sharpe": med(s0) - med(bh10), "cost_sharpe": med(s10) - med(s0),
        "timing_vs_bh_pct": 100 * (med(c0) - med(bhc10)), "cost_pct": 100 * (med(c10) - med(c0)),
        "timing_vs_bh_sharpe_paired": med(against_bh(s0, bh10)), "cost_sharpe_paired": med(s10 - s0),
        # LC8 / G-lag: difference of the medians (§V8 wording); the paired median as a robustness check
        "sharpe_10bp_lag1": med(l10), "lag_delta": med(l10) - med(s10), "lag_delta_paired": med(l10 - s10),
        "timing_ic": med(d["timing_ic"]), "timing_sharpe": med(d["timing_sharpe"]),
        "mean_exposure": med(d["mean_exposure"]), "switches_per_100": med(d["switches_per_100"]),
        "time_at_anchor": med(d["time_at_anchor"]),
        "h3_share_mdd_below_matched": float(np.mean(d["mdd_agent"] < d["mdd_matched"] - MDD_TOL)),
        "h3_dmdd_matched": med(d["mdd_agent"] - d["mdd_matched"]),
        "h3_dmdd_volmatched": med(d["mdd_agent"] - d["mdd_volmatched"]),
    }
    row["g_lag_pass"] = bool(row["lag_delta"] >= -G_LAG_TOL)
    return row, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--skip-q", action="store_true", help="skip the Q-value inference (LC5/LC7)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cfg0 = load_config()
    primary = cfg0["evaluation"]["costs"]["primary_bps"]
    sets0 = ticker_sets(cfg0)
    allp = load_prices(sets0["all"], cfg0)
    sigmas_all = {t: ex_ante_vol(df["Close"].to_numpy(), 60) for t, df in allp.items()}
    bh = load_result(latest_dir("baseline_buy_and_hold"))

    # --- the new descriptive baseline: run once, logged as a BASELINE row (never a trial) --------
    if latest_dir("baseline_vol_target") is None:
        print("[0a] running the descriptive vol-target baseline (logged as a baseline row)")
        run_experiment(load_config("config/experiments/baseline_vol_target.yaml"),
                       notes="descriptive baseline (PROTOCOL Part II step 0a)", verbose=False)

    # --- agents ----------------------------------------------------------------------------------
    from rl.policy import StoredExposurePolicy
    trials = tr.load_trials()
    agents = trials[trials["kind"] == "agent"].drop_duplicates("name", keep="last")
    rows, qrows, lcrows = [], [], []
    for _, t in agents.iterrows():
        run_dir = repo_path(t["output_dir"])
        res = load_result(run_dir)
        es = (res.cfg.get("eval_sets") or ["single"])[0]
        print(f"[0a] {t['name']} ({es})")
        row, d = score(t["name"], res, StoredExposurePolicy(run_dir), es, bh, sigmas_all, primary)
        d.assign(strategy=t["name"]).to_csv(os.path.join(OUT, f"diag_{t['name']}.csv"), index=False)
        agent_dir = os.path.join(run_dir, "agent")
        lc = []
        for j in sorted(os.listdir(agent_dir)):
            info = json.load(open(os.path.join(agent_dir, j, "info.json")))
            lc.append(learning_curve_stats(pd.read_csv(os.path.join(agent_dir, j, "curve.csv")), info))
        lc = pd.DataFrame(lc)
        row.update({k: med(lc[k]) for k in lc.columns})
        lcrows.append(lc.assign(strategy=t["name"]))
        if not args.skip_q:
            from rl.diagnostics import q_diagnostics_for_run
            q = q_diagnostics_for_run(run_dir, workers=args.workers, threads=args.threads)
            q = q[q["ticker"].isin(ticker_sets(res.cfg)[es])]
            qrows.append(q.assign(strategy=t["name"]))
            row.update({"action_gap_ratio": med(q["action_gap_ratio"]),
                        "q_minus_G": med(q["q_minus_G_median"]), "q_over_G": med(q["q_over_G"]),
                        "q_bias_sd": med(q["q_bias_sd"]),
                        "q_G_corr": med(q["q_G_corr"]), "reproduced": float(np.nanmean(q["reproduced"]))})
        rows.append(row)
    agents_df = pd.DataFrame(rows)
    agents_df.to_csv(os.path.join(OUT, "agents.csv"), index=False)
    if qrows:
        pd.concat(qrows).to_csv(os.path.join(OUT, "q_diagnostics_per_sleeve.csv"), index=False)
    pd.concat(lcrows).to_csv(os.path.join(OUT, "learning_curves_per_job.csv"), index=False)

    # --- baselines on the two main sets --------------------------------------------------------
    brows = []
    for name in BASELINES:
        d = latest_dir(name)
        if d is None:
            print(f"[0a] skip {name}: not logged")
            continue
        res = load_result(d)
        for es in ("single", "train"):
            print(f"[0a] {name} ({es})")
            pol = BaselinePolicy(res.cfg["policy"], res.cfg.get("policy_params"))
            row, _ = score(name.replace("baseline_", ""), res, pol, es, bh, sigmas_all, primary)
            brows.append(row)
    base_df = pd.DataFrame(brows)
    base_df.to_csv(os.path.join(OUT, "baselines.csv"), index=False)
    shutil.rmtree(repo_path("experiments", "runs", "_unlogged"), ignore_errors=True)
    write_report(agents_df, base_df, primary)


def fmt(v, d=2, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.{d}f}"


def write_report(agents_df, base_df, primary):
    L = ["# v2 step 0a — Re-scoring of all logged trials (generated by scripts/rescore_v2.py)", "",
         "No retraining and no new trials: agents are measured through their stored exposures and saved "
         "weights. Medians over 10 seeds × 5 folds on each trial's primary evaluation set (^GDAXI = `single`, "
         f"26 ETFs = `train`), net of {primary} bp + 1 bp unless stated. Definitions: PROTOCOL Part II §V7 and "
         "`harness/diagnostics.py`. Per-row CSVs: `experiments/reports/v2_0a/`.", "",
         "## 1. Performance, attribution and execution lag", "",
         "LC4: *timing* = median gross (0 bp) Sharpe − median B&H Sharpe; *cost* = median net − median gross, so "
         "net − B&H = timing + cost. LC8: *Δ lag* = median lag-1 net Sharpe − median lag-0 net Sharpe; gate G-lag "
         f"(§V8) passes if Δ lag ≥ −{G_LAG_TOL}. It is binding only for the configuration selected for M5. "
         "Baselines are deterministic, so their medians are over 5 folds only and Δ lag is noisy "
         "(a one-day delay moves each 2-year fold's Sharpe by ~0.1 for a strategy that switches 13×/yr). "
         "Paired medians are in the CSVs (`*_paired`).", "",
         "| Strategy | Set | Sharpe 10 bp | 0 bp | B&H | Timing vs B&H (Sharpe / %·yr) | Cost (Sharpe / %·yr) | Lag-1 Sharpe | Δ lag | G-lag |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for df, mark in ((agents_df, ""), (base_df, "*")):
        for _, r in df.iterrows():
            nm = f"{mark}{r['strategy']}{mark}"
            L.append(f"| {nm} | {r['eval_set']} | {fmt(r['sharpe_10bp'])} | {fmt(r['sharpe_0bp'])} | {fmt(r['bh_sharpe'])} | "
                     f"{fmt(r['timing_vs_bh_sharpe'])} / {fmt(r['timing_vs_bh_pct'], 1)} | {fmt(r['cost_sharpe'])} / {fmt(r['cost_pct'], 1)} | "
                     f"{fmt(r['sharpe_10bp_lag1'])} | {fmt(r['lag_delta'])} | {'pass' if r['g_lag_pass'] else 'FAIL'} |")
    L += ["", "## 2. Timing skill and behaviour", "",
          "| Strategy | Set | Timing IC | Timing Sharpe | Mean exposure | Switches / 100 bars | Time at 100 % |",
          "|---|---|---|---|---|---|---|"]
    for df, mark in ((agents_df, ""), (base_df, "*")):
        for _, r in df.iterrows():
            L.append(f"| {mark}{r['strategy']}{mark} | {r['eval_set']} | {fmt(r['timing_ic'], 3)} | {fmt(r['timing_sharpe'])} | "
                     f"{fmt(r['mean_exposure'], pct=True)} | {fmt(r['switches_per_100'], 1)} | {fmt(r['time_at_anchor'], pct=True)} |")
    if "action_gap_ratio" in agents_df:
        L += ["", "## 3. Learning curves and value estimates (agents)", "",
              "LC1 from `curve.csv` (inner-validation Sharpe; AUC = mean over training; slope per unit of training "
              "progress over the second half). LC5/LC7 from re-running the selected checkpoint; G truncated at the "
              "block end, bars with < 300 bars left excluded. *Action-gap ratio* < 1 = the greedy choice is decided "
              "by noise. *Over-estimation* = (mean Q − mean G) / SD(G), scale-free so it compares across rewards "
              "(Q − G in reward units and Q / G are in the CSVs). *Reproduced* = share of decision bars whose "
              "re-computed exposure equals the stored one.", "",
              "| Strategy | LC1 AUC | Last-half slope | Best ckpt at | Action-gap ratio | Over-estimation (SD of G) | corr(Q, G) | Reproduced |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in agents_df.iterrows():
            L.append(f"| {r['strategy']} | {fmt(r['lc_auc'])} | {fmt(r['lc_last_half_slope'])} | {fmt(r['lc_best_at'], pct=True)} | "
                     f"{fmt(r['action_gap_ratio'], 3)} | {fmt(r['q_bias_sd'])} | {fmt(r['q_G_corr'])} | "
                     f"{fmt(r['reproduced'], pct=True)} |")
    L += ["", "## 4. H3 preview (descriptive): drawdown beyond lower exposure", "",
          "Exposure-matched buy-and-hold = each sleeve earns (agent's mean exposure) × asset return, daily "
          "rebalanced, no costs. Vol-matched = the same with the agent's realised volatility instead of its mean "
          "exposure. Negative ΔMDD = smaller drawdown than the benchmark. Descriptive only; the H3 test itself is "
          "pre-registered for the v2 configurations (§V4).", "",
          "| Strategy | Set | Share of (seed, fold) with MDD lower by > 0.1 pp | median ΔMDD vs matched | vs vol-matched |",
          "|---|---|---|---|---|"]
    for df, mark in ((agents_df, ""), (base_df, "*")):
        for _, r in df.iterrows():
            L.append(f"| {mark}{r['strategy']}{mark} | {r['eval_set']} | {fmt(r['h3_share_mdd_below_matched'], pct=True)} | "
                     f"{fmt(100 * r['h3_dmdd_matched'], 1)} pp | {fmt(100 * r['h3_dmdd_volmatched'], 1)} pp |")
    out = repo_path("experiments", "reports", "V2_0a_rescore.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("report ->", out)


if __name__ == "__main__":
    main()
