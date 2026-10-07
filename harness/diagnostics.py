"""Timing and behaviour diagnostics from exposures (PROTOCOL Part II §V7; step 0a).

All quantities are computed from the SAME exposures and dates the harness
backtests, so they explain the reported Sharpe ratios instead of measuring
something else. For a sleeve (one ticker) and one evaluation block:

    e_t      exposure decided at the close of bar t (held t -> t+1)
    r_t+1    next-day return close[t+1] / close[t] - 1
    sigma_t  ex-ante daily volatility known at bar t (EWMA, rl/features.ex_ante_vol)

    LC3  timing IC       Spearman rho(e_t, r_t+1 / sigma_t). Positive = more exposure
                         before better (vol-adjusted) days. 0 = no timing.
         timing return   (e_t - mean(e)) * r_t+1. Removes the effect of simply being
                         long on average; its Sharpe is the pure timing contribution
                         (gross, no costs).
    LC6  switches / 100  changes of exposure per 100 decision bars (incl. the entry)
         time at anchor  share of decision bars at full exposure (e = 1)
    H3 preview           drawdown of the agent (net) vs exposure-matched buy-and-hold:
                         mean(e) * r_t+1, rebalanced daily without costs (conservative);
                         and vs volatility-matched buy-and-hold (sensitivity)

Portfolio level: sleeves are combined with fixed equal weights exactly as in
harness.backtest.portfolio; the IC and switch statistics are averaged over sleeves.

strategy_scores() adds the result-based diagnostics of one strategy (LC2, LC4, LC8
with gate G-lag) and summarises everything above as one row (Step 0a, v2 reports).
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats as _st

from harness import backtest as bt
from harness import metrics as mt
from harness.data import load_prices, ticker_sets
from harness.config import repo_path
from harness.experiment import run_experiment
from harness.splits import block_positions, folds_from_config

MDD_TOL = 0.001          # H3 preview: "lower drawdown" = lower by more than 0.1 pp (ties are not wins)
G_LAG_TOL = 0.05         # gate G-lag, PROTOCOL Part II §V8


def sleeve_timing(close, exposure, pos, sigma):
    """Diagnostics of one sleeve over one block (pos = return positions, decision at pos-1)."""
    close = np.asarray(close, dtype=np.float64)
    e = np.asarray(exposure, dtype=np.float64)[pos - 1]
    r = close[pos] / close[pos - 1] - 1.0
    z = r / np.asarray(sigma, dtype=np.float64)[pos - 1]
    ic = float(_st.spearmanr(e, z)[0]) if np.std(e) > 0 and np.std(z) > 0 else np.nan
    ebar = float(e.mean())
    switches = np.abs(np.diff(np.r_[0.0, e])) > 1e-12
    return {"ic": ic, "ebar": ebar, "switches_per_100": 100.0 * switches.mean(),
            "time_at_anchor": float(np.mean(np.abs(e - 1.0) < 1e-12)),
            "timing_ret": (e - ebar) * r, "matched_ret": ebar * r, "asset_ret": r}


def diagnose(policy, cfg, prices, folds, seeds, tickers_by_set, sigmas, cost, lag=0):
    """One row per (eval_set, seed, fold) with LC3 / LC6 / H3-preview statistics.

    policy  : anything with .exposures(prices, fold, seed, cfg, tickers) (harness interface)
    sigmas  : {ticker: ex-ante daily volatility array aligned with prices[ticker]}
    cost    : cost model for the agent's NET returns (the H3 comparison uses net)
    """
    bpy = cfg["data"]["bars_per_year"]
    all_t = sorted({t for ts in tickers_by_set.values() for t in ts})
    rows = []
    for fold in folds:
        fp = {t: df[df.index <= fold.val_end] for t, df in prices.items()}
        for seed in seeds:
            expo = policy.exposures(fp, fold, seed, cfg, all_t)
            per = {}
            for t in all_t:
                df = fp[t]
                pos = block_positions(df.index, fold.val_start, fold.val_end)
                if len(pos) == 0:
                    continue
                d = sleeve_timing(df["Close"].to_numpy(), expo[t], pos, sigmas[t][:len(df)])
                net = bt.backtest(df["Close"].to_numpy(), expo[t], pos, cost, lag=lag)["net"]
                idx = df.index[pos]
                d["frames"] = {k: pd.Series(d[k], index=idx) for k in ("timing_ret", "matched_ret", "asset_ret")}
                d["frames"]["net"] = pd.Series(net, index=idx)
                per[t] = d
            for es, ts in tickers_by_set.items():
                sub = [per[t] for t in ts if t in per]
                if not sub:
                    continue

                def port(key):
                    wide = pd.concat([s["frames"][key] for s in sub], axis=1).sort_index()
                    return wide.fillna(0.0).mean(axis=1).to_numpy()

                net, timing, matched, asset = port("net"), port("timing_ret"), port("matched_ret"), port("asset_ret")
                sd_net, sd_bh = np.std(net, ddof=1), np.std(asset, ddof=1)
                volmatched = asset * (sd_net / sd_bh if sd_bh > 0 else 0.0)
                rows.append({
                    "eval_set": es, "seed": seed, "fold": fold.name,
                    "timing_ic": float(np.nanmean([s["ic"] for s in sub])),
                    "timing_sharpe": mt.sharpe(timing, bpy),
                    "mean_exposure": float(np.mean([s["ebar"] for s in sub])),
                    "switches_per_100": float(np.mean([s["switches_per_100"] for s in sub])),
                    "time_at_anchor": float(np.mean([s["time_at_anchor"] for s in sub])),
                    "mdd_agent": mt.max_drawdown(net),
                    "mdd_matched": mt.max_drawdown(matched),
                    "mdd_volmatched": mt.max_drawdown(volmatched),
                })
    return pd.DataFrame(rows)


def learning_curve_stats(curve, info):
    """LC1 from one training curve: area under the inner-Sharpe curve, last-half slope, best position."""
    c = curve.copy()
    c["p"] = c["update"] / max(1, info["updates"])
    c = c.sort_values("p")
    x, y = c["p"].to_numpy(), c["inner_sharpe"].to_numpy()
    auc = float(np.trapz(y, x) / max(x[-1] - x[0], 1e-9)) if len(x) > 1 else float(y.mean())
    late = c[c["p"] > 0.5]
    slope = float(np.polyfit(late["p"], late["inner_sharpe"], 1)[0]) if len(late) >= 3 else np.nan
    return {"lc_auc": auc, "lc_last_half_slope": slope,
            "lc_best_at": info["best_update"] / max(1, info["updates"])}



def med(x):
    """Median ignoring NaN (NaN if nothing is finite)."""
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def _by(res, es, cost, metric):
    return res.runs(es, cost).set_index(["seed", "fold"])[metric]


def _against_bh(series, bh_series):
    """Pair (seed, fold) values with buy-and-hold by fold (B&H is the same for every seed)."""
    bh = bh_series.groupby(level="fold").first()
    return series - series.index.get_level_values("fold").map(bh).to_numpy()


def strategy_scores(name, res, policy, es, bh, sigmas, primary):
    """§V7 summary row of one strategy on evaluation set `es` (PROTOCOL Part II §V12.1).

    res    : the strategy's logged ExperimentResult; policy : its exposures (stored or baseline)
    bh     : buy-and-hold ExperimentResult; sigmas : {ticker: ex-ante vol}; primary : cost in bp
    LC4 (timing = gross - B&H, cost = net - gross) and LC8 (lag-1 - lag-0) are differences of
    the reported medians, so the table adds up and G-lag reads as worded in §V8; paired
    medians are kept as *_paired. Returns (row, per-(seed, fold) diagnostics frame).
    """
    cfg = res.cfg
    tickers = ticker_sets(cfg)[es]
    rate = bt.cost_rate(primary, cfg["evaluation"]["costs"]["half_spread_bps"])
    s0, s10, bh10 = _by(res, es, 0, "sharpe"), _by(res, es, primary, "sharpe"), _by(bh, es, primary, "sharpe")
    c0, c10, bhc10 = _by(res, es, 0, "cagr"), _by(res, es, primary, "cagr"), _by(bh, es, primary, "cagr")
    lag1 = run_experiment(cfg, policy=policy, eval_sets=[es], cost_levels=[0, primary], cost_scenarios=[],
                          execution_lag=1, log=False, verbose=False)
    l10 = _by(lag1, es, primary, "sharpe")
    d = diagnose(policy, cfg, load_prices(tickers, cfg), folds_from_config(cfg), sorted(res.metrics["seed"].unique()),
                 {es: tickers}, {t: sigmas[t] for t in tickers}, cost=rate)
    row = {
        "strategy": name, "eval_set": es,
        "sharpe_10bp": med(s10), "sharpe_0bp": med(s0), "bh_sharpe": med(bh10),
        "timing_vs_bh_sharpe": med(s0) - med(bh10), "cost_sharpe": med(s10) - med(s0),
        "timing_vs_bh_pct": 100 * (med(c0) - med(bhc10)), "cost_pct": 100 * (med(c10) - med(c0)),
        "timing_vs_bh_sharpe_paired": med(_against_bh(s0, bh10)), "cost_sharpe_paired": med(s10 - s0),
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


def agent_scores(name, res, bh, sigmas, primary, es=None, workers=1, threads=2, q_values=True):
    """strategy_scores() of an agent trial from its STORED exposures, plus LC1 (learning curves)
    and, if q_values, LC5/LC7 and the reproduction check from its SAVED checkpoints
    (rl/diagnostics.py, CPU only). Nothing is retrained. `es` defaults to the trial's
    primary (first) evaluation set.
    Returns (row, per-(seed, fold) frame, per-job learning-curve frame, per-sleeve Q frame or None).
    """
    from rl.policy import StoredExposurePolicy                 # late: rl imports the harness
    run = repo_path(res.output_dir)
    es = es or (res.cfg.get("eval_sets") or ["single"])[0]
    row, d = strategy_scores(name, res, StoredExposurePolicy(run), es, bh, sigmas, primary)
    jobs = os.path.join(run, "agent")
    lc = pd.DataFrame([learning_curve_stats(pd.read_csv(os.path.join(jobs, j, "curve.csv")),
                                            json.load(open(os.path.join(jobs, j, "info.json"))))
                       for j in sorted(os.listdir(jobs))])
    row.update({k: med(lc[k]) for k in lc.columns})
    q = None
    if q_values:
        from rl.diagnostics import q_diagnostics_for_run
        q = q_diagnostics_for_run(run, workers=workers, threads=threads)
        q = q[q["ticker"].isin(ticker_sets(res.cfg)[es])]
        row.update({"action_gap_ratio": med(q["action_gap_ratio"]), "q_minus_G": med(q["q_minus_G_median"]),
                    "q_over_G": med(q["q_over_G"]), "q_bias_sd": med(q["q_bias_sd"]),
                    "q_G_corr": med(q["q_G_corr"]), "reproduced": float(np.nanmean(q["reproduced"]))})
    return row, d, lc, q

