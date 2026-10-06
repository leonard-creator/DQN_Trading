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
"""

import numpy as np
import pandas as pd
from scipy import stats as _st

from harness import backtest as bt
from harness import metrics as mt
from harness.splits import block_positions


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
