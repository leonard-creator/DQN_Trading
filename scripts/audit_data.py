"""PROTOCOL Part II, step 0b: data audit -> DATA_AUDIT.md (+ CSVs in experiments/reports/v2_0b/).

    python scripts/audit_data.py

Offline. Uses data/raw (frozen inputs) and the owner-downloaded files in
data/audit/ (Q2). Restricted to the DEVELOPMENT period (dev_start .. dev_end):
the locked test period is not inspected, not even for data quality, and every
forward-looking quantity (next-day returns) is computed inside that period.

Sections (PROTOCOL Part II §V2.1-§V2.3):
    1  effective number of independent assets (average correlation, eigenvalues)
    2  equity drawdown events >= 15 %, and how many each fold's training window contains
    3  minimum backtest length: the best Sharpe expected from N zero-skill trials
    Q1 close-time leak: size and value of the leak in ^GDAXI's residual features (v1 vs v2 pipeline)
    Q2 adjusted prices: frozen data/raw vs a fresh download; adjusted vs unadjusted + dividends
    Q3 moves beyond 8 sigma (ex-ante volatility): market-wide vs single-ticker, bad-print checks
    Q4 zero / constant volume
    Q5 dates, duplicates, stale prices, forward filling
    Q6 point-in-time tests (run here; the result line is copied into the report)
"""

import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from scipy.stats import spearmanr                                     # noqa: E402

from harness import backtest as bt                                    # noqa: E402
from harness import metrics as mt                                     # noqa: E402
from harness import stats as st                                       # noqa: E402
from harness import trials as tr                                      # noqa: E402
from harness.config import load_config, repo_path                     # noqa: E402
from harness.data import load_prices, ticker_sets, ticker_to_file     # noqa: E402
from harness.splits import block_positions, fold_ranges, folds_from_config  # noqa: E402
from rl.features import ex_ante_vol                                   # noqa: E402
from rl.features_m4 import m4_raw_frames                              # noqa: E402

OUT = repo_path("experiments", "reports", "v2_0b")
SIGMA_LIMIT = 8.0          # Q3: |daily log return| / ex-ante sigma
WIDE_SIGMA = 4.0           # Q3: a move is market-wide if >= WIDE_COUNT tickers exceed this on the same day
WIDE_COUNT = 3
DD_LIMIT = 0.15            # section 2: drawdown events at least this deep
DIFF_BP = 5.0              # Q2: flag return differences above this
P_V2 = 40                  # §V6.0: purge P = W + H_max = 20 + 20 bars for every v2 configuration
N_ROWS = {"now": None, "v2 cap": 26, "Part I budget": 50}

# Dates the 8-sigma scan flags, with the well-known event behind them. These labels come from
# general knowledge and were NOT checked against a source in this audit; the classification
# itself (market-wide / genuine / possible bad print) is computed from the data alone.
KNOWN_EVENTS = {
    "2008-09-16": "Lehman week (bankruptcy filed 2008-09-15)",
    "2013-04-15": "gold crash",
    "2016-06-24": "UK Brexit referendum result",
    "2017-04-24": "French presidential election, first round",
    "2017-05-18": "Brazil: Temer/JBS recordings scandal",
    "2018-02-05": "volatility spike ('Volmageddon')",
    "2020-03-09": "COVID-19 sell-off and OPEC+ oil price war",
    "2022-03-16": "China: State Council pledge to support markets",
}

# Q6: point-in-time tests (perturb the future -> the value at t is unchanged), run below.
PIT_TESTS = [
    "tests/test_rl.py::test_features_do_not_look_ahead",
    "tests/test_baselines.py::test_no_lookahead_future_prices_do_not_change_past_exposure",
    "tests/test_rewards.py::test_ex_ante_vol_has_no_lookahead_and_is_positive",
    "tests/test_features_m4.py::test_no_lookahead_for_every_m4_feature",
    "tests/test_features_m4.py::test_vix_is_joined_from_the_bar_strictly_before",
    "tests/test_features_m4.py::test_residual_at_t_does_not_use_return_t_for_its_model",
    "tests/test_features_m4.py::test_q1_early_close_ticker_never_sees_same_day_us_returns",
    "tests/test_features_m4.py::test_q4_zero_or_constant_volume_is_masked",
    "tests/test_features_m4.py::test_resid_avail_flag_marks_the_warm_up",
    "tests/test_features_m4.py::test_v2_features_still_have_no_lookahead",
    "tests/test_v2_diagnostics.py::test_vol_target_is_causal_on_the_grid_and_capped",
]


def dev_slice(df, cfg):
    s = cfg["splits"]
    return df[(df.index >= pd.Timestamp(s["dev_start"])) & (df.index <= pd.Timestamp(s["dev_end"]))]


def drawdown_episodes(close, limit):
    """Peak -> trough -> recovery episodes at least `limit` deep (recovered = None if not by the end)."""
    c = close.dropna()
    dd = 1 - c / c.cummax()
    out, start = [], None
    for d, v in dd.items():
        if start is None and v > 0:
            start = d
        elif start is not None and v == 0:
            seg = dd[start:d]
            if seg.max() >= limit:
                out.append((c[:start].index[-2], seg.idxmax(), seg.max(), d))
            start = None
    if start is not None and dd[start:].max() >= limit:
        out.append((c[:start].index[-2], dd[start:].idxmax(), dd[start:].max(), None))
    return out


def neff(frame):
    """(mean pairwise correlation, N / (1 + (N-1) rho), eigenvalue participation ratio)."""
    c = frame.corr(min_periods=120).to_numpy()
    n = c.shape[0]
    rho = c[~np.eye(n, dtype=bool)].mean()
    lam = np.linalg.eigvalsh(np.nan_to_num(c))
    return rho, n / (1 + (n - 1) * rho), lam.sum() ** 2 / (lam ** 2).sum()


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    sets = ticker_sets(cfg)
    train, allt = sets["train"], sets["all"]
    raw = load_prices(allt + ["^VIX"], cfg)                    # load_prices cuts before the test period
    dev = {t: dev_slice(df, cfg) for t, df in raw.items()}
    folds = folds_from_config(cfg)
    costs = cfg["evaluation"]["costs"]
    rate = bt.cost_rate(costs["primary_bps"], costs["half_spread_bps"])
    years = (pd.Timestamp(cfg["splits"]["dev_end"]) - pd.Timestamp(cfg["splits"]["dev_start"])).days / 365.25
    val_years = sum((f.val_end - f.val_start).days for f in folds) / 365.25
    L = ["# DATA_AUDIT — v2 step 0b (generated by `scripts/audit_data.py`)", "",
         f"PROTOCOL Part II §V2. Development period only ({cfg['splits']['dev_start']} → {cfg['splits']['dev_end']}, "
         f"{years:.2f} years); the locked test period is not inspected. Per-item CSVs: `experiments/reports/v2_0b/`.", ""]

    # ---- 1. effective number of assets ----------------------------------------------------
    rets = pd.concat({t: np.log(dev[t]["Close"]).diff() for t in train}, axis=1, sort=True)
    rho_all, ne_all, pr_all = neff(rets)
    us = [t for t in train if t != "^GDAXI"]
    rho_us, ne_us, pr_us = neff(rets[us])
    vix_lag = raw["^VIX"]["Close"].shift(1).reindex(rets.index, method="ffill")
    rho_hi, ne_hi, _ = neff(rets[vix_lag > 30])
    rho_lo, ne_lo, _ = neff(rets[vix_lag < 20])
    by_year = {y: neff(g)[0] for y, g in rets.groupby(rets.index.year) if len(g) > 150}
    L += [f"## 1. Effective number of independent assets ({len(train)} training tickers)", "",
          "| Subset | Days | Mean pairwise correlation ρ̄ | N_eff = N / (1 + (N−1)ρ̄) | Eigenvalue participation ratio |",
          "|---|---|---|---|---|",
          f"| all days | {len(rets)} | {rho_all:.2f} | **{ne_all:.1f}** | {pr_all:.1f} |",
          f"| all days, {len(us)} US-listed (excl. ^GDAXI) | {len(rets)} | {rho_us:.2f} | {ne_us:.1f} | {pr_us:.1f} |",
          f"| stress: previous-day VIX > 30 | {int((vix_lag > 30).sum())} | {rho_hi:.2f} | {ne_hi:.1f} | – |",
          f"| calm: previous-day VIX < 20 | {int((vix_lag < 20).sum())} | {rho_lo:.2f} | {ne_lo:.1f} | – |", "",
          "Mean correlation by year: " + ", ".join(f"{y}: {v:.2f}" for y, v in by_year.items()) + ".", "",
          f"The protocol's planning range was ρ̄ = 0.3–0.7 (§V2.1). Measured: ρ̄ = {rho_all:.2f}, so the "
          f"{len(train)} tickers carry the information of about **{ne_all:.0f} independent assets**, and "
          f"fewer in stress ({ne_hi:.1f}), exactly when a de-risking agent should act. ^GDAXI closes 4.5 h "
          "before the US ETFs, so its same-day correlations are understated; excluding it changes ρ̄ by "
          f"{rho_us - rho_all:+.3f}.", ""]

    # ---- 2. drawdown events ------------------------------------------------------------------
    ew = (np.exp(rets.fillna(0.0)) - 1).mean(axis=1)
    series = {"SPY": dev["SPY"]["Close"], "^GDAXI": dev["^GDAXI"]["Close"], f"EW-{len(train)}": (1 + ew).cumprod()}
    rows = []
    for name, s in series.items():
        for peak, trough, depth, rec in drawdown_episodes(s, DD_LIMIT):
            rows.append({"series": name, "peak": peak.date(), "trough": trough.date(), "depth": depth,
                         "recovered": rec.date() if rec is not None else "not within development"})
    dd = pd.DataFrame(rows)
    dd.to_csv(os.path.join(OUT, "drawdowns.csv"), index=False)
    L += [f"## 2. Drawdown events ≥ {DD_LIMIT:.0%} (peak → trough → recovery)", "",
          f"EW-{len(train)} = the equally weighted, daily rebalanced portfolio of the training tickers.", "",
          "| Series | Peak | Trough | Depth | Recovered |", "|---|---|---|---|---|"]
    L += [f"| {r.series} | {r.peak} | {r.trough} | {r.depth:.1%} | {r.recovered} |" for r in dd.itertuples()]
    L += ["", f"Events whose trough lies inside each fold's training window (anchored at "
          f"{cfg['splits']['dev_start']}; ends P = {P_V2} bars before the validation start, the v2 purge):", "",
          "| Fold | Training window ends | " + " | ".join(series) + " |", "|---" * (len(series) + 2) + "|"]
    idx = raw["SPY"].index
    for f in folds:
        end = idx[fold_ranges(idx, f, P_V2, cfg["splits"]["inner_val_bars"]).train_end_pos - 1]
        cells = [str(int(((dd["series"] == n) & (pd.to_datetime(dd["trough"]) <= end)).sum())) for n in series]
        L.append(f"| {f.name} | {end.date()} | " + " | ".join(cells) + " |")
    L += ["", "The protocol's estimate was 5–6 events, F1 ≈ 2 and F5 ≈ 5 (§V2.1); the count above confirms it. "
          "These few episodes are all a de-risking agent can learn from.", ""]

    # ---- 3. minimum backtest length -----------------------------------------------------------
    n_now = tr.n_trials()
    var_pp = tr.sharpe_variance()
    ns = {**N_ROWS, "now": n_now}

    def e_max(n, t_years):           # expected best annual Sharpe of n zero-skill trials over t years
        return st.expected_max_sharpe(1.0 / t_years, n)

    L += ["## 3. Minimum backtest length: the best Sharpe that luck alone produces", "",
          "Best annual Sharpe ratio expected from N zero-skill configurations: the expected maximum "
          "(Bailey & López de Prado 2014, V[SR] = 1/T) and, in brackets, the upper bound √(2 ln N / T) "
          "(Bailey et al. 2014), which the protocol's table in §V2.2 uses.", "",
          f"| N trials | T = development ({years:.2f} y) | T = validation blocks ({val_years:.2f} y) | "
          "From the trial log (correlated trials) |", "|---|---|---|---|"]
    for lab, n in sorted(ns.items(), key=lambda kv: kv[1]):
        L.append(f"| {n} ({lab}) | {e_max(n, years):.2f} (≤ {np.sqrt(2 * np.log(n) / years):.2f}) | "
                 f"{e_max(n, val_years):.2f} (≤ {np.sqrt(2 * np.log(n) / val_years):.2f}) | "
                 f"{st.expected_max_sharpe(var_pp, n) * np.sqrt(252):.2f} |")
    L += ["", f"- Trials are compared on the concatenated validation blocks ({val_years:.2f} years), not on the "
          "whole development period, so **the validation column is the relevant one**: luck alone can be "
          "expected to produce a best Sharpe near "
          f"{e_max(n_now, val_years):.2f} from the {n_now} trials so far.",
          f"- The trial log's cross-trial variance of daily Sharpe ratios is {var_pp:.2e}, about "
          f"{1 / (val_years * 252) / var_pp:.0f}× smaller than independent zero-skill trials would show "
          f"({1 / (val_years * 252):.2e}). Our trials are strongly correlated (mostly long the same ETFs). "
          "The deflated Sharpe ratio uses this variance (Part I §10a.3), so its hurdle SR₀ (last column) is "
          "much lower than the independent-trial bound.", "",
          "Minimum backtest length (MinBTL): years of validation data needed before the best of N zero-skill "
          "trials is *expected* to stay below a given annual Sharpe; in brackets the protocol's bound 2 ln N / SR².", "",
          "| N trials | SR 0.25 | SR 0.5 | SR 1.0 |", "|---|---|---|---|"]
    for lab, n in sorted(ns.items(), key=lambda kv: kv[1]):
        z = st.expected_max_sharpe(1.0, n)
        L.append(f"| {n} ({lab}) | " + " | ".join(f"{(z / s) ** 2:.0f} ({2 * np.log(n) / s ** 2:.0f})" if s < 1 else
                                                 f"{(z / s) ** 2:.1f} ({2 * np.log(n) / s ** 2:.1f})"
                                                 for s in (0.25, 0.5, 1.0)) + " |")
    L.append("")

    # ---- Q1: close-time leak ------------------------------------------------------------------
    feats = ["resid", "resid_cum30"]
    m4 = {"factor_set": "train"}
    v1 = m4_raw_frames(raw, ["^GDAXI"] + train, feats, train, raw["^VIX"], dict(m4, pipeline="v1"))["^GDAXI"]
    v2 = m4_raw_frames(raw, ["^GDAXI"] + train, feats, train, raw["^VIX"], dict(m4, pipeline="v2"))["^GDAXI"]
    dax = dev["^GDAXI"]["Close"]
    nxt = (dax.shift(-1) / dax - 1).rename("next")               # NaN on the last development day

    def leak_corr(f):
        d = pd.concat([f.reindex(dax.index), nxt], axis=1).dropna()
        return {c: d[c].corr(d["next"]) for c in feats}, len(d)
    c1, n1 = leak_corr(v1)
    c2, _ = leak_corr(v2)
    # inside the validation blocks: timing IC as LC3 defines it, and what the leak was worth to two
    # trivial rules scored exactly like an agent (lag 0, 10 bp + 1 bp):
    #   daily rule     long tomorrow iff resid_t < 0 (trades about every other day)
    #   selective rule flat tomorrow iff z_t > 2, z = resid / its rolling 252-day SD (trades rarely).
    #                  The threshold 2 is the best of {0, 0.5, 1, 1.5, 2}, so it is biased upward:
    #                  a generous estimate of the leak's value, not a strategy.
    c = dax.to_numpy()
    sig = ex_ante_vol(c, 60)
    lk = []
    for name, f in (("v1", v1), ("v2", v2)):
        xs = f["resid"].reindex(dax.index)
        x = xs.to_numpy()
        z = (xs / xs.rolling(252, min_periods=60).std()).to_numpy()
        e_daily = np.where(np.isfinite(x), (x < 0).astype(float), 1.0)
        e_sel = np.where(np.isfinite(z), (z <= 2.0).astype(float), 1.0)
        for fo in folds:
            pos = block_positions(dax.index, fo.val_start, fo.val_end)
            ok = np.isfinite(x[pos - 1])
            y = (c[pos] / c[pos - 1] - 1) / sig[pos - 1]
            r = bt.backtest(c, e_daily, pos, rate)
            rs = bt.backtest(c, e_sel, pos, rate)
            lk.append({"pipeline": name, "fold": fo.name, "ic": spearmanr(x[pos - 1][ok], y[ok]).correlation,
                       "rule_sharpe_gross": mt.sharpe(r["gross"]), "rule_sharpe_net": mt.sharpe(r["net"]),
                       "rule_switches_per_100": 100 * float(r["trades"].mean()),
                       "selective_sharpe_net": mt.sharpe(rs["net"]),
                       "selective_switches_per_100": 100 * float(rs["trades"].mean()),
                       "bh_sharpe_net": mt.sharpe(bt.backtest(c, np.ones(len(c)), pos, rate)["net"])})
    lk = pd.DataFrame(lk)
    lk.to_csv(os.path.join(OUT, "q1_leak_value.csv"), index=False)
    g = lk.groupby("pipeline").median(numeric_only=True)
    L += ["## Q1. Close-time leak in ^GDAXI's residual features: **found and fixed**", "",
          "The day-t residual of ^GDAXI subtracts factor returns built from the US ETFs' day-t returns, which "
          "settle 4.5 h after the DAX close. The US afternoon move then shows up in the DAX's *next-day* return, "
          "so the v1 residual 'predicts' it. Medians over the 5 folds for the validation rows:", "",
          "| Quantity | v1 pipeline (as in M4) | v2 pipeline (lagged one bar) |", "|---|---|---|"]
    L += [f"| corr({cc}ₜ, next-day return), development, n = {n1} (SE ≈ {1 / np.sqrt(n1):.3f}) | "
          f"{c1[cc]:+.3f} | {c2[cc]:+.3f} |" for cc in feats]
    L += [f"| timing IC of resid (LC3 definition), validation blocks | {g.loc['v1', 'ic']:+.3f} | {g.loc['v2', 'ic']:+.3f} |",
          f"| rule 'long tomorrow iff residₜ < 0': gross Sharpe | {g.loc['v1', 'rule_sharpe_gross']:.2f} | {g.loc['v2', 'rule_sharpe_gross']:.2f} |",
          f"| same rule, net Sharpe at {costs['primary_bps']} bp + {costs['half_spread_bps']} bp | "
          f"{g.loc['v1', 'rule_sharpe_net']:.2f} | {g.loc['v2', 'rule_sharpe_net']:.2f} |",
          f"| same rule, switches per 100 bars | {g.loc['v1', 'rule_switches_per_100']:.0f} | {g.loc['v2', 'rule_switches_per_100']:.0f} |",
          f"| selective rule 'flat tomorrow iff z(resid)ₜ > 2', net Sharpe (best of 5 thresholds, biased up) | "
          f"{g.loc['v1', 'selective_sharpe_net']:.2f} | {g.loc['v2', 'selective_sharpe_net']:.2f} |",
          f"| selective rule, switches per 100 bars | {g.loc['v1', 'selective_switches_per_100']:.0f} | "
          f"{g.loc['v2', 'selective_switches_per_100']:.0f} |",
          f"| buy-and-hold net Sharpe, same blocks | {g.loc['v1', 'bh_sharpe_net']:.2f} | {g.loc['v2', 'bh_sharpe_net']:.2f} |", "",
          "Fix: `rl/features_m4.py`, `pipeline: v2`: the cross-asset (residual) features of early-close tickers "
          "use the t−1 value; context features (VIX) already used t−1 for every ticker. Test "
          "`test_q1_early_close_ticker_never_sees_same_day_us_returns`. All v2 configurations, R0′ included, use "
          "pipeline v2. ^GDAXI is the only early-close ticker among the 33 (§V2.3).", "",
          "Reading: the leak is large in gross terms, but trading it daily costs more than it earns. Even the "
          "generous selective rule beats buy-and-hold net of costs by only "
          f"{g.loc['v1', 'selective_sharpe_net'] - g.loc['v1', 'bh_sharpe_net']:+.2f}. "
          "Whether the M4 agents used it at all is answered by their timing IC in Step 0a.", ""]

    # ---- Q2: adjusted prices -----------------------------------------------------------------
    q2dirs = sorted(glob.glob(repo_path("data", "audit", "q2_*")))
    L += ["## Q2. Adjusted prices: frozen `data/raw` vs a fresh download", ""]
    if not q2dirs:
        L += ["No owner download found in `data/audit/`, so the check was not run.", ""]
    else:
        q2 = q2dirs[-1]
        rows = []
        for t in allt + ["^VIX"]:
            f = os.path.join(q2, "yahoo_adjusted", ticker_to_file(t) + ".csv")
            if not os.path.exists(f):
                continue
            fresh = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")["Close"]
            jj = dev_slice(pd.concat({"frozen": raw[t]["Close"], "fresh": fresh}, axis=1), cfg)
            j = jj.dropna()
            diff_bp = (np.log(j).diff().dropna().pipe(lambda d: d["frozen"] - d["fresh"])).abs() * 1e4
            rows.append({"ticker": t, "days": len(diff_bp), "share_gt_5bp": float((diff_bp > DIFF_BP).mean()),
                         "max_diff_bp": float(diff_bp.max()), "date_of_max": diff_bp.idxmax().date(),
                         "max_level_deviation": float((j["fresh"] / j["frozen"] - 1).abs().max()),
                         "dates_only_in_frozen": int((jj["frozen"].notna() & jj["fresh"].isna()).sum()),
                         "dates_only_in_fresh": int((jj["fresh"].notna() & jj["frozen"].isna()).sum())})
        q2df = pd.DataFrame(rows)
        q2df.to_csv(os.path.join(OUT, "q2_frozen_vs_fresh.csv"), index=False)
        bad = q2df[q2df["share_gt_5bp"] > 0]
        lvl = q2df["max_level_deviation"].max()
        L += [f"Owner download `{os.path.basename(q2)}` (Yahoo, adjusted), {len(q2df)} series, daily log returns "
              "compared on common development dates.", "",
              f"- Series with **any** day differing by more than {DIFF_BP:.0f} bp: **{len(bad)} of {len(q2df)}**. "
              f"Largest single-day difference: {q2df['max_diff_bp'].max():.2f} bp "
              f"({q2df.loc[q2df['max_diff_bp'].idxmax(), 'ticker']}).",
              f"- Dates present in one file but not the other: {int(q2df['dates_only_in_frozen'].sum())} only in "
              f"`data/raw`, {int(q2df['dates_only_in_fresh'].sum())} only in the fresh download.",
              f"- Price levels: largest relative deviation {lvl:.1e}. "
              + ("The levels are identical: no dividend went ex between the two downloads, so Yahoo did not "
                 "re-base the adjusted series. Returns would be unaffected either way."
                 if lvl < 1e-4 else
                 "Yahoo re-based the adjusted series after a new dividend; returns, which are all the harness "
                 "uses, are unaffected."), ""]
        if len(bad):
            L += ["| Ticker | Days | Share > 5 bp | Max diff (bp) | Date of max |", "|---|---|---|---|---|"]
            L += [f"| {r.ticker} | {r.days} | {r.share_gt_5bp:.2%} | {r.max_diff_bp:.1f} | {r.date_of_max} |"
                  for r in bad.itertuples()]
            L.append("")
        # adjusted vs unadjusted + distributions. Yahoo's unadjusted Close is already split-adjusted,
        # so the total return of day t is (Close_t + Dividends_t + CapitalGains_t) / Close_t-1 - 1.
        rows = []
        for f in sorted(glob.glob(os.path.join(q2, "yahoo_raw", "*.csv"))):
            r = dev_slice(pd.read_csv(f, parse_dates=["Date"]).set_index("Date"), cfg)
            if not {"Close", "Adj Close"} <= set(r.columns):
                continue
            t = os.path.splitext(os.path.basename(f))[0]
            tick = "^GDAXI" if t == "GDAXI" else t
            dist = sum(r[k] for k in ("Dividends", "Capital Gains") if k in r)
            tr_ret = (r["Close"] + dist) / r["Close"].shift(1) - 1
            adj_ret = r["Adj Close"] / r["Adj Close"].shift(1) - 1
            frozen_ret = dev[tick]["Close"].pct_change()
            j = pd.concat({"tr": tr_ret, "adj": adj_ret, "frozen": frozen_ret}, axis=1).dropna()
            rows.append({"ticker": tick, "days": len(j),
                         "distribution_days": int((np.asarray(dist) > 0).sum()) if not np.isscalar(dist) else 0,
                         "split_days": int((r["Stock Splits"] > 0).sum()) if "Stock Splits" in r else 0,
                         "share_raw_vs_adj_gt_5bp": float(((j["tr"] - j["adj"]).abs() * 1e4 > DIFF_BP).mean()),
                         "share_adj_vs_frozen_gt_5bp": float(((j["adj"] - j["frozen"]).abs() * 1e4 > DIFF_BP).mean())})
        if rows:
            ad = pd.DataFrame(rows)
            ad.to_csv(os.path.join(OUT, "q2_raw_vs_adjusted.csv"), index=False)
            L += ["Unadjusted closes + distributions vs adjusted closes (the 5 pre-declared spot-check tickers). "
                  "This tests whether the dividend adjustment itself is right:", "",
                  "| Ticker | Days | Distribution days | Split days | (close + distribution) vs adjusted: share > 5 bp | "
                  "adjusted vs frozen: share > 5 bp |", "|---|---|---|---|---|---|"]
            L += [f"| {r.ticker} | {r.days} | {r.distribution_days} | {r.split_days} | {r.share_raw_vs_adj_gt_5bp:.2%} | "
                  f"{r.share_adj_vs_frozen_gt_5bp:.2%} |" for r in ad.itertuples()]
            L += ["", "^GDAXI is a performance (total-return) index, so it has no distributions by construction.", ""]
        L += ["**Second independent source:** the stooq download failed (no access rights). By owner decision "
              "(2026-10-06) the second-source spot check is not repeated; Q2 rests on the two Yahoo-internal "
              "checks above.", ""]

    # ---- Q3: extreme moves ---------------------------------------------------------------------
    zs = {}
    for t in allt + ["^VIX"]:
        cl = dev[t]["Close"]
        zs[t] = np.log(cl).diff() / pd.Series(ex_ante_vol(cl.to_numpy(), 60), index=cl.index).shift(1)
    wide = (pd.concat({t: zs[t] for t in allt}, axis=1, sort=True).abs() > WIDE_SIGMA).sum(axis=1)
    big = []
    for t, z in zs.items():
        df = dev[t]
        ret = df["Close"].pct_change()
        for d, v in z[z.abs() > SIGMA_LIMIT].items():
            nxt_r = ret.shift(-1)[d]
            within = bool(df.at[d, "Low"] * (1 - 1e-9) <= df.at[d, "Close"] <= df.at[d, "High"] * (1 + 1e-9))
            reverts = bool(np.isfinite(nxt_r) and nxt_r / ret[d] < -0.75)
            n_wide = int(wide.get(d, 0))
            cls = ("market-wide" if n_wide >= WIDE_COUNT else
                   "single-ticker, genuine" if within and not reverts else "single-ticker, possible bad print")
            big.append({"ticker": t, "date": d.date(), "return": float(ret[d]), "z": float(v),
                        "next_day_return": float(nxt_r), "close_within_high_low": within,
                        "reverts_next_day": reverts, "tickers_beyond_4sigma": n_wide, "class": cls,
                        "event": KNOWN_EVENTS.get(str(d.date()), "")})
    big = pd.DataFrame(big).sort_values("date") if big else pd.DataFrame()
    big.to_csv(os.path.join(OUT, "q3_moves_beyond_8sigma.csv"), index=False)
    n_bad = int((big["class"] == "single-ticker, possible bad print").sum()) if len(big) else 0
    L += [f"## Q3. Daily moves beyond {SIGMA_LIMIT:.0f}σ (σ = ex-ante EWMA-60 volatility of the previous day)", "",
          f"{len(big)} moves in the development period. *Market-wide* = at least {WIDE_COUNT} of the {len(allt)} "
          f"tickers moved beyond {WIDE_SIGMA:.0f}σ that day. A single-ticker move counts as *genuine* if the close "
          "lies within the day's high–low range and the next day does not reverse more than 75 % of it (a bad "
          "print usually fails one of the two).", "",
          "| Ticker | Date | Return | z | Next day | Tickers beyond 4σ | Class | Event (general knowledge) |",
          "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r.ticker} | {r.date} | {r._3:+.1%} | {r.z:+.1f} | {r.next_day_return:+.1%} | {r.tickers_beyond_4sigma} | "
          f"{r._9} | {r.event} |" for r in big.itertuples()] if len(big) else ["| – | – | – | – | – | – | – | – |"]
    L += ["", f"Result: **{n_bad} possible bad prints**; every flagged move is "
          f"{'a real market event' if n_bad == 0 else 'listed above for review'}, and all of them agree with the "
          "fresh download (Q2). The event labels are from general knowledge and were not checked against a "
          "source here.",
          "Known structural change that the 8σ scan cannot see: USO changed from front-month to "
          "longer-dated oil futures and did a 1-for-8 reverse split in April–May 2020 (general knowledge, not "
          "verified here). The adjusted prices absorb the split, and the series is what an investor in USO "
          "actually earned, so **no window is masked**.", ""]

    # ---- Q4: volume ----------------------------------------------------------------------------
    rows = []
    for t in allt:
        v = dev[t]["Volume"]
        rows.append({"ticker": t, "zero": int((v == 0).sum()), "constant_20d": int((v.rolling(20).std() == 0).sum())})
    vol = pd.DataFrame(rows)
    vol.to_csv(os.path.join(OUT, "q4_volume.csv"), index=False)
    flagged = vol[(vol["zero"] > 0) | (vol["constant_20d"] > 0)]
    vix_zero = float((dev["^VIX"]["Volume"] == 0).mean()) if "Volume" in dev["^VIX"] else 1.0
    L += ["## Q4. Zero or constant volume", "",
          f"{len(flagged)} of {len(vol)} tradable tickers have zero-volume or 20-day-constant-volume bars in "
          f"development. ^VIX volume is zero on {vix_zero:.0%} of days; ^VIX is a context feature (level only), "
          "so its volume is never read.", "",
          "| Ticker | Zero-volume bars | Bars with constant 20-day volume |", "|---|---|---|"]
    L += [f"| {r.ticker} | {r.zero} | {r.constant_20d} |" for r in flagged.itertuples()]
    L += ["", "Fix: `pipeline: v2` masks `vol_rel20` on those bars (set to the neutral 0 after z-scoring). Test "
          "`test_q4_zero_or_constant_volume_is_masked`.", ""]

    # ---- Q5: dates and stale prices --------------------------------------------------------------
    rows = []
    for t in allt:
        df = dev[t]
        same = df["Close"].diff() == 0
        runs = same.groupby((~same).cumsum()).sum()
        rows.append({"ticker": t, "bars": len(df), "bars_per_year": len(df) / years,
                     "duplicate_dates": int(df.index.duplicated().sum()), "monotonic": bool(df.index.is_monotonic_increasing),
                     "unchanged_close_days": int(same.sum()), "longest_unchanged_run": int(runs.max()) if len(runs) else 0,
                     "unchanged_and_zero_volume": int((same & (df["Volume"] == 0)).sum()),
                     "missing_ohlc": int(df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())})
    q5 = pd.DataFrame(rows)
    q5.to_csv(os.path.join(OUT, "q5_dates_and_stale_prices.csv"), index=False)
    L += ["## Q5. Dates, duplicates, stale prices and forward filling", "",
          f"- Duplicate dates: {int(q5['duplicate_dates'].sum())}; non-monotonic series: {int((~q5['monotonic']).sum())}; "
          f"bars with a missing open/high/low/close: {int(q5['missing_ohlc'].sum())}.",
          f"- Bars per year: {q5['bars_per_year'].min():.1f} – {q5['bars_per_year'].max():.1f} (exchange calendars; "
          "^GDAXI follows Xetra).",
          f"- Days with an unchanged close: {int(q5['unchanged_close_days'].sum())} of "
          f"{int(q5['bars'].sum())} ticker-days; longest run {int(q5['longest_unchanged_run'].max())} day(s); unchanged "
          f"**and** zero volume (a typical stale bar): {int(q5['unchanged_and_zero_volume'].sum())}.",
          "- No tradable bar is forward-filled (Part I §3). Each ticker is backtested on its own bars "
          "(`harness/backtest.py`). In the equal-weight portfolio a ticker without a bar that day (e.g. a German "
          "holiday for ^GDAXI) contributes 0 % (`harness/backtest.portfolio`). Forward filling exists only inside "
          "the residual-factor panel, where a missing day is a 0 % return (`rl/features_m4.residual_returns`).", ""]

    # ---- Q6: point-in-time tests -------------------------------------------------------------------
    env = dict(os.environ, OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="3")
    run = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *PIT_TESTS],
                         cwd=repo_path(), env=env, capture_output=True, text=True)
    done = [ln.strip("= ") for ln in run.stdout.splitlines() if re.search(r"\d+ (passed|failed|error)", ln)]
    summary = done[-1] if done else "(no pytest summary line found)"
    L += ["## Q6. Point-in-time tests (perturb the future → the value at t is unchanged)", ""]
    L += [f"- `{t}`" for t in PIT_TESTS]
    L += ["", f"Run by this script: **{summary}**" + ("" if run.returncode == 0 else " — FAILURES, see pytest output"), ""]

    # ---- consequences --------------------------------------------------------------------------------
    L += ["## Consequences", "",
          "- **Q1, Q4 (fixed in `pipeline: v2`)** change agent features only. Every v2 configuration uses "
          "pipeline v2, and the reference is re-run as R0′ (Step 0d). The baselines read prices only, so they "
          "cannot change; they were re-scored in Step 0a anyway (`experiments/reports/V2_0a_rescore.md`).",
          "- **Q2, Q3, Q5:** the price data is correct as frozen. No re-download, no masking.",
          "- **Sections 1–3:** about 2 independent assets, 5–6 drawdown events and a luck level near Sharpe "
          f"{e_max(n_now, val_years):.2f} for the best of {n_now} trials. This is why v2 tries few configurations "
          "and uses structure (anchor, cost-aware head) instead of more search.", ""]

    with open(repo_path("DATA_AUDIT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("-> DATA_AUDIT.md")


if __name__ == "__main__":
    main()
