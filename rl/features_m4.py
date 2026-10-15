"""M4 feature pipeline (spec Phase 4 + plan §7 Track A1). Fully causal.

Every feature value at bar t is computed from data up to and including bar t
only. Nothing is fitted on a training range: normalisation is a TRAILING
z-score (spec Phase 4: "rolling z-score that uses past data only"), so the
same feature matrix is valid for every fold. tests/test_features_m4.py
perturbs future prices and checks that no past value changes.

Per-ticker features (names used in configs)
-------------------------------------------
    log_ret       log(C_t / C_t-1)
    p_sma20       C / SMA20(C) - 1
    p_sma50       C / SMA50(C) - 1
    bb_pctb       Bollinger %B: (C - (SMA20 - 2 sd20)) / (4 sd20)
    rsi14         Wilder RSI(14) / 100
    macd_hist     (MACD(12, 26) - signal(9)) / C
    vol_rel20     log(V / SMA20(V)); 0 when volume is missing or zero
    atr14_p       Wilder ATR(14) / C
    sigma20       rolling std of log returns over 20 bars
Context (^VIX), joined AS-OF WITH A ONE-BAR LAG (spec W8):
    vix_lag1      log VIX close of the last VIX bar strictly BEFORE bar t
    vix_chg_lag1  change of that log VIX against the VIX bar before it
    The VIX closes at 16:15 New York time, after the close of most bars here
    (the DAX closes 6 hours earlier), so only the previous VIX bar is known.
Residual features (plan §7 Track A1, after Guijarro-Ordonez, Pelger & Zanotti 2026):
    resid         out-of-sample residual return of the ticker at bar t after
                  removing its exposure to the first k principal components
                  of the factor universe (default: the 26 training ETFs)
    resid_cum30   sum of the last 30 residual returns (the paper's L = 30)
    At each date t:
      1. correlation-matrix PCA of the factor tickers' log returns over the
         `corr_window` days BEFORE t -> k eigenportfolios (weights v / sigma)
      2. factor returns of those eigenportfolios over the `beta_window` days
         before t; OLS (no intercept) of each ticker's returns on them -> betas
      3. residual at t = r_t - betas . F_t, i.e. strictly out of sample.
    Tickers outside the factor universe (the leave-out set) get residuals
    against the same factors; their own returns never enter the PCA.

All features are then z-scored with a trailing window, clipped to +-clip, and
NaN warm-up values are set to 0.

Pipeline v2 (params `pipeline: v2`; PROTOCOL Part II §V2.3 and §V6.0). Off by
default, so every v1 trial keeps exactly its features:
    Q1  close-time lag: for tickers whose exchange closes BEFORE the US close
        (`early_close_tickers`, default ["^GDAXI"]), the cross-asset features
        (resid, resid_cum30) use the previous bar's value. Their day-t value
        contains US returns of day t, which settle 4.5 h after the DAX close.
    Q4  relative volume is masked (set to unavailable) where volume is zero or
        constant over the 20-bar window (index volumes are unreliable).
    flag  resid_avail = 1 once both residual features have a valid z-score AND the
        ticker has `warmup_bars` (default 450) bars of history; before that both
        residual features are 0 (warm-up rule, §V0 item 6). Binary, not z-scored.
    flag  vix_avail = 1 where both VIX features are known, else 0 and both are 0 (VIX
        exists only from 1990; the long-history pretraining of Step 4, §V6.4).
    z   the trailing z-score needs `z_min_periods` = 126 bars (default for v2;
        expanding until the window is full). The window stays 252 bars as in M4.

V3 signal pack (PROTOCOL Part II §V6.3; §V12.1 item 26), all optional by name:
  D2 per ticker, return-derived (also in the long history):
    tsr21 .. tsr252   log return over h bars / (EWMA-60 daily vol * sqrt(h)), h = 21, 63, 126, 252
    macd8/16/32       (EWMA_S - EWMA_3S)(close) / 63-bar SD of the close, S = 8, 16, 32 (Lim et al. 2019)
    cp21_*, cp126_*   Bayesian online changepoint detection (Adams & MacKay 2007) on r_t / sigma_t-1 with a
                      Normal-Gamma prior (0, 1, 1, 1) and hazard 1/lambda, lambda = 21, 126; run length
                      truncated at 500: `recent` = P(run length <= 5), `age` = min(E[run length] / lambda, 1)
    flag sig_avail    1 once the ticker has `warmup_bars` of history; before that D2 is 0
  D1 cross-asset context, from TRAINING tickers only (IEF is a leave-out ticker, so TIP replaces it):
    bond_mom21/63/252 mean log return of TLT and TIP;  eq_mom21/63/252 the same for SPY and EFA
    credit20/60       log return LQD - TIP;  term20/60  log return TLT - TIP
    breadth           share of the factor tickers above their 200-bar SMA
    dispersion        cross-sectional SD of the factor tickers' 21-bar log returns
    corr60            average pairwise correlation of the factor tickers' daily returns over 60 bars
    rank63, rank252   share of factor tickers whose 63/252-bar return is below the ticker's own
    flag ctx_avail    1 where all context inputs exist (never in the long history, which has no ETFs)
    The bounded features (changepoint scores, breadth, ranks; all in [0, 1]) are mapped to 2x - 1 instead of being
    z-scored: a saturated score is constant for long stretches, where a trailing z-score is undefined.
    Context values are joined AS-OF: the same day for tickers that close with the US market, the previous US
    bar for early-close tickers (Q1), exactly like the VIX features.
"""

import numpy as np
import pandas as pd

TREND_FEATURES = ("tsr21", "tsr63", "tsr126", "tsr252", "macd8", "macd16", "macd32",
                  "cp21_recent", "cp21_age", "cp126_recent", "cp126_age")                     # D2
CTX_FEATURES = ("bond_mom21", "bond_mom63", "bond_mom252", "eq_mom21", "eq_mom63", "eq_mom252", "credit20",
                "credit60", "term20", "term60", "breadth", "dispersion", "corr60", "rank63", "rank252")   # D1
CTX_TICKERS = ("TLT", "TIP", "LQD", "SPY", "EFA")
M4_FEATURES = (["log_ret", "p_sma20", "p_sma50", "bb_pctb", "rsi14", "macd_hist", "vol_rel20",
                "atr14_p", "sigma20", "vix_lag1", "vix_chg_lag1", "resid", "resid_cum30", "resid_avail",
                "vix_avail"] + list(TREND_FEATURES) + ["sig_avail"] + list(CTX_FEATURES) + ["ctx_avail"])
FLAG_FEATURES = ("resid_avail", "vix_avail", "sig_avail", "ctx_avail")   # binary availability flags: never z-scored
# already bounded in [0, 1] (and often constant for long stretches, e.g. a saturated regime age, where a trailing
# z-score would divide by zero): mapped to 2x - 1 instead of z-scored
BOUNDED_FEATURES = ("cp21_recent", "cp21_age", "cp126_recent", "cp126_age", "breadth", "rank63", "rank252")
RESID_FEATURES = ("resid", "resid_cum30")
VIX_FEATURES = ("vix_lag1", "vix_chg_lag1")


# ---------------------------------------------------------------------------
# per-ticker indicators
# ---------------------------------------------------------------------------
def indicators(df):
    """Raw (un-normalised) per-ticker features from OHLCV, indexed like df."""
    c, h, l, v = (df[k].astype(float) for k in ("Close", "High", "Low", "Volume"))
    out = pd.DataFrame(index=df.index)
    lr = np.log(c).diff()
    out["log_ret"] = lr
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    out["p_sma20"] = c / sma20 - 1.0
    out["p_sma50"] = c / sma50 - 1.0
    sd20 = c.rolling(20).std()
    out["bb_pctb"] = ((c - (sma20 - 2 * sd20)) / (4 * sd20)).where(sd20 > 0)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 1.0 - 1.0 / (1.0 + gain / loss)
    out["rsi14"] = rsi.where(loss > 0, 1.0)
    out.loc[out.index[:14], "rsi14"] = np.nan                      # Wilder warm-up
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    out.loc[out.index[:35], "macd_hist"] = np.nan                  # 26 + 9 warm-up
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = np.log(v / v.rolling(20).mean())
    out["vol_rel20"] = vr.replace([np.inf, -np.inf], 0.0).where(v.rolling(20).count() == 20)
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    out["atr14_p"] = (atr / c).where(prev.notna())
    out.loc[out.index[:14], "atr14_p"] = np.nan
    out["sigma20"] = lr.rolling(20).std()
    return out


# ---------------------------------------------------------------------------
# VIX, as-of with a one-bar lag
# ---------------------------------------------------------------------------
def vix_features(dates, vix):
    """vix_lag1 / vix_chg_lag1 for each date: values of the last VIX bar strictly before it."""
    vdates = pd.DatetimeIndex(vix.index)
    lv = np.log(vix["Close"].to_numpy(dtype=float))
    idx = vdates.searchsorted(pd.DatetimeIndex(dates), side="left") - 1    # strictly before
    lag1 = np.where(idx >= 0, lv[np.maximum(idx, 0)], np.nan)
    lag2 = np.where(idx >= 1, lv[np.maximum(idx - 1, 0)], np.nan)
    return pd.DataFrame({"vix_lag1": lag1, "vix_chg_lag1": lag1 - lag2}, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# residual returns vs PCA factors (out of sample)
# ---------------------------------------------------------------------------
def residual_returns(closes, factor_tickers, k=3, corr_window=252, beta_window=60):
    """Out-of-sample residual log returns, a DataFrame dates x tickers.

    closes : DataFrame (union of dates x tickers) of closing prices; a ticker's
             missing dates (holidays) are forward-filled, i.e. 0 return.
    """
    px = closes.sort_index().ffill()
    R = np.log(px).diff()
    ret = R.to_numpy(dtype=float)
    fcols = [closes.columns.get_loc(t) for t in factor_tickers]
    T, N = ret.shape
    out = np.full((T, N), np.nan)
    start = max(corr_window, beta_window) + 1
    for t in range(start, T):
        win = ret[t - corr_window:t][:, fcols]                  # days BEFORE t
        ok = ~np.isnan(win).any(axis=0)
        if ok.sum() < k + 2:
            continue
        w = win[:, ok]
        sd = w.std(axis=0, ddof=1)
        good = sd > 1e-12
        if good.sum() < k + 2:
            continue
        w, sd = w[:, good], sd[good]
        z = (w - w.mean(axis=0)) / sd
        _, _, vt = np.linalg.svd(z, full_matrices=False)        # rows = eigenvectors of the correlation matrix
        q = vt[:k] / sd                                         # eigenportfolio weights on raw returns
        cols = np.array(fcols)[ok][good]
        f_hist = ret[t - beta_window:t][:, cols] @ q.T          # (beta_window, k) factor returns before t
        f_now = ret[t, cols] @ q.T                              # (k,)
        if np.isnan(f_hist).any() or np.isnan(f_now).any():
            continue
        y = ret[t - beta_window:t]                              # (beta_window, N) all tickers
        have = ~np.isnan(y).any(axis=0) & ~np.isnan(ret[t])
        if not have.any():
            continue
        beta, *_ = np.linalg.lstsq(f_hist, y[:, have], rcond=None)   # (k, n_have)
        out[t, have] = ret[t, have] - f_now @ beta
    return pd.DataFrame(out, index=px.index, columns=closes.columns)


# ---------------------------------------------------------------------------
# V3 signal pack: D2 trend + changepoints (per ticker), D1 cross-asset context
# ---------------------------------------------------------------------------
def bocpd(x, lam, rmax=500, prior=(0.0, 1.0, 1.0, 1.0)):
    """Bayesian online changepoint detection (Adams & MacKay 2007), filtering only (causal).

    x: 1-D series (NaN = no update). Gaussian observations with unknown mean and precision, Normal-Gamma
    prior (mu0, kappa0, alpha0, beta0), constant hazard 1/lam, run length truncated at rmax.
    Returns (P(run length <= 5), min(E[run length] / lam, 1)) per bar.
    """
    from scipy.special import gammaln
    mu0, k0, a0, b0 = prior
    h = 1.0 / lam
    p = np.array([1.0]); mu, kap, al, be = (np.array([v]) for v in prior)
    recent, age = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for t, v in enumerate(x):
        if np.isfinite(v):
            nu, sc2 = 2 * al, be * (kap + 1) / (al * kap)              # Student-t predictive of each run length
            lp = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi * sc2)
                  - (nu + 1) / 2 * np.log1p((v - mu) ** 2 / (nu * sc2)))
            w = p * np.exp(lp - lp.max())
            p = np.concatenate(([w.sum() * h], w * (1 - h)))[:rmax + 1]   # changepoint, growth; truncate
            p /= p.sum()
            mu, kap, al, be = (np.concatenate(([mu0], (kap * mu + v) / (kap + 1))),
                               np.concatenate(([k0], kap + 1)), np.concatenate(([a0], al + 0.5)),
                               np.concatenate(([b0], be + kap * (v - mu) ** 2 / (2 * (kap + 1)))))
            mu, kap, al, be = mu[:rmax + 1], kap[:rmax + 1], al[:rmax + 1], be[:rmax + 1]
        recent[t] = p[:6].sum()
        age[t] = min((np.arange(len(p)) * p).sum() / lam, 1.0)
    return recent, age


def trend_features(df):
    """D2 raw features of one ticker (multi-horizon vol-scaled trend, MACD at three scales, BOCPD)."""
    c = df["Close"].astype(float)
    lr = np.log(c).diff()
    sig = lr.ewm(span=60, min_periods=30).std()
    out = pd.DataFrame(index=df.index)
    for h in (21, 63, 126, 252):
        out[f"tsr{h}"] = np.log(c / c.shift(h)) / (sig * np.sqrt(h))
    for s_ in (8, 16, 32):
        macd = c.ewm(span=s_, adjust=False).mean() - c.ewm(span=3 * s_, adjust=False).mean()
        out[f"macd{s_}"] = (macd / c.rolling(63).std()).where(np.arange(len(c)) >= 3 * s_)
    x = (lr / sig.shift(1)).to_numpy()
    for lam in (21, 126):
        out[f"cp{lam}_recent"], out[f"cp{lam}_age"] = bocpd(x, lam)
    return out


def context_frame(prices, factor_tickers):
    """D1 global context on the union of the context tickers' dates, plus the factor tickers' 63/252-bar
    returns (for the ranks); None if a context ticker is missing (e.g. the long history)."""
    if not all(t in prices for t in CTX_TICKERS):
        return None, None
    names = sorted(set(factor_tickers) | set(CTX_TICKERS))
    C = pd.concat({t: prices[t]["Close"] for t in names}, axis=1).sort_index().ffill()
    L = np.log(C)
    ret = lambda t, h: L[t] - L[t].shift(h)                                          # noqa: E731
    out = pd.DataFrame(index=C.index)
    for h in (21, 63, 252):
        out[f"bond_mom{h}"] = (ret("TLT", h) + ret("TIP", h)) / 2
        out[f"eq_mom{h}"] = (ret("SPY", h) + ret("EFA", h)) / 2
    for h in (20, 60):
        out[f"credit{h}"] = ret("LQD", h) - ret("TIP", h)
        out[f"term{h}"] = ret("TLT", h) - ret("TIP", h)
    F = C[list(factor_tickers)]
    out["breadth"] = (F > F.rolling(200).mean()).astype(float).where(F.rolling(200).count() == 200).mean(axis=1)
    out["dispersion"] = (np.log(F) - np.log(F.shift(21))).std(axis=1)
    R = np.log(F).diff()
    z = R / R.rolling(60).std()
    n = z.notna().sum(axis=1)
    out["corr60"] = ((n * z.mean(axis=1)).rolling(60).var() / n - 1) / (n - 1)       # Var(mean z) = (1 + (n-1) rho) / n
    rets = {h: np.log(F) - np.log(F.shift(h)) for h in (63, 252)}
    return out, rets


def _asof(frame, dates, strict):
    """Rows of `frame` (dates x cols) as of each date: the last row on or (strict) before it."""
    idx = frame.index.searchsorted(pd.DatetimeIndex(dates), side="left" if strict else "right") - 1
    vals = frame.to_numpy(dtype=float)[np.maximum(idx, 0)]
    vals[idx < 0] = np.nan
    return pd.DataFrame(vals, index=pd.DatetimeIndex(dates), columns=frame.columns)


# ---------------------------------------------------------------------------
# normalisation and assembly
# ---------------------------------------------------------------------------
def rolling_z(frame, window=252, min_periods=60):
    """Trailing z-score: (x_t - mean(x_{t-w+1..t})) / std(...). Past data only."""
    mean = frame.rolling(window, min_periods=min_periods).mean()
    std = frame.rolling(window, min_periods=min_periods).std()
    return (frame - mean) / std.where(std > 0)


def _params(params):
    p = {"pca_k": 3, "corr_window": 252, "beta_window": 60, "cum_window": 30,
         "z_window": 252, "z_min_periods": 60, "clip": 5.0, "pipeline": "v1", **(params or {})}
    v2 = p["pipeline"] == "v2"
    p.setdefault("early_close_tickers", ["^GDAXI"] if v2 else [])
    p.setdefault("mask_volume", v2)
    p.setdefault("warmup_bars", 450 if v2 else 0)          # §V0 item 6: long-lookback features
    if v2 and "z_min_periods" not in (params or {}):
        p["z_min_periods"] = 126                           # §V0 item 6: expanding z-score, >= 126 bars
    return p


def m4_raw_frames(prices, tickers, features, factor_tickers, vix=None, params=None):
    """{ticker: DataFrame of RAW feature values} (before normalisation).

    Separated from build_m4_features so tests can check the raw values for
    NaN after the warm-up (the final matrix replaces warm-up NaN with 0).
    """
    p = _params(params)
    unknown = [f for f in features if f not in M4_FEATURES]
    if unknown:
        raise KeyError(f"unknown M4 features {unknown}")
    if any(f in VIX_FEATURES for f in features) and vix is None:
        raise ValueError("VIX features requested but no ^VIX data given")
    if "resid_avail" in features and not all(f in features for f in RESID_FEATURES):
        raise ValueError("resid_avail needs both residual features")
    if "vix_avail" in features and not all(f in features for f in VIX_FEATURES):
        raise ValueError("vix_avail needs both VIX features")
    features = [f for f in features if f not in FLAG_FEATURES]       # flags are derived after z-scoring
    early = set(p["early_close_tickers"])
    ctx, ctx_rets = (context_frame(prices, factor_tickers) if any(f in CTX_FEATURES for f in features)
                     else (None, None))
    resid = None
    if any(f in RESID_FEATURES for f in features):
        names = sorted(set(tickers) | set(factor_tickers))
        closes = pd.concat({t: prices[t]["Close"] for t in names}, axis=1)
        resid = residual_returns(closes, list(factor_tickers), p["pca_k"], p["corr_window"], p["beta_window"])
    out = {}
    for t in tickers:
        df = prices[t]
        raw = indicators(df)
        if vix is not None:
            raw = raw.join(vix_features(df.index, vix))
        if resid is not None:
            r = resid[t].reindex(df.index)
            raw["resid"] = r
            raw["resid_cum30"] = r.rolling(p["cum_window"], min_periods=p["cum_window"]).sum()
            if t in early:
                # Q1: the day-t residual uses US returns of day t, unknown at this ticker's close
                raw[["resid", "resid_cum30"]] = raw[["resid", "resid_cum30"]].shift(1)
        if p["mask_volume"] and "vol_rel20" in raw:
            v = df["Volume"].astype(float)
            raw["vol_rel20"] = raw["vol_rel20"].where((v > 0) & (v.rolling(20).std() > 0))
        if any(f in TREND_FEATURES for f in features):
            raw = raw.join(trend_features(df))
        if any(f in CTX_FEATURES for f in features):
            if ctx is None:                                   # no context tickers (long history): unavailable
                raw = raw.join(pd.DataFrame(np.nan, index=df.index, columns=list(CTX_FEATURES)))
            else:
                strict = t in early                           # Q1: early-close tickers see the previous US bar
                raw = raw.join(_asof(ctx, df.index, strict))
                own = np.log(df["Close"].astype(float))
                for h, R in ctx_rets.items():
                    mine = (own - own.shift(h)).to_numpy()[:, None]
                    peers = _asof(R, df.index, strict).to_numpy()
                    raw[f"rank{h}"] = np.where(np.isfinite(mine[:, 0]),
                                               (peers < mine).sum(axis=1) / np.isfinite(peers).sum(axis=1).clip(1), np.nan)
        out[t] = raw[features]
    return out


def build_m4_features(prices, tickers, features, factor_tickers, vix=None, params=None):
    """{ticker: (T_i, F) float32} aligned with prices[ticker].index.

    prices         : {ticker: OHLCV DataFrame indexed by date}; must contain
                     every ticker in `tickers` and in `factor_tickers`
    features       : feature names from M4_FEATURES (order = column order)
    factor_tickers : universe used for the residual PCA (training ETFs)
    vix            : ^VIX DataFrame (required if a VIX feature is requested)
    params         : pca_k, corr_window, beta_window, cum_window, z_window,
                     z_min_periods, clip, pipeline (v1 | v2), early_close_tickers,
                     mask_volume, warmup_bars
    """
    p = _params(params)
    out = {}
    for t, raw in m4_raw_frames(prices, tickers, features, factor_tickers, vix, p).items():
        z = rolling_z(raw, p["z_window"], p["z_min_periods"])
        z = z.clip(-p["clip"], p["clip"])
        for f in (f for f in BOUNDED_FEATURES if f in z):
            z[f] = 2.0 * raw[f] - 1.0
        if "resid_avail" in features:
            # the residual GROUP is available only when both members are and the ticker
            # has warmup_bars of history; until then both stay 0 and the flag is 0
            # (PROTOCOL Part II §V0 item 6)
            ok = z["resid"].notna() & z["resid_cum30"].notna()
            ok &= np.arange(len(z)) >= int(p["warmup_bars"])
            z.loc[~ok, list(RESID_FEATURES)] = np.nan
            z["resid_avail"] = ok.astype(float)
        if "vix_avail" in features:
            ok = z[list(VIX_FEATURES)].notna().all(axis=1)
            z.loc[~ok, list(VIX_FEATURES)] = np.nan
            z["vix_avail"] = ok.astype(float)
        for flag, group in (("sig_avail", TREND_FEATURES), ("ctx_avail", CTX_FEATURES)):
            if flag in features:                              # the group counts only when complete and warm
                cols = [f for f in group if f in features]
                ok = z[cols].notna().all(axis=1) & (np.arange(len(z)) >= int(p["warmup_bars"]))
                z.loc[~ok, cols] = np.nan
                z[flag] = ok.astype(float)
        x = z[list(features)].to_numpy(dtype=float)               # column order = config order
        out[t] = np.nan_to_num(x, nan=0.0).astype(np.float32)
    return out
