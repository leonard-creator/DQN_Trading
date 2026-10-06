"""Per-ticker feature matrices for the agent.

M2 deliberately keeps the feature set of the existing code (log return, log
volume, ROC12, MFI14, realised volatility, Close/SMA20, Close/SMA50), so that
M2 measures the algorithm changes only. The spec's full stationary feature set
and the rolling z-score arrive in M4.

Two pieces of existing code are reused, not copied:
  * scrape_data.compute_indicators: indicators from raw OHLCV (all rolling,
    i.e. backward-looking)
  * functions.FeatureScaler: per-feature transforms + z-score statistics

Leakage rule: the scaler is fitted on the bars the network is TRAINED on
(the fold's inner-training range) of each ticker, and then applied to the
whole history of that ticker. Statistics never come from validation or test
bars.
"""

import numpy as np

from functions import FeatureScaler
from scrape_data import compute_indicators


def ticker_features(ohlcv, features, fit_lo, fit_hi, clip=10.0):
    """(T, F) float32 feature matrix for one ticker.

    ohlcv          : DataFrame with Open/High/Low/Close/Volume, indexed by Date
    features       : feature column names (see functions.FEATURE_TRANSFORMS)
    fit_lo, fit_hi : bar range [fit_lo, fit_hi) used to fit the scaler
    clip           : z-scores are clipped to +-clip (a zero-volume day on an
                     index would otherwise create a -20 sigma outlier)

    Indicator warm-up rows at the very start of the history are NaN and are set
    to 0; tests/test_rl_env.py checks that no decision bar ever reads one.
    """
    ind = compute_indicators(ohlcv)
    missing = [f for f in features if f not in ind.columns]
    if missing:
        raise KeyError(f"unknown features {missing}")
    fit = ind.iloc[int(fit_lo):int(fit_hi)]
    if fit[features].isna().any().any():
        raise ValueError("NaN inside the scaler fit range: training starts inside the indicator warm-up")
    scaler = FeatureScaler(features).fit(fit)
    x = scaler.transform(ind)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -clip, clip).astype(np.float32)


def vol_scale(close, fit_lo, fit_hi, horizon=20):
    """Typical size of a `horizon`-bar move, from the fit range only.

    Used to normalise the unrealised P&L in the position vector, so that +5 %
    means the same thing for a bond ETF as for an equity index.
    """
    c = np.asarray(close[int(fit_lo):int(fit_hi)], dtype=np.float64)
    s = float(np.std(np.diff(np.log(c))) * np.sqrt(horizon)) if len(c) > 2 else 0.0
    return s if s > 1e-6 else 1e-2


def ex_ante_vol(close, span=60, fit_lo=None, fit_hi=None, floor=1e-4):
    """Daily volatility known at the close of each bar (no look-ahead).

    sigma[t] = EWMA standard deviation (span `span`) of the log returns up to
    and including bar t. Used by the volatility-scaled rewards (M3): a reward
    measured in "typical daily moves" means the same for a bond ETF and an
    equity index, and keeps the mean-variance risk aversion lambda scale-free
    (plan weakness W9). Warm-up bars without enough history get the std of the
    fit range (or of the whole series if no fit range is given).
    """
    import pandas as pd
    c = np.asarray(close, dtype=np.float64)
    lr = pd.Series(np.r_[np.nan, np.diff(np.log(c))])
    sig = lr.ewm(span=int(span), min_periods=int(span) // 2).std().to_numpy()
    seg = lr.iloc[fit_lo:fit_hi] if fit_lo is not None else lr
    fill = float(np.nanstd(seg)) if np.isfinite(np.nanstd(seg)) else 0.01
    sig = np.where(np.isfinite(sig), sig, fill)
    return np.maximum(sig, floor)
