"""Baseline strategies (PROTOCOL §6), expressed as exposure series.

Every function returns an array `exposure` with one value per bar of the
ticker's history: exposure[t] is the fraction of the sleeve's capital held
from close t to close t+1, decided with information up to bar t only.
All signals use pandas rolling/ewm operations that look backwards, and
tests/test_baselines.py checks that changing future prices never changes a
past exposure.

Exposures live on the same grid as the agent: {0, 1/K, ..., 1} long-only
(K = evaluation.position_levels). The momentum and MACD rules are all-in or
flat, i.e. 0 or 1, which is on that grid for every K.
"""

import zlib

import numpy as np
import pandas as pd


def buy_and_hold(close):
    """Fully invested on every bar."""
    return np.ones(len(close), dtype=np.float64)


def momentum(close, lookback, allow_short=False):
    """Time-series momentum: long if the past `lookback`-bar return is > 0.

    Flat otherwise (short instead of flat if allow_short). Bars without
    enough history are flat.
    """
    c = pd.Series(np.asarray(close, dtype=np.float64))
    past_ret = c / c.shift(int(lookback)) - 1.0              # uses bars t-lookback .. t only
    down = -1.0 if allow_short else 0.0
    e = np.where(past_ret > 0, 1.0, down)
    e[past_ret.isna().to_numpy()] = 0.0
    return e


def macd_crossover(close, fast=12, slow=26, signal=9, allow_short=False):
    """MACD crossover: long while MACD line > signal line, else flat.

    MACD = EMA_fast(close) - EMA_slow(close); signal = EMA_signal(MACD).
    `adjust=False` gives the standard recursive EMA, which only uses past values.
    The first `slow + signal` bars are flat (EMAs still warming up).
    """
    c = pd.Series(np.asarray(close, dtype=np.float64))
    macd = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    down = -1.0 if allow_short else 0.0
    e = np.where(macd > sig, 1.0, down)
    e[: int(slow) + int(signal)] = 0.0
    return e


def vol_target(close, span=60, min_history=252, levels=4):
    """Volatility-managed exposure (descriptive baseline, PROTOCOL Part II step 0a).

    exposure_t = min(1, (sigma_bar_t / sigma_t)^2), after Moreira & Muir (2017):
      sigma_t     = EWMA daily volatility (span `span`) of log returns up to bar t
      sigma_bar_t = median of sigma over all bars up to t (expanding, >= min_history bars)
    Rounded to the 1/levels grid (long-only, capped at 1). Fully causal; flat
    until min_history bars of volatility exist.
    """
    c = pd.Series(np.asarray(close, dtype=np.float64))
    lr = np.log(c).diff()
    sig = lr.ewm(span=int(span), min_periods=int(span) // 2).std()
    sig_bar = sig.expanding(min_periods=int(min_history)).median()
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.minimum(1.0, (sig_bar / sig) ** 2)
    e = np.round(raw.to_numpy() * levels) / levels
    e[~np.isfinite(e)] = 0.0
    return e


def mean_reversion(close, decisions, cost, horizons=(2, 5, 10, 20, 60), min_history=504):
    """Estimated W-revert oracle (descriptive baseline, PROTOCOL Part II §V12.1 item 25, realism check).

    Fits the temporary-component model of harness/synthetic.py (random walk + transitory AR(1) part) to the
    ticker's OWN daily log returns before the block (least squares on the variance ratios at `horizons`),
    then runs the causal Kalman contrarian rule synthetic.kalman_revert with those parameters and the
    evaluation cost: long unless the filtered transitory deviation predicts a loss beyond the cost band.
    No reversion in the past data (fitted share ~ 0) means holding. Exposure 0 or 1; flat before
    `min_history` bars.
    """
    from scipy.optimize import minimize
    from harness.synthetic import kalman_revert
    r = np.diff(np.log(np.asarray(close, dtype=np.float64)), prepend=np.nan)
    past = r[1:int(decisions[0]) + 1]                         # returns known at the first decision of the block
    past = past[np.isfinite(past)] - np.nanmean(past)
    q = np.asarray(horizons)
    vr = np.array([np.convolve(past, np.ones(h), "valid").var() / (h * past.var()) for h in q])
    fit = minimize(lambda x: np.sum((1 - x[0] + x[0] * (1 - x[1] ** q) / (q * (1 - x[1])) - vr) ** 2), [0.2, 0.9],
                   bounds=[(0.0, 0.9), (0.0, 0.99)])
    share, phi = fit.x
    v, mu = np.nanstd(r[1:int(decisions[0]) + 1]), np.nanmean(r[1:int(decisions[0]) + 1])
    e = kalman_revert(np.nan_to_num(r, nan=mu), mu, phi, np.sqrt(1 - share) * v,
                      np.sqrt(share * (1 + phi) / 2) * v, cost * (1 - phi))[:, 0]
    e[:min_history] = 0.0
    return e


def stable_seed(*parts):
    """Deterministic integer seed from strings/ints.

    Python's built-in hash() is randomised per process, so it would make the
    random baseline irreproducible. crc32 is stable everywhere.
    """
    return [zlib.crc32(str(p).encode()) for p in parts]


def random_agent(n_bars, decision_positions, levels, seed, allow_short=False):
    """Random agent in the SAME action space as the DQN.

    Actions: 0 = hold, 1 = buy (+1/K), 2 = sell (-1/K). At each decision bar
    it picks uniformly among the VALID actions (the same action masking the
    agent uses: no buy at max long, no sell at the lower limit). It starts flat
    at the first decision bar of the block.

    n_bars             : length of the ticker's history (size of the output)
    decision_positions : bars at which decisions are taken (block return positions - 1)
    levels             : K
    seed               : int or list of ints (use stable_seed(seed, ticker, fold))

    Returns an exposure array; bars outside `decision_positions` are NaN, so
    an accidental read of a non-decision bar fails loudly in backtest().
    """
    rng = np.random.default_rng(seed)
    k_min = -levels if allow_short else 0
    k = 0
    out = np.full(int(n_bars), np.nan)
    for t in np.asarray(decision_positions, dtype=np.int64):
        valid = [0]
        if k < levels:
            valid.append(1)
        if k > k_min:
            valid.append(2)
        a = valid[rng.integers(len(valid))]
        k += 1 if a == 1 else (-1 if a == 2 else 0)
        out[t] = k / levels
    return out


# Registry used by harness.experiment. Each entry maps a policy name to a
# function (close, decision_positions, seed, ticker, fold_name, cfg, params)
# -> exposure array. Deterministic baselines ignore seed/ticker/fold.
def _bh(close, dec, seed, ticker, fold, cfg, p):
    return buy_and_hold(close)


def _mom(close, dec, seed, ticker, fold, cfg, p):
    lb = p.get("lookback", cfg["evaluation"]["baselines"]["momentum_lookback"])
    return momentum(close, lb, cfg["evaluation"]["allow_short"])


def _macd(close, dec, seed, ticker, fold, cfg, p):
    m = dict(cfg["evaluation"]["baselines"]["macd"], **p)
    return macd_crossover(close, m["fast"], m["slow"], m["signal"], cfg["evaluation"]["allow_short"])


def _rand(close, dec, seed, ticker, fold, cfg, p):
    ev = cfg["evaluation"]
    return random_agent(len(close), dec, ev["position_levels"],
                        stable_seed(seed, ticker, fold), ev["allow_short"])


def _vt(close, dec, seed, ticker, fold, cfg, p):
    return vol_target(close, levels=cfg["evaluation"]["position_levels"], **p)


def _mr(close, dec, seed, ticker, fold, cfg, p):
    c = cfg["evaluation"]["costs"]
    return mean_reversion(close, dec, (c["primary_bps"] + c["half_spread_bps"]) / 1e4, **p)


BASELINES = {
    "buy_and_hold": _bh,
    "momentum": _mom,
    "macd": _macd,
    "random": _rand,
    "vol_target": _vt,          # descriptive only (v2 step 0a); not one of the four H1 baselines
    "mean_reversion": _mr,      # descriptive only (v2 realism check, §V12.1 item 25); not an H1 baseline
}

# Deterministic baselines give identical results for every seed, so the
# harness computes them once and reuses them across seeds.
DETERMINISTIC = {"buy_and_hold", "momentum", "macd", "vol_target", "mean_reversion"}
