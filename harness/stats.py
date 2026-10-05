"""Overfitting-aware statistics (PROTOCOL §1, §8).

    deflated_sharpe   Bailey & Lopez de Prado (2014), J. Portfolio Management 40(5).
                      Probability that the true Sharpe ratio is above the best
                      Sharpe you would expect from N_trials worthless strategies.
    pbo_cscv          Bailey, Borwein, Lopez de Prado & Zhu (2015), J. Comput. Finance.
                      Probability of backtest overfitting via combinatorially
                      symmetric cross-validation.
    permutation_test  paired sign-flip test of "agent beats baseline".
    holm              Holm (1979) step-down family-wise correction.

All Sharpe ratios inside this module are PER PERIOD (not annualised), as in
the original papers.
"""

import itertools

import numpy as np
from scipy import stats as _st

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Deflated Sharpe ratio
# ---------------------------------------------------------------------------
def probabilistic_sharpe(sr, sr_benchmark, n_obs, skew, kurt):
    """PSR(SR*) = Phi( (SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) ).

    sr, sr_benchmark : per-period Sharpe ratios
    n_obs            : T, number of return observations
    skew, kurt       : skewness g3 and NON-excess kurtosis g4 of the returns
    """
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if n_obs < 2 or denom <= 0:
        return np.nan
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1.0) / np.sqrt(denom)
    return float(_st.norm.cdf(z))


def expected_max_sharpe(var_sr, n_trials):
    """SR* = sqrt(V[SR]) * ((1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e))).

    The Sharpe ratio the best of `n_trials` zero-skill strategies would reach
    by luck, given the cross-trial variance `var_sr` of their Sharpe ratios.
    With one trial there is no selection, so SR* = 0.
    """
    n = int(n_trials)
    if n <= 1 or not np.isfinite(var_sr) or var_sr <= 0:
        return 0.0
    g = EULER_GAMMA
    return float(np.sqrt(var_sr) * ((1 - g) * _st.norm.ppf(1 - 1.0 / n)
                                    + g * _st.norm.ppf(1 - 1.0 / (n * np.e))))


def deflated_sharpe(returns, n_trials, var_sr_trials):
    """Deflated Sharpe ratio of one return series.

    returns       : per-period net returns of the SELECTED strategy (out of sample)
    n_trials      : number of configurations tried (from experiments/trials.csv)
    var_sr_trials : variance of the per-period Sharpe ratios of those trials

    Returns a dict: deflated_sharpe (the probability, compare to 0.95), the
    observed per-period Sharpe, the benchmark SR*, and the moments used.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    sd = np.std(r, ddof=1) if n > 1 else 0.0
    sd = sd if sd > 1e-12 else 0.0                 # same zero-tolerance as harness.metrics
    sr = float(r.mean() / sd) if sd > 0 else 0.0
    c = r - r.mean()
    m2 = np.mean(c ** 2) if sd > 0 else 0.0
    skew = float(np.mean(c ** 3) / m2 ** 1.5) if m2 > 0 else 0.0
    kurt = float(np.mean(c ** 4) / m2 ** 2) if m2 > 0 else 3.0
    sr0 = expected_max_sharpe(var_sr_trials, n_trials)
    return {"deflated_sharpe": probabilistic_sharpe(sr, sr0, n, skew, kurt),
            "sr_pp": sr, "sr0_pp": sr0, "n_obs": n, "n_trials": int(n_trials),
            "skew": skew, "kurtosis": kurt}


# ---------------------------------------------------------------------------
# PBO via CSCV
# ---------------------------------------------------------------------------
def _sharpe_from_sums(s1, s2, n):
    mean = s1 / n
    var = (s2 - n * mean ** 2) / np.maximum(n - 1, 1)
    sd = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mean / sd, 0.0)
    return out


def pbo_cscv(returns_matrix, n_blocks=16):
    """Probability of backtest overfitting.

    returns_matrix : (T, N) array, T aligned return observations of N
                     configurations (columns). Needs N >= 2.
    n_blocks       : S, even. The T rows are cut into S equal contiguous blocks
                     (the last T mod S rows are dropped).

    For every way of choosing S/2 blocks as "in-sample" (IS) and the other S/2
    as "out-of-sample" (OOS):
      1. pick the configuration with the best IS Sharpe,
      2. find its relative rank w in OOS (w = rank / (N + 1), rank 1 = worst),
      3. logit l = ln(w / (1 - w)).
    PBO = share of splits with l <= 0, i.e. the IS winner ends up at or below
    the OOS median. PBO near 0 = selection works; near 1 = it picks noise.

    Uses block sums and one matrix product per moment, so all C(16, 8) = 12,870
    splits are evaluated at once.
    """
    m = np.asarray(returns_matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("pbo_cscv needs a (T, N) matrix with N >= 2 configurations")
    if n_blocks % 2:
        raise ValueError("n_blocks must be even")
    t_use = (m.shape[0] // n_blocks) * n_blocks
    if t_use < 2 * n_blocks:
        raise ValueError("too few observations for this many blocks")
    m = np.nan_to_num(m[:t_use])
    blocks = m.reshape(n_blocks, -1, m.shape[1])            # (S, T/S, N)
    s1_b = blocks.sum(axis=1)                               # (S, N)
    s2_b = (blocks ** 2).sum(axis=1)
    n_b = np.full(n_blocks, blocks.shape[1], dtype=np.float64)

    combos = np.array(list(itertools.combinations(range(n_blocks), n_blocks // 2)))
    mask = np.zeros((len(combos), n_blocks))
    mask[np.arange(len(combos))[:, None], combos] = 1.0     # (C, S) IS indicator

    is_s1, is_s2, is_n = mask @ s1_b, mask @ s2_b, mask @ n_b
    oos_s1, oos_s2, oos_n = s1_b.sum(0) - is_s1, s2_b.sum(0) - is_s2, n_b.sum() - is_n
    sr_is = _sharpe_from_sums(is_s1, is_s2, is_n[:, None])
    sr_oos = _sharpe_from_sums(oos_s1, oos_s2, oos_n[:, None])

    best = np.argmax(sr_is, axis=1)                         # IS winner per split
    ranks = _st.rankdata(sr_oos, axis=1)                    # 1 = worst, ties averaged
    w = ranks[np.arange(len(best)), best] / (m.shape[1] + 1.0)
    logits = np.log(w / (1.0 - w))
    winner_oos = sr_oos[np.arange(len(best)), best]
    return {"pbo": float(np.mean(logits <= 0)),
            "n_splits": int(len(combos)),
            "n_configs": int(m.shape[1]),
            "logits": logits,
            "prob_oos_loss": float(np.mean(winner_oos < 0))}


# ---------------------------------------------------------------------------
# Paired permutation test and Holm correction
# ---------------------------------------------------------------------------
def permutation_test(diffs, n_permutations=10000, seed=0):
    """One-sided paired sign-flip test of H1: mean(diffs) > 0.

    diffs : agent metric minus baseline metric, one value per (seed, fold).
    Under H0 each difference is symmetric around 0, so flipping its sign is
    equally likely. p = (1 + #{permuted mean >= observed mean}) / (1 + B).
    Note: the seeds within one fold share the same baseline value, so the
    pairs are not fully independent. The test is about this design, not
    about new market periods.
    """
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.all(d == 0):
        return {"p_value": 1.0, "mean_diff": 0.0, "n_pairs": int(len(d))}
    rng = np.random.default_rng(seed)
    observed = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(int(n_permutations), len(d)))
    perm = (signs * d).mean(axis=1)
    p = (1.0 + np.sum(perm >= observed - 1e-15)) / (1.0 + n_permutations)
    return {"p_value": float(p), "mean_diff": float(observed), "n_pairs": int(len(d))}


def holm(p_values):
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj
