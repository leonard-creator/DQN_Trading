"""Synthetic positive controls (PROTOCOL Part II §V6.5, Step 0c).

Three 26-ticker one-factor worlds with a KNOWN best policy (the "oracle") test
whether an agent configuration learns timing that exists (W-vol, W-regime) and
leaves alone timing that does not (W-null):

    r_i,t = mu_i,t + s_i * sigma_t * (sqrt(rho) z_t + sqrt(1 - rho) u_i,t)

with z, u standardised Student-t (nu = 5), s_i log-uniform on [0.5, 2] (one
fixed draw), rho = 0.48 as measured in DATA_AUDIT.md. Prices compound from 100
(no intraday range, no volume: volume features are masked as in Q4). A noisy
VIX proxy 100 sqrt(252) sigma_t+1|t exp(0.25 eta) enters with the pipeline's
usual one-bar lag.

    W-null    GARCH(1,1) variance; mu_i,t = lambda_i (s_i sigma_t)^2, so mu / var is
              constant: no timing beats buy-and-hold. Oracle: buy-and-hold.
    W-vol     the same GARCH, constant mu_i (Sharpe 0.6 at the unconditional vol).
              Oracle: e_t = min(1, sigma_bar^2 / sigma^2_t+1|t) on the 1/4 grid.
    W-regime  2-state Markov switching of the factor (bull / bear), no GARCH. Oracle:
              Hamilton filter -> mean-variance weight relative to the bull weight,
              on the 1/4 grid with one-step hysteresis.

Run k (k = 0..4) trains agent seed k on path 1000+k (504 warm-up + 4,090 bars;
checkpoints on its last 252 bars, purged) and is scored on the independent 20-year
path 2000+k, both laid out one after the other per ticker so the normal harness
(fold, purge, features, backtest) does the rest. Results: experiments/synthetic.csv
and experiments/reports/V2_0c.md. Never logged as trials (0 trials, §V5).
"""

import copy
import datetime as _dt
import os

import numpy as np
import pandas as pd

from harness.config import code_hash, config_hash, load_config, repo_path
from harness.experiment import BaselinePolicy, run_experiment
from harness.splits import Fold

WORLDS = ("W-null", "W-vol", "W-regime")
N_TICKERS, NU, RHO, SHARPE, LEVELS = 26, 5, 0.48, 0.6, 4
WARMUP, TRAIN_BARS, EVAL_BARS = 504, 4090, 20 * 252
# beta 0.90 -> 0.91 by the calibration rule (W-vol oracle gain +0.08 -> +0.19; 2026-10-07), frozen
GARCH = {"alpha": 0.08, "beta": 0.91, "vol": 0.01}                         # daily unconditional vol 1 %
REGIME = {"mu": (0.15, -0.20), "sigma": (0.14, 0.25), "stay": (0.998, 0.99)}  # (bull, bear), annual
TICKERS = [f"S{i:02d}" for i in range(N_TICKERS)]
SCALES = np.exp(np.random.default_rng(0).uniform(np.log(0.5), np.log(2.0), N_TICKERS))
CSV = repo_path("experiments", "synthetic.csv")


def _shocks(rng, size):
    return rng.standard_t(NU, size) * np.sqrt((NU - 2) / NU)               # Student-t, unit variance


def _grid(x):
    return np.round(np.clip(x, 0.0, 1.0) * LEVELS) / LEVELS


def garch_forecast(z, alpha, beta, vol):
    """(sigma_t, sigma_t+1|t) of the GARCH(1,1) common variance driven by the shocks z (causal)."""
    var = np.empty(len(z) + 1)
    var[0] = vol ** 2
    omega = vol ** 2 * (1 - alpha - beta)
    for t in range(len(z)):
        var[t + 1] = omega + alpha * var[t] * z[t] ** 2 + beta * var[t]
    return np.sqrt(var[:-1]), np.sqrt(var[1:])


def hamilton_target(x, mu, sd, stay):
    """Oracle exposure of W-regime from the observed factor proxy x (causal).

    P(bull) is filtered with the true parameters and predicted one bar ahead; the target is
    the mean-variance weight relative to the bull weight, clipped to [0, 1], on the 1/4 grid
    with hysteresis: the exposure moves to the rounded target only once the target is at
    least 3/4 step away (plain rounding would move at 1/2 step). A full-step band would never
    return to 1, because the target stays just below it while P(bull) < 1.
    """
    p_bb, p_rr = stay
    pi = (1 - p_rr) / ((1 - p_bb) + (1 - p_rr))                            # stationary P(bull)
    e, prev = np.empty(len(x)), 1.0
    for t, v in enumerate(x):
        prior = pi * p_bb + (1 - pi) * (1 - p_rr)
        lb, lr = (np.exp(-0.5 * ((v - mu[k]) / sd[k]) ** 2) / sd[k] for k in (0, 1))
        pi = prior * lb / (prior * lb + (1 - prior) * lr)
        p = pi * p_bb + (1 - pi) * (1 - p_rr)                              # P(bull tomorrow | data <= t)
        w = (p * mu[0] + (1 - p) * mu[1]) / (p * sd[0] ** 2 + (1 - p) * sd[1] ** 2)
        target = np.clip(w / (mu[0] / sd[0] ** 2), 0.0, 1.0)
        prev = float(_grid(target)) if abs(target - prev) >= 0.75 / LEVELS else prev
        e[t] = prev
    return e


def simulate(world, n, seed):
    """One path: returns (n x 26), VIX proxy (n,), oracle exposure decided at each bar (n,)."""
    rng = np.random.default_rng(seed)
    z, u, eta = _shocks(rng, n), _shocks(rng, (n, N_TICKERS)), rng.standard_normal(n)
    if world == "W-regime":
        mu_d = np.array(REGIME["mu"]) / 252
        sd_d = np.array(REGIME["sigma"]) / np.sqrt(252)
        state, flips = np.zeros(n, dtype=int), rng.random(n)              # 0 = bull, 1 = bear
        for t in range(1, n):
            state[t] = state[t - 1] if flips[t] < REGIME["stay"][state[t - 1]] else 1 - state[t - 1]
        sig, ahead = sd_d[state], sd_d[state]                              # VIX: the current regime's vol
        mu = mu_d[state][:, None] * SCALES
    else:
        sig, ahead = garch_forecast(z, **GARCH)
        v = GARCH["vol"]
        drift = (SHARPE / np.sqrt(252)) * (sig ** 2 / v if world == "W-null" else np.full(n, v))
        mu = drift[:, None] * SCALES
    r = mu + SCALES * sig[:, None] * (np.sqrt(RHO) * z[:, None] + np.sqrt(1 - RHO) * u)
    if world == "W-null":
        oracle = np.ones(n)
    elif world == "W-vol":
        oracle = _grid(np.minimum(1.0, np.median(ahead) ** 2 / ahead ** 2))
    else:
        x = (r / SCALES).mean(axis=1)                                      # observable factor proxy
        oracle = hamilton_target(x, mu_d, sd_d * np.sqrt(RHO + (1 - RHO) / N_TICKERS), REGIME["stay"])
    vix = 100 * np.sqrt(252) * ahead * np.exp(0.25 * eta)
    return np.maximum(r, -0.95), vix, oracle


def world_data(world, seed, start="1950-01-02"):
    """Prices (26 tickers + ^VIX), oracle exposures and the fold of synthetic run `seed`."""
    n1, n2 = WARMUP + TRAIN_BARS, WARMUP + EVAL_BARS
    parts = [simulate(world, n1, 1000 + seed), simulate(world, n2, 2000 + seed)]
    r, vix, oracle = (np.concatenate([p[i] for p in parts]) for i in range(3))
    dates = pd.bdate_range(start, periods=n1 + n2)

    def frame(close, volume):
        return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                             "Volume": volume}, index=pd.Index(dates, name="Date"))

    close = 100 * np.cumprod(1 + r, axis=0)
    prices = {t: frame(close[:, i], 0.0) for i, t in enumerate(TICKERS)}
    prices["^VIX"] = frame(vix, 0.0)
    # train on path 1 only (cut at the junction, purged); score path 2 after its own warm-up
    fold = Fold("SYN", dates[WARMUP], dates[n1 + WARMUP], dates[-1], dates[n1])
    return prices, {t: oracle for t in TICKERS}, fold


def synthetic_config(cfg):
    """The agent configuration re-pointed at the synthetic universe; agent, costs, budget and purge unchanged."""
    c = copy.deepcopy(cfg)
    c["name"] = f"syn_{cfg['name']}"
    c["universe"] = {"buckets": {"synthetic": list(TICKERS)}, "leave_out": [],
                     "single_asset": TICKERS[0], "context": ["^VIX"]}
    c["eval_sets"], c["extra_ticker_sets"], c["cost_scenarios"] = ["train"], {}, []
    c.setdefault("logging", {})["wandb"] = False                           # logged in synthetic.csv
    return c


class FixedPolicy:
    """Exposures known in advance (the oracle): {ticker: array over the ticker's bars}."""
    deterministic = True

    def __init__(self, exposures):
        self.e = exposures

    def exposures(self, prices, fold, seed, cfg, tickers):
        return {t: self.e[t][:len(prices[t])] for t in tickers}


ROOT = repo_path("experiments", "runs", "synthetic")


def _score(cfg, policy, prices, fold, seed, root=ROOT):
    """(net Sharpe, gross Sharpe, turnover/yr) of one policy on the evaluation path."""
    p = cfg["evaluation"]["costs"]["primary_bps"]
    res = run_experiment(cfg, policy=policy, prices=prices, folds=[fold], seeds=[seed], cost_levels=[0, p],
                         cost_scenarios=[], log=False, verbose=False, out_root=root)
    net, gross = res.runs("train", p).iloc[0], res.runs("train", 0).iloc[0]
    return float(net["sharpe"]), float(gross["sharpe"]), float(net["turnover"])


def calibrate(config="config/v2/R0prime.yaml", seeds=range(5)):
    """Calibration rule (§V6.5): the oracle's median net Sharpe gain over buy-and-hold on the
    evaluation paths must be >= 0.15 in W-vol and W-regime. No training, not a trial."""
    cfg = synthetic_config(load_config(config) if isinstance(config, str) else config)
    out = {}
    for w in WORLDS:
        gains = []
        for s in seeds:
            prices, oracle, fold = world_data(w, s)
            gains.append(_score(cfg, FixedPolicy(oracle), prices, fold, s)[0]
                         - _score(cfg, BaselinePolicy("buy_and_hold"), prices, fold, s)[0])
        out[w] = float(np.median(gains))
        print(f"[0c] calibration {w}: oracle net Sharpe gain over buy-and-hold, median {out[w]:+.2f} "
              f"(per path {np.round(gains, 2).tolist()})")
    return out


def run(config, seeds=range(5), worlds=WORLDS, workers=None, log=True, root=ROOT):
    """Step 0c for one agent configuration: train all (world, seed) runs in one parallel pool,
    score agent, buy-and-hold and oracle on the evaluation paths, append experiments/synthetic.csv
    and refresh the report (log=False: return the rows only, e.g. for tests)."""
    from rl.policy import StoredExposurePolicy, make_jobs, train_jobs  # late: rl imports the harness
    base = load_config(config) if isinstance(config, str) else config
    cfg = synthetic_config(base)
    runtime = dict(cfg["agent"].get("runtime", {}), **({"workers": workers} if workers else {}))
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = os.path.join(root, f"{stamp}_{cfg['name']}_{config_hash(base)}")
    runs, jobs = {}, []
    for w in worlds:
        for s in seeds:
            prices, oracle, fold = world_data(w, s)
            out = os.path.join(root, f"{w}_s{s}")
            os.makedirs(out, exist_ok=True)
            runs[(w, s)] = (prices, oracle, fold, out)
            jobs += make_jobs(cfg, prices, [fold], [s], list(TICKERS), out)
    train_jobs(jobs, runtime)
    rows = []
    for (w, s), (prices, oracle, fold, out) in runs.items():
        a, bh, orc = (_score(cfg, p, prices, fold, s, root) for p in
                      (StoredExposurePolicy(out), BaselinePolicy("buy_and_hold"), FixedPolicy(oracle)))
        rows.append({"timestamp": stamp, "config": base["name"], "config_hash": config_hash(base),
                     "code_hash": code_hash(), "world": w, "seed": s,
                     "agent_sharpe": a[0], "agent_sharpe_gross": a[1], "agent_turnover": a[2],
                     "bh_sharpe": bh[0], "oracle_sharpe": orc[0], "oracle_turnover": orc[2],
                     "output_dir": os.path.relpath(out, repo_path()) if out.startswith(repo_path()) else out})
    new = pd.DataFrame(rows)
    if log:
        new.to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
        write_report()
    return new


def verdicts(df):
    """§V6.5 pass criteria per (config, world), medians over the seeds of its LATEST run."""
    df = df[df["timestamp"] == df.groupby("config")["timestamp"].transform("max")]
    out = []
    for (c, w), g in df.groupby(["config", "world"], sort=False):
        gain, gain_o = (g["agent_sharpe"] - g["bh_sharpe"]).median(), (g["oracle_sharpe"] - g["bh_sharpe"]).median()
        turn = g["agent_turnover"].median()
        ok = (turn <= 2.0 and gain >= -0.05) if w == "W-null" else bool(gain_o > 0 and gain >= 0.5 * gain_o)
        out.append({"config": c, "world": w, "runs": len(g), "agent_sharpe": g["agent_sharpe"].median(),
                    "agent_gross": g["agent_sharpe_gross"].median(), "bh_sharpe": g["bh_sharpe"].median(),
                    "oracle_sharpe": g["oracle_sharpe"].median(), "gain": gain, "oracle_gain": gain_o,
                    "turnover": turn, "pass": ok})
    return pd.DataFrame(out)


def write_report():
    """experiments/reports/V2_0c.md from experiments/synthetic.csv."""
    v = verdicts(pd.read_csv(CSV))
    L = ["# v2 Step 0c — Synthetic positive controls (generated by harness/synthetic.py)", "",
         "PROTOCOL Part II §V6.5. Medians over 5 runs (agent seed k trained on path 1000+k, scored on the "
         "independent 20-year path 2000+k), net of 10 bp + 1 bp. *Gain* = net Sharpe − buy-and-hold. "
         "Pass: W-null turnover ≤ 2/yr and gain ≥ −0.05; W-vol / W-regime agent gain ≥ 50 % of the oracle's. "
         "Not trials (§V5); raw rows in `experiments/synthetic.csv`.", "",
         "| Config | World | Net Sharpe | Gross | B&H | Oracle | Gain | Oracle gain | Turnover/yr | Pass |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r.config} | {r.world} | {r.agent_sharpe:.2f} | {r.agent_gross:.2f} | {r.bh_sharpe:.2f} | "
          f"{r.oracle_sharpe:.2f} | {r.gain:+.2f} | {r.oracle_gain:+.2f} | {r.turnover:.1f} | "
          f"**{'pass' if r.pass_ else 'FAIL'}** |" for r in v.rename(columns={"pass": "pass_"}).itertuples()]
    with open(repo_path("experiments", "reports", "V2_0c.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def real_data_allowed(configs=("v2_R0prime", "v2_V1")):
    """Stopping rule 1 (§V8): real-data trials pause if EVERY config failed both W-vol and W-regime
    in its latest synthetic run. Missing results also pause (safe default). Prints the reason."""
    v = verdicts(pd.read_csv(CSV)) if os.path.exists(CSV) else pd.DataFrame(columns=["config", "world", "pass"])
    v = v[v["config"].isin(configs) & v["world"].isin(["W-vol", "W-regime"])]
    passed = {c: bool(v.loc[v["config"] == c, "pass"].any()) for c in configs}
    complete = all((v["config"] == c).sum() == 2 for c in configs)
    ok = complete and any(passed.values())
    print(f"[0c] stopping rule 1: {'continue' if ok else 'PAUSE real-data trials'} "
          f"(results complete: {complete}; passed W-vol or W-regime: {passed})")
    return ok
