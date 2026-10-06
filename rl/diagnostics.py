"""Q-value diagnostics from saved weights (PROTOCOL Part II §V7: LC5, LC7; step 0a).

For every finished (seed, fold) job of a trial, the SELECTED checkpoint
(best.weights.h5) is rolled greedily through its validation block once more,
with the identical data pipeline (rl/policy.build_job_data), recording the
Q-values. Per evaluation sleeve:

    LC5 action gap   median over decision bars of Q(best valid) - Q(second-best valid),
                     divided by the TD-error standard deviation at the selected
                     checkpoint (from curve.csv: SD ~= mean|TD| * sqrt(pi/2), Gaussian
                     approximation). Below 1: the greedy choice is decided by noise.
    LC7 over-estimation   predicted Q(s_t, a_t) vs the realised discounted return
                     G_t = sum_k gamma^k r_t+k / reward_scale along the same greedy path,
                     with r the agent's own training reward. G is truncated at the end
                     of the block, so only bars with >= `horizon` bars left are used.
                     Reported as Q - G (reward units), Q / G (ratio of means) and the
                     scale-free (mean Q - mean G) / SD(G).
    reproduction     share of decision bars whose exposure equals the stored one
                     (exposures.npz). 1.0 = the pipeline reproduces the trial exactly.

All quantities are in the agent's normalised reward units. Nothing is retrained
and no new trial is created.
"""

import json
import math
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

TRUNC_HORIZON = 300          # bars of future needed before G_t is trusted (gamma^300 ~ 0.05)


def _curve_row(job_dir, best_update):
    c = pd.read_csv(os.path.join(job_dir, "curve.csv"))
    return c.iloc[int((c["update"] - best_update).abs().idxmin())]


def job_q_diagnostics(args):
    """Diagnostics of one job. `args` = (job dict, job_dir). Runs in a worker process."""
    job, job_dir = args
    from rl.agent import DQNAgent
    from rl.env import N_ACTIONS, VecTradingEnv
    from rl.policy import build_job_data, eval_market, eval_ranges

    cfg = job["cfg"]
    a = cfg["agent"]
    info = json.load(open(os.path.join(job_dir, "info.json")))
    row = _curve_row(job_dir, info["best_update"])
    td_sd = float(row["abs_td"]) * math.sqrt(math.pi / 2)
    scale = float(row.get("reward_scale", 1.0)) or 1.0
    gamma = float(a["algo"]["gamma"])

    jd = build_job_data(job)
    eval_t = jd["eval_t"]
    sub = eval_market(jd, eval_t)
    fold = job["folds"][0]
    ranges = eval_ranges(job, jd, fold, eval_t)
    env = VecTradingEnv(sub, ranges, a["env"], jd["K"], jd["short"],
                        [jd["cost_of"](t, len(eval_t)) for t in eval_t], mode="eval")
    agent = DQNAgent(jd["W"], sub.n_feat, env.pos_dim, N_ACTIONS, a["network"], a["algo"], total_updates=1)
    agent.online.load_weights(os.path.join(job_dir, "best.weights.h5"))

    flat = np.full(len(sub.close), np.nan)
    steps = []
    while env.active.any():
        g, pv, mk = env.observe()
        live = env.active.copy()
        q = agent.q_values(sub.windows(np.where(live, g, g.min())), pv)
        act = np.argmax(np.where(mk, q, -np.inf), axis=1)
        out = env.step(act)
        flat[g[live]] = out["exposure"][live]
        steps.append((live, q, mk, act, out["reward"].astype(np.float64)))

    stored = np.load(os.path.join(job_dir, "exposures.npz"))
    rows = []
    for b, t in enumerate(eval_t):
        live = np.array([s[0][b] for s in steps])
        q = np.array([s[1][b] for s in steps])[live]
        mk = np.array([s[2][b] for s in steps])[live]
        act = np.array([s[3][b] for s in steps])[live]
        r = np.array([s[4][b] for s in steps])[live] / scale
        qv = np.where(mk, q, -np.inf)
        top2 = np.sort(qv, axis=1)[:, -2:]
        gap = top2[:, 1] - top2[:, 0]
        gap = gap[np.isfinite(gap)]
        G = np.zeros(len(r))
        acc = 0.0
        for i in range(len(r) - 1, -1, -1):          # discounted return, truncated at the block end
            acc = r[i] + gamma * acc
            G[i] = acc
        use = np.arange(len(r)) <= len(r) - 1 - TRUNC_HORIZON
        pred = q[np.arange(len(q)), act]
        key = f"selected|{fold.name}|{t}"
        mine = flat[sub.offsets[b]:sub.offsets[b] + sub.lengths[b]]
        ref = stored[key][:len(mine)] if key in stored.files else np.full(len(mine), np.nan)
        dec = np.isfinite(ref)
        rows.append({
            "ticker": t, "fold": fold.name, "seed": job["seed"],
            "gap_median": float(np.median(gap)) if len(gap) else np.nan,
            "td_sd": td_sd,
            "action_gap_ratio": float(np.median(gap) / td_sd) if len(gap) and td_sd > 0 else np.nan,
            "q_minus_G_median": float(np.median(pred[use] - G[use])) if use.any() else np.nan,
            "q_over_G": float(np.mean(pred[use]) / np.mean(G[use])) if use.any() and abs(np.mean(G[use])) > 1e-9 else np.nan,
            # scale-free over-estimation: (mean Q - mean G) in units of SD(G); comparable across
            # rewards, unlike Q - G (reward units) or Q / G (unstable when mean G is near 0)
            "q_bias_sd": float((np.mean(pred[use]) - np.mean(G[use])) / np.std(G[use]))
                         if use.sum() > 2 and np.std(G[use]) > 0 else np.nan,
            "q_G_corr": float(np.corrcoef(pred[use], G[use])[0, 1]) if use.sum() > 2 and np.std(G[use]) > 0 else np.nan,
            "reproduced": float(np.mean(np.abs(mine[dec] - ref[dec]) < 1e-9)) if dec.any() else np.nan,
        })
    return rows


def q_diagnostics_for_run(run_dir, workers=1, threads=2):
    """Per-sleeve LC5 / LC7 / reproduction rows for every job of a logged agent trial."""
    from harness.data import load_prices, ticker_sets
    from harness.experiment import load_result
    from harness.splits import folds_from_config
    from rl.policy import _configure_tf, _init_worker, experiment_features, make_jobs

    cfg = load_result(run_dir).cfg
    sets = ticker_sets(cfg)
    eval_t = sorted({t for es in (cfg.get("eval_sets") or ["train"]) for t in sets[es]})
    a = cfg["agent"]
    load = set(eval_t) | set(sets[a["train_tickers"]]) | set(cfg["universe"].get("context", []))
    if a.get("m4", {}).get("factor_set"):
        load |= set(sets[a["m4"]["factor_set"]])
    prices = load_prices(sorted(load), cfg)
    agent_dir = os.path.join(run_dir, "agent")
    seeds = sorted({int(json.load(open(os.path.join(agent_dir, d, "info.json")))["seed"])
                    for d in os.listdir(agent_dir)})
    feats = experiment_features(cfg, prices, eval_t)
    jobs = make_jobs(cfg, prices, folds_from_config(cfg), seeds, eval_t, run_dir, features=feats)
    tasks = []
    for j in jobs:
        d = os.path.join(agent_dir, f"{'+'.join(f.name for f in j['folds'])}_s{j['seed']}")
        if os.path.exists(os.path.join(d, "best.weights.h5")):
            tasks.append((j, d))
    if workers <= 1:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        _configure_tf(threads, False)
        results = [job_q_diagnostics(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        with ctx.Pool(workers, initializer=_init_worker, initargs=(counter, [], threads, False)) as pool:
            results = list(pool.imap_unordered(job_q_diagnostics, tasks, chunksize=1))
    return pd.DataFrame([r for rows in results for r in rows])
