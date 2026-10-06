"""DQNPolicy: the harness policy interface for the agent, with parallel training.

`harness.experiment.run_experiment` calls

    policy.prepare(prices, folds, seeds, cfg, tickers, out_dir)   # once
    policy.exposures(prices, fold, seed, cfg, tickers)            # per (fold, seed)

prepare() trains one agent per (seed, training cut) and rolls it greedily
through every evaluation block that shares that cut (normally one fold; the
three test sub-blocks share one cut, PROTOCOL §10a.8). Jobs run in parallel
worker processes:

  * each worker is pinned to ONE GPU (CUDA_VISIBLE_DEVICES) with memory
    growth on, so TensorFlow does not grab ~31 GB on every GPU
  * each worker is limited to `threads_per_worker` CPU threads, and the total
    is capped at 90 % of the machine's cores (shared server rule)
  * op determinism is on by default, so a (config, seed, fold) gives the same
    result no matter how many workers run

Per job, under <out_dir>/agent/<fold>_s<seed>/:
    curve.csv           training curve (rl/trainer.py)
    info.json           best/last inner-validation Sharpe, update counts, timing
    exposures.npz       exposures of the SELECTED and of the LAST checkpoint
    best.weights.h5, last.weights.h5
"""

import json
import multiprocessing as mp
import os
import subprocess

import numpy as np

from harness.config import config_hash
from harness.data import ticker_sets
from harness.splits import fold_ranges, purge_bars

MAX_SHARE = 0.9          # never use more than 90 % of CPU cores (project rule)


# ---------------------------------------------------------------------------
# resource planning
# ---------------------------------------------------------------------------
def free_gpus(max_used_mb=2000):
    """GPUs whose memory use is below `max_used_mb` right now (others are busy)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode()
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, used = [int(x) for x in line.split(",")]
        if used < max_used_mb:
            gpus.append(idx)
    return gpus


def plan_workers(runtime, n_jobs):
    """(n_workers, gpu list, threads per worker) within the 90 % CPU cap."""
    threads = int(runtime.get("threads_per_worker", 3))
    cap = max(1, int(MAX_SHARE * (os.cpu_count() or 1)) // threads)
    n = max(1, min(int(runtime.get("workers", 1)), cap, n_jobs))
    gpus = runtime.get("gpus", "auto")
    gpus = free_gpus() if gpus == "auto" else [int(g) for g in (gpus or [])]
    return n, gpus, threads


def _init_worker(counter, gpus, threads, deterministic):
    """Runs once in each fresh worker process, BEFORE TensorFlow is imported."""
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[i % len(gpus)]) if gpus else ""
    for var in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("WANDB_SILENT", "true")
    _configure_tf(threads, deterministic)


def _configure_tf(threads, deterministic):
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(threads))
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass                                    # already initialised (in-process runs)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------------------
# one job = one (seed, training cut)
# ---------------------------------------------------------------------------
def _seed_everything(seed):
    import random
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _wandb_logger(cfg, job_name, out_dir):
    log = cfg.get("logging", {})
    if not log.get("wandb", False):
        return None, None
    try:
        import wandb
        run = wandb.init(project=log.get("wandb_project", "dqn-trading-protocol"),
                         group=f"{cfg.get('name', 'run')}_{config_hash(cfg)}", name=job_name,
                         config=cfg, mode=log.get("wandb_mode", "online"), dir=out_dir,
                         reinit=True)
        return run.log, run
    except Exception as exc:                    # logging must never kill a training run
        print(f"[wandb] disabled for {job_name}: {type(exc).__name__}: {exc}")
        return None, None


def run_job(job):
    """Train one agent and produce exposures for all its evaluation blocks.

    `job` is a plain dict (picklable): cfg, seed, folds (sharing one cut),
    prices {ticker: DataFrame}, train_tickers, eval_tickers, out_dir.
    """
    from rl.env import MarketData
    from rl.features import ex_ante_vol, ticker_features, vol_scale
    from rl.trainer import greedy_exposures, train_run
    from harness import backtest as bt

    cfg, seed, folds = job["cfg"], job["seed"], job["folds"]
    a, ev = cfg["agent"], cfg["evaluation"]
    _seed_everything(seed)
    name = f"{'+'.join(f.name for f in folds)}_s{seed}"
    job_dir = os.path.join(job["out_dir"], "agent", name)
    os.makedirs(job_dir, exist_ok=True)

    W = int(a["window"])
    P = max(purge_bars(cfg), W + int(a["algo"].get("n_step", 1)))
    inner = int(cfg["splits"]["inner_val_bars"])
    train_t, eval_t = list(job["train_tickers"]), list(job["eval_tickers"])
    tickers = train_t + [t for t in eval_t if t not in train_t]     # training tickers first

    closes, feats, vols, dates, sigmas, fr = [], [], [], [], [], {}
    vol_span = int(a["env"].get("vol_span", 60))
    for t in tickers:
        df = job["prices"][t]
        r = fold_ranges(df.index, folds[0], P, inner)                # same cut for all folds
        fr[t] = r
        closes.append(df["Close"].to_numpy())
        if job.get("features") is not None:
            # M4: causal features computed once for the whole experiment; a
            # job's price history is a prefix of it, so slice to its length
            feats.append(job["features"][t][:len(df)])
        else:
            # legacy features: scaler fitted on this fold's inner-training bars
            feats.append(ticker_features(df, a["features"], r.train_start_pos, r.inner_train_end_pos,
                                         a.get("feature_clip", 10.0)))
        vols.append(vol_scale(df["Close"].to_numpy(), r.train_start_pos, r.inner_train_end_pos))
        sigmas.append(ex_ante_vol(df["Close"].to_numpy(), vol_span, r.train_start_pos, r.inner_train_end_pos))
        dates.append(df.index)

    # decisions t in [lo, hi): the last one reads close[hi], which stays inside its range
    train_ranges = [(max(fr[t].train_start_pos, W - 1), fr[t].inner_train_end_pos - 1) for t in train_t]
    select_ranges = [(fr[t].inner_val_start_pos - 1, fr[t].train_end_pos - 1) for t in train_t]
    train_data = MarketData(train_t, closes[:len(train_t)], feats[:len(train_t)], W,
                            vols[:len(train_t)], dates[:len(train_t)], sigmas[:len(train_t)])

    # Costs the agent trains under: the PROTOCOL level, or a named scenario
    # (e.g. neo_broker: EUR 1 per transaction, spread, TER). With a scenario, the
    # capital is shared by all training tickers, so each sleeve holds C / N.
    bpy = cfg["data"]["bars_per_year"]
    scen_name = a["env"].get("cost_scenario")
    if scen_name:
        scen = bt.load_scenarios()[scen_name]
        cost_of = lambda t, n: bt.scenario_cost(scen, t, n, bpy)          # noqa: E731
    else:
        rate = bt.cost_rate(ev["costs"]["primary_bps"], ev["costs"]["half_spread_bps"])
        cost_of = lambda t, n: rate                                       # noqa: E731
    train_costs = [cost_of(t, len(train_t)) for t in train_t]

    logger, wb = _wandb_logger(cfg, name, job_dir)
    agent, info, curve, last_w = train_run(cfg, train_data, train_ranges, select_ranges, seed,
                                           train_costs, logger)
    curve.to_csv(os.path.join(job_dir, "curve.csv"), index=False)
    agent.online.save_weights(os.path.join(job_dir, "best.weights.h5"))

    # greedy exposures on every evaluation block, for the selected AND the last weights
    K, short = a["env"].get("levels", ev["position_levels"]), ev["allow_short"]
    result = {}
    for tag, weights in (("selected", None), ("last", last_w)):
        if weights is not None:
            agent.online.set_weights(weights)
            agent.online.save_weights(os.path.join(job_dir, "last.weights.h5"))
        for f in folds:
            idx = [tickers.index(t) for t in eval_t]
            ranges = []
            for t in eval_t:
                val = fold_ranges(job["prices"][t].index, f, P, inner).val
                ranges.append((int(val[0]) - 1, int(val[-1])))
            sub = MarketData([tickers[i] for i in idx], [closes[i] for i in idx], [feats[i] for i in idx],
                             W, [vols[i] for i in idx], [dates[i] for i in idx], [sigmas[i] for i in idx])
            # costs do not influence greedy actions; passed only because the env needs them
            flat = greedy_exposures(agent, sub, ranges, a["env"], K, short,
                                    [cost_of(t, len(eval_t)) for t in eval_t])
            for j, t in enumerate(eval_t):
                result[f"{tag}|{f.name}|{t}"] = flat[sub.offsets[j]:sub.offsets[j] + sub.lengths[j]]
    np.savez_compressed(os.path.join(job_dir, "exposures.npz"), **result)
    info.update(job=name, seed=seed, purge=P, train_tickers=train_t)
    with open(os.path.join(job_dir, "info.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=float)
    if wb is not None:
        wb.summary.update({k: v for k, v in info.items() if isinstance(v, (int, float))})
        wb.finish()
    return {"seed": seed, "folds": [f.name for f in folds], "dir": job_dir, "info": info}


# ---------------------------------------------------------------------------
# the harness policy
# ---------------------------------------------------------------------------
class DQNPolicy:
    deterministic = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.exp = {}            # (fold name, seed) -> {ticker: exposure array}
        self.jobs_done = []

    def prepare(self, prices, folds, seeds, cfg, tickers, out_dir):
        a = cfg["agent"]
        sets = ticker_sets(cfg)
        train_t = sets[a["train_tickers"]]
        features = None
        if a.get("feature_mode", "legacy") == "m4":
            from rl.features_m4 import build_m4_features
            m4 = a.get("m4", {})
            need = sorted(set(train_t) | set(tickers))
            print(f"  [dqn] building M4 features for {len(need)} tickers (factor set '{m4.get('factor_set', 'train')}')")
            features = build_m4_features(prices, need, a["features"], sets[m4.get("factor_set", "train")],
                                         vix=prices.get("^VIX"), params=m4)
        groups = {}
        for f in folds:
            groups.setdefault(f.cut, []).append(f)
        jobs = []
        for seed in seeds:
            for group in groups.values():
                end = max(f.val_end for f in group)
                need = sorted(set(train_t) | set(tickers))
                jobs.append({"cfg": cfg, "seed": int(seed), "folds": group,
                             "prices": {t: prices[t][prices[t].index <= end] for t in need},
                             "train_tickers": train_t, "eval_tickers": list(tickers),
                             "out_dir": out_dir,
                             "features": None if features is None else
                             {t: features[t][:int((prices[t].index <= end).sum())] for t in need}})

        rt = a.get("runtime", {})
        n, gpus, threads = plan_workers(rt, len(jobs))
        deterministic = bool(rt.get("deterministic_ops", True))
        print(f"  [dqn] {len(jobs)} training jobs on {n} worker(s), GPUs {gpus or 'none (CPU)'}, "
              f"{threads} threads each")
        if n == 1 and not rt.get("force_subprocess", False):
            _configure_tf(threads, deterministic)          # in-process (tests, debugging)
            results = [run_job(j) for j in jobs]
        else:
            ctx = mp.get_context("spawn")
            counter = ctx.Value("i", 0)
            with ctx.Pool(n, initializer=_init_worker, initargs=(counter, gpus, threads, deterministic)) as pool:
                results = []
                for r in pool.imap_unordered(run_job, jobs, chunksize=1):
                    results.append(r)
                    i = r["info"]
                    print(f"  [dqn] done {len(results)}/{len(jobs)}: {'+'.join(r['folds'])} seed {r['seed']} "
                          f"| best inner Sharpe {i['best_inner_sharpe']:+.2f} @ upd {i['best_update']} "
                          f"| last {i['last_inner_sharpe']:+.2f} | {i['seconds']:.0f}s")
        self.jobs_done = results
        self.exp = load_exposures([r["dir"] for r in results])

    def exposures(self, prices, fold, seed, cfg, tickers):
        got = self.exp[(fold.name, seed)]
        # arrays cover the job's price history; the harness passes prefixes of it
        return {t: got[t][:len(prices[t])] for t in tickers}


def load_exposures(job_dirs, tag="selected"):
    """{(fold name, seed): {ticker: exposure array}} from finished job folders."""
    out = {}
    for d in job_dirs:
        with open(os.path.join(d, "info.json")) as fh:
            seed = int(json.load(fh)["seed"])
        data = np.load(os.path.join(d, "exposures.npz"))
        for key in data.files:
            k_tag, fold, t = key.split("|")
            if k_tag == tag:
                out.setdefault((fold, seed), {})[t] = data[key]
    return out


class StoredExposurePolicy(DQNPolicy):
    """Replays the exposures a finished DQN run saved (no TensorFlow, no training).

    Used to re-score an existing trial under other cost settings. The strategy
    is unchanged, so such a re-score is NOT a new trial and is not logged.
    tag = "selected" (the checkpoint the harness evaluated) or "last".
    """

    def __init__(self, run_dir, tag="selected"):
        super().__init__(cfg=None)
        agent_dir = os.path.join(run_dir, "agent")
        self.exp = load_exposures([os.path.join(agent_dir, d) for d in sorted(os.listdir(agent_dir))], tag)

    def prepare(self, **kwargs):
        """Nothing to train: exposures were loaded in __init__."""
