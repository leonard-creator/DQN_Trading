"""One training run of the DQN agent, with checkpoint selection on inner validation.

    collect  : B parallel episodes, masked epsilon-greedy, one network call per step
    store    : n-step transitions into (prioritised) replay
    learn    : `update_ratio` gradient steps per collected transition
    select   : every `eval_every_updates` steps, roll the GREEDY policy through
               the inner-validation slice (last 252 bars of the fold's training
               range, purged; PROTOCOL §4.2) and score it with the harness
               backtest at the primary cost. The best-scoring weights are kept.
               The outer validation block is never looked at here.

Everything that is logged per evaluation point (the "curve") is what the M2
stability analysis uses: loss, mean Q, |TD|, epsilon, learning rate, training
episode returns, and the inner-validation Sharpe / turnover / exposure.
"""

import math
import time

import numpy as np
import pandas as pd

from harness import backtest as bt
from harness import metrics as mt
from rl.agent import DQNAgent
from rl.env import N_ACTIONS, VecTradingEnv
from rl.replay import NStepCollector, PrioritizedReplay, ReplayBuffer


class RunningStd:
    """Welford running standard deviation of the raw rewards seen so far.

    Used to put rewards on unit scale before the Huber loss, so that
    delta = 1 means something (spec W9). Applied at sample time, so all
    replayed transitions use the same (current) scale.
    """

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        for v in np.asarray(x, np.float64).ravel():
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self.m2 += d * (v - self.mean)

    @property
    def std(self):
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 1.0


# ---------------------------------------------------------------------------
# greedy evaluation
# ---------------------------------------------------------------------------
def greedy_exposures(agent, data, ranges, env_cfg, levels, allow_short, cost):
    """Roll the greedy policy once through each (lo, hi) decision range.

    ranges[i] belongs to ticker i of `data`. Each pass starts flat at bar lo.
    Returns a flat array over data's global bars: the exposure decided at each
    visited decision bar, NaN elsewhere (harness.backtest reads exposure[t] at
    decision bars only and fails loudly on NaN there).
    """
    env = VecTradingEnv(data, ranges, env_cfg, levels, allow_short, cost, mode="eval")
    out = np.full(len(data.close), np.nan)
    while env.active.any():
        g, pv, mk = env.observe()
        live = env.active
        a = agent.act(data.windows(np.where(live, g, g.min())), pv, mk, 0.0, None)
        step = env.step(a)
        out[g[live]] = step["exposure"][live]
    return out


def score_ranges(data, exposure_flat, ranges, cost, bars_per_year=252):
    """Harness metrics of an equal-weight portfolio over decision ranges.

    Decisions lo..hi-1 earn the returns at bars lo+1..hi. `cost` is one
    CostModel/float for all tickers or a list with one per ticker.
    """
    costs = cost if isinstance(cost, (list, tuple)) else [cost] * len(ranges)
    frames = {}
    for i, (lo, hi) in enumerate(ranges):
        off, n = data.offsets[i], data.lengths[i]
        close = data.close[off:off + n]
        expo = exposure_flat[off:off + n]
        pos = np.arange(lo + 1, hi + 1)
        res = bt.backtest(close, expo, pos, costs[i])
        idx = data.dates[i][pos] if data.dates is not None else pd.RangeIndex(len(pos))
        frames[data.tickers[i]] = bt.to_frame(res, idx)
    port = bt.portfolio(frames)
    return mt.portfolio_metrics(port, frames, bars_per_year)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_run(cfg, data, train_ranges, select_ranges, seed, costs, logger=None, verbose=False):
    """Train one agent; return (agent with SELECTED weights, info dict, curve DataFrame).

    data          : MarketData; the first len(train_ranges) tickers are trained on
    train_ranges  : (lo, hi) decision ranges for training episodes, one per training ticker
    select_ranges : (lo, hi) decision ranges of the inner-validation slice, same tickers
    costs         : one harness.backtest.CostModel per training ticker. The same
                    costs enter the training reward AND the checkpoint score.
    logger        : optional callable(dict) for live logging (wandb)
    """
    a, ev = cfg["agent"], cfg["evaluation"]
    env_cfg, algo, tr = a["env"], a["algo"], a["train"]
    cost = costs                                   # per training ticker
    K, short = env_cfg.get("levels", ev["position_levels"]), ev["allow_short"]

    env = VecTradingEnv(data, train_ranges, env_cfg, K, short, cost,
                        n_envs=tr["n_envs"], mode="train", seed=seed)
    n_envs = env.B
    total = int(tr["transitions"])
    ticks = math.ceil(total / n_envs)
    ratio = float(tr["update_ratio"])
    total_updates = int(total * ratio)
    agent = DQNAgent(data.window, data.n_feat, env.pos_dim, N_ACTIONS, a["network"], algo, total_updates)

    per = algo.get("per", {})
    if per.get("enabled", False):
        buffer = PrioritizedReplay(algo["buffer_size"], env.pos_dim, N_ACTIONS,
                                   per.get("alpha", 0.6), per.get("eps", 1e-6), seed=seed)
    else:
        buffer = ReplayBuffer(algo["buffer_size"], env.pos_dim, N_ACTIONS, seed=seed)
    nstep = NStepCollector(n_envs, algo.get("n_step", 1), algo["gamma"])
    rnorm = RunningStd()
    use_norm = algo.get("reward_norm", "none") == "running_std"
    rng = np.random.default_rng(seed + 10_000)

    sel_env_cfg = dict(env_cfg)          # greedy scoring uses the same env rules
    eval_every = int(tr["eval_every_updates"])
    batch_size = int(algo["batch_size"])
    learn_after = max(int(tr["learning_starts"]), batch_size)
    eps0, eps1 = float(tr["eps_start"]), float(tr["eps_end"])
    eps_steps = max(1.0, float(tr["eps_decay_frac"]) * total)
    b0, b1 = float(per.get("beta_start", 0.4)), float(per.get("beta_end", 1.0))

    curve, best = [], {"sharpe": -np.inf, "update": -1, "weights": None}
    # per-interval statistics; rc_* = mean reward components per live env step (spec Phase 3)
    RC = ("rc_pnl", "rc_cost", "rc_shadow", "rc_risk", "rc_hold", "reward_raw")
    window_stats = {"loss": [], "q": [], "td": [], **{k: [] for k in RC}}
    credit = 0.0
    t_start = time.time()

    def evaluate(tag):
        expo = greedy_exposures(agent, data, select_ranges, sel_env_cfg, K, short, cost)
        m = score_ranges(data, expo, select_ranges, cost, cfg["data"]["bars_per_year"])
        fin = env.finished_returns
        row = {"update": agent.updates, "transitions": tick_now[0] * n_envs, "epsilon": eps_now[0],
               "lr": agent.current_lr(), "reward_scale": rnorm.std if use_norm else 1.0,
               "loss": float(np.mean(window_stats["loss"])) if window_stats["loss"] else np.nan,
               "mean_q": float(np.mean(window_stats["q"])) if window_stats["q"] else np.nan,
               "abs_td": float(np.mean(window_stats["td"])) if window_stats["td"] else np.nan,
               "train_episode_return": float(np.mean(fin)) if fin else np.nan,
               **{k: (float(np.mean(window_stats[k])) if window_stats[k] else np.nan) for k in RC},
               "inner_sharpe": m["sharpe"], "inner_cagr": m["cagr"], "inner_turnover": m["turnover"],
               "inner_exposure": m["exposure"], "elapsed_s": time.time() - t_start, "tag": tag}
        env.finished_returns = []
        for v in window_stats.values():
            v.clear()
        curve.append(row)
        if logger:
            logger({k: v for k, v in row.items() if k != "tag"})
        if verbose:
            print(f"    upd {row['update']:6d} eps {row['epsilon']:.3f} loss {row['loss']:.4f} "
                  f"Q {row['mean_q']:.3f} inner Sharpe {row['inner_sharpe']:+.3f} "
                  f"turn {row['inner_turnover']:.1f} ({row['elapsed_s']:.0f}s)")
        if m["sharpe"] > best["sharpe"]:
            best.update(sharpe=m["sharpe"], update=agent.updates, weights=agent.online.get_weights())

    tick_now, eps_now = [0], [eps0]                 # mutable cells read by evaluate()
    g, pv, mk = env.observe()
    for tick in range(ticks):
        tick_now[0] = tick
        done_frac = min(1.0, tick * n_envs / eps_steps)
        eps_now[0] = eps0 + done_frac * (eps1 - eps0)
        actions = agent.act(data.windows(g), pv, mk, eps_now[0], rng)
        step = env.step(actions)
        live = step["live"]
        if use_norm:
            rnorm.update(step["reward"][live])
        if live.any():
            for k in RC[:-1]:
                window_stats[k].append(float(np.mean(step[k][live])))
            window_stats["reward_raw"].append(float(np.mean(step["reward"][live])))
        batch = nstep.push(g, pv, mk, actions, step)
        if batch is not None:
            buffer.add_batch(**batch)
        g, pv, mk = env.observe()

        if len(buffer) >= learn_after:
            credit += ratio * n_envs
            while credit >= 1.0:
                credit -= 1.0
                frac = agent.updates / max(1, total_updates)
                sample = buffer.sample(batch_size, beta=b0 + frac * (b1 - b0))
                td, loss, mq = agent.learn(sample, data, rnorm.std if use_norm else 1.0)
                buffer.update_priorities(sample["idx"], td)
                window_stats["loss"].append(loss)
                window_stats["q"].append(mq)
                window_stats["td"].append(float(np.mean(np.abs(td))))
                if agent.updates % eval_every == 0:
                    evaluate("periodic")

    evaluate("last")
    last_weights = agent.online.get_weights()
    select = tr.get("select", "best_inner_val")
    if select == "best_inner_val" and best["weights"] is not None:
        agent.online.set_weights(best["weights"])
    elif select not in ("best_inner_val", "last"):
        raise ValueError(f"unknown select '{select}'")
    info = {"best_update": best["update"], "best_inner_sharpe": float(best["sharpe"]),
            "last_inner_sharpe": float(curve[-1]["inner_sharpe"]), "updates": agent.updates,
            "transitions": ticks * n_envs, "seconds": time.time() - t_start, "select": select}
    return agent, info, pd.DataFrame(curve), last_weights
