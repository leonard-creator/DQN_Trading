"""V1 decision structure (PROTOCOL Part II §V6.1): target exposures, exact cost head,
exogenous replay and a lazy buy-and-hold anchor. Selected by agent.replay.mode: exogenous.

Prices do not depend on the agent's position (no market impact). With target-exposure
actions e' in {0, 1/K, ..., 1} the action value therefore splits EXACTLY into

    Q(m, p, e') = U(m, e') - kappa(m) |e' - p| - phi(m) 1[e' != p]
    kappa = rate / sigma_t,   phi = fee_frac / sigma_t     (costs in vol units, like the reward)

The network sees the market window m only and outputs the K+1 values U(m, .). Every
(ticker, training day) yields targets for ALL exposures ("exogenous replay"):

    y(e') = clip((e' r_t+1 - hold e') / sigma_t) - eta 1[e' != 1]
            + gamma [U_target(m_t+1, e*) - kappa_t+1 |e* - e'| - phi_t+1 1[e* != e']]
    e*    = argmax_e'' U_online(m_t+1, e'') - kappa_t+1 |e'' - e'| - phi_t+1 1[e'' != e']   (Double DQN)

With algo.n_step = n > 1 the target holds e' for n bars before bootstrapping:
    y(e') = e' sum_k<n gamma^k clip(z_t+k) - eta 1[e' != 1] sum_k<n gamma^k + gamma^n [ ...as above at t+n... ]
a lower bound on U (switching inside the n bars is ignored) that propagates slow regime
value n times faster; n <= H_max = 20 keeps it inside the purge P = 40 (§V0 item 4).

With env.reward = mean_variance the immediate term is x - (lambda/2) x^2, x = e' z, summed
over the n bars; env.mv_lambda = auto sets lambda = mean(z) / mean(z^2) on the TRAINING
samples, the value at which full exposure is optimal on average. Because z is vol-scaled,
the optimum e' ~ E[z | m] / lambda then falls when volatility rises (vol targeting) and
turns to 0 when the expected return turns negative. vol_scaled_pnl (lambda = 0) is risk
neutral: with a positive drift it never prefers less than full exposure.

With algo.heads = K > 1 (V2, §V6.2) the network has K U-heads on one trunk (Bootstrapped
DQN): each head learns from its own Bernoulli(1/2) subset of the samples and bootstraps from
its own target head. Acting uses the gated rule: move from p to the e' with the largest mean
gain only if mean_k[dQ] > gate_z * sd_k[dQ], dQ = Q_k(e') - Q_k(p); otherwise stay.

This is the env's vol_scaled_pnl reward split by exposure. eta, the lazy anchor (training
reward only), charges every bar away from full exposure, so a deviation must be expected to
beat holding; the output biases start at the buy-and-hold value of each exposure. The greedy
policy trades only if U(e') - U(p) exceeds the cost: a no-trade band (Garleanu & Pedersen
2013). No environment loop and no epsilon: the data set is the training range itself.
Same interface and outputs as rl.trainer.train_run, so checkpoint selection, saving,
exposures and the harness are unchanged.
"""

import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from harness import backtest as bt
from rl.agent import make_schedule
from rl.networks import build_q_network
from rl.trainer import score_ranges


def cost_arrays(data, costs):
    """(rate, fee_frac, hold) per global bar of `data`, from one CostModel/float per ticker."""
    cms = [bt.as_cost_model(c) for c in costs]
    return tuple(np.repeat([getattr(c, k) for c in cms], data.lengths) for k in ("rate", "fee_frac", "hold"))


def decide(u, p, grid, kappa, phi, z=0.0):
    """Index of the exposure to hold next from U (E,) or K heads (K, E) at position p.

    One head: argmax Q = U - kappa |e' - p| - phi 1[e' != p] (the no-trade band). Several heads:
    the e' with the largest mean gain dQ over staying, taken only if mean dQ > z * sd dQ.
    """
    q = np.atleast_2d(u) - (kappa * np.abs(grid - p) + phi * (grid != p))
    stay = int(np.argmin(np.abs(grid - p)))
    dq = q - q[:, [stay]]                                  # gain over staying, per head
    k = int(np.argmax(dq.mean(axis=0)))
    if len(q) > 1 and k != stay and dq[:, k].mean() <= z * dq[:, k].std(ddof=1):
        return stay
    return k


class UAgent:
    """Online and target U networks (market input only): `heads` x one output per target exposure."""

    def __init__(self, window, n_feat, levels, net_cfg, algo, total_updates, bias=None, discount=None):
        self.grid = np.arange(levels + 1) / levels
        self.heads, E = int(algo.get("heads", 1)), levels + 1
        self.online = build_q_network(window, n_feat, 0, self.heads * E, net_cfg)
        self.target = build_q_network(window, n_feat, 0, self.heads * E, net_cfg)
        if bias is not None:
            kernel = self.online.layers[-1].get_weights()[0]
            self.online.layers[-1].set_weights([kernel, np.tile(np.asarray(bias, np.float32), self.heads)])
        self.target.set_weights(self.online.get_weights())
        self.gamma, self.tau = float(algo["gamma"]), float(algo.get("tau", 0.005))
        self.discount = self.gamma if discount is None else float(discount)   # gamma^n for n-step targets
        self.delta = float(algo.get("huber_delta", 1.0)) if algo.get("loss", "huber") == "huber" else None
        self.schedule = make_schedule(algo, total_updates)
        clip = algo.get("grad_clip_norm")
        self.opt = keras.optimizers.Adam(learning_rate=self.schedule,
                                         **({"global_clipnorm": float(clip)} if clip else {}))
        grid = tf.constant(self.grid, tf.float32)
        self.dist = tf.abs(grid[None, :] - grid[:, None])          # [e', e''] = |e'' - e'|
        self.move = tf.cast(self.dist > 0, tf.float32)              # [e', e''] = 1[e'' != e']
        self.updates = 0
        self._step = tf.function(self._train_step, reduce_retracing=True)
        self._u = tf.function(lambda m: self._split(self.online(m, training=False)), reduce_retracing=True)

    def _split(self, out):
        return tf.reshape(out, (-1, self.heads, len(self.grid)))            # (B, heads, E)

    def u_values(self, market):
        """(B, E) for one head, (B, heads, E) for several."""
        u = self._u(tf.convert_to_tensor(market, tf.float32)).numpy()
        return u[:, 0] if self.heads == 1 else u

    def current_lr(self):
        s = self.schedule
        return float(s(self.opt.iterations)) if callable(s) else float(s)

    def targets(self, rew, m2, kappa2, phi2):
        """y (B, heads, E) from the immediate terms rew (B, E) and next-bar cost factors kappa2, phi2 (B,).
        Each head picks e* with its online head and is evaluated by its own target head."""
        cost = (kappa2[:, None, None] * self.dist + phi2[:, None, None] * self.move)[:, None]   # (B, 1, e', e'')
        u_on, u_tg = self._split(self.online(m2, training=False)), self._split(self.target(m2, training=False))
        best = tf.argmax(u_on[:, :, None, :] - cost, axis=3, output_type=tf.int32)            # (B, heads, e')
        cost = tf.broadcast_to(cost, tf.concat([tf.shape(best), [len(self.grid)]], 0))
        nxt = tf.gather(u_tg, best, batch_dims=2) - tf.gather(cost, best, batch_dims=3)
        return rew[:, None, :] + self.discount * nxt

    def _train_step(self, m, rew, m2, kappa2, phi2, mask):
        y = tf.stop_gradient(self.targets(rew, m2, kappa2, phi2))
        with tf.GradientTape() as tape:
            u = self._split(self.online(m, training=True))
            td = y - u                                                # all heads and exposures at once
            if self.delta is None:
                err = tf.square(td)
            else:
                a = tf.abs(td)
                q = tf.minimum(a, self.delta)
                err = 0.5 * q ** 2 + self.delta * (a - q)
            w = mask[:, :, None]                                      # bootstrap mask (B, heads)
            loss = tf.reduce_sum(w * err) / tf.maximum(1.0, tf.reduce_sum(w) * len(self.grid))
        grads = tape.gradient(loss, self.online.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.online.trainable_variables))
        for t_var, o_var in zip(self.target.weights, self.online.weights):
            t_var.assign(self.tau * o_var + (1.0 - self.tau) * t_var)
        return td, loss, tf.reduce_mean(u)

    def learn(self, m, rew, m2, kappa2, phi2, mask=None):
        f = lambda x: tf.convert_to_tensor(x, tf.float32)            # noqa: E731
        mask = np.ones((len(m), self.heads)) if mask is None else mask
        td, loss, mean_u = self._step(f(m), f(rew), f(m2), f(kappa2), f(phi2), f(mask))
        self.updates += 1
        return td.numpy(), float(loss), float(mean_u)


def band_exposures(agent, data, ranges, costs, z=0.0):
    """Greedy rollout of decide() through each (lo, hi) range, starting flat. Flat array over
    data's bars: the exposure decided at each decision bar, NaN elsewhere."""
    rate, fee, _ = cost_arrays(data, costs)
    out = np.full(len(data.close), np.nan)
    for i, (lo, hi) in enumerate(ranges):
        g = data.offsets[i] + np.arange(lo, hi)
        u, p = agent.u_values(data.windows(g)), 0.0
        for j, gj in enumerate(g):
            p = out[gj] = agent.grid[decide(u[j], p, agent.grid, rate[gj] / data.sigma[gj], fee[gj] / data.sigma[gj], z)]
    return out


def train_exogenous(cfg, data, train_ranges, select_ranges, seed, costs, logger=None, verbose=False):
    """Train one V1 agent; returns (agent with SELECTED weights, info, curve, last weights) like train_run.

    Budget: transitions x update_ratio gradient steps of `batch_size` (ticker, day) samples,
    the same number of steps as the environment-loop agent with the same config.
    """
    a, algo, tr = cfg["agent"], cfg["agent"]["algo"], cfg["agent"]["train"]
    if cfg["evaluation"]["allow_short"] or a["network"].get("dueling", False):
        raise NotImplementedError("exogenous replay: long-only, plain output head (PROTOCOL Part II §V6.1)")
    levels = a["env"].get("levels", cfg["evaluation"]["position_levels"])
    total, batch = int(int(tr["transitions"]) * float(tr["update_ratio"])), int(algo["batch_size"])
    gamma, eta, clip = float(algo["gamma"]), float(a.get("anchor_eta", 0.0)), a["env"].get("reward_clip")
    n = int(algo.get("n_step", 1))
    grid = np.arange(levels + 1) / levels
    rate, fee, hold = cost_arrays(data, costs)
    # samples t with t + n inside the ticker's range; z = vol-scaled net return of each bar
    G = np.concatenate([data.offsets[i] + np.arange(lo, hi - n + 1) for i, (lo, hi) in enumerate(train_ranges)])
    z = lambda g: np.clip((data.close[g + 1] / data.close[g] - 1.0 - hold[g]) / data.sigma[g],  # noqa: E731
                          -(clip or np.inf), clip or np.inf)
    zsum = sum(gamma ** k * z(G + k) for k in range(n))               # discounted n-bar sums per sample
    z2sum = sum(gamma ** k * z(G + k) ** 2 for k in range(n))
    anchor = eta * sum(gamma ** k for k in range(n)) * (grid != 1.0)
    lam = 0.0
    if a["env"].get("reward") == "mean_variance":
        lam = a["env"].get("mv_lambda", 0.5)
        lam = max(float(z(G).mean()), 1e-3) / float((z(G) ** 2).mean()) if lam == "auto" else float(lam)

    def immediate(idx):                                               # (B, E) training reward of each e'
        return (grid[None, :] * zsum[idx][:, None] - 0.5 * lam * grid[None, :] ** 2 * z2sum[idx][:, None]
                - anchor[None, :])

    # prior: U(e') = value of holding e' forever at the average training reward (anchor included)
    bias = (grid * z(G).mean() - 0.5 * lam * grid ** 2 * (z(G) ** 2).mean() - eta * (grid != 1.0)) / (1.0 - gamma)
    agent = UAgent(data.window, data.n_feat, levels, a["network"], algo, total, bias, discount=gamma ** n)
    rng = np.random.default_rng(seed + 10_000)
    z_gate = float(a.get("gate_z", 0.0))
    masks = (rng.random((len(G), agent.heads)) < 0.5).astype(np.float32) if agent.heads > 1 else None
    eval_every = int(tr["eval_every_updates"])
    curve, best, stats, t0 = [], {"sharpe": -np.inf, "update": -1, "weights": None}, ([], [], []), time.time()

    def evaluate(tag):
        m = score_ranges(data, band_exposures(agent, data, select_ranges, costs, z_gate), select_ranges, costs,
                         cfg["data"]["bars_per_year"])
        row = {"update": agent.updates, "transitions": agent.updates * batch, "epsilon": 0.0,
               "lr": agent.current_lr(), "reward_scale": 1.0,
               **{k: float(np.mean(v)) if v else np.nan for k, v in zip(("loss", "mean_q", "abs_td"), stats)},
               "inner_sharpe": m["sharpe"], "inner_cagr": m["cagr"], "inner_turnover": m["turnover"],
               "inner_exposure": m["exposure"], "elapsed_s": time.time() - t0, "tag": tag}
        for v in stats:
            v.clear()
        curve.append(row)
        if logger:
            logger({k: v for k, v in row.items() if k != "tag"})
        if verbose:
            print(f"    upd {row['update']:6d} loss {row['loss']:.4f} U {row['mean_q']:.3f} "
                  f"inner Sharpe {row['inner_sharpe']:+.3f} turn {row['inner_turnover']:.1f}")
        if m["sharpe"] > best["sharpe"]:
            best.update(sharpe=m["sharpe"], update=agent.updates, weights=agent.online.get_weights())

    for _ in range(total):
        idx = rng.integers(len(G), size=batch)
        g2 = G[idx] + n                                               # same ticker: t + n <= hi
        td, loss, mu = agent.learn(data.windows(G[idx]), immediate(idx), data.windows(g2),
                                   rate[g2] / data.sigma[g2], fee[g2] / data.sigma[g2],
                                   None if masks is None else masks[idx])
        for v, x in zip(stats, (loss, mu, float(np.mean(np.abs(td))))):
            v.append(x)
        if agent.updates % eval_every == 0:
            evaluate("periodic")
    evaluate("last")
    last_weights = agent.online.get_weights()
    if tr.get("select", "best_inner_val") == "best_inner_val" and best["weights"] is not None:
        agent.online.set_weights(best["weights"])
    info = {"best_update": best["update"], "best_inner_sharpe": float(best["sharpe"]),
            "last_inner_sharpe": float(curve[-1]["inner_sharpe"]), "updates": agent.updates,
            "transitions": total * batch, "seconds": time.time() - t0, "select": tr.get("select", "best_inner_val"),
            "samples": int(len(G)), "mv_lambda": lam}
    return agent, info, pd.DataFrame(curve), last_weights
