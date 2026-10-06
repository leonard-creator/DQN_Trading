"""Vectorised trading environment (spec Phases 2-3, PROTOCOL §5 position sizing).

B environments are stepped in lockstep with NumPy, so the agent needs ONE
network call per step for all of them. The old code made one
`predict_on_batch` call per environment step, which dominated run time.

Market data
-----------
`MarketData` holds every ticker's prices and feature rows in one flat array.
A decision is identified by its GLOBAL bar index g = offset[ticker] + t. The
replay buffer stores g instead of a copy of the (window x features) block,
and the window is cut out of the flat array when a batch is sampled. That
is ~280x less memory per transition and works unchanged for many tickers.

Positions and actions (PROTOCOL §5, decision 3)
-----------------------------------------------
Exposure = k / K of the capital, k integer, K = evaluation.position_levels (4).
Long-only: k in {0..K}. With allow_short: k in {-K..K}.
Actions: 0 = hold, 1 = buy (k += 1), 2 = sell (k -= 1).

Action masking (spec Phase 2.4): buy at k = k_max and sell at k = k_min are
INVALID. With masking on, the env reports a valid-action mask that the agent
applies to epsilon-greedy and to the target argmax/max. With masking off
(legacy), an invalid action is silently executed as hold, as in the old env.

Per-step return, identical to harness/backtest.py
-------------------------------------------------
For a decision at bar t that moves the exposure from e_old to e_new:
    R_t = e_new * (close[t+1] / close[t] - 1)
          - rate * |e_new - e_old|              proportional cost
          - fee_frac * 1[e_new != e_old]        fixed fee per transaction (each buy and each sell)
          - hold * |e_new|                      holding cost per bar (e.g. TER)
with one harness.backtest.CostModel per ticker (a plain float = proportional
rate only, the PROTOCOL levels). tests/test_rl.py checks that R over an
evaluation block equals the harness backtest of the same exposures exactly.

Rewards (env.reward; spec Phase 3)
----------------------------------
R is the net per-step return above. sigma_t is the ex-ante daily volatility
of the ticker (EWMA of past log returns, rl/features.ex_ante_vol), known at
the decision bar. Vol-scaled rewards are in units of "typical daily moves".

    diff_sharpe    : differential Sharpe ratio of R (Moody & Saffell 1998), EMA
                     rate eta, moments reset at each episode start, clipped
    pnl            : R
    profit         : LEGACY. Realised gain when a unit is sold, (price/avg - 1)/K,
                     0 otherwise, costs not included (as in the original q-trader)
    mean_variance  : x - (lambda/2) x^2  with x = R / sigma_t   (plan Phase B)
    vol_scaled_pnl : R / sigma_t   (reward side of Zhang, Zohren & Roberts 2019;
                     our exposures stay on the PROTOCOL grid, not vol-targeted)
    active_return  : (R - r_bench) / sigma_t, r_bench = close[t+1]/close[t] - 1,
                     i.e. the return BEYOND being fully invested (added in M3:
                     it rewards exactly what H1 tests, beating buy-and-hold)

Training-only shaping (never changes the measured return R):
    cost_penalty_mult m : the reward sees R - (m - 1) * cost, a "shadow cost"
                     that makes trades look m times as expensive (added in M3
                     against the overtrading found in M2). Default 1 = off.
    holding_penalty     : rho(h) subtracted every bar a position is held, h =
                     bars held; none (default) | linear: k*h/252 |
                     exp: k*(exp(alpha*h/252) - 1)   (plan Phase B, ablation only)

Every step also returns the reward components (rc_*) so the trainer can log
them separately (spec Phase 3).

State: position vector (spec Phase 3)
-------------------------------------
    one-hot(k)                      K+1 entries (2K+1 with shorting)
    unrealised P&L                  sign(k) * (price/avg_entry - 1) / vol_scale, clipped to +-5
    holding time                    log(1 + bars held) / log(1 + 252)
vol_scale = typical 20-bar move of the ticker over the fit range (rl/features.py).

Episodes
--------
    random_window : start bar uniform in the training range, fixed `horizon`
                    bars, then a new (ticker, start). Time-limit ends are
                    TRUNCATIONS, not terminal: the agent still bootstraps
                    through them (the market does not end there).
    full_series   : LEGACY. Every episode walks the whole training range from
                    its first bar, and the last step is terminal (old env).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HOLD, BUY, SELL = 0, 1, 2
N_ACTIONS = 3


class MarketData:
    """Prices and features of several tickers, flattened (see module docstring)."""

    def __init__(self, tickers, closes, features, window, vol_scales, dates=None, sigmas=None):
        self.tickers = list(tickers)
        self.dates = list(dates) if dates is not None else None     # one DatetimeIndex per ticker
        # ex-ante daily volatility per bar (only needed by vol-scaled rewards)
        self.sigma = (np.concatenate([np.asarray(s, np.float64) for s in sigmas])
                      if sigmas is not None else None)
        self.window = int(window)
        self.lengths = np.array([len(c) for c in closes], dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths)[:-1])).astype(np.int64)
        self.close = np.concatenate([np.asarray(c, np.float64) for c in closes])
        self.feat = np.ascontiguousarray(np.concatenate(features).astype(np.float32))
        self.vol_scale = np.asarray(vol_scales, dtype=np.float64)
        self.n_feat = self.feat.shape[1]
        # (N - W + 1, F, W) read-only view; row j holds bars j .. j+W-1
        self._win = sliding_window_view(self.feat, self.window, axis=0)

    def windows(self, g):
        """Observation windows (B, W, F) ending at global bars g (inclusive)."""
        g = np.asarray(g, dtype=np.int64)
        return np.ascontiguousarray(self._win[g - self.window + 1].transpose(0, 2, 1))

    def global_index(self, asset, t):
        return self.offsets[asset] + t


class VecTradingEnv:
    """B environments over MarketData.

    ranges : one (lo, hi) per ticker: decisions are taken at bars t with
             lo <= t < hi. The last decision (hi - 1) reads close[hi], so `hi`
             must be a bar the caller is allowed to use.
    mode   : "train" (random episodes, auto-reset) or "eval" (env b = ticker b,
             one pass from lo to hi, starting flat)
    """

    def __init__(self, data, ranges, env_cfg, levels, allow_short, cost,
                 n_envs=None, mode="train", seed=0):
        self.d = data
        self.ranges = np.asarray(ranges, dtype=np.int64)          # (n_tickers, 2)
        if np.any(self.ranges[:, 0] < data.window - 1):
            raise ValueError("a decision range starts before a full observation window exists")
        if np.any(self.ranges[:, 1] >= data.lengths):
            raise ValueError("a decision range reads past the end of a ticker's data")
        if np.any(self.ranges[:, 1] <= self.ranges[:, 0]):
            raise ValueError("empty decision range")
        self.cfg = env_cfg
        self.K = int(levels)
        self.k_min = -self.K if allow_short else 0
        self.k_max = self.K
        self.n_levels = self.k_max - self.k_min + 1
        self.pos_dim = self.n_levels + 2
        # one CostModel per ticker (or one shared); stored as per-ticker arrays
        from harness.backtest import as_cost_model
        cms = [as_cost_model(c) for c in cost] if isinstance(cost, (list, tuple)) \
            else [as_cost_model(cost)] * len(self.ranges)
        if len(cms) != len(self.ranges):
            raise ValueError("need one cost model per ticker range")
        self.rate_a = np.array([c.rate for c in cms])
        self.fee_a = np.array([c.fee_frac for c in cms])
        self.hold_a = np.array([c.hold for c in cms])
        self.mode = mode
        self.masking = bool(env_cfg.get("action_masking", True))
        self.reward_type = env_cfg.get("reward", "diff_sharpe")
        self.eta = float(env_cfg.get("diff_sharpe_eta", 0.01))
        self.clip = env_cfg.get("reward_clip")
        self.mv_lambda = float(env_cfg.get("mv_lambda", 0.5))
        self.shadow_mult = float(env_cfg.get("cost_penalty_mult", 1.0))
        hp = env_cfg.get("holding_penalty") or {"type": "none"}
        self.hold_type = hp.get("type", "none")
        self.hold_k, self.hold_alpha = float(hp.get("k", 0.0)), float(hp.get("alpha", 1.0))
        if self.reward_type not in ("diff_sharpe", "pnl", "profit", "mean_variance",
                                    "vol_scaled_pnl", "active_return"):
            raise ValueError(f"unknown reward '{self.reward_type}'")
        if self.hold_type not in ("none", "linear", "exp"):
            raise ValueError(f"unknown holding_penalty type '{self.hold_type}'")
        self.needs_sigma = self.reward_type in ("mean_variance", "vol_scaled_pnl", "active_return")
        if self.needs_sigma and data.sigma is None:
            raise ValueError(f"reward '{self.reward_type}' needs MarketData(sigmas=...)")
        self.episode = env_cfg.get("episode", "random_window")
        self.horizon = int(env_cfg.get("horizon", 252))
        self.rng = np.random.default_rng(seed)
        self.B = len(self.ranges) if mode == "eval" else int(n_envs)
        # length-weighted ticker sampling gives every BAR the same chance
        w = (self.ranges[:, 1] - self.ranges[:, 0]).astype(np.float64)
        self.p_asset = (w / w.sum()) if env_cfg.get("asset_sampling") == "length" else None

        z = lambda dt=np.float64: np.zeros(self.B, dtype=dt)          # noqa: E731
        self.asset, self.t, self.end = z(np.int64), z(np.int64), z(np.int64)
        self.k, self.hold = z(np.int64), z(np.int64)
        self.avg, self.A, self.Bm = z(), z(), z()                     # entry price, diff-Sharpe moments
        self.ep_ret = z()
        self.finished_returns = []                                     # episode net returns (train mode)
        self.reset_all()

    # ------------------------------------------------------------------ reset
    def reset_all(self):
        if self.mode == "eval":
            self.asset = np.arange(self.B, dtype=np.int64)
            self.t = self.ranges[:, 0].copy()
            self.end = self.ranges[:, 1].copy()
            self._clear(np.ones(self.B, dtype=bool))
        else:
            self._reset(np.ones(self.B, dtype=bool))

    def _clear(self, idx):
        """Flat position and fresh reward statistics for envs `idx` (bool mask)."""
        self.k[idx] = 0
        self.hold[idx] = 0
        self.avg[idx] = 0.0
        self.A[idx] = 0.0
        self.Bm[idx] = 0.0
        self.ep_ret[idx] = 0.0

    def _reset(self, idx):
        n = int(idx.sum())
        if n == 0:
            return
        a = self.rng.choice(len(self.ranges), size=n, p=self.p_asset)
        lo, hi = self.ranges[a, 0], self.ranges[a, 1]
        if self.episode == "full_series":
            start, end = lo, hi
        else:
            latest = np.maximum(lo, hi - self.horizon)                # inclusive latest start
            start = lo + (self.rng.random(n) * (latest - lo + 1)).astype(np.int64)
            end = np.minimum(start + self.horizon, hi)
        self.asset[idx], self.t[idx], self.end[idx] = a, start, end
        self._clear(idx)

    # ------------------------------------------------------------ observation
    @property
    def active(self):
        return self.t < self.end

    def g(self):
        return self.d.offsets[self.asset] + self.t

    def valid_mask(self):
        """(B, 3) bool: hold always; buy below k_max; sell above k_min."""
        m = np.ones((self.B, N_ACTIONS), dtype=bool)
        if self.masking:
            m[:, BUY] = self.k < self.k_max
            m[:, SELL] = self.k > self.k_min
        return m

    def position_vector(self):
        pv = np.zeros((self.B, self.pos_dim), dtype=np.float32)
        pv[np.arange(self.B), self.k - self.k_min] = 1.0
        price = self.d.close[np.minimum(self.g(), len(self.d.close) - 1)]
        with np.errstate(divide="ignore", invalid="ignore"):
            unreal = np.where(self.k != 0, np.sign(self.k) * (price / self.avg - 1.0), 0.0)
        unreal = unreal / self.d.vol_scale[self.asset]
        pv[:, -2] = np.clip(np.nan_to_num(unreal), -5.0, 5.0)
        pv[:, -1] = np.log1p(self.hold) / np.log1p(252.0)
        return pv

    def observe(self):
        """(global bar index, position vector, valid-action mask) for acting."""
        return self.g(), self.position_vector(), self.valid_mask()

    # ------------------------------------------------------------------ step
    def _diff_sharpe(self, R):
        """Differential Sharpe ratio increment (uses the PREVIOUS moments)."""
        dA = R - self.A
        dB = R * R - self.Bm
        var = self.Bm - self.A ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.where(var > 1e-12, (self.Bm * dA - 0.5 * self.A * dB) / np.maximum(var, 1e-12) ** 1.5, 0.0)
        self.A = self.A + self.eta * dA
        self.Bm = self.Bm + self.eta * dB
        return D

    def step(self, actions):
        """Apply one action per env. Inactive envs (eval mode, finished) are frozen.

        Returns a dict of arrays (length B):
          reward, net_return (R), cost, exposure (after the action), turnover,
          done (episode ended), terminal (bootstrap must stop),
          g_next, pos_next, mask_next (the TRUE next observation; in train
          mode finished envs are reset AFTER these are recorded).
        """
        act = np.asarray(actions, dtype=np.int64)
        live = self.active
        g = self.g()
        n = len(self.d.close)
        p0 = self.d.close[np.minimum(g, n - 1)]
        p1 = self.d.close[np.minimum(g + 1, n - 1)]

        k_old = self.k.copy()
        buy = live & (act == BUY) & (k_old < self.k_max)               # invalid buy  -> hold
        sell = live & (act == SELL) & (k_old > self.k_min)             # invalid sell -> hold
        k_new = k_old + buy.astype(np.int64) - sell.astype(np.int64)
        e_old, e_new = k_old / self.K, k_new / self.K
        turnover = np.abs(e_new - e_old)
        a_ = self.asset
        cost = (self.rate_a[a_] * turnover + self.fee_a[a_] * (turnover > 1e-12)
                + self.hold_a[a_] * np.abs(e_new))
        R = np.where(live, e_new * (p1 / p0 - 1.0) - cost, 0.0)

        # realised gain of the unit closed this step (legacy "profit" reward)
        with np.errstate(divide="ignore", invalid="ignore"):
            realised = np.where(sell & (k_old > 0), (p0 / self.avg - 1.0) / self.K, 0.0) \
                     + np.where(buy & (k_old < 0), (1.0 - p0 / self.avg) / self.K, 0.0)
        realised = np.nan_to_num(realised)

        # average entry price: updated when |k| grows, cleared when flat
        grow = (buy & (k_old >= 0)) | (sell & (k_old <= 0))
        n_old, n_new = np.abs(k_old), np.abs(k_new)
        self.avg = np.where(grow, (self.avg * n_old + p0) / np.maximum(n_new, 1), self.avg)
        self.avg = np.where(k_new == 0, 0.0, self.avg)
        self.hold = np.where(k_new == 0, 0, np.where(k_old == 0, 1, self.hold + 1))
        self.hold = np.where(live, self.hold, 0)
        self.k = k_new

        # training-only shadow cost: the reward sees trades as m times as expensive
        shadow = (self.shadow_mult - 1.0) * cost
        R_train = R - shadow
        sig = self.d.sigma[np.minimum(g, n - 1)] if self.needs_sigma else None
        risk = np.zeros(self.B)
        if self.reward_type == "diff_sharpe":
            reward = self._diff_sharpe(R_train)
        elif self.reward_type == "pnl":
            reward = R_train
        elif self.reward_type == "profit":
            reward = realised - shadow
        elif self.reward_type == "mean_variance":
            x = R_train / sig
            risk = 0.5 * self.mv_lambda * x * x
            reward = x - risk
        elif self.reward_type == "vol_scaled_pnl":
            reward = R_train / sig
        else:                                                      # active_return
            bench = p1 / p0 - 1.0                                  # being fully invested
            reward = (R_train - bench) / sig

        # holding penalty rho(h), h = bars the current position has been held
        h = self.hold.astype(np.float64)
        if self.hold_type == "linear":
            rho = self.hold_k * h / 252.0
        elif self.hold_type == "exp":
            rho = self.hold_k * (np.exp(self.hold_alpha * h / 252.0) - 1.0)
        else:
            rho = np.zeros(self.B)
        rho = np.where(self.k != 0, rho, 0.0)
        reward = reward - rho
        if self.clip is not None:
            reward = np.clip(reward, -float(self.clip), float(self.clip))
        reward = np.where(live, reward, 0.0)

        self.ep_ret = self.ep_ret + R
        self.t = np.where(live, self.t + 1, self.t)
        done = live & (self.t >= self.end)
        terminal = done & (self.episode == "full_series")

        out = {"reward": reward.astype(np.float32), "net_return": R, "cost": cost,
               "exposure": e_new, "turnover": turnover, "done": done, "terminal": terminal,
               "live": live, "g_next": self.g(), "pos_next": self.position_vector(),
               "mask_next": self.valid_mask(),
               # reward components for logging (spec Phase 3); zero where not live
               "rc_pnl": np.where(live, R, 0.0), "rc_cost": np.where(live, cost, 0.0),
               "rc_shadow": np.where(live, shadow, 0.0), "rc_risk": np.where(live, risk, 0.0),
               "rc_hold": np.where(live, rho, 0.0)}

        if self.mode == "train" and done.any():
            self.finished_returns.extend(self.ep_ret[done].tolist())
            self._reset(done)
        return out
