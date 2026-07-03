"""Single-stock trading environment.

All buy/sell/reward logic lives here so that training and evaluation share one
implementation (the previous code duplicated it across train.py / evaluate.py
and the copies had drifted). Key design points:

* Actions: 0 = hold, 1 = buy 1 unit, 2 = sell 1 unit.
* Inventory uses an AVERAGE COST BASIS (avg_cost); selling closes one unit and
  realizes (price - avg_cost). The average is what the model is informed about.
* A flat TRANSACTION COST is charged on every executed buy/sell.
* Reward = DIFFERENTIAL SHARPE RATIO (Moody & Saffell, 1998) of the per-step
  mark-to-market P&L (net of cost). The Sharpe ratio is scale-invariant, so the
  reward stays well-scaled across stocks of very different price levels.
"""

import numpy as np
from functions import get_window


class TradingEnv:
    def __init__(self, raw_close, feature_matrix, window,
                 transaction_cost=1.0, max_position=10,
                 dsr_eta=0.01, dsr_clip=5.0, inventory_penalty=0.0):
        # dsr_clip bounds the reward: during the first ~1/eta steps the DSR
        # variance estimate is tiny and D_t can spike enormously, so we clip it.
        assert len(raw_close) == len(feature_matrix), "price/feature length mismatch"
        self.close = np.asarray(raw_close, dtype=np.float64)
        self.features = np.asarray(feature_matrix, dtype=np.float32)
        self.window = int(window)
        self.cost = float(transaction_cost)
        self.max_position = int(max_position)
        self.eta = float(dsr_eta)                 # DSR EMA adaptation rate
        self.dsr_clip = dsr_clip                  # optional |reward| clip
        self.inv_penalty = float(inventory_penalty)
        self.n_pos = 3                            # position feature vector length
        self.reset()

    @property
    def length(self):
        # Decision steps per episode. We need close[t+1] to score the last
        # action, so the final decidable index is len-2 -> len-1 steps.
        return len(self.close) - 1

    def reset(self):
        self.t = 0
        self.position = 0
        self.avg_cost = 0.0
        self.realized_pnl = 0.0    # gross realized trading P&L (excludes costs)
        self.total_cost = 0.0      # cumulative transaction costs paid
        self.trade_count = 0
        self.A = 0.0                              # EMA of returns  (1st moment)
        self.B = 0.0                              # EMA of returns^2 (2nd moment)
        return self._obs()

    # --- observation -----------------------------------------------------
    def _running_sharpe(self):
        var = self.B - self.A * self.A
        return self.A / np.sqrt(var + 1e-8) if var > 1e-12 else 0.0

    def _obs(self):
        """Observation = (market window, position vector).

        Position vector gives the agent what it is currently holding, which the
        old state lacked entirely (the MDP was not observable before):
          [ position / max_position,
            unrealized P&L fraction vs average cost,
            running Sharpe estimate ]
        """
        market = get_window(self.features, self.t, self.window)
        price = self.close[self.t]
        unreal = (price - self.avg_cost) / self.avg_cost if self.position > 0 else 0.0
        pos_vec = np.array(
            [self.position / self.max_position, unreal, self._running_sharpe()],
            dtype=np.float32,
        )
        return (market, pos_vec)

    # --- differential Sharpe ratio reward --------------------------------
    def _dsr(self, R):
        """Differential Sharpe Ratio increment for step return R.

        Uses the PREVIOUS moment estimates (A_{t-1}, B_{t-1}) in the closed
        form, then updates the EMAs. Returns 0 during warm-up while the
        variance estimate is not yet defined.
        """
        dA = R - self.A
        dB = R * R - self.B
        denom = self.B - self.A * self.A
        if denom > 1e-12:
            D = (self.B * dA - 0.5 * self.A * dB) / (denom ** 1.5)
        else:
            D = 0.0
        # update EMAs for next step
        self.A += self.eta * dA
        self.B += self.eta * dB
        if self.dsr_clip is not None:
            D = float(np.clip(D, -self.dsr_clip, self.dsr_clip))
        return D

    # --- environment step ------------------------------------------------
    def step(self, action):
        price = self.close[self.t]
        traded = False
        trade_type = None

        # Apply the action against the average-cost-basis inventory.
        if action == 1 and self.position < self.max_position:        # BUY 1 unit
            self.avg_cost = (self.avg_cost * self.position + price) / (self.position + 1)
            self.position += 1
            traded, trade_type = True, "buy"
        elif action == 2 and self.position > 0:                      # SELL 1 unit
            self.realized_pnl += price - self.avg_cost               # realize vs average
            self.position -= 1
            if self.position == 0:
                self.avg_cost = 0.0
            traded, trade_type = True, "sell"
        # hold (action 0) or an invalid trade is a no-op
        if traded:
            self.trade_count += 1
            self.total_cost += self.cost

        # Per-step P&L: mark-to-market of the position we now hold over the
        # next price move, minus the flat cost, minus optional inventory penalty.
        next_price = self.close[self.t + 1]
        R = self.position * (next_price - price)
        if traded:
            R -= self.cost
        if self.inv_penalty:
            R -= self.inv_penalty * self.position ** 2

        reward = self._dsr(R)                                        # learning signal
        self.t += 1
        done = self.t >= self.length

        # net equity = realized P&L net of costs + open unrealized position value
        equity = self.realized_pnl - self.total_cost + self.position * (next_price - self.avg_cost)
        info = {
            "price": price, "traded": traded, "trade_type": trade_type,
            "position": self.position, "avg_cost": self.avg_cost,
            "realized_pnl": self.realized_pnl, "total_cost": self.total_cost,
            "net_pnl": self.realized_pnl - self.total_cost,   # profit after transaction costs
            "step_pnl": R, "equity": equity,
        }
        return self._obs(), reward, done, info
