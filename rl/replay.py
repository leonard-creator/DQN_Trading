"""Replay buffers (spec Phase 2.6) and the n-step return collector.

Each stored transition holds the GLOBAL bar indices of s and s' (see
rl/env.py) instead of the observation windows; the windows are cut out of
MarketData when a batch is sampled.

Stored per transition:
    g, pos, mask           state s: bar index, position vector, valid-action mask
    action
    reward                 RAW reward (sum of n discounted rewards for n-step);
                           reward normalisation is applied at sample time, so
                           old and new transitions are always on the same scale
    g2, pos2, mask2        the next state s' (n steps later for n-step)
    discount               gamma^n * (1 - terminal): 0 stops the bootstrap

PrioritizedReplay is proportional PER (Schaul et al. 2016):
    P(i) = p_i^alpha / sum_j p_j^alpha,  p_i = |TD error_i| + eps
    importance weight w_i = (N * P(i))^-beta / max_j w_j, beta annealed to 1.
New transitions get the current maximum priority, so each is replayed at
least once with high probability.
"""

from collections import deque

import numpy as np


class ReplayBuffer:
    """Uniform ring buffer."""

    def __init__(self, capacity, pos_dim, n_actions=3, seed=0):
        self.capacity = int(capacity)
        c = self.capacity
        self.g = np.zeros(c, np.int64)
        self.g2 = np.zeros(c, np.int64)
        self.pos = np.zeros((c, pos_dim), np.float32)
        self.pos2 = np.zeros((c, pos_dim), np.float32)
        self.mask = np.zeros((c, n_actions), bool)
        self.mask2 = np.zeros((c, n_actions), bool)
        self.action = np.zeros(c, np.int32)
        self.reward = np.zeros(c, np.float32)
        self.discount = np.zeros(c, np.float32)
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add_batch(self, g, pos, mask, action, reward, g2, pos2, mask2, discount):
        """Append a batch of transitions; returns the slots they were written to."""
        n = len(g)
        slots = (self.idx + np.arange(n)) % self.capacity
        self.g[slots], self.pos[slots], self.mask[slots] = g, pos, mask
        self.action[slots], self.reward[slots] = action, reward
        self.g2[slots], self.pos2[slots], self.mask2[slots] = g2, pos2, mask2
        self.discount[slots] = discount
        self.idx = int((self.idx + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))
        return slots

    def _gather(self, i):
        return {"idx": i, "g": self.g[i], "pos": self.pos[i], "mask": self.mask[i],
                "action": self.action[i], "reward": self.reward[i], "g2": self.g2[i],
                "pos2": self.pos2[i], "mask2": self.mask2[i], "discount": self.discount[i],
                "weights": np.ones(len(i), np.float32)}

    def sample(self, batch_size, beta=None):
        return self._gather(self.rng.integers(0, self.size, size=batch_size))

    def update_priorities(self, idx, td):
        """No-op for uniform replay (same call signature as PER)."""


class SumTree:
    """Binary tree over `capacity` leaves; each node holds the sum of its children."""

    def __init__(self, capacity):
        self.n = 1
        while self.n < capacity:
            self.n *= 2
        self.tree = np.zeros(2 * self.n, np.float64)     # leaves at [n, 2n)

    def total(self):
        return self.tree[1]

    def set(self, leaf_idx, values):
        """Set leaf values (vectorised) and refresh all ancestors."""
        pos = np.asarray(leaf_idx, np.int64) + self.n
        self.tree[pos] = values
        pos = np.unique(pos // 2)
        while pos[0] >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            if pos[0] == 1:
                break
            pos = np.unique(pos // 2)

    def find(self, mass):
        """Leaf index for each cumulative mass value (vectorised descent)."""
        pos = np.ones(len(mass), np.int64)
        mass = np.asarray(mass, np.float64).copy()
        while pos[0] < self.n:
            left = 2 * pos
            go_right = mass > self.tree[left]
            mass = np.where(go_right, mass - self.tree[left], mass)
            pos = np.where(go_right, left + 1, left)
        return pos - self.n


class PrioritizedReplay(ReplayBuffer):
    """Proportional prioritised replay (see module docstring)."""

    def __init__(self, capacity, pos_dim, n_actions=3, alpha=0.6, eps=1e-6, seed=0):
        super().__init__(capacity, pos_dim, n_actions, seed)
        self.alpha, self.eps = float(alpha), float(eps)
        self.tree = SumTree(self.capacity)
        self.max_p = 1.0

    def add_batch(self, *args, **kwargs):
        slots = super().add_batch(*args, **kwargs)
        self.tree.set(slots, np.full(len(slots), self.max_p ** self.alpha))
        return slots

    def sample(self, batch_size, beta=0.4):
        total = self.tree.total()
        # stratified: one draw from each of batch_size equal slices of the mass
        bounds = np.linspace(0.0, total, batch_size + 1)
        mass = bounds[:-1] + self.rng.random(batch_size) * np.diff(bounds)
        i = np.minimum(self.tree.find(np.minimum(mass, total * (1 - 1e-12))), self.size - 1)
        p = self.tree.tree[i + self.tree.n] / total
        w = (self.size * np.maximum(p, 1e-12)) ** (-beta)
        out = self._gather(i)
        out["weights"] = (w / w.max()).astype(np.float32)
        return out

    def update_priorities(self, idx, td):
        p = np.abs(np.asarray(td, np.float64)) + self.eps
        self.max_p = max(self.max_p, float(p.max()))
        self.tree.set(idx, p ** self.alpha)


class NStepCollector:
    """Turns per-env 1-step transitions into n-step transitions.

    For each env it keeps the last n steps. A transition from s_t is emitted
    once n rewards are known (R = sum_i gamma^i r_{t+i}, next state s_{t+n},
    discount gamma^n), or earlier when the episode ends: then the remaining
    steps are flushed with shorter sums and discount gamma^k * (1 - terminal).
    With n = 1 this is the ordinary 1-step transition.
    """

    def __init__(self, n_envs, n, gamma):
        self.n, self.gamma = int(n), float(gamma)
        self.q = [deque() for _ in range(n_envs)]

    def push(self, g, pos, mask, action, step):
        """Add one vectorised env step; returns a dict of emitted transitions (or None)."""
        if self.n == 1:                                   # vectorised fast path
            live = step["live"]
            if not live.any():
                return None
            return {"g": g[live], "pos": pos[live], "mask": mask[live], "action": action[live],
                    "reward": step["reward"][live], "g2": step["g_next"][live],
                    "pos2": step["pos_next"][live], "mask2": step["mask_next"][live],
                    "discount": (self.gamma * (1.0 - step["terminal"][live])).astype(np.float32)}
        out = {k: [] for k in ("g", "pos", "mask", "action", "reward", "g2", "pos2", "mask2", "discount")}
        for b in range(len(self.q)):
            if not step["live"][b]:
                continue
            q = self.q[b]
            q.append((g[b], pos[b], mask[b], action[b], float(step["reward"][b])))
            nxt = (step["g_next"][b], step["pos_next"][b], step["mask_next"][b])
            end = bool(step["done"][b])
            term = bool(step["terminal"][b])
            while q and (len(q) == self.n or end):
                R = sum(self.gamma ** i * tr[4] for i, tr in enumerate(q))
                disc = self.gamma ** len(q) * (0.0 if term else 1.0)
                s = q.popleft()
                for key, v in zip(("g", "pos", "mask", "action", "reward", "g2", "pos2", "mask2", "discount"),
                                  (s[0], s[1], s[2], s[3], R, *nxt, disc)):
                    out[key].append(v)
                if not end:
                    break
        if not out["g"]:
            return None
        return {k: np.asarray(v) for k, v in out.items()}
