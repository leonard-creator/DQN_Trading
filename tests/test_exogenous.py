"""V1 decision structure (rl/exogenous.py, PROTOCOL Part II §V6.1): targets, no-trade band, end to end."""

import numpy as np
import pytest
import tensorflow as tf

from rl.env import MarketData
from rl.exogenous import UAgent, band_exposures, decide

NET = {"arch": "mlp", "hidden": [8]}
ALGO = {"gamma": 0.9, "lr": 1e-3, "batch_size": 4}


def test_targets_match_the_formula():
    """y(e') = rew(e') + gamma [U_tgt(m', e*) - kappa' |e* - e'|], e* chosen by the online net."""
    agent = UAgent(3, 2, 4, NET, ALGO, total_updates=10)
    rng = np.random.default_rng(0)
    m2 = rng.standard_normal((6, 3, 2)).astype(np.float32)
    rew = rng.standard_normal((6, 5)).astype(np.float32)
    kappa = np.abs(rng.standard_normal(6)).astype(np.float32)
    y = agent.targets(tf.constant(rew), tf.constant(m2), tf.constant(kappa), tf.zeros(6)).numpy()[:, 0]
    on, tg = agent.online(m2).numpy(), agent.target(m2).numpy()
    g = agent.grid
    for b in range(6):
        for i, e in enumerate(g):
            cost = kappa[b] * np.abs(g - e)
            k = int(np.argmax(on[b] - cost))
            assert y[b, i] == pytest.approx(rew[b, i] + 0.9 * (tg[b, k] - cost[k]), rel=1e-5, abs=1e-5)


class FixedU:
    """Stand-in agent with given U values per bar."""
    grid = np.arange(5) / 4

    def __init__(self, u):
        self.u = np.asarray(u, float)

    def u_values(self, market):
        return self.u[: len(market)]


def test_band_trades_only_when_the_value_gain_beats_the_cost():
    n = 6
    data = MarketData(["X"], [np.full(n + 1, 100.0)], [np.zeros((n + 1, 1))], 1, [1.0],
                      sigmas=[np.full(n + 1, 0.01)])
    rate = 0.0011                                            # kappa = 0.11 vol units per unit traded
    small = np.tile([0.0, 0.0, 0.0, 0.0, 0.05], (n, 1))      # buying everything gains 0.05 < 0.11
    big = np.tile([0.0, 0.0, 0.0, 0.0, 0.20], (n, 1))        # 0.20 > 0.11
    assert np.all(band_exposures(FixedU(small), data, [(0, n)], [rate])[:n] == 0.0)
    assert np.all(band_exposures(FixedU(big), data, [(0, n)], [rate])[:n] == 1.0)
    # once invested it holds: the way back costs as much and gains nothing
    back = np.vstack([big[:3], np.tile([0.06, 0.0, 0.0, 0.0, 0.0], (3, 1))])
    np.testing.assert_array_equal(band_exposures(FixedU(back), data, [(0, n)], [rate])[:n], 1.0)


def test_v1_trains_through_the_harness_and_its_diagnostics_reproduce_it(synthetic_agent_cfg, monkeypatch, tmp_path):
    from harness import trials as tr
    from harness.config import deep_merge
    from harness.experiment import run_experiment
    from rl.diagnostics import q_diagnostics_for_run
    monkeypatch.setattr(tr, "append_trial", lambda row: None)
    cfg = deep_merge(synthetic_agent_cfg, {"agent": {"replay": {"mode": "exogenous"}, "anchor_eta": 0.01,
                                                     "env": {"reward": "vol_scaled_pnl", "reward_clip": 10.0},
                                                     "algo": {"reward_norm": "none"}}})
    res = run_experiment(cfg, seeds=[0], cost_levels=[10], out_root=str(tmp_path), verbose=False)
    assert np.isfinite(res.metrics["sharpe"]).all()
    diag = q_diagnostics_for_run(res.output_dir, workers=1)
    assert (diag["reproduced"] == 1.0).all() and diag["gap_median"].notna().all()


def test_n_step_targets_hold_the_exposure_and_stay_inside_the_range(synthetic_agent_cfg):
    """n-step: samples need t + n <= hi; the immediate term is the discounted n-bar return."""
    from rl.exogenous import train_exogenous
    rng = np.random.default_rng(3)
    n_bars = 400
    close = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(n_bars))
    data = MarketData(["X"], [close], [rng.standard_normal((n_bars, 2))], 5, [1.0],
                      sigmas=[np.full(n_bars, 0.01)])
    cfg = {**synthetic_agent_cfg}
    cfg["agent"] = {**cfg["agent"], "network": NET, "anchor_eta": 0.0,
                    "algo": {**ALGO, "n_step": 5, "batch_size": 8}, "env": {"reward_clip": 10.0},
                    "train": {"transitions": 40, "update_ratio": 1.0, "eval_every_updates": 20}}
    agent, info, curve, _ = train_exogenous(cfg, data, [(10, 300)], [(320, 390)], 0, [0.0011])
    assert info["samples"] == 300 - 10 - 5 + 1 and agent.discount == pytest.approx(0.9 ** 5)
    assert len(curve) >= 2 and np.isfinite(curve["inner_sharpe"]).all()


def test_gated_switching_needs_the_heads_to_agree():
    """Several heads: move only if the mean gain over staying beats z x its spread across heads."""
    g = np.arange(5) / 4
    agree = np.array([[0, 0, 0, 0, 1.0], [0, 0, 0, 0, 1.2], [0, 0, 0, 0, 0.9]])   # all prefer 100 %
    split = np.array([[0, 0, 0, 0, 3.0], [0, 0, 0, 0, -1.5], [0, 0, 0, 0, 0.0]])  # same mean, no consensus
    assert g[decide(agree, 0.0, g, 0.11, 0.0, z=1.0)] == 1.0
    assert g[decide(split, 0.0, g, 0.11, 0.0, z=1.0)] == 0.0
    assert g[decide(split, 0.0, g, 0.11, 0.0, z=0.0)] == 1.0             # without the gate: mean only
    assert g[decide(agree[0], 0.0, g, 0.11, 0.0)] == 1.0                  # one head: the plain band


def test_bootstrapped_heads_train_and_gate(synthetic_agent_cfg):
    from rl.exogenous import train_exogenous
    rng = np.random.default_rng(4)
    close = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(300))
    data = MarketData(["X"], [close], [rng.standard_normal((300, 2))], 5, [1.0], sigmas=[np.full(300, 0.01)])
    cfg = {**synthetic_agent_cfg}
    cfg["agent"] = {**cfg["agent"], "network": {**NET, "layer_norm": True}, "anchor_eta": 0.01, "gate_z": 1.0,
                    "algo": {**ALGO, "heads": 4, "loss": "mse", "batch_size": 8}, "env": {"reward_clip": 10.0},
                    "train": {"transitions": 30, "update_ratio": 1.0, "eval_every_updates": 15}}
    agent, info, curve, _ = train_exogenous(cfg, data, [(10, 220)], [(240, 290)], 0, [0.0011])
    assert agent.u_values(data.windows(np.arange(20, 30))).shape == (10, 4, 5)
    e = band_exposures(agent, data, [(240, 290)], [0.0011], z=1.0)[240:290]
    assert set(np.unique(e)) <= set(agent.grid) and np.isfinite(curve["inner_sharpe"]).all()


def test_mean_variance_auto_lambda_makes_full_exposure_optimal_on_average(synthetic_agent_cfg):
    from rl.exogenous import train_exogenous
    rng = np.random.default_rng(5)
    close = 100 * np.cumprod(1 + 0.003 + 0.01 * rng.standard_normal(300))       # clearly positive drift
    data = MarketData(["X"], [close], [rng.standard_normal((300, 2))], 5, [1.0], sigmas=[np.full(300, 0.01)])
    cfg = {**synthetic_agent_cfg}
    cfg["agent"] = {**cfg["agent"], "network": NET, "anchor_eta": 0.0,
                    "algo": {**ALGO, "loss": "mse", "batch_size": 8},
                    "env": {"reward": "mean_variance", "mv_lambda": "auto", "reward_clip": 10.0},
                    "train": {"transitions": 0, "update_ratio": 1.0, "eval_every_updates": 10}}
    agent, info, _, _ = train_exogenous(cfg, data, [(10, 250)], [(260, 290)], 0, [0.0011])
    z = (close[11:251] / close[10:250] - 1) / 0.01
    assert info["mv_lambda"] == pytest.approx(z.mean() / (z ** 2).mean(), rel=1e-6)
    prior = agent.online.layers[-1].get_weights()[1]                  # untrained: the bias prior
    assert int(np.argmax(prior)) == 4                                 # e' = 1 maximises x - lambda/2 x^2 on average
