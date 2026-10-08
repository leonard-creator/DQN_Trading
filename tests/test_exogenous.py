"""V1 decision structure (rl/exogenous.py, PROTOCOL Part II §V6.1): targets, no-trade band, end to end."""

import os

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


def test_fine_tuning_starts_from_the_seeds_pretrained_checkpoint(synthetic_agent_cfg, tmp_path):
    """Step 4: with agent.pretrain the U network starts from <dir>/agent/PRE_s<seed>/best.weights.h5."""
    from rl.exogenous import train_exogenous
    rng = np.random.default_rng(6)
    close = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(300))
    data = MarketData(["X"], [close], [rng.standard_normal((300, 2))], 5, [1.0], sigmas=[np.full(300, 0.01)])
    algo = {**ALGO, "loss": "mse", "batch_size": 8}
    pre = UAgent(5, 2, 4, NET, algo, total_updates=1)
    os.makedirs(tmp_path / "agent" / "PRE_s3")
    pre.online.save_weights(str(tmp_path / "agent" / "PRE_s3" / "best.weights.h5"))
    cfg = {**synthetic_agent_cfg}
    cfg["agent"] = {**cfg["agent"], "network": NET, "anchor_eta": 0.01, "algo": algo, "env": {"reward_clip": 10.0},
                    "pretrain": {"dir": str(tmp_path)},
                    "train": {"transitions": 0, "update_ratio": 1.0, "eval_every_updates": 10}}
    agent, info, _, _ = train_exogenous(cfg, data, [(10, 250)], [(260, 290)], 3, [0.0011])
    m = data.windows(np.arange(20, 30))
    np.testing.assert_allclose(agent.u_values(m), pre.u_values(m), rtol=1e-6)
    assert info["pretrained"].endswith("PRE_s3/best.weights.h5")


def test_hl_gauss_target_is_a_smoothed_histogram_with_the_target_as_its_mean():
    """HL-Gauss (Farebrother et al. 2024): bin masses sum to 1, their mean is the target, far-out targets clip."""
    from rl.exogenous import hl_probs
    edges = tf.constant(np.linspace(-5.0, 5.0, 52), tf.float32)
    centres = (edges[1:] + edges[:-1]).numpy() / 2
    p = hl_probs(tf.constant([-1.3, 0.0, 2.2, 40.0]), edges, 0.75 * 10 / 51).numpy()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(p[:3] @ centres, [-1.3, 0.0, 2.2], atol=1e-3)
    assert np.isfinite(p).all() and p[3].argmax() == 50                  # clipped into the top bin


def test_hl_gauss_buy_and_hold_prior_starts_each_value_at_its_prior():
    agent = UAgent(3, 2, 4, NET, {**ALGO, "loss": "hl_gauss", "heads": 3}, total_updates=1,
                   bias=np.array([-0.5, 0.0, 0.5, 1.0, 1.5]), support=(-4.0, 6.0))
    kernel, b = agent.online.layers[-1].get_weights()
    agent.online.layers[-1].set_weights([np.zeros_like(kernel), b])         # output = the bias logits only
    u = agent.u_values(np.random.default_rng(0).standard_normal((2, 3, 2)).astype(np.float32))
    np.testing.assert_allclose(u, np.broadcast_to([-0.5, 0.0, 0.5, 1.0, 1.5], (2, 3, 5)), atol=1e-3)


def test_hl_gauss_agent_takes_its_bins_from_the_training_samples_and_rebuilds(synthetic_agent_cfg, tmp_path):
    """The support comes from the training data, is stored in info, and a rebuilt agent (diagnostics) reproduces U."""
    from rl.exogenous import train_exogenous
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + 0.01 * rng.standard_normal(300))
    data = MarketData(["X"], [close], [rng.standard_normal((300, 2))], 5, [1.0], sigmas=[np.full(300, 0.01)])
    cfg = {**synthetic_agent_cfg}
    algo = {**ALGO, "loss": "hl_gauss", "bins": 11, "heads": 2, "n_step": 3, "batch_size": 8}
    cfg["agent"] = {**cfg["agent"], "network": NET, "anchor_eta": 0.0, "gate_z": 0.5, "algo": algo,
                    "env": {"reward_clip": 10.0}, "train": {"transitions": 40, "update_ratio": 1.0, "eval_every_updates": 20}}
    agent, info, curve, _ = train_exogenous(cfg, data, [(10, 250)], [(260, 290)], 0, [0.0011])
    lo, hi = info["hl_support"]
    m = data.windows(np.arange(20, 30))
    u = agent.u_values(m)
    assert lo < 0 < hi and u.shape == (10, 2, 5) and (u > lo).all() and (u < hi).all()
    assert curve["loss"].notna().any() and np.isfinite(curve["inner_sharpe"]).all()
    agent.online.save_weights(str(tmp_path / "w.h5"))
    again = UAgent(5, 2, 4, NET, algo, total_updates=1, support=info["hl_support"])
    again.online.load_weights(str(tmp_path / "w.h5"))
    np.testing.assert_allclose(again.u_values(m), u, rtol=1e-5, atol=1e-6)


def test_random_prior_is_frozen_added_to_the_output_and_saved(tmp_path):
    """prior_scale beta: out = base(m) + beta prior(m); only the base learns; the prior is in the weights file."""
    algo = {**ALGO, "loss": "mse", "heads": 4, "prior_scale": 3.0, "batch_size": 8}
    agent = UAgent(5, 2, 4, NET, algo, total_updates=10)
    nets = [layer for layer in agent.online.layers if isinstance(layer, tf.keras.Model)]
    (base,), (prior,) = [[n for n in nets if n.trainable is flag] for flag in (True, False)]
    assert len(agent.online.trainable_weights) == len(base.weights)          # only the base learns
    rng = np.random.default_rng(0)
    m = rng.standard_normal((8, 5, 2)).astype(np.float32)
    np.testing.assert_allclose(agent.online(m), base(m) + 3.0 * prior(m), rtol=1e-5, atol=1e-5)
    frozen, learnt = ([w.numpy().copy() for w in ws] for ws in (prior.weights, base.weights))
    for _ in range(3):
        agent.learn(m, rng.standard_normal((8, 5)), m, np.full(8, 0.1), np.zeros(8))
    assert all(np.array_equal(a, w.numpy()) for a, w in zip(frozen, prior.weights))
    assert not all(np.array_equal(a, w.numpy()) for a, w in zip(learnt, base.weights))
    tprior = [layer for layer in agent.target.layers if isinstance(layer, tf.keras.Model) and not layer.trainable][0]
    assert all(np.array_equal(a, w.numpy()) for a, w in zip(frozen, tprior.weights))   # target: same prior
    agent.online.save_weights(str(tmp_path / "w.h5"))
    again = UAgent(5, 2, 4, NET, algo, total_updates=1)
    again.online.load_weights(str(tmp_path / "w.h5"))
    np.testing.assert_allclose(again.u_values(m), agent.u_values(m), rtol=1e-5, atol=1e-6)


def test_weight_decay_uses_adamw_and_prior_none_skips_the_buy_and_hold_start(synthetic_agent_cfg):
    from rl.exogenous import train_exogenous
    assert type(UAgent(5, 2, 4, NET, {**ALGO, "weight_decay": 0.1}, total_updates=1).opt).__name__ == "AdamW"
    assert type(UAgent(5, 2, 4, NET, ALGO, total_updates=1).opt).__name__ == "Adam"
    with pytest.raises(ValueError):
        UAgent(5, 2, 4, NET, {**ALGO, "loss": "msee"}, total_updates=1)          # typos fail loudly
    rng = np.random.default_rng(8)
    close = 100 * np.cumprod(1 + 0.003 + 0.01 * rng.standard_normal(300))       # clearly positive drift
    data = MarketData(["X"], [close], [rng.standard_normal((300, 2))], 5, [1.0], sigmas=[np.full(300, 0.01)])
    cfg = {**synthetic_agent_cfg}
    cfg["agent"] = {**cfg["agent"], "network": NET, "anchor_eta": 0.01, "prior": "none",
                    "algo": {**ALGO, "loss": "mse"}, "env": {"reward_clip": 10.0},
                    "train": {"transitions": 0, "update_ratio": 1.0, "eval_every_updates": 10}}
    agent, _, _, _ = train_exogenous(cfg, data, [(10, 250)], [(260, 290)], 0, [0.0011])
    assert np.all(agent.online.layers[-1].get_weights()[1] == 0.0)           # Keras' zero bias, no prior
