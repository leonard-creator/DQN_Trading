"""M3 reward designs (spec Phase 3): formulas, shaping terms, logging, no look-ahead."""

import numpy as np
import pytest

from harness.backtest import backtest
from rl.env import BUY, HOLD, SELL, MarketData, VecTradingEnv
from rl.features import ex_ante_vol

SIG = 0.01      # constant ex-ante volatility used in the formula checks


def market(n=300, seed=0, sigma=SIG):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    return MarketData(["X"], [close], [rng.standard_normal((n, 2)).astype(np.float32)], 5, [0.05],
                      sigmas=[np.full(n, sigma)])


def run(env_cfg, actions, rate=0.001, d=None):
    d = d or market()
    env = VecTradingEnv(d, [(10, 250)], env_cfg, 4, False, rate, mode="eval")
    outs, ts = [], []
    for a in actions:
        ts.append(env.t[0])
        outs.append(env.step([a]))
    return d, outs, ts


ACTIONS = [BUY, BUY, HOLD, HOLD, SELL, HOLD, BUY, SELL, SELL, HOLD]


@pytest.mark.parametrize("reward", ["vol_scaled_pnl", "mean_variance", "active_return"])
def test_vol_scaled_reward_formulas(reward):
    d, outs, ts = run({"reward": reward, "mv_lambda": 0.5}, ACTIONS)
    for o, t in zip(outs, ts):
        R = o["net_return"][0]
        x = R / SIG
        if reward == "vol_scaled_pnl":
            expected = x
        elif reward == "mean_variance":
            expected = x - 0.25 * x * x
        else:
            bench = d.close[t + 1] / d.close[t] - 1
            expected = (R - bench) / SIG
        assert o["reward"][0] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_active_return_is_zero_when_fully_invested_without_trading():
    d = market()
    env = VecTradingEnv(d, [(10, 250)], {"reward": "active_return"}, 1, False, 0.0, mode="eval")
    env.step([BUY])                                  # K = 1: one buy = fully invested
    out = env.step([HOLD])
    assert out["reward"][0] == pytest.approx(0.0, abs=1e-12)


def test_shadow_cost_changes_the_reward_but_never_the_measured_return():
    _, base, _ = run({"reward": "vol_scaled_pnl"}, ACTIONS)
    _, shad, _ = run({"reward": "vol_scaled_pnl", "cost_penalty_mult": 3}, ACTIONS)
    for b, s in zip(base, shad):
        assert s["net_return"][0] == b["net_return"][0]                     # real return unchanged
        assert s["reward"][0] == pytest.approx(b["reward"][0] - 2 * b["cost"][0] / SIG)
        assert s["rc_shadow"][0] == pytest.approx(2 * b["cost"][0])


def test_env_still_equals_harness_with_shaping_on():
    d = market()
    env = VecTradingEnv(d, [(10, 250)], {"reward": "mean_variance", "cost_penalty_mult": 4,
                                         "holding_penalty": {"type": "linear", "k": 1.0}},
                        4, False, 0.0011, mode="eval")
    rng = np.random.default_rng(1)
    expo, R = np.full(len(d.close), np.nan), []
    while env.active.any():
        t = env.t[0]
        o = env.step(rng.integers(0, 3, size=1))
        expo[t] = o["exposure"][0]
        R.append(o["net_return"][0])
    np.testing.assert_allclose(R, backtest(d.close, expo, np.arange(11, 251), 0.0011)["net"], atol=1e-12)


@pytest.mark.parametrize("kind,expected", [("linear", lambda h: 0.5 * h / 252),
                                           ("exp", lambda h: 0.5 * (np.exp(2.0 * h / 252) - 1))])
def test_holding_penalty_grows_with_time_held_and_is_zero_when_flat(kind, expected):
    cfg = {"reward": "pnl", "holding_penalty": {"type": kind, "k": 0.5, "alpha": 2.0}}
    _, outs, _ = run(cfg, [BUY, HOLD, HOLD, SELL, HOLD], rate=0.0)
    holds = [1, 2, 3, 0, 0]                                  # bars held after each step
    for o, h in zip(outs, holds):
        assert o["rc_hold"][0] == pytest.approx(expected(h) if h else 0.0)
        assert o["reward"][0] == pytest.approx(o["net_return"][0] - o["rc_hold"][0])


def test_mean_variance_risk_component_is_logged():
    _, outs, _ = run({"reward": "mean_variance", "mv_lambda": 1.0}, ACTIONS)
    for o in outs:
        x = o["net_return"][0] / SIG
        assert o["rc_risk"][0] == pytest.approx(0.5 * x * x)


def test_bad_reward_configs_fail_loudly():
    d = market()
    no_sigma = MarketData(["X"], [d.close], [d.feat], 5, [0.05])
    with pytest.raises(ValueError):
        VecTradingEnv(no_sigma, [(10, 250)], {"reward": "mean_variance"}, 4, False, 0.0, mode="eval")
    with pytest.raises(ValueError):
        VecTradingEnv(d, [(10, 250)], {"reward": "sharpe_typo"}, 4, False, 0.0, mode="eval")


def test_ex_ante_vol_has_no_lookahead_and_is_positive():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(500)))
    base = ex_ante_vol(c, 60, 0, 200)
    t = 350
    pert = c.copy()
    pert[t + 1:] *= np.exp(0.2 * rng.standard_normal(len(c) - t - 1))
    np.testing.assert_array_equal(base[: t + 1], ex_ante_vol(pert, 60, 0, 200)[: t + 1])
    assert np.all(base > 0) and np.all(np.isfinite(base))


@pytest.mark.parametrize("reward", ["mean_variance", "active_return"])
def test_agent_trains_with_vol_scaled_rewards_end_to_end(synthetic_agent_cfg, reward):
    from harness.config import deep_merge
    from harness.experiment import run_experiment
    cfg = deep_merge(synthetic_agent_cfg, {"agent": {"env": {"reward": reward, "reward_clip": 10.0,
                                                             "cost_penalty_mult": 2}}})
    res = run_experiment(cfg, seeds=[0], cost_levels=[10], log=False, verbose=False)
    assert len(res.metrics) == 2 and np.isfinite(res.metrics["sharpe"]).all()
