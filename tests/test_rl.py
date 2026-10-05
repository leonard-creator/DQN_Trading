"""Agent-side tests required by the spec: masking, Double-DQN target, PER,
plus env <-> harness consistency, n-step returns, feature look-ahead and
end-to-end determinism."""

import numpy as np
import pandas as pd
import pytest

from harness.backtest import backtest, cost_rate
from rl.env import BUY, HOLD, SELL, MarketData, VecTradingEnv
from rl.features import ticker_features
from rl.replay import NStepCollector, PrioritizedReplay, SumTree
from tests.conftest import make_prices

ENV = {"reward": "pnl", "episode": "random_window", "horizon": 50, "action_masking": True}


def market(n=400, n_tickers=2, window=5, seed=0):
    rng = np.random.default_rng(seed)
    closes = [100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n))) for _ in range(n_tickers)]
    feats = [rng.standard_normal((n, 3)).astype(np.float32) for _ in range(n_tickers)]
    return MarketData([f"T{i}" for i in range(n_tickers)], closes, feats, window, [0.05] * n_tickers)


# --- environment ---------------------------------------------------------------
def test_windows_are_the_right_rows_of_the_right_ticker():
    d = market()
    g = d.global_index(np.array([1, 0]), np.array([10, 4]))
    w = d.windows(g)
    np.testing.assert_array_equal(w[0], d.feat[d.offsets[1] + 6:d.offsets[1] + 11])
    np.testing.assert_array_equal(w[1], d.feat[0:5])


def test_env_net_return_equals_the_harness_backtest():
    """Training and evaluation must measure the same thing."""
    d = market(n_tickers=1)
    rate = cost_rate(10, 1)
    env = VecTradingEnv(d, [(20, 300)], ENV, levels=4, allow_short=False, cost=rate, mode="eval")
    rng = np.random.default_rng(1)
    expo, R = np.full(len(d.close), np.nan), []
    while env.active.any():
        t = env.t[0]
        out = env.step(rng.integers(0, 3, size=1))
        expo[t] = out["exposure"][0]
        R.append(out["net_return"][0])
    harness = backtest(d.close, expo, np.arange(21, 301), rate)["net"]
    np.testing.assert_allclose(R, harness, atol=1e-12)


def test_action_mask_and_invalid_actions():
    d = market(n_tickers=1)
    env = VecTradingEnv(d, [(10, 300)], ENV, levels=2, allow_short=False, cost=0.0, mode="eval")
    assert env.valid_mask().tolist() == [[True, True, False]]          # flat: no sell
    env.step([BUY]); env.step([BUY])
    assert env.k[0] == 2 and env.valid_mask().tolist() == [[True, False, True]]   # max long: no buy
    off = dict(ENV, action_masking=False)
    env2 = VecTradingEnv(d, [(10, 300)], off, levels=2, allow_short=False, cost=0.0, mode="eval")
    assert env2.valid_mask().all()
    out = env2.step([SELL])                                             # invalid -> hold
    assert env2.k[0] == 0 and out["turnover"][0] == 0


def test_position_vector_layout():
    d = market(n_tickers=1)
    env = VecTradingEnv(d, [(10, 300)], ENV, levels=4, allow_short=False, cost=0.0, mode="eval")
    env.step([BUY]); env.step([HOLD])
    pv = env.position_vector()[0]
    assert pv.shape == (4 + 1 + 2,)
    assert pv[1] == 1.0 and pv[:5].sum() == 1.0                         # one-hot k = 1
    assert pv[-1] == pytest.approx(np.log1p(2) / np.log1p(252))         # held 2 bars


def test_random_episodes_stay_inside_their_ranges():
    d = market()
    env = VecTradingEnv(d, [(30, 200), (50, 120)], ENV, 4, False, 0.0, n_envs=16, seed=3)
    for _ in range(500):
        assert np.all(env.t >= env.ranges[env.asset, 0]) and np.all(env.t < env.ranges[env.asset, 1])
        env.step(np.random.default_rng(0).integers(0, 3, 16))


def test_legacy_full_series_is_terminal_at_the_end():
    d = market(n_tickers=1)
    env = VecTradingEnv(d, [(10, 20)], dict(ENV, episode="full_series", reward="profit"), 4, False,
                        0.0, n_envs=1, seed=0)
    outs = [env.step([HOLD]) for _ in range(10)]
    assert outs[-1]["done"][0] and outs[-1]["terminal"][0] and not outs[-2]["done"][0]


# --- n-step ---------------------------------------------------------------------
def _step(r, done=False, terminal=False, g_next=0):
    return {"live": np.array([True]), "reward": np.array([r], np.float32), "done": np.array([done]),
            "terminal": np.array([terminal]), "g_next": np.array([g_next]),
            "pos_next": np.zeros((1, 2), np.float32), "mask_next": np.ones((1, 3), bool)}


def test_nstep_sums_and_flushes_with_correct_discounts():
    c = NStepCollector(1, n=3, gamma=0.5)
    z = np.zeros((1, 2), np.float32)
    m = np.ones((1, 3), bool)
    assert c.push(np.array([0]), z, m, np.array([0]), _step(1.0, g_next=1)) is None
    assert c.push(np.array([1]), z, m, np.array([0]), _step(2.0, g_next=2)) is None
    out = c.push(np.array([2]), z, m, np.array([0]), _step(4.0, g_next=3))
    assert out["reward"][0] == pytest.approx(1 + 0.5 * 2 + 0.25 * 4) and out["discount"][0] == 0.125
    assert out["g"][0] == 0 and out["g2"][0] == 3
    out = c.push(np.array([3]), z, m, np.array([0]), _step(8.0, done=True, terminal=True, g_next=4))
    # flush: from s1 (3 rewards), s2 (2), s3 (1); terminal -> discount 0
    np.testing.assert_allclose(out["reward"], [2 + 0.5 * 4 + 0.25 * 8, 4 + 0.5 * 8, 8])
    np.testing.assert_array_equal(out["discount"], [0, 0, 0])


# --- prioritised replay ---------------------------------------------------------------
def test_sum_tree_totals_and_lookup():
    t = SumTree(5)
    t.set(np.arange(5), np.array([1.0, 2.0, 3.0, 4.0, 0.0]))
    assert t.total() == pytest.approx(10.0)
    assert t.find(np.array([0.5, 1.5, 3.5, 9.9])).tolist() == [0, 1, 2, 3]


def test_per_samples_proportionally_and_weights_are_normalised():
    buf = PrioritizedReplay(8, pos_dim=2, alpha=1.0, eps=0.0, seed=0)
    n = 4
    buf.add_batch(np.arange(n), np.zeros((n, 2)), np.ones((n, 3), bool), np.zeros(n), np.zeros(n),
                  np.arange(n), np.zeros((n, 2)), np.ones((n, 3), bool), np.ones(n))
    buf.update_priorities(np.arange(n), np.array([1.0, 1.0, 1.0, 7.0]))
    counts = np.bincount(np.concatenate([buf.sample(10, beta=0.5)["idx"] for _ in range(2000)]), minlength=n)
    assert counts[3] / counts.sum() == pytest.approx(0.7, abs=0.03)
    w = buf.sample(10, beta=1.0)["weights"]
    assert w.max() == pytest.approx(1.0) and np.all(w > 0)


# --- agent: masking and Double-DQN targets -----------------------------------------------
@pytest.fixture
def agent():
    from rl.agent import DQNAgent
    algo = {"gamma": 0.9, "double": True, "loss": "huber", "lr": 1e-3, "lr_schedule": "constant"}
    return DQNAgent(5, 3, 4, 3, {"arch": "mlp", "hidden": [8]}, algo, total_updates=10)


def test_masked_acting_never_picks_invalid_actions(agent):
    rng = np.random.default_rng(0)
    m = rng.standard_normal((200, 5, 3)).astype(np.float32)
    p = rng.standard_normal((200, 4)).astype(np.float32)
    mask = np.ones((200, 3), bool)
    mask[:, SELL] = False
    for eps in (0.0, 1.0):                                   # greedy and fully random
        assert not np.any(agent.act(m, p, mask, eps, rng) == SELL)
    q = agent.q_values(m, p)
    greedy = agent.act(m, p, mask, 0.0, rng)
    np.testing.assert_array_equal(greedy, np.argmax(np.where(mask, q, -np.inf), axis=1))


@pytest.mark.parametrize("double", [True, False])
def test_double_dqn_target_matches_a_manual_computation(agent, double):
    import tensorflow as tf
    agent.double = double
    # make the target net differ from the online net
    agent.target.set_weights([w + 0.3 for w in agent.online.get_weights()])
    rng = np.random.default_rng(1)
    m2 = rng.standard_normal((64, 5, 3)).astype(np.float32)
    p2 = rng.standard_normal((64, 4)).astype(np.float32)
    r = rng.standard_normal(64).astype(np.float32)
    disc = np.where(rng.random(64) < 0.2, 0.0, 0.9).astype(np.float32)
    q_on = agent.online([m2, p2]).numpy()
    q_tg = agent.target([m2, p2]).numpy()
    mask2 = np.ones((64, 3), bool)
    mask2[np.arange(64), np.argmax(q_on, axis=1)] = False    # the unmasked argmax is now INVALID
    y = agent.targets(tf.constant(r), tf.constant(m2), tf.constant(p2), tf.constant(mask2),
                      tf.constant(disc)).numpy()
    if double:
        a_star = np.argmax(np.where(mask2, q_on, -np.inf), axis=1)
        expected = r + disc * q_tg[np.arange(64), a_star]
    else:
        expected = r + disc * np.max(np.where(mask2, q_tg, -np.inf), axis=1)
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-5)


def test_soft_target_update_moves_by_tau(agent):
    agent.soft, agent.tau = True, 0.5
    agent.target.set_weights([np.zeros_like(w) for w in agent.online.get_weights()])
    batch = {"g": None}
    before = [w.copy() for w in agent.online.get_weights()]
    d = market(n_tickers=1)
    batch = {"g": np.array([10, 11]), "pos": np.zeros((2, 4), np.float32), "action": np.array([0, 1]),
             "reward": np.zeros(2, np.float32), "g2": np.array([11, 12]), "pos2": np.zeros((2, 4), np.float32),
             "mask2": np.ones((2, 3), bool), "discount": np.zeros(2, np.float32), "weights": np.ones(2, np.float32)}
    agent.learn(batch, d)
    after = agent.online.get_weights()
    for t, b, a in zip(agent.target.get_weights(), before, after):
        np.testing.assert_allclose(t, 0.5 * a, rtol=1e-5, atol=1e-6)   # 0.5*online + 0.5*0
        assert not np.allclose(a, b) or np.allclose(b, 0)            # online actually moved


# --- features: no look-ahead ------------------------------------------------------------
def test_features_do_not_look_ahead():
    dates = pd.bdate_range("2010-01-01", periods=400)
    df = make_prices(dates, seed=4).set_index("Date")
    feats = ["Close", "Volume", "ROC12", "MFI14", "FVolatility", "SMA20_Rel", "SMA50_Rel"]
    base = ticker_features(df, feats, 60, 250)
    t = 300
    pert = df.copy()
    noise = np.exp(0.3 * np.random.default_rng(5).standard_normal(len(df) - t - 1))
    for c in ("Open", "High", "Low", "Close"):
        pert.iloc[t + 1:, pert.columns.get_loc(c)] *= noise
    pert.iloc[t + 1:, pert.columns.get_loc("Volume")] *= 3
    np.testing.assert_array_equal(base[: t + 1], ticker_features(pert, feats, 60, 250)[: t + 1])


def test_real_data_has_no_nan_features_after_warmup():
    from harness.config import load_config
    from harness.data import load_prices
    from harness.splits import fold_ranges, folds_from_config
    cfg = load_config()
    try:
        prices = load_prices(["^GDAXI", "SLV"], cfg)          # SLV has the shortest history
    except FileNotFoundError:
        pytest.skip("data/raw not downloaded")
    feats = ["Close", "Volume", "ROC12", "MFI14", "FVolatility", "SMA20_Rel", "SMA50_Rel"]
    for t, df in prices.items():
        r = fold_ranges(df.index, folds_from_config(cfg)[0], 25, 252)
        raw = __import__("scrape_data").compute_indicators(df)[feats].iloc[r.train_start_pos - 20:]
        assert not raw.isna().any().any(), t


# --- end to end ---------------------------------------------------------------------------
def test_dqn_experiment_runs_and_is_deterministic(synthetic_agent_cfg):
    from harness.experiment import run_experiment
    a = run_experiment(synthetic_agent_cfg, seeds=[0], cost_levels=[10], log=False, verbose=False)
    b = run_experiment(synthetic_agent_cfg, seeds=[0], cost_levels=[10], log=False, verbose=False)
    pd.testing.assert_frame_equal(a.metrics, b.metrics, check_exact=True)
    assert len(a.metrics) == 2                                       # 2 folds x 1 seed x 1 cost
    expo = a.metrics["exposure"].to_numpy()
    assert np.all((expo >= 0) & (expo <= 1))
