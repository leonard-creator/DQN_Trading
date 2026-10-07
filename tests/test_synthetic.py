"""Step 0c synthetic worlds (PROTOCOL Part II §V6.5): processes, oracles, causality, layout, end to end."""

import numpy as np
import pytest

from harness import synthetic as sy
from harness.config import load_config


def test_processes_match_their_specification():
    for w in sy.WORLDS:
        r, vix, o = sy.simulate(w, 40000, 1)
        sharpe = np.median(r.mean(0) / r.std(0)) * np.sqrt(252)
        assert 0.35 < sharpe < 0.9, w                                  # target 0.6 (W-regime 0.56)
        assert abs(np.corrcoef(r[:, 0], r[:, 1])[0, 1] - sy.RHO) < 0.05, w
        assert set(np.unique(o)) <= {0.0, 0.25, 0.5, 0.75, 1.0} and np.isfinite(vix).all()
    assert np.all(sy.simulate("W-null", 500, 2)[2] == 1.0)                # no timing exists in W-null


def test_oracles_use_only_the_past():
    rng = np.random.default_rng(0)
    z, x, t = rng.standard_normal(600), 0.01 * rng.standard_normal(600), 400
    z2, x2 = z.copy(), x.copy()
    z2[t + 1:] *= 3
    x2[t + 1:] -= 0.05
    np.testing.assert_array_equal(sy.garch_forecast(z, **sy.GARCH)[1][: t + 1],
                                  sy.garch_forecast(z2, **sy.GARCH)[1][: t + 1])
    mu, sd = np.array([6e-4, -8e-4]), np.array([0.006, 0.011])
    np.testing.assert_array_equal(sy.hamilton_target(x, mu, sd, (0.998, 0.99))[: t + 1],
                                  sy.hamilton_target(x2, mu, sd, (0.998, 0.99))[: t + 1])


def test_hamilton_oracle_is_invested_in_bulls_and_flat_in_bears():
    rng = np.random.default_rng(1)
    mu, sd = np.array([6e-4, -8e-4]), np.array([0.006, 0.011])
    x = np.r_[mu[0] + sd[0] * rng.standard_normal(500), mu[1] + sd[1] * rng.standard_normal(200)]
    e = sy.hamilton_target(x, mu, sd, (0.998, 0.99))
    assert e[100:500].mean() > 0.9 and e[560:].mean() < 0.25


def test_world_data_trains_on_path_1_and_scores_path_2():
    prices, oracle, fold = sy.world_data("W-vol", 0)
    idx, n1 = prices["S00"].index, sy.WARMUP + sy.TRAIN_BARS
    assert len(idx) == n1 + sy.WARMUP + sy.EVAL_BARS and set(prices) == set(sy.TICKERS) | {"^VIX"}
    assert fold.cut == idx[n1] and fold.val_start == idx[n1 + sy.WARMUP] and fold.val_end == idx[-1]
    assert (prices["S00"]["Volume"] == 0).all() and len(oracle["S00"]) == len(idx)


def test_synthetic_run_end_to_end(monkeypatch, tmp_path):
    """A tiny R0'-type agent on a shortened W-null world, in-process on CPU, nothing logged."""
    for k, v in (("WARMUP", 150), ("TRAIN_BARS", 500), ("EVAL_BARS", 200)):
        monkeypatch.setattr(sy, k, v)
    cfg = load_config("config/v2/R0prime.yaml", {
        "splits": {"inner_val_bars": 100},
        "agent": {"window": 10, "network": {"transformer_dim": 4, "hidden": [8]},
                  "m4": {"pca_k": 2, "corr_window": 60, "beta_window": 20, "cum_window": 10,
                         "z_window": 60, "z_min_periods": 20, "warmup_bars": 0},
                  "algo": {"batch_size": 32, "buffer_size": 2000},
                  "train": {"transitions": 400, "n_envs": 4, "learning_starts": 100, "eval_every_updates": 50},
                  "runtime": {"workers": 1, "threads_per_worker": 1}}})
    rows = sy.run(cfg, seeds=[0], worlds=["W-null"], workers=1, log=False, root=str(tmp_path))
    assert len(rows) == 1
    r = rows.iloc[0]
    assert np.isfinite([r.agent_sharpe, r.bh_sharpe, r.oracle_sharpe]).all()
    assert r.oracle_sharpe == pytest.approx(r.bh_sharpe)                   # the W-null oracle is buy-and-hold
