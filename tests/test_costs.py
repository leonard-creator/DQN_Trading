"""Cost model: PROTOCOL bp levels (unchanged) and named scenarios (neo_broker)."""

import numpy as np
import pandas as pd
import pytest

from harness import backtest as bt
from harness.backtest import CostModel, backtest, load_scenarios, scenario_cost

CLOSE = 100 * np.cumprod(np.r_[1.0, 1 + np.array([0.01, -0.02, 0.03, 0.005, 0.01, -0.01, 0.02])])
POS = np.arange(1, len(CLOSE))


def test_fixed_fee_on_every_buy_and_every_sell_never_on_hold():
    e = np.array([0.25, 0.25, 0.5, 0.5, 0.25, 0.0, 0.0, 0.0])     # buy, hold, buy, hold, sell, sell, hold
    res = backtest(CLOSE, e, POS, CostModel(fee_frac=0.001))
    np.testing.assert_array_equal(res["trades"], [1, 0, 1, 0, 1, 1, 0])
    np.testing.assert_allclose(res["cost"], 0.001 * res["trades"])


def test_round_trip_pays_the_fee_twice():
    e = np.zeros(len(CLOSE))
    e[2] = 1.0                                                     # buy at bar 2, sell at bar 3
    res = backtest(CLOSE, e, POS, CostModel(fee_frac=0.002))
    assert res["trades"].sum() == 2
    assert res["cost"].sum() == pytest.approx(0.004)


def test_holding_cost_accrues_only_while_invested():
    e = np.array([0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    res = backtest(CLOSE, e, POS, CostModel(hold=0.0001))
    np.testing.assert_allclose(res["cost"], 0.0001 * np.abs(res["exposure"]))
    assert res["cost"][0] == 0 and res["cost"][-1] == 0


def test_a_plain_rate_is_unchanged_protocol_behaviour():
    e = np.array([0.25, 0.5, 0.5, 0.0, 1.0, 1.0, 0.75, 0.0])
    a = backtest(CLOSE, e, POS, 0.0011)
    b = backtest(CLOSE, e, POS, CostModel(rate=0.0011))
    np.testing.assert_array_equal(a["net"], b["net"])
    np.testing.assert_allclose(a["cost"], 0.0011 * a["turnover"])


def test_neo_broker_scenario_values_and_fee_scaling():
    sc = load_scenarios()
    nb = sc["neo_broker"]
    assert nb["fee_eur"] == 1.0 and nb["capital_eur"] == 10000
    one = scenario_cost(nb, "SPY", n_positions=1)
    many = scenario_cost(nb, "SPY", n_positions=26)
    assert one.fee_frac == pytest.approx(1 / 10000)                # EUR 1 on EUR 10,000
    assert many.fee_frac == pytest.approx(26 / 10000)              # EUR 1 on EUR 385 per sleeve
    assert one.rate == pytest.approx(3 / 1e4)                       # 0 bp commission + 3 bp half-spread
    # TER only where the series is an index without fees
    assert scenario_cost(nb, "^GDAXI", 1).hold == pytest.approx(15 / 1e4 / 252)
    assert one.hold == 0.0
    # sensitivity variants inherit everything except the changed key
    assert sc["neo_broker_1k"]["capital_eur"] == 1000 and sc["neo_broker_1k"]["fee_eur"] == 1.0
    assert sc["neo_broker_spread5"]["half_spread_bps"] == 5 and sc["neo_broker_spread5"]["capital_eur"] == 10000


def test_env_net_return_equals_harness_with_fee_and_ter():
    from rl.env import MarketData, VecTradingEnv
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(300)))
    d = MarketData(["X"], [close], [rng.standard_normal((300, 2)).astype(np.float32)], 5, [0.05])
    cm = CostModel(rate=0.0003, fee_frac=0.0005, hold=0.15 / 1e2 / 252)
    env = VecTradingEnv(d, [(10, 250)], {"reward": "pnl"}, 4, False, [cm], mode="eval")
    expo, R = np.full(300, np.nan), []
    while env.active.any():
        t = env.t[0]
        out = env.step(rng.integers(0, 3, size=1))
        expo[t] = out["exposure"][0]
        R.append(out["net_return"][0])
    np.testing.assert_allclose(R, backtest(close, expo, np.arange(11, 251), cm)["net"], atol=1e-12)


def test_run_experiment_reports_scenarios_with_per_portfolio_fees(synthetic_cfg, monkeypatch, tmp_path):
    from harness import trials as tr
    from harness.experiment import load_result, run_experiment
    monkeypatch.setattr(tr, "append_trial", lambda row: None)     # never touch the real trial log
    cfg = dict(synthetic_cfg, name="bh", kind="baseline", policy="buy_and_hold")
    res = run_experiment(cfg, eval_sets=["single", "train"], cost_levels=[10], cost_scenarios=["neo_broker"],
                         seeds=[0], out_root=str(tmp_path), verbose=False)
    single = res.runs("single", "neo_broker")
    train = res.runs("train", "neo_broker")
    assert len(single) == 2 and len(train) == 2                    # 2 folds
    # buy-and-hold trades once per block; the per-sleeve fee is N times larger in an N-ticker portfolio
    assert single["trades_per_year"].iloc[0] > 0
    assert (res.runs("single", 10)["cost_bps"] == 10).all()
    # outputs round-trip through load_result, scenario files included
    back = load_result(res.output_dir)
    pd.testing.assert_frame_equal(back.returns[("single", "neo_broker")], res.returns[("single", "neo_broker")],
                                  check_freq=False)
    assert "single@neo_broker" in res.summary and "single@10bp" in res.summary


def test_agent_trains_under_a_cost_scenario_end_to_end(synthetic_agent_cfg):
    """A tiny agent with agent.env.cost_scenario runs through the harness and
    reports both the PROTOCOL level and its training scenario."""
    from harness.config import deep_merge
    from harness.experiment import run_experiment
    cfg = deep_merge(synthetic_agent_cfg, {"agent": {"env": {"cost_scenario": "neo_broker", "levels": 2}}})
    res = run_experiment(cfg, seeds=[0], cost_levels=[10], log=False, verbose=False)
    assert set(res.metrics["cost"]) == {"10bp", "neo_broker"}
    expo = res.metrics["exposure"].to_numpy()
    assert np.all((expo >= 0) & (expo <= 1))


def test_cost_positions_sizes_the_fee_to_the_deployment_account():
    """agent.env.cost_positions: the EUR 1 fee is a share of the owner's position size (2 x EUR 5,000 = 2 bp),
    not of the capital split over all training sleeves (EUR 10,000 / 26 = 26 bp)."""
    from harness.config import load_config
    from rl.policy import cost_function
    cfg = load_config("config/v2/V1b.yaml", {"agent": {"env": {"cost_scenario": "neo_broker"}}})
    assert cost_function(cfg)("SPY", 26).fee_frac == pytest.approx(26 / 10000)
    cfg["agent"]["env"]["cost_positions"] = 2
    assert cost_function(cfg)("SPY", 26).fee_frac == pytest.approx(1 / 5000)
    assert cost_function(load_config("config/v2/V1b.yaml"))("SPY", 26) == pytest.approx(11e-4)   # PROTOCOL level
