"""End-to-end harness tests on synthetic data: guard, trial log, reproducibility, no leakage."""

import numpy as np
import pandas as pd
import pytest

from harness import data as hd
from harness import trials as tr
from harness.backtest import backtest, cost_rate
from harness.experiment import BaselinePolicy, run_experiment
from harness.metrics import sharpe
from harness.splits import folds_from_config, final_test_fold


# --- test-period guard --------------------------------------------------------
def test_load_prices_never_returns_test_period_without_the_token(synthetic_cfg):
    prices = hd.load_prices(["AAA", "CCC"], synthetic_cfg)
    test_start = pd.Timestamp(synthetic_cfg["splits"]["test_start"])
    assert all(df.index.max() < test_start for df in prices.values())


def test_unlock_works_once_and_fake_tokens_are_rejected(synthetic_cfg, tmp_path):
    lock = tmp_path / "FINAL_TEST.lock"
    with pytest.raises(RuntimeError):
        hd.load_prices(["AAA"], synthetic_cfg, unlock=object())            # not a real token
    token = hd.unlock_test_period({"who": "test"}, lock_path=str(lock))
    full = hd.load_prices(["AAA"], synthetic_cfg, unlock=token)["AAA"]
    assert full.index.max() >= pd.Timestamp(synthetic_cfg["splits"]["test_start"])
    with pytest.raises(RuntimeError):
        hd.unlock_test_period({"who": "again"}, lock_path=str(lock))      # second use refused


# --- trial log ----------------------------------------------------------------
def test_trial_log_counts_distinct_agent_trials_only(tmp_path):
    path = str(tmp_path / "trials.csv")
    base = {k: "" for k in tr.FIELDS}
    rows = [dict(base, kind="baseline", config_hash="b1", code_hash="c", sr_pp=0.01),
            dict(base, kind="agent", config_hash="a1", code_hash="c", sr_pp=0.02),
            dict(base, kind="agent", config_hash="a1", code_hash="c", sr_pp=0.02),   # identical rerun
            dict(base, kind="agent", config_hash="a2", code_hash="c", sr_pp=0.05),
            dict(base, kind="agent", config_hash="a1", code_hash="d", sr_pp=0.03)]   # code changed
    for r in rows:
        tr.append_trial(r, path)
    assert tr.n_trials(path) == 3
    assert tr.sharpe_variance(path) == pytest.approx(np.var([0.02, 0.05, 0.03], ddof=1))
    with pytest.raises(KeyError):
        tr.append_trial({"not_a_field": 1}, path)


# --- run_experiment -------------------------------------------------------------
class SpyPolicy(BaselinePolicy):
    """Buy-and-hold that records the latest date it was shown."""

    def __init__(self):
        super().__init__("buy_and_hold")
        self.deterministic = False
        self.max_seen = {}

    def exposures(self, prices, fold, seed, cfg, tickers):
        self.max_seen[fold.name] = max(df.index.max() for df in prices.values())
        return super().exposures(prices, fold, seed, cfg, tickers)


def test_policy_never_sees_data_after_the_evaluation_block(synthetic_cfg):
    spy = SpyPolicy()
    cfg = dict(synthetic_cfg, name="spy", kind="baseline", policy="buy_and_hold")
    run_experiment(cfg, policy=spy, eval_sets=["train"], log=False, verbose=False)
    for f in folds_from_config(synthetic_cfg):
        assert spy.max_seen[f.name] <= f.val_end


def test_run_experiment_is_reproducible_and_shapes_are_right(synthetic_cfg):
    cfg = dict(synthetic_cfg, name="rnd", kind="baseline", policy="random")
    a = run_experiment(cfg, eval_sets=["train", "leave_out", "single"], log=False, verbose=False)
    b = run_experiment(cfg, eval_sets=["train", "leave_out", "single"], log=False, verbose=False)
    pd.testing.assert_frame_equal(a.metrics, b.metrics, check_exact=True)
    # 3 eval sets x 2 cost levels x 3 seeds x 2 folds
    assert len(a.metrics) == 3 * 2 * 3 * 2
    assert a.metrics.groupby(["eval_set", "cost_bps", "fold"])["sharpe"].nunique().max() > 1  # seeds differ


def test_buy_and_hold_single_asset_matches_a_direct_computation(synthetic_cfg):
    cfg = dict(synthetic_cfg, name="bh", kind="baseline", policy="buy_and_hold")
    res = run_experiment(cfg, eval_sets=["single"], cost_levels=[10], log=False, verbose=False)
    df = hd.load_prices(["AAA"], synthetic_cfg)["AAA"]
    f = folds_from_config(synthetic_cfg)[0]
    pos = np.flatnonzero((df.index >= f.val_start) & (df.index <= f.val_end))
    direct = backtest(df["Close"].to_numpy(), np.ones(len(df)), pos, cost_rate(10, 1))["net"]
    got = res.runs("single", 10).query("fold == 'F1' and seed == 0")["sharpe"].iloc[0]
    assert got == pytest.approx(sharpe(direct))


def test_test_fold_cannot_be_evaluated_without_unlock(synthetic_cfg):
    cfg = dict(synthetic_cfg, name="bh", kind="baseline", policy="buy_and_hold")
    with pytest.raises(ValueError):          # the data simply stops before the test period
        run_experiment(cfg, folds=[final_test_fold(synthetic_cfg)], eval_sets=["single"], log=False, verbose=False)
