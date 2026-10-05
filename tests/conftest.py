"""Shared pytest fixtures.

`synthetic_cfg` builds a tiny self-contained world in a temporary directory:
three random-walk tickers with business-day calendars (one with extra
holidays), a protocol with two short folds and a test period. Harness tests run
against it, so they are fast, deterministic and never touch the real data in
data/raw or the real experiments/trials.csv.
"""

import os
import sys

# Tests run on CPU: deterministic, fast for tiny networks, and they never take
# GPU memory on the shared server. Must be set before TensorFlow is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from harness.config import deep_merge, load_config  # noqa: E402


def make_prices(dates, seed, drift=0.0003, vol=0.01, start=100.0):
    """Geometric random walk with plausible OHLCV columns."""
    rng = np.random.default_rng(seed)
    r = drift + vol * rng.standard_normal(len(dates))
    close = start * np.exp(np.cumsum(r))
    return pd.DataFrame({
        "Date": dates,
        "Open": close * (1 - 0.001),
        "High": close * (1 + 0.005),
        "Low": close * (1 - 0.005),
        "Close": close,
        "Volume": rng.integers(1_000, 10_000, len(dates)).astype(float),
    })


@pytest.fixture
def synthetic_cfg(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    days = pd.bdate_range("2010-01-01", "2014-12-31")
    # ticker C trades on a different calendar: drop every 20th day as a "holiday"
    holidays = days[::20]
    for i, (t, dates) in enumerate({"AAA": days, "BBB": days, "CCC": days.difference(holidays)}.items()):
        make_prices(dates, seed=i).to_csv(raw / f"{t}.csv", index=False, date_format="%Y-%m-%d")

    override = {
        "data": {"raw_dir": str(raw), "usable_start": "2010-06-01"},
        "universe": {"buckets": {"x": ["AAA", "BBB", "CCC"]}, "leave_out": ["CCC"],
                     "single_asset": "AAA", "context": []},
        "splits": {"dev_start": "2010-06-01", "dev_end": "2013-12-31",
                   "test_start": "2014-01-02", "test_end": "2014-12-31",
                   "folds": [{"name": "F1", "val_start": "2012-01-02", "val_end": "2012-12-31"},
                             {"name": "F2", "val_start": "2013-01-01", "val_end": "2013-12-31"}],
                   "purge_window": 10, "purge_horizon": 2, "inner_val_bars": 60},
        "evaluation": {"seeds": [0, 1, 2],
                       "costs": {"primary_bps": 10, "half_spread_bps": 1, "sweep_bps": [0, 10]},
                       "statistics": {"n_permutations": 2000, "pbo_blocks": 8}},
    }
    cfg = deep_merge(load_config(), override)
    cfg["universe"]["buckets"] = override["universe"]["buckets"]   # replace, not merge
    cfg["splits"]["folds"] = override["splits"]["folds"]
    return cfg


@pytest.fixture
def synthetic_agent_cfg(synthetic_cfg):
    """The M2 base agent on the synthetic world, shrunk to run in seconds on CPU."""
    base = load_config("config/experiments/m2_dqn_base.yaml")
    cfg = deep_merge(synthetic_cfg, {k: base[k] for k in ("name", "kind", "policy", "agent")})
    return deep_merge(cfg, {
        "eval_sets": ["single"],
        "agent": {"train_tickers": "train", "window": 10,
                  "env": {"horizon": 60},
                  "network": {"conv_filters": [8, 8], "hidden": [16]},
                  "algo": {"batch_size": 32, "buffer_size": 5000},
                  "train": {"transitions": 1600, "n_envs": 8, "learning_starts": 200,
                            "eval_every_updates": 100},
                  "runtime": {"workers": 1, "threads_per_worker": 1}},
        "logging": {"wandb": False},
    })
