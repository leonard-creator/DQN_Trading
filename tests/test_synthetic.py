"""Step 0c synthetic worlds (PROTOCOL Part II §V6.5): processes, oracles, causality, layout, end to end."""

import json

import numpy as np
import pandas as pd
import pytest

from harness import synthetic as sy
from harness.config import deep_merge, load_config


def test_processes_match_their_specification():
    for w in sy.ALL_WORLDS:
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


def test_long_history_loader_cuts_the_window_and_compounds(tmp_path):
    """harness/longhistory: French CSV tables -> price frames inside the window only (Step 4)."""
    from harness import longhistory as lh
    ind = ("header text\n\n  Average Value Weighted Returns -- Daily\n,Cnsmr,Manuf,HiTec,Hlth,Other\n"
           "19260701,1.00,0.00,0.00,0.00,0.00\n19260702,1.00,-99.99,0.00,0.00,0.00\n"
           "19260706,-1.00,0.00,0.00,0.00,0.00\n20070702,5.00,0.00,0.00,0.00,0.00\n\n"
           "  Average Equal Weighted Returns -- Daily\n,Cnsmr,Manuf,HiTec,Hlth,Other\n19260701,9,9,9,9,9\n")
    ff = "text\n\n,Mkt-RF,SMB,HML,RF\n19260701,0.50,0,0,0.01\n19260702,0.50,0,0,0.01\n19260706,0.50,0,0,0.01\n"
    (tmp_path / "5_Industry_Portfolios_Daily.csv").write_text(ind)
    (tmp_path / "F-F_Research_Data_Factors_daily.csv").write_text(ff)
    p = lh.load(("1926-07-01", "2007-06-29"), folder=str(tmp_path))
    assert set(p) == {"FF_Cnsmr", "FF_Manuf", "FF_HiTec", "FF_Hlth", "FF_Other", "FF_Mkt", "^VIX"}
    c = p["FF_Cnsmr"]["Close"]
    assert len(c) == 2 and c.index[-1] == pd.Timestamp("1926-07-06")   # missing row dropped, 2007 cut
    assert c.iloc[0] == pytest.approx(101.0) and c.iloc[1] == pytest.approx(101.0 * 0.99)
    assert p["FF_Mkt"]["Close"].iloc[0] == pytest.approx(100 * 1.0051) and p["^VIX"]["Close"].isna().all()


def test_swing_oracle_follows_each_tickers_own_drift_and_uses_only_the_past():
    """W-swing (§V12.1 item 23): a per-ticker Kalman filter; one ticker steps aside, the other stays in."""
    rng = np.random.default_rng(2)
    phi, mu, sd, noise, band = 0.5 ** 0.1, 4e-4, 1.5e-3, 0.01, 3e-5
    y = mu + noise * rng.standard_normal((800, 2))
    y[300:400, 0] -= 5e-3                                             # ticker 0: a clearly negative drift spell
    y[300:400, 1] += 5e-3                                             # ticker 1: a clearly positive one
    e = sy.kalman_target(y, mu, phi, sd, noise, band)
    assert e[350:400, 0].mean() < 0.1 and e[350:400, 1].mean() > 0.9 and set(np.unique(e)) <= {0.0, 1.0}
    y2 = y.copy()
    y2[501:] += 0.05
    np.testing.assert_array_equal(e[:501], sy.kalman_target(y2, mu, phi, sd, noise, band)[:501])
    o = sy.simulate("W-swing", 3000, 5)[2]
    assert o.shape == (3000, sy.N_TICKERS) and not (o == o[:, :1]).all()   # each ticker has its own timing


def test_run_many_trains_several_configs_in_one_pool_and_builds_features_once(monkeypatch, tmp_path):
    """Two learner settings share each path's features (built once per (world, seed)); W-swing end to end."""
    import rl.policy as rp
    for k, v in (("WARMUP", 150), ("TRAIN_BARS", 500), ("EVAL_BARS", 200)):
        monkeypatch.setattr(sy, k, v)
    calls, build = [], rp.experiment_features
    monkeypatch.setattr(rp, "experiment_features", lambda *a, **kw: calls.append(1) or build(*a, **kw))
    base = load_config("config/v2/V1b.yaml", {
        "splits": {"inner_val_bars": 100},
        "agent": {"window": 10, "network": {"transformer_dim": 4, "hidden": [8]},
                  "m4": {"pca_k": 2, "corr_window": 60, "beta_window": 20, "cum_window": 10,
                         "z_window": 60, "z_min_periods": 20, "warmup_bars": 0},
                  "algo": {"batch_size": 16, "heads": 2},
                  "train": {"transitions": 160, "eval_every_updates": 20},
                  "runtime": {"workers": 1, "threads_per_worker": 1}}})
    other = deep_merge(base, {"name": "v2_V1b_hl", "agent": {"anchor_eta": 0.0, "prior": "none",
                                                            "algo": {"loss": "hl_gauss", "heads": 1}}})
    rows = sy.run_many([base, other], seeds=[0], worlds=["W-null", "W-swing"], workers=1, log=False,
                       root=str(tmp_path))
    assert len(calls) == 2 and len(rows) == 4                            # features once per path, 2 x 2 runs
    assert set(rows["config"]) == {"v2_V1b", "v2_V1b_hl"} and np.isfinite(rows["agent_sharpe"]).all()
    assert (rows.loc[rows["world"] == "W-swing", "oracle_turnover"] > 0).all()


def test_hpo_design_is_balanced_and_overrides_scale_the_budget():
    from harness import hpo
    s = hpo.sample()
    assert len(s) == 32 and len({json.dumps(x, sort_keys=True) for x in s}) == 32 and s == hpo.sample()
    for k, opts in hpo.SPACE.items():
        counts = [sum(x[k] == o for x in s) for o in opts]
        assert max(counts) - min(counts) <= 1, k                         # every option equally often
    base = load_config("config/v2/V1b.yaml")
    cfg = deep_merge(base, hpo.overrides(s[0], 0.25, base))
    a, costs = cfg["agent"], cfg["evaluation"]["costs"]
    assert a["train"]["transitions"] == 25000 and a["train"]["eval_every_updates"] == 1250   # ~5 checkpoints
    assert a["anchor_eta"] == 0.0 and a["prior"] == "none" and costs["primary_bps"] + costs["half_spread_bps"] == 5
    assert a["network"]["hidden"] == hpo.WIDTH[s[0]["width"]]["hidden"] and a["algo"]["n_step"] == s[0]["n_step"]


def _hpo_rows(cfgs, seeds, worlds, null_gain=0.0):
    """Fake synthetic rows: the capture of candidate cNN grows with NN; W-null gain as given."""
    rows = []
    for c in cfgs:
        name = c if isinstance(c, str) else c["name"]
        for s in seeds:
            for w in worlds:
                gain = null_gain if w == "W-null" else 0.04 * (1 + int(name[-2:]))
                rows.append({"config": name, "world": w, "seed": s, "bh_sharpe": 0.5, "agent_sharpe": 0.5 + gain,
                             "oracle_sharpe": 0.5 + (0.0 if w == "W-null" else 0.4), "agent_turnover": 1.0})
    return pd.DataFrame(rows)


def test_hpo_score_is_the_mean_capture_minus_a_noise_trading_penalty():
    from harness import hpo
    rows = pd.concat([_hpo_rows(["x_c01"], range(3), hpo.WORLDS), _hpo_rows(["x_c04"], range(3), hpo.WORLDS, -0.2)])
    sc = hpo.score(rows).set_index("config")
    assert sc.loc["x_c01", "score"] == pytest.approx(0.2) and sc.loc["x_c01", "capture_swing"] == pytest.approx(0.2)
    assert sc.loc["x_c04", "score"] == pytest.approx(0.5 - 1.0)            # better capture, but trades noise
    assert list(hpo.score(rows)["config"]) == ["x_c01", "x_c04"]


def test_hpo_halves_the_field_each_round_and_resumes(monkeypatch, tmp_path):
    from harness import hpo
    monkeypatch.setattr(hpo, "SIZES", (4, 2, 1))
    monkeypatch.setattr(hpo, "ROUNDS", ((range(1), 0.25), (range(2), 0.5), (range(2, 3), 1.0)))
    seen = []
    monkeypatch.setattr(hpo.sy, "run_many", lambda cfgs, seeds, worlds, workers, log, root:
                        seen.append([c["name"] for c in cfgs]) or _hpo_rows(cfgs, seeds, worlds))
    base, out = load_config("config/v2/V1b.yaml"), str(tmp_path / "v2_hpo")
    final = hpo.run(base, out=out, root=str(tmp_path / "runs"))
    assert seen == [["hpo_r1_c00", "hpo_r1_c01", "hpo_r1_c02", "hpo_r1_c03"], ["hpo_r2_c03", "hpo_r2_c02"],
                    ["hpo_r3_c03"]]
    assert list(final["id"]) == ["c03"] and (tmp_path / "V2_hpo.md").exists()
    hpo.run(base, out=out, root=str(tmp_path / "runs"))                      # finished rounds are re-used
    assert len(seen) == 3


def test_hpo_writes_the_real_data_configs_of_the_best_two(monkeypatch, tmp_path):
    """H1/H2 = V1b + the finalist's setting at the full budget, protocol costs; the nb twins add neo_broker x 2."""
    from harness import hpo
    s = hpo.sample()
    final = pd.DataFrame([{**s[3], "id": "c03", "score": 0.4}, {**s[1], "id": "c01", "score": 0.3}])
    final.to_csv(tmp_path / f"round{len(hpo.ROUNDS)}.csv", index=False)
    paths = hpo.write_configs(out=str(tmp_path), folder=str(tmp_path))
    assert [p.split("/")[-1] for p in paths] == ["H1.yaml", "H1nb.yaml", "H2.yaml", "H2nb.yaml"]
    h1, h1nb = load_config(paths[0]), load_config(paths[1])
    assert h1["name"] == "v2_H1" and h1["agent"]["algo"]["n_step"] == s[3]["n_step"]
    assert h1["agent"]["prior"] == "none" and h1["agent"]["anchor_eta"] == 0.0
    assert h1["agent"]["train"]["transitions"] == 100000 and h1["evaluation"]["costs"]["primary_bps"] == 10
    assert h1nb["agent"]["env"]["cost_scenario"] == "neo_broker" and h1nb["agent"]["env"]["cost_positions"] == 2
    assert h1nb["agent"]["algo"] == h1["agent"]["algo"]


def test_hpo_guard_pauses_the_real_data_trials_unless_both_finalists_qualify(tmp_path):
    from harness import hpo
    path = tmp_path / f"round{len(hpo.ROUNDS)}.csv"
    assert not hpo.real_data_allowed(out=str(tmp_path))                          # search not finished
    for scores, steps, ok in (((0.4, 0.2), (20, 5), True), ((0.4, -0.1), (20, 5), False),
                              ((0.4, 0.2), (40, 5), False)):
        pd.DataFrame({"id": ["c03", "c01"], "score": scores, "n_step": steps}).to_csv(path, index=False)
        assert hpo.real_data_allowed(out=str(tmp_path)) is ok
