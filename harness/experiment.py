"""run_experiment(config, seeds): the single entry point of the harness.

One call =
    for every fold (walk-forward, PROTOCOL §4)
      for every seed (PROTOCOL §5)
        policy -> exposures for every evaluation ticker        (trained once per seed x fold)
        for every evaluation set (train tickers, leave-out tickers, single asset)
          for every cost level (0 / 5 / 10 / 25 bp)
            backtest each ticker -> equal-weight portfolio -> metrics
    -> experiments/runs/<trial_id>/   metrics.csv, returns_*.csv, config.yaml, summary.json
    -> one appended row in experiments/trials.csv

Policies
--------
A policy is any object with

    policy.exposures(prices, fold, seed, cfg, tickers) -> {ticker: exposure array}

where `prices` is {ticker: DataFrame} (history up to the end of the block,
never into the locked test period) and the exposure array has one entry per
bar of that ticker. `BaselinePolicy` wraps harness/baselines.py. The DQN agent
(milestone M2) implements the same method: it trains on the purged training
range of the fold (harness.splits.fold_ranges) and then acts greedily.
"""

import datetime as _dt
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml

from harness import backtest as bt
from harness import metrics as mt
from harness import stats as st
from harness import trials as tr
from harness.baselines import BASELINES, DETERMINISTIC
from harness.config import code_hash, config_hash, load_config, repo_path
from harness.data import load_prices, ticker_sets
from harness.splits import block_positions, folds_from_config


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------
class BaselinePolicy:
    """Adapter from the BASELINES registry to the policy interface."""

    def __init__(self, name, params=None):
        if name not in BASELINES:
            raise KeyError(f"unknown baseline '{name}', choose from {sorted(BASELINES)}")
        self.name = name
        self.params = params or {}
        self.deterministic = name in DETERMINISTIC

    def exposures(self, prices, fold, seed, cfg, tickers):
        out = {}
        for t in tickers:
            df = prices[t]
            dec = block_positions(df.index, fold.val_start, fold.val_end) - 1
            out[t] = BASELINES[self.name](df["Close"].to_numpy(), dec, seed, t, fold.name, cfg, self.params)
        return out


def make_policy(cfg):
    """Build the policy named in the config.

    kind: baseline -> BaselinePolicy (harness/baselines.py)
    kind: agent    -> DQNPolicy (rl/policy.py), imported lazily so baseline
                      runs never load TensorFlow.
    """
    kind = cfg.get("kind", "baseline")
    if kind == "baseline":
        return BaselinePolicy(cfg["policy"], cfg.get("policy_params"))
    if kind == "agent" and cfg.get("policy") == "dqn":
        from rl.policy import DQNPolicy
        return DQNPolicy(cfg)
    raise NotImplementedError(f"policy kind '{kind}' / '{cfg.get('policy')}' is not implemented")


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class ExperimentResult:
    cfg: dict
    trial_id: str
    metrics: pd.DataFrame          # one row per (eval_set, cost_bps, seed, fold)
    returns: dict                  # {(eval_set, cost_bps): DataFrame dates x seeds of portfolio net returns}
    summary: dict = field(default_factory=dict)
    output_dir: str = ""
    code_hash: str = ""            # hash of the code when the run started

    def runs(self, eval_set, cost):
        """Per-(seed, fold) metric rows of one evaluation set and cost setting.

        cost : a number (PROTOCOL bp level) or a scenario name (str).
        """
        m = self.metrics
        if isinstance(cost, str):
            return m[(m["eval_set"] == eval_set) & (m["cost"] == cost)]
        return m[(m["eval_set"] == eval_set) & (m["cost_bps"] == cost)]


def cost_label(cost):
    """'10bp' for a PROTOCOL level, the scenario name for a scenario."""
    return cost if isinstance(cost, str) else f"{cost}bp"


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def run_experiment(config, seeds=None, eval_sets=None, cost_levels=None, folds=None,
                   policy=None, unlock=None, log=True, notes="", out_root=None, verbose=True,
                   cost_scenarios=None):
    """Evaluate one configuration over seeds x folds x eval sets x cost levels.

    config      : path to an experiment YAML, or an already merged config dict
    seeds       : iterable of ints (default: PROTOCOL seeds 0-9)
    eval_sets   : list of names from harness.data.ticker_sets; the FIRST one is
                  the primary set reported in trials.csv (default: config
                  'eval_sets' or ['train'])
    cost_levels : list of c in bp (default: the PROTOCOL sweep); the primary
                  cost must be included
    folds       : list of harness.splits.Fold (default: the 5 development folds)
    policy      : policy object (default: built from the config)
    unlock      : test-period token (only scripts/final_test.py passes this)
    log         : append a row to experiments/trials.csv and write outputs
    cost_scenarios : names from config/cost_scenarios.yaml evaluated in addition
                  to the bp levels (default: config 'cost_scenarios' plus the
                  agent's training scenario, if any). Secondary results only.
    """
    cfg = load_config(config) if isinstance(config, str) else config
    ev = cfg["evaluation"]
    seeds = list(ev["seeds"] if seeds is None else seeds)
    eval_sets = list(eval_sets or cfg.get("eval_sets") or ["train"])
    costs = ev["costs"]
    cost_levels = list(costs["sweep_bps"] if cost_levels is None else cost_levels)
    primary_cost = costs["primary_bps"]
    if primary_cost not in cost_levels:
        cost_levels.append(primary_cost)
    if cost_scenarios is None:
        cost_scenarios = list(cfg.get("cost_scenarios") or [])
        train_scen = cfg.get("agent", {}).get("env", {}).get("cost_scenario")
        if train_scen and train_scen not in cost_scenarios:
            cost_scenarios.append(train_scen)
    scenarios = bt.load_scenarios() if cost_scenarios else {}
    unknown = [c for c in cost_scenarios if c not in scenarios]
    if unknown:
        raise KeyError(f"unknown cost scenario(s) {unknown}; see config/cost_scenarios.yaml")
    # cost keys: numbers = PROTOCOL bp levels, strings = named scenarios
    cost_keys = list(cost_levels) + list(cost_scenarios)
    folds = list(folds or folds_from_config(cfg))
    policy = policy or make_policy(cfg)
    bpy = cfg["data"]["bars_per_year"]
    run_code_hash = code_hash()                # code as it is when the run STARTS

    sets = ticker_sets(cfg)
    tickers_by_set = {s: sets[s] for s in eval_sets}
    all_eval = sorted({t for ts in tickers_by_set.values() for t in ts})
    # tickers a learning policy trains on (may differ from the evaluation set)
    train_set = cfg.get("agent", {}).get("train_tickers")
    load = sorted(set(all_eval) | set(sets[train_set] if train_set else []))
    if train_set:
        # context series (^VIX) are features for agents; same test-period guard
        load = sorted(set(load) | set(cfg["universe"].get("context", [])))
        # tickers whose returns define the residual factors (M4 features)
        fset = cfg["agent"].get("m4", {}).get("factor_set")
        if fset:
            load = sorted(set(load) | set(sets[fset]))
    prices = load_prices(load, cfg, unlock=unlock)

    # The trial id and output folder exist before the policy runs, so a
    # learning policy can store its artifacts (curves, weights) next to the
    # metrics. Unlogged runs write to experiments/runs/_unlogged/.
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_id = f"{stamp}_{cfg.get('name', 'run')}_{config_hash(cfg)}"
    root = out_root or repo_path("experiments", "runs")
    out_dir = os.path.join(root if log else os.path.join(root, "_unlogged"), trial_id)
    if log or hasattr(policy, "prepare"):
        os.makedirs(out_dir, exist_ok=True)

    # Learning policies train every (seed, fold) up front, in parallel; the
    # loop below then only reads their exposures.
    if hasattr(policy, "prepare"):
        policy.prepare(prices=prices, folds=folds, seeds=seeds, cfg=cfg,
                       tickers=all_eval, out_dir=out_dir)

    rows = []
    daily = {}                                 # (eval_set, cost, seed) -> list of Series (one per fold)
    cache = {}                                 # deterministic baselines: fold -> exposures
    for fold in folds:
        # never hand a policy more history than the end of the evaluation block
        fold_prices = {t: df[df.index <= fold.val_end] for t, df in prices.items()}
        for seed in seeds:
            if getattr(policy, "deterministic", False) and fold.name in cache:
                expo = cache[fold.name]
            else:
                expo = policy.exposures(fold_prices, fold, seed, cfg, all_eval)
                if getattr(policy, "deterministic", False):
                    cache[fold.name] = expo
            for ck in cost_keys:
                def frame(t, cost):
                    df = fold_prices[t]
                    pos = block_positions(df.index, fold.val_start, fold.val_end)
                    if len(pos) == 0:
                        return None
                    res = bt.backtest(df["Close"].to_numpy(), expo[t], pos, cost)
                    return bt.to_frame(res, df.index[pos])

                if isinstance(ck, str):
                    # scenario: a fixed fee depends on how many positions share
                    # the capital, so each evaluation set is backtested separately
                    per_set = {}
                    for es, ts in tickers_by_set.items():
                        fr = {t: frame(t, bt.scenario_cost(scenarios[ck], t, len(ts), bpy)) for t in ts}
                        per_set[es] = {t: f for t, f in fr.items() if f is not None}
                else:
                    rate = bt.cost_rate(ck, costs["half_spread_bps"])
                    common = {t: f for t in all_eval if (f := frame(t, rate)) is not None}
                    per_set = {es: {t: common[t] for t in ts if t in common}
                               for es, ts in tickers_by_set.items()}
                if not any(per_set.values()):
                    raise ValueError(
                        f"fold {fold.name}: no price data between {fold.val_start.date()} and "
                        f"{fold.val_end.date()} (is this the locked test period?)")
                for es, sub in per_set.items():
                    port = bt.portfolio(sub)
                    m = mt.portfolio_metrics(port, sub, bpy)
                    rows.append({"eval_set": es, "cost_bps": ck if not isinstance(ck, str) else np.nan,
                                 "cost": cost_label(ck), "seed": seed, "fold": fold.name, **m})
                    daily.setdefault((es, ck, seed), []).append(port["net"])
        if verbose:
            print(f"  [{cfg.get('name', policy.__class__.__name__)}] fold {fold.name} done "
                  f"({len(seeds)} seeds, {len(cost_keys)} cost settings)")

    metrics = pd.DataFrame(rows)
    returns = {}
    for (es, c, seed), parts in daily.items():
        returns.setdefault((es, c), {})[seed] = pd.concat(parts).sort_index()
    returns = {k: pd.DataFrame(v) for k, v in returns.items()}

    result = ExperimentResult(cfg=cfg, trial_id=trial_id, metrics=metrics, returns=returns,
                              output_dir=out_dir, code_hash=run_code_hash)
    result.summary = summarize(result, eval_sets, cost_keys, primary_cost)
    if log:
        _persist(result, eval_sets[0], primary_cost, seeds, folds, notes)
    return result


def summarize(result, eval_sets, cost_levels, primary_cost):
    """Median/IQR over seeds x folds for every (eval_set, cost) + per-period Sharpe."""
    out = {}
    for es in eval_sets:
        for c in cost_levels:
            runs = result.runs(es, c)
            agg = mt.aggregate(runs)
            # per-period Sharpe of each seed's concatenated out-of-sample series;
            # the median over seeds is what the trial log stores as sr_pp
            r = result.returns[(es, c)]
            sr = [mt.return_metrics(r[s].dropna().to_numpy())["sharpe_pp"] for s in r.columns]
            agg["sr_pp_median"] = float(np.median(sr))
            agg["fold_sharpe_median"] = runs.groupby("fold")["sharpe"].median().round(4).to_dict()
            out[f"{es}@{cost_label(c)}"] = agg
    return out


def _persist(result, primary_set, primary_cost, seeds, folds, notes):
    cfg = result.cfg
    chash = config_hash(cfg)
    out_dir = result.output_dir

    with open(os.path.join(out_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    result.metrics.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    for (es, c), df in result.returns.items():
        tag = f"scen-{c}" if isinstance(c, str) else f"c{c}"      # scenario vs bp level
        df.to_csv(os.path.join(out_dir, f"returns_{es}_{tag}.csv"), index_label="Date")
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(result.summary, fh, indent=2, default=float)

    s = result.summary[f"{primary_set}@{primary_cost}bp"]
    tr.append_trial({
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "trial_id": result.trial_id,
        "kind": cfg.get("kind", "baseline"),
        "name": cfg.get("name", ""),
        "policy": cfg.get("policy", ""),
        "config_hash": chash,
        "code_hash": result.code_hash or code_hash(),
        "git_commit": tr.git_commit(),
        "protocol_version": cfg.get("protocol_version", ""),
        "seeds": f"{min(seeds)}-{max(seeds)} (n={len(seeds)})",
        "folds": ",".join(f.name for f in folds),
        "eval_set": primary_set,
        "cost_bps": primary_cost,
        "n_runs": len(seeds) * len(folds),
        "sharpe_median": s["sharpe_median"],
        "sharpe_iqr": s["sharpe_iqr"],
        "cagr_median": s["cagr_median"],
        "max_drawdown_median": s["max_drawdown_median"],
        "turnover_median": s["turnover_median"],
        "sr_pp": s["sr_pp_median"],
        "fold_metrics": s["fold_sharpe_median"],
        "output_dir": os.path.relpath(out_dir, repo_path()),
        "versions": tr.library_versions(),
        "notes": notes,
    })


# ---------------------------------------------------------------------------
# comparisons (H1) and selection statistics
# ---------------------------------------------------------------------------
def compare_to_baselines(agent, baselines, eval_set="train", cost_bps=None,
                         metric="sharpe", n_permutations=None, seed=0):
    """Paired permutation tests of agent vs each baseline, Holm-corrected.

    agent     : ExperimentResult of the agent
    baselines : {name: ExperimentResult}; must cover the same folds.
                Pairs are matched on (seed, fold) when the baseline was run
                with the agent's seeds. A baseline run with a single seed is
                matched on fold only (valid for deterministic baselines, whose
                value is the same for every seed).
    Returns a DataFrame: baseline, mean_diff, p_value, p_holm, n_pairs.
    """
    cfg = agent.cfg
    cost_bps = cfg["evaluation"]["costs"]["primary_bps"] if cost_bps is None else cost_bps
    n_perm = n_permutations or cfg["evaluation"]["statistics"]["n_permutations"]
    a = agent.runs(eval_set, cost_bps)[["seed", "fold", metric]]
    out = []
    for name, res in baselines.items():
        b = res.runs(eval_set, cost_bps)[["seed", "fold", metric]]
        if set(a["seed"]) <= set(b["seed"]):
            merged = a.merge(b, on=["seed", "fold"], suffixes=("_a", "_b"))
        elif b["seed"].nunique() == 1:
            merged = a.merge(b.drop(columns="seed"), on="fold", suffixes=("_a", "_b"))
        else:
            raise ValueError(f"baseline '{name}' was run with seeds that do not match the agent's")
        test = st.permutation_test(merged[f"{metric}_a"] - merged[f"{metric}_b"], n_perm, seed)
        out.append({"baseline": name, **test})
    df = pd.DataFrame(out)
    df["p_holm"] = st.holm(df["p_value"].to_numpy()) if len(df) else []
    return df


def load_result(output_dir):
    """Rebuild an ExperimentResult from a run folder (no recomputation).

    Lets reports recompute statistics from stored outputs, e.g.
    load_result(trials row 'output_dir').
    """
    if not os.path.isabs(output_dir):
        output_dir = repo_path(output_dir)
    with open(os.path.join(output_dir, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    metrics = pd.read_csv(os.path.join(output_dir, "metrics.csv"))
    returns = {}
    for name in os.listdir(output_dir):
        if name.startswith("returns_") and name.endswith(".csv"):
            stem = name[len("returns_"):-len(".csv")]
            if "_scen-" in stem:
                es, c = stem.split("_scen-", 1)                 # scenario file
            else:
                es, c = stem.rsplit("_c", 1)
                c = int(c)
            df = pd.read_csv(os.path.join(output_dir, name), index_col="Date", parse_dates=True)
            df.columns = [int(s) for s in df.columns]
            returns[(es, c)] = df
    summary = {}
    if os.path.exists(os.path.join(output_dir, "summary.json")):
        with open(os.path.join(output_dir, "summary.json")) as fh:
            summary = json.load(fh)
    return ExperimentResult(cfg=cfg, trial_id=os.path.basename(output_dir), metrics=metrics,
                            returns=returns, summary=summary, output_dir=output_dir)


def seed_mean_returns(result, eval_set, cost_bps):
    """Daily portfolio returns averaged over seeds (one series per configuration)."""
    return result.returns[(eval_set, cost_bps)].mean(axis=1)


def pbo_over(results, eval_set="train", cost_bps=None, n_blocks=None):
    """PBO via CSCV across several configurations' out-of-sample returns."""
    cfg = next(iter(results.values())).cfg
    cost_bps = cfg["evaluation"]["costs"]["primary_bps"] if cost_bps is None else cost_bps
    n_blocks = n_blocks or cfg["evaluation"]["statistics"]["pbo_blocks"]
    mat = pd.concat({k: seed_mean_returns(r, eval_set, cost_bps) for k, r in results.items()},
                    axis=1).dropna()
    return st.pbo_cscv(mat.to_numpy(), n_blocks)


def deflated_sharpe_of(result, eval_set="train", cost_bps=None, n_trials=None, var_sr=None):
    """Deflated Sharpe ratio of one configuration (median over its seeds).

    Each seed's concatenated out-of-sample daily returns get their own
    deflated Sharpe; the median over seeds is reported. N_trials and the
    cross-trial Sharpe variance come from experiments/trials.csv unless given.
    """
    cfg = result.cfg
    cost_bps = cfg["evaluation"]["costs"]["primary_bps"] if cost_bps is None else cost_bps
    n = tr.n_trials() if n_trials is None else n_trials
    v = tr.sharpe_variance() if var_sr is None else var_sr
    r = result.returns[(eval_set, cost_bps)]
    per_seed = [st.deflated_sharpe(r[s].dropna().to_numpy(), n, v)["deflated_sharpe"] for s in r.columns]
    return {"deflated_sharpe_median": float(np.nanmedian(per_seed)), "per_seed": per_seed,
            "n_trials": n, "var_sr": v}
