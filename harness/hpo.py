"""Hyperparameter search on the synthetic worlds (PROTOCOL Part II §V12.1 item 23). Never a trial.

Owner objective (2026-10-07): timing on 1-4 week horizons at small costs instead of buy-and-hold.
Every candidate keeps V1b's structure (target exposures, exact cost band, exogenous replay) without
any buy-and-hold bias (anchor_eta 0, agent.prior none) and trains and is scored at 5 bp per unit
traded (2 bp + 3 bp half-spread) in three worlds (harness/synthetic.py):
W-null (no timing exists), W-regime (bull / bear) and W-swing (1-4 week drifts).

    score = mean over W-swing, W-regime of  median(agent gain) / median(oracle gain)
            - 1 if the median W-null gain is below -0.05 (noise trading)
    gain  = net Sharpe - buy-and-hold net Sharpe on the evaluation path; medians over the seeds

Successive halving: round 1 = 32 settings (balanced random design over SPACE) x seeds 0-4 at 1/4
of the training budget -> best 8; round 2 = seeds 0-9 at 1/2 -> best 2; round 3 = those 2 on fresh
seeds 10-19 at the full budget (confirmation). eval_every_updates keeps ~5 checkpoints per run.
All jobs of a round train in one pool (synthetic.run_many). Candidates, raw rows and scores go to
experiments/reports/v2_hpo/, the report to experiments/reports/V2_hpo.md. A round whose scores
exist is re-used, so an interrupted search restarts where it stopped (delete v2_hpo/ to start over).

    python scripts/run_agent.py --config config/v2/V1b.yaml --hpo --workers 12
"""

import json
import os

import numpy as np
import pandas as pd

from harness import synthetic as sy
from harness.config import deep_merge, load_config, repo_path

SPACE = {                                                   # HANDOVER §4.1, owner-approved 2026-10-07
    "loss": ["mse", "hl_gauss"],
    "n_step": [5, 10, 20, 40],
    "gamma": [0.9, 0.95, 0.99],
    "width": [1, 4],                                        # x1 = V1b's network, x4 = BBF scale
    "weight_decay": [0.0, 0.1],
    "heads": [1, 10, 20],
    "prior_scale": [0.0, 3.0],
    "gate_z": [0.0, 0.5, 1.0],
    "lr": [1e-4, 3e-4],
}
WIDTH = {1: {"transformer_dim": 8, "hidden": [64, 32]}, 4: {"transformer_dim": 32, "hidden": [256, 128]}}
COMMON = {"agent": {"anchor_eta": 0.0, "prior": "none"},
          "evaluation": {"costs": {"primary_bps": 2, "half_spread_bps": 3}}}       # 5 bp per unit traded
WORLDS = ("W-null", "W-regime", "W-swing")
SIZES = (32, 8, 2)                                          # settings per round
ROUNDS = ((range(5), 0.25), (range(10), 0.5), (range(10, 20), 1.0))   # (seeds, share of the budget)
CHECKPOINTS, NULL_FLOOR, SEED = 5, -0.05, 20261007
OUT = repo_path("experiments", "reports", "v2_hpo")


def sample(n=SIZES[0], seed=SEED):
    """n distinct settings in which every option of every dimension appears equally often (+-1),
    each dimension shuffled independently (a balanced random design)."""
    rng = np.random.default_rng(seed)
    while True:
        cols = {k: rng.permutation(np.resize(np.array(v, dtype=object), n)) for k, v in SPACE.items()}
        rows = [{k: cols[k][i] for k in SPACE} for i in range(n)]
        if len({json.dumps(r, sort_keys=True) for r in rows}) == n:
            return rows


def overrides(s, share, base):
    """Config overrides of setting `s` at `share` of the base training budget."""
    tr = base["agent"]["train"]
    transitions = int(round(int(tr["transitions"]) * share))
    every = max(1, int(transitions * float(tr["update_ratio"])) // CHECKPOINTS)
    algo = {k: s[k] for k in ("loss", "n_step", "gamma", "weight_decay", "heads", "prior_scale", "lr")}
    return deep_merge(COMMON, {"agent": {"gate_z": s["gate_z"], "network": dict(WIDTH[s["width"]]), "algo": algo,
                                         "train": {"transitions": transitions, "eval_every_updates": every}}})


def score(rows):
    """One row per configuration: per-world medians and the search score (module docstring)."""
    d = rows.assign(gain=rows.agent_sharpe - rows.bh_sharpe, oracle_gain=rows.oracle_sharpe - rows.bh_sharpe)
    m = d.groupby(["config", "world"])[["gain", "oracle_gain", "agent_turnover"]].median().unstack("world")
    out = pd.DataFrame(index=m.index)
    for w, tag in (("W-swing", "swing"), ("W-regime", "regime")):
        out[f"capture_{tag}"] = m[("gain", w)] / m[("oracle_gain", w)]
        out[f"gain_{tag}"], out[f"oracle_gain_{tag}"] = m[("gain", w)], m[("oracle_gain", w)]
        out[f"turnover_{tag}"] = m[("agent_turnover", w)]
    out["null_gain"], out["null_turnover"] = m[("gain", "W-null")], m[("agent_turnover", "W-null")]
    out["score"] = out[["capture_swing", "capture_regime"]].mean(axis=1) - (out["null_gain"] < NULL_FLOOR)
    return out.reset_index().sort_values("score", ascending=False, kind="stable")


def run(base, workers=None, out=OUT, root=None):
    """The search (module docstring); returns the scores of the last round."""
    base = load_config(base) if isinstance(base, str) else base
    root = root or repo_path("experiments", "runs", "hpo")
    os.makedirs(out, exist_ok=True)
    cands = pd.DataFrame(sample(SIZES[0])).set_axis([f"c{i:02d}" for i in range(SIZES[0])])
    cands.to_csv(os.path.join(out, "candidates.csv"), index_label="id")
    keep = list(cands.index)
    for r, ((seeds, share), n) in enumerate(zip(ROUNDS, SIZES), 1):
        path = os.path.join(out, f"round{r}.csv")
        if not os.path.exists(path):
            cfgs = []
            for c in keep[:n]:
                s = {k: v.item() if hasattr(v, "item") else v for k, v in cands.loc[c].items()}   # plain types
                cfgs.append(deep_merge(base, {**overrides(s, share, base), "name": f"hpo_r{r}_{c}"}))
            print(f"[hpo] round {r}: {len(cfgs)} settings x {len(WORLDS)} worlds x seeds "
                  f"{min(seeds)}-{max(seeds)} at {share:.0%} of the budget", flush=True)
            rows = sy.run_many(cfgs, seeds=seeds, worlds=WORLDS, workers=workers, log=False, root=root)
            rows.to_csv(os.path.join(out, f"runs_r{r}.csv"), index=False)
            sc = score(rows).assign(id=lambda d: d["config"].str.split("_").str[-1])
            sc.merge(cands, left_on="id", right_index=True).to_csv(path, index=False)
        keep = list(pd.read_csv(path)["id"])                    # sorted by score
        write_report(out)
    return pd.read_csv(os.path.join(out, f"round{len(ROUNDS)}.csv"))


def write_configs(out=OUT, base="config/v2/V1b.yaml", folder=None):
    """config/v2/H1.yaml, H2.yaml (+ neo-broker twins H1nb, H2nb) from the final round's best two settings,
    written before any real-data run of them (§V12.1 item 23): V1b + the setting at the full budget as in
    the last round, but at the protocol's costs (H1, H2) or under neo_broker sized to a 2-position
    account (H1nb, H2nb). Returns the written paths."""
    import yaml
    folder = folder or repo_path("config", "v2")
    final, b, paths = pd.read_csv(os.path.join(out, f"round{len(ROUNDS)}.csv")), load_config(base), []
    for rank, row in enumerate(final.head(2).itertuples(), 1):
        s = {k: v.item() if hasattr(v, "item") else v for k, v in ((k, getattr(row, k)) for k in SPACE)}
        o = overrides(s, ROUNDS[-1][1], b)
        o.pop("evaluation")                                  # real data: train at the protocol's costs
        docs = {f"H{rank}": {"inherit": base, "name": f"v2_H{rank}", "agent": o["agent"]}}
        docs[f"H{rank}nb"] = {"inherit": os.path.relpath(os.path.join(folder, f"H{rank}.yaml"), repo_path()),
                              "name": f"v2_H{rank}nb",
                              "agent": {"env": {"cost_scenario": "neo_broker", "cost_positions": 2}}}
        for name, doc in docs.items():
            path = os.path.join(folder, f"{name}.yaml")
            with open(path, "w") as fh:
                fh.write(f"# PROTOCOL Part II §V12.1 item 23: rank {rank} of the synthetic hyperparameter search "
                         f"(id {row.id}, round-{len(ROUNDS)} score {row.score:+.2f}),\n# written by "
                         f"harness/hpo.write_configs before any real-data run of it. +1 trial.\n")
                yaml.safe_dump(doc, fh, sort_keys=False)
            paths.append(path)
    return paths


def real_data_allowed(out=OUT):
    """Guard of the chained real-data trials (owner decision 2026-10-07): the final round must be complete
    and both finalists must have a score > 0 (they learnt something) and n_step <= 20 (purge P stays 40;
    n = 40 would need P = 60, which the owner decides first). Prints the reason; False pauses the trials."""
    path = os.path.join(out, f"round{len(ROUNDS)}.csv")
    final = pd.read_csv(path).head(2) if os.path.exists(path) else pd.DataFrame(columns=["score", "n_step"])
    ok = len(final) == 2 and bool((final["score"] > 0).all() and (final["n_step"] <= 20).all())
    print(f"[hpo] real-data trials: {'start' if ok else 'PAUSED'} (finalists: "
          f"{final[['score', 'n_step']].to_dict('records') if len(final) else 'none'})")
    return ok


def write_report(out=OUT):
    """V2_hpo.md next to `out` (experiments/reports/) from the round files in `out`."""
    L = ["# v2 hyperparameter search on the synthetic worlds (generated by harness/hpo.py)", "",
         "PROTOCOL Part II §V12.1 item 23. Synthetic worlds only, **no trials**. All settings: V1b's structure, "
         "anchor 0, no buy-and-hold prior, costs 2 bp + 3 bp half-spread. *Capture* = median gain / median oracle "
         "gain over the seeds (gain = net Sharpe − buy-and-hold). Score = mean capture over W-swing and W-regime, "
         f"−1 if the median W-null gain is below {NULL_FLOOR}. Raw rows: `experiments/reports/v2_hpo/runs_r*.csv`.", ""]
    for r, ((seeds, share), n) in enumerate(zip(ROUNDS, SIZES), 1):
        path = os.path.join(out, f"round{r}.csv")
        if not os.path.exists(path):
            break
        sc = pd.read_csv(path)
        L += [f"## Round {r}: {len(sc)} settings, seeds {min(seeds)}–{max(seeds)}, {share:.0%} of the training budget",
              "", f"Oracle gain (median): W-swing {sc['oracle_gain_swing'].median():+.2f}, "
              f"W-regime {sc['oracle_gain_regime'].median():+.2f}.", "",
              "| Rank | Id | Score | Capture W-swing (turns/yr) | Capture W-regime (turns/yr) | W-null gain (turns/yr) | "
              + " | ".join(SPACE) + " |", "|" + "---|" * (6 + len(SPACE))]
        L += [f"| {i} | {x.id} | {x.score:+.2f} | {x.capture_swing:+.0%} ({x.turnover_swing:.1f}) | "
              f"{x.capture_regime:+.0%} ({x.turnover_regime:.1f}) | {x.null_gain:+.2f} ({x.null_turnover:.1f}) | "
              + " | ".join(str(getattr(x, k)) for k in SPACE) + " |" for i, x in enumerate(sc.itertuples(), 1)]
        L.append("")
    with open(os.path.join(os.path.dirname(os.path.abspath(out)), "V2_hpo.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
