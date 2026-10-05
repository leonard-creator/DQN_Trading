"""Milestone M5: the ONE evaluation on the frozen test period (PROTOCOL §9).

    python scripts/final_test.py --config config/experiments/<agent>.yaml --i-am-sure

Rules enforced here
  * Without --i-am-sure the script only explains itself and exits.
  * It refuses baseline-only configs, so the lock cannot be used up by accident.
  * It writes experiments/FINAL_TEST.lock BEFORE loading any test data. If the
    lock already exists it refuses to start: the test period is used once.
  * The agent and the four baselines are evaluated in the same unlocked run
    on the three 12-month sub-blocks T1-T3 (PROTOCOL §10a.8). All share one
    training cut at the test start, so the agent is trained once per seed.
  * H1's deflated-Sharpe and Holm criteria are evaluated over 10 seeds x 3
    sub-blocks. The PBO criterion is a development-time statistic over all
    logged configurations (RESULTS.md); it is reported there, not recomputed
    here.

Run this only after the project owner has approved M5.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.config import code_hash, config_hash, load_config, repo_path     # noqa: E402
from harness.data import unlock_test_period                                   # noqa: E402
from harness.experiment import (compare_to_baselines, deflated_sharpe_of,    # noqa: E402
                                make_policy, run_experiment)
from harness.metrics import return_metrics                                    # noqa: E402
from harness.splits import final_test_blocks                                  # noqa: E402
from harness.trials import git_commit                                         # noqa: E402

BASELINES = ["buy_and_hold", "momentum", "macd", "random"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="the selected AGENT configuration")
    p.add_argument("--i-am-sure", action="store_true", help="confirm the one-time test evaluation")
    args = p.parse_args()

    cfg = load_config(args.config)
    if cfg.get("kind") != "agent":
        sys.exit("final_test.py only evaluates an AGENT configuration; refusing to use the lock.")
    if not args.i_am_sure:
        sys.exit("Refusing: the frozen test period can be evaluated ONCE. Re-run with --i-am-sure "
                 "after M5 has been approved.")
    policy = make_policy(cfg)          # fail before unlocking if the agent cannot be built

    token = unlock_test_period({"config": args.config, "config_hash": config_hash(cfg),
                                "code_hash": code_hash(), "git_commit": git_commit()})
    blocks = final_test_blocks(cfg)
    eval_sets = ["train", "leave_out", "single"]
    agent = run_experiment(cfg, folds=blocks, eval_sets=eval_sets, policy=policy, unlock=token,
                           notes="FINAL TEST (PROTOCOL §9)")
    base = {b: run_experiment(load_config(f"config/experiments/baseline_{b}.yaml"), folds=blocks,
                              eval_sets=eval_sets, unlock=token, notes="FINAL TEST baseline")
            for b in BASELINES}

    stats = cfg["evaluation"]["statistics"]
    primary = cfg["evaluation"]["costs"]["primary_bps"]
    comp = compare_to_baselines(agent, base, eval_set="train")
    dsr = deflated_sharpe_of(agent, eval_set="train")
    passes = (dsr["deflated_sharpe_median"] > stats["deflated_sharpe_threshold"]
              and bool((comp["p_holm"] < stats["alpha"]).all()))

    # full-period view: the three sub-blocks' daily returns concatenated, per seed
    def full_period(res):
        r = res.returns[("train", primary)]
        sharpe = [return_metrics(r[s].dropna().to_numpy())["sharpe"] for s in r.columns]
        return float(sorted(sharpe)[len(sharpe) // 2])
    full = {"agent": full_period(agent), **{b: full_period(res) for b, res in base.items()}}

    report = {"dsr_and_holm_criteria_pass": passes, "deflated_sharpe": dsr,
              "comparisons": comp.to_dict("records"), "agent_summary": agent.summary,
              "full_period_median_sharpe": full,
              "note": "H1 also requires PBO < 0.5 over the development trials; see RESULTS.md."}
    out = repo_path("experiments", "final_test_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps({"dsr_and_holm_criteria_pass": passes,
                      "deflated_sharpe_median": dsr["deflated_sharpe_median"],
                      "full_period_median_sharpe": full}, indent=2))
    print(comp.to_string(index=False))
    print("report ->", out)


if __name__ == "__main__":
    main()
