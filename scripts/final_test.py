"""Milestone M5: the ONE evaluation on the frozen test period (PROTOCOL §9).

    python scripts/final_test.py --config config/experiments/<agent>.yaml --i-am-sure

Rules enforced here
  * Without --i-am-sure the script only explains itself and exits.
  * It refuses baseline-only configs, so the lock cannot be used up by accident.
  * It writes experiments/FINAL_TEST.lock BEFORE loading any test data. If the
    lock already exists it refuses to start: the test period is used once.
  * The agent and the four baselines are evaluated in the same unlocked run,
    then H1 is evaluated exactly as PROTOCOL §1 states.

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
from harness.splits import final_test_fold                                    # noqa: E402
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
    fold = final_test_fold(cfg)
    eval_sets = ["train", "leave_out", "single"]
    agent = run_experiment(cfg, folds=[fold], eval_sets=eval_sets, policy=policy, unlock=token,
                           notes="FINAL TEST (PROTOCOL §9)")
    base = {b: run_experiment(load_config(f"config/experiments/baseline_{b}.yaml"), folds=[fold],
                              eval_sets=eval_sets, unlock=token, notes="FINAL TEST baseline")
            for b in BASELINES}

    stats = cfg["evaluation"]["statistics"]
    comp = compare_to_baselines(agent, base, eval_set="train")
    dsr = deflated_sharpe_of(agent, eval_set="train")
    h1 = (dsr["deflated_sharpe_median"] > stats["deflated_sharpe_threshold"]
          and bool((comp["p_holm"] < stats["alpha"]).all()))
    report = {"h1_pass": h1, "deflated_sharpe": dsr, "comparisons": comp.to_dict("records"),
              "agent_summary": agent.summary,
              "note": "PBO is computed over the development trials (RESULTS.md), not on the test period."}
    out = repo_path("experiments", "final_test_report.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps({"h1_pass": h1, "deflated_sharpe_median": dsr["deflated_sharpe_median"]}, indent=2))
    print(comp.to_string(index=False))
    print("report ->", out)


if __name__ == "__main__":
    main()
