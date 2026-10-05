"""Run an agent configuration through the harness (logged as a trial).

    python scripts/run_agent.py --config config/experiments/m2_dqn_base.yaml
    python scripts/run_agent.py --config config/legacy.yaml
    python scripts/run_agent.py --config config/experiments/m2_dqn_base.yaml --smoke

--smoke is for mechanics and timing only: one seed, fold F1, a short training
budget, no trial log, no wandb. It prints run time and the INNER-validation
curve, and deliberately does not print outer-validation metrics, so smoke runs
cannot feed into design decisions (they would be uncounted trials).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.config import load_config                  # noqa: E402
from harness.experiment import run_experiment           # noqa: E402
from harness.splits import folds_from_config            # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="*", help="default: PROTOCOL seeds 0-9")
    p.add_argument("--workers", type=int, help="override agent.runtime.workers")
    p.add_argument("--smoke", action="store_true", help="timing/mechanics check, nothing logged")
    p.add_argument("--smoke-transitions", type=int, default=20000)
    args = p.parse_args()

    over = {}
    if args.workers:
        over = {"agent": {"runtime": {"workers": args.workers}}}
    if args.smoke:
        over = {"agent": {"train": {"transitions": args.smoke_transitions},
                          "runtime": {"workers": 1, "force_subprocess": True}},
                "logging": {"wandb": False}}
    cfg = load_config(args.config, over)
    folds = folds_from_config(cfg)[:1] if args.smoke else None
    seeds = [0] if args.smoke else args.seeds

    t0 = time.time()
    res = run_experiment(cfg, seeds=seeds, folds=folds, log=not args.smoke,
                         notes="smoke" if args.smoke else "")
    minutes = (time.time() - t0) / 60
    if args.smoke:
        import pandas as pd
        job_dirs = sorted(os.path.join(res.output_dir, "agent", d)
                          for d in os.listdir(os.path.join(res.output_dir, "agent")))
        curve = pd.read_csv(os.path.join(job_dirs[0], "curve.csv"))
        print(curve[["update", "transitions", "epsilon", "loss", "mean_q", "inner_sharpe",
                     "inner_turnover", "elapsed_s"]].to_string(index=False))
        print(f"smoke run: {minutes:.1f} min wall time for one (seed, fold) job "
              f"with {cfg['agent']['train']['transitions']} transitions")
        return
    s = res.summary
    key = f"{res.cfg['eval_sets'][0]}@{res.cfg['evaluation']['costs']['primary_bps']}bp"
    print(f"{res.trial_id}: {key} median Sharpe {s[key]['sharpe_median']:.3f} "
          f"(IQR {s[key]['sharpe_iqr']:.3f}) | {minutes:.1f} min")


if __name__ == "__main__":
    main()
