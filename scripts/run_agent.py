"""Run an agent configuration through the harness (logged as a trial).

    python scripts/run_agent.py --config config/experiments/m2_dqn_base.yaml
    python scripts/run_agent.py --config config/legacy.yaml
    python scripts/run_agent.py --config config/experiments/m2_dqn_base.yaml --smoke
    python scripts/run_agent.py --config config/v2/R0prime.yaml --synthetic    # v2 Step 0c
    python scripts/run_agent.py --config config/v2/R0prime.yaml --calibrate    # its calibration rule

--smoke is for mechanics and timing only: one seed, fold F1, a short training
budget, no trial log, no wandb. It prints run time and the INNER-validation
curve, and deliberately does not print outer-validation metrics, so smoke runs
cannot feed into design decisions (they would be uncounted trials).

--pretrain trains the configuration on the long history (PROTOCOL Part II Step 4,
harness/longhistory.py) once per seed; its fine-tuning run is then a normal run.

--synthetic trains and scores the configuration on the three synthetic worlds of
PROTOCOL Part II §V6.5 (harness/synthetic.py). Logged in experiments/synthetic.csv,
never as a trial. --calibrate only checks the worlds' calibration rule (no training).

--hpo runs the hyperparameter search on the synthetic worlds around the configuration
(PROTOCOL Part II §V12.1 item 23, harness/hpo.py): never a trial, results in
experiments/reports/v2_hpo/ and V2_hpo.md.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml                                             # noqa: E402

from harness.config import deep_merge, load_config      # noqa: E402
from harness.experiment import run_experiment           # noqa: E402
from harness.splits import folds_from_config            # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="*", help="default: PROTOCOL seeds 0-9")
    p.add_argument("--workers", type=int, help="override agent.runtime.workers")
    p.add_argument("--smoke", action="store_true", help="timing/mechanics check, nothing logged")
    p.add_argument("--smoke-transitions", type=int, default=20000)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. agent.algo.loss=mse name=v2_V1_mse (YAML values; part of the hash)")
    p.add_argument("--synthetic", action="store_true", help="v2 Step 0c on the synthetic worlds (no trial)")
    p.add_argument("--worlds", nargs="*", help="--synthetic: subset of W-null W-vol W-regime")
    p.add_argument("--calibrate", action="store_true", help="v2 Step 0c calibration rule only")
    p.add_argument("--pretrain", action="store_true", help="v2 Step 4: long-history pretraining into agent.pretrain.dir")
    p.add_argument("--hpo", action="store_true", help="v2 hyperparameter search on the synthetic worlds (no trial)")
    args = p.parse_args()
    sets = {}
    for kv in args.set:                                  # a.b.c=value -> {"a": {"b": {"c": value}}}
        key, value = kv.split("=", 1)
        node = sets
        *path, last = key.split(".")
        for k in path:
            node = node.setdefault(k, {})
        node[last] = yaml.safe_load(value)
    if args.hpo:
        from harness import hpo
        print(hpo.run(load_config(args.config, sets), workers=args.workers).to_string(index=False))
        return
    if args.pretrain:
        from harness import longhistory
        longhistory.run(load_config(args.config, sets), seeds=args.seeds if args.seeds else range(10),
                        workers=args.workers)
        return
    if args.synthetic or args.calibrate:
        from harness import synthetic
        cfg = load_config(args.config, sets)
        if args.calibrate:
            synthetic.calibrate(cfg, seeds=args.seeds if args.seeds else range(5),
                                worlds=args.worlds or synthetic.WORLDS)
        else:
            rows = synthetic.run(cfg, workers=args.workers, worlds=args.worlds or synthetic.WORLDS,
                                 seeds=args.seeds if args.seeds else range(5))
            print(rows.to_string(index=False))
        return

    over = sets
    if args.workers:
        over = deep_merge(over, {"agent": {"runtime": {"workers": args.workers}}})
    if args.smoke:
        over = deep_merge(over, {"agent": {"train": {"transitions": args.smoke_transitions},
                                           "runtime": {"workers": 1, "force_subprocess": True}},
                                 "logging": {"wandb": False}})
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
