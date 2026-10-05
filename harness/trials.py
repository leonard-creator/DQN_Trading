"""Append-only trial log: experiments/trials.csv (spec W12, PROTOCOL §5).

Every call of harness.experiment.run_experiment() appends exactly one row here,
including runs that turn out badly. Rows are never edited or deleted by code.
The deflated Sharpe ratio takes its N_trials and the cross-trial Sharpe
variance from this file, so "trying one more configuration" always costs
something in the final statistics.

What counts as a trial
----------------------
* kind = "agent": each distinct (config_hash, code_hash) pair is one trial.
  Re-running the identical configuration on identical code does not add a
  trial. Changing the config OR the code does.
* kind = "baseline": logged for completeness, but NOT counted in N_trials.
  The baselines are fixed in PROTOCOL §6 and are never selected among.
"""

import csv
import json
import os
import subprocess

import numpy as np
import pandas as pd

from harness.config import repo_path

TRIALS_CSV = repo_path("experiments", "trials.csv")

FIELDS = [
    "timestamp", "trial_id", "kind", "name", "policy", "config_hash", "code_hash",
    "git_commit", "protocol_version", "seeds", "folds", "eval_set", "cost_bps",
    "n_runs", "sharpe_median", "sharpe_iqr", "cagr_median", "max_drawdown_median",
    "turnover_median", "sr_pp", "fold_metrics", "output_dir", "versions", "notes",
]


def git_commit():
    """Current commit hash, with '+dirty' if the working tree has changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=repo_path(), stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"], cwd=repo_path(),
                                stderr=subprocess.DEVNULL) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def library_versions():
    """Versions of the libraries that change numbers (logged with every trial)."""
    import platform
    import scipy
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__, "scipy": scipy.__version__}
    try:                                       # TF is optional for baseline-only runs
        import sys
        if "tensorflow" in sys.modules:
            out["tensorflow"] = sys.modules["tensorflow"].__version__
    except Exception:
        pass
    return out


def append_trial(row, path=TRIALS_CSV):
    """Append one row. Unknown keys are rejected so the schema cannot drift silently."""
    unknown = set(row) - set(FIELDS)
    if unknown:
        raise KeyError(f"unknown trial fields: {sorted(unknown)}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    clean = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
             for k, v in row.items()}
    with open(path, "a", newline="") as fh:              # 'a' = append only
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(clean)


def load_trials(path=TRIALS_CSV):
    if not os.path.exists(path):
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(path)


def agent_trials(path=TRIALS_CSV):
    """One row per distinct agent trial (latest row of each config/code pair)."""
    df = load_trials(path)
    df = df[df["kind"] == "agent"]
    if df.empty:
        return df
    return df.drop_duplicates(["config_hash", "code_hash"], keep="last")


def n_trials(path=TRIALS_CSV):
    """N_trials for the deflated Sharpe ratio (at least 1)."""
    return max(1, len(agent_trials(path)))


def sharpe_variance(path=TRIALS_CSV):
    """Variance of the per-period Sharpe ratios across distinct agent trials."""
    sr = pd.to_numeric(agent_trials(path)["sr_pp"], errors="coerce").dropna()
    return float(sr.var(ddof=1)) if len(sr) > 1 else 0.0
