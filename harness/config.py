"""YAML configuration loading, merging and hashing.

Every experiment is described by one YAML file. It is merged on top of
`config/protocol.yaml`, so protocol values (dates, costs, seeds, ...) are
always present and can only be changed in one place.

The CONFIG HASH identifies a trial in `experiments/trials.csv`. It covers the
fully merged config, so a change anywhere (including the protocol) gives a new
hash. Seeds are excluded on purpose: running more seeds of the same
configuration is the same trial, not a new one.
"""

import copy
import hashlib
import json
import os

import yaml

# Repository root = parent of this file's directory. All relative paths in the
# configs (data/raw, experiments/, ...) are resolved against it, so scripts work
# no matter which directory they are started from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL_PATH = os.path.join(REPO_ROOT, "config", "protocol.yaml")


def repo_path(*parts):
    """Absolute path inside the repository."""
    return os.path.join(REPO_ROOT, *parts)


def load_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base, override):
    """Recursively merge `override` into a copy of `base` (override wins).

    Dicts are merged key by key; any other type (lists included) is replaced
    as a whole, so a config can redefine e.g. a ticker list completely.
    """
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_with_inherit(path, depth=0):
    """Load an experiment YAML, first merging the file named in its `inherit:` key.

    This keeps ablation configs to a few lines: e.g. m2_dqn_dueling.yaml says
    `inherit: config/experiments/m2_dqn_base.yaml` and then changes one flag,
    so it is obvious that exactly one thing differs.
    """
    if depth > 10:                 # v2 configs sit 7 levels below m2_dqn_base
        raise RecursionError(f"inherit chain too deep at {path}")
    if not os.path.isabs(path):
        path = repo_path(path)
    own = load_yaml(path)
    parent = own.pop("inherit", None)
    if parent is None:
        return own
    return deep_merge(_load_with_inherit(parent, depth + 1), own)


def load_config(path=None, overrides=None):
    """Return protocol.yaml merged with an experiment YAML and optional overrides.

    path      : experiment YAML (relative paths resolve against the repo root),
                or None to get the bare protocol. May use `inherit:`.
    overrides : dict merged last, e.g. {"evaluation": {"costs": {"primary_bps": 5}}}.
    """
    cfg = load_yaml(PROTOCOL_PATH)
    if path is not None:
        cfg = deep_merge(cfg, _load_with_inherit(path))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


# Keys that only control HOW a run executes (parallel workers, GPUs, live
# logging), not WHAT it computes. They are left out of the config hash, so
# running the same configuration with more workers is still the same trial.
EXECUTION_KEYS = ("runtime", "logging")


def config_hash(cfg, length=12):
    """Stable short hash of a config (seeds and execution keys excluded)."""
    clean = copy.deepcopy(cfg)
    clean.get("evaluation", {}).pop("seeds", None)
    for k in EXECUTION_KEYS:
        clean.pop(k, None)
        clean.get("agent", {}).pop(k, None)
    # sort_keys + fixed separators -> identical dicts always give identical text
    text = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:length]


# Source files that can change what a run COMPUTES. Drivers and reports
# (scripts/), tests and the original single-asset code are excluded: editing an
# analysis script must not make an identical training run look like new code.
# (Before 2026-10-05 17:20 the hash covered every .py file, including scripts/,
# so M2's m2_dqn_base row differs from the other four rows only because
# scripts/analyze_m2.py was edited in between.)
CODE_PATHS = ("harness", "rl", "functions.py", "scrape_data.py")


def code_hash(length=12):
    """Hash of the source files in CODE_PATHS.

    Stored next to each trial so that 'same config, different code' can be
    told apart even when the change is not committed yet.
    """
    h = hashlib.sha256()
    files = []
    for p in CODE_PATHS:
        full = os.path.join(REPO_ROOT, p)
        if os.path.isfile(full):
            files.append(full)
        for root, dirs, names in os.walk(full):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            files += [os.path.join(root, n) for n in names if n.endswith(".py")]
    for full in sorted(set(files)):
        h.update(os.path.relpath(full, REPO_ROOT).encode())
        with open(full, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:length]
