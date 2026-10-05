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


def load_config(path=None, overrides=None):
    """Return protocol.yaml merged with an experiment YAML and optional overrides.

    path      : experiment YAML (relative paths resolve against the repo root),
                or None to get the bare protocol.
    overrides : dict merged last, e.g. {"evaluation": {"costs": {"primary_bps": 5}}}.
    """
    cfg = load_yaml(PROTOCOL_PATH)
    if path is not None:
        if not os.path.isabs(path):
            path = repo_path(path)
        cfg = deep_merge(cfg, load_yaml(path))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def config_hash(cfg, length=12):
    """Stable short hash of a config (seeds excluded, see module docstring)."""
    clean = copy.deepcopy(cfg)
    clean.get("evaluation", {}).pop("seeds", None)
    # sort_keys + fixed separators -> identical dicts always give identical text
    text = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def code_hash(length=12):
    """Hash of every tracked-looking Python source file in the repo.

    Stored next to each trial so that 'same config, different code' can be
    told apart even when the change is not committed yet. Tests and caches are
    skipped because they do not change what a run computes.
    """
    h = hashlib.sha256()
    skip_dirs = {".git", "__pycache__", "wandb", "tests", "data", "experiments",
                 "models", "graphs", "train_data", "test_data"}
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs and not d.startswith("."))
        for name in sorted(files):
            if name.endswith(".py"):
                full = os.path.join(root, name)
                h.update(os.path.relpath(full, REPO_ROOT).encode())
                with open(full, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()[:length]
