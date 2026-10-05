"""Walk-forward folds with purge, embargo and inner validation (PROTOCOL §4).

Conventions used everywhere in the harness
------------------------------------------
* A **bar** is one daily row of one ticker. Positions are integer row numbers
  in that ticker's own index (tickers have different holiday calendars, so
  bar counts are always per ticker; cut DATES are shared by all tickers).
* A **decision** is taken at the close of bar t using information up to and
  including bar t. The exposure chosen at t earns the return from close t to
  close t+1. Nothing at bar t may depend on bar t+1 or later.
* An **evaluation block** [start, end] is the set of RETURN dates inside it.
  The first decision of a block is therefore taken at the last bar *before*
  `start`. Adjacent folds then chain without losing a day.

Purge (W4)
----------
A training decision at bar t looks back W bars (its observation window) and
forward H bars (its reward horizon). The first decision of an evaluation block
is made at bar v-1 (v = first bar of the block) and looks back to bar v-W.
For no training label to reach into any evaluation window we need
t + H < v - W, i.e. t <= v - (W + H) - 1. So the P = W + H bars just before
the block, [v-P, v-1], are dropped from training. `train_end_pos` below is the
exclusive end of the usable training data.

Embargo
-------
In anchored walk-forward, training never comes after an evaluation block, so
the embargo has nothing to remove. `embargo_mask()` is provided for the CSCV
variant and any non-anchored ablation: it removes the E bars right after an
evaluation block from a training mask.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """Cut dates of one walk-forward fold (shared by all tickers)."""
    name: str
    train_start: pd.Timestamp   # first bar a training decision may use (dev_start)
    val_start: pd.Timestamp     # first RETURN date of the evaluation block
    val_end: pd.Timestamp       # last RETURN date of the evaluation block (inclusive)


def folds_from_config(cfg):
    """The 5 development folds of PROTOCOL §4.2."""
    s = cfg["splits"]
    start = pd.Timestamp(s["dev_start"])
    return [Fold(f["name"], start, pd.Timestamp(f["val_start"]), pd.Timestamp(f["val_end"]))
            for f in s["folds"]]


def final_test_fold(cfg):
    """The frozen test period as a fold (training = whole development period).

    Only usable together with the unlock token from harness.data.unlock_test_period(),
    because without it the price data simply ends before test_start.
    """
    s = cfg["splits"]
    return Fold("TEST", pd.Timestamp(s["dev_start"]),
                pd.Timestamp(s["test_start"]), pd.Timestamp(s["test_end"]))


def purge_bars(cfg, window=None, horizon=None):
    """P = W + H in bars; falls back to the protocol defaults."""
    s = cfg["splits"]
    w = s["purge_window"] if window is None else window
    h = s["purge_horizon"] if horizon is None else horizon
    return int(w) + int(h)


def block_positions(index, start, end):
    """Positions of the RETURN dates of a block [start, end] in `index`.

    Returns an integer array (possibly empty). The decision for return
    position i is taken at position i-1.
    """
    index = pd.DatetimeIndex(index)
    lo = index.searchsorted(pd.Timestamp(start), side="left")
    hi = index.searchsorted(pd.Timestamp(end), side="right")
    return np.arange(lo, hi)


@dataclass(frozen=True)
class FoldRanges:
    """Per-ticker integer ranges for one fold. All ends are EXCLUSIVE.

    train      : [train_start_pos, train_end_pos)        all usable training bars
    inner_train: [train_start_pos, inner_train_end_pos)  used to fit when an inner validation is used
    inner_val  : [inner_val_start_pos, train_end_pos)    checkpoint selection only
    val        : return positions of the evaluation block (decision at pos-1)
    """
    train_start_pos: int
    train_end_pos: int
    inner_train_end_pos: int
    inner_val_start_pos: int
    val: np.ndarray

    @property
    def first_decision_pos(self):
        """Bar at which the first evaluation decision is taken (v - 1)."""
        return int(self.val[0]) - 1 if len(self.val) else None


def fold_ranges(index, fold, purge, inner_val_bars):
    """Turn a Fold's dates into purged per-ticker positions.

    index          : the ticker's DatetimeIndex (full history, may include warm-up)
    purge          : P = W + H (bars)
    inner_val_bars : length of the inner validation slice at the end of training

    The inner validation slice is itself purged from inner training with the
    same P, so checkpoint selection never sees a window that overlaps a bar
    the network was trained on.
    """
    index = pd.DatetimeIndex(index)
    val = block_positions(index, fold.val_start, fold.val_end)
    if len(val) == 0:
        raise ValueError(f"{fold.name}: no bars between {fold.val_start.date()} and {fold.val_end.date()}")
    v = int(val[0])
    train_start = int(index.searchsorted(fold.train_start, side="left"))
    train_end = v - purge                         # exclusive: bars [v-P, v-1] are purged
    if train_end <= train_start:
        raise ValueError(f"{fold.name}: purge {purge} leaves no training data")
    inner_val_start = max(train_start, train_end - int(inner_val_bars))
    inner_train_end = inner_val_start - purge     # purge between inner train and inner val
    if inner_train_end <= train_start:
        raise ValueError(f"{fold.name}: training range too short for inner validation + purge")
    return FoldRanges(train_start, train_end, inner_train_end, inner_val_start, val)


def embargo_mask(n_bars, eval_positions, embargo):
    """Boolean mask over [0, n_bars): False for the `embargo` bars after an evaluation block.

    Use it to remove those bars from a training set that comes AFTER the block
    (CSCV / non-anchored folds). Bars inside the block itself must be excluded
    separately by the caller.
    """
    mask = np.ones(int(n_bars), dtype=bool)
    if len(eval_positions) and embargo > 0:
        end = int(np.max(eval_positions))
        mask[end + 1:min(int(n_bars), end + 1 + int(embargo))] = False
    return mask
