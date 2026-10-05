"""Purge / embargo / fold correctness (spec Phase 4 tests: 'purge/embargo correctness')."""

import numpy as np
import pandas as pd
import pytest

from harness.config import load_config
from harness.splits import (Fold, block_positions, embargo_mask, fold_ranges,
                            folds_from_config, purge_bars, final_test_fold, final_test_blocks)

INDEX = pd.bdate_range("2005-01-03", "2012-12-31")


@pytest.mark.parametrize("window,horizon", [(20, 1), (20, 5), (5, 3), (60, 1)])
def test_purge_keeps_training_labels_out_of_every_eval_window(window, horizon):
    fold = Fold("F", pd.Timestamp("2006-01-02"), pd.Timestamp("2010-01-04"), pd.Timestamp("2010-12-31"))
    P = window + horizon
    r = fold_ranges(INDEX, fold, purge=P, inner_val_bars=100)
    v = int(r.val[0])
    last_train = r.train_end_pos - 1
    first_eval_decision = v - 1
    earliest_bar_in_any_eval_window = first_eval_decision - window + 1
    # the last training decision's label (t + H) must end before any eval window starts
    assert last_train + horizon < earliest_bar_in_any_eval_window
    # and exactly P bars are removed, not more
    assert r.train_end_pos == v - P


def test_inner_validation_is_purged_and_inside_training():
    fold = Fold("F", pd.Timestamp("2006-01-02"), pd.Timestamp("2010-01-04"), pd.Timestamp("2010-12-31"))
    r = fold_ranges(INDEX, fold, purge=25, inner_val_bars=252)
    assert r.train_end_pos - r.inner_val_start_pos == 252
    assert r.inner_val_start_pos - r.inner_train_end_pos == 25
    assert r.train_start_pos < r.inner_train_end_pos < r.inner_val_start_pos < r.train_end_pos <= int(r.val[0])


def test_block_positions_are_return_dates_inside_the_block():
    pos = block_positions(INDEX, "2010-01-01", "2010-01-31")      # Jan 1 is not a bar
    dates = INDEX[pos]
    assert dates.min() >= pd.Timestamp("2010-01-01") and dates.max() <= pd.Timestamp("2010-01-31")
    assert len(pos) == len(INDEX[(INDEX >= "2010-01-01") & (INDEX <= "2010-01-31")])
    assert np.all(np.diff(pos) == 1)


def test_embargo_removes_bars_after_the_block_only():
    m = embargo_mask(100, np.arange(40, 50), embargo=10)
    assert not m[50:60].any()
    assert m[:50].all() and m[60:].all()


def test_protocol_folds_are_ordered_disjoint_and_before_the_test_period():
    cfg = load_config()
    folds = folds_from_config(cfg)
    test_start = pd.Timestamp(cfg["splits"]["test_start"])
    assert [f.name for f in folds] == ["F1", "F2", "F3", "F4", "F5"]
    for a, b in zip(folds, folds[1:]):
        assert a.val_end < b.val_start                      # disjoint, in time order
        assert a.train_start == b.train_start               # anchored walk-forward
    assert folds[-1].val_end < test_start
    assert pd.Timestamp(cfg["splits"]["dev_end"]) == folds[-1].val_end
    assert final_test_fold(cfg).val_start == test_start
    assert purge_bars(cfg) == 25


def test_test_sub_blocks_cover_the_test_period_and_share_one_training_cut():
    cfg = load_config()
    blocks = final_test_blocks(cfg)
    s = cfg["splits"]
    assert [b.name for b in blocks] == ["T1", "T2", "T3"]
    assert blocks[0].val_start == pd.Timestamp(s["test_start"])
    assert blocks[-1].val_end == pd.Timestamp(s["test_end"])
    for a, b in zip(blocks, blocks[1:]):
        assert a.val_end < b.val_start and (b.val_start - a.val_end).days <= 4   # contiguous
    assert {b.cut for b in blocks} == {pd.Timestamp(s["test_start"])}
    # same cut -> identical purged training range for every sub-block
    idx = pd.bdate_range("2006-01-02", "2026-09-30")
    ranges = [fold_ranges(idx, b, purge=25, inner_val_bars=252) for b in blocks]
    assert len({(r.train_start_pos, r.train_end_pos) for r in ranges}) == 1
    assert ranges[0].train_end_pos == int(idx.searchsorted(pd.Timestamp(s["test_start"]))) - 25


def test_fold_too_short_for_purge_raises():
    fold = Fold("F", pd.Timestamp("2010-01-04"), pd.Timestamp("2010-01-20"), pd.Timestamp("2010-02-26"))
    with pytest.raises(ValueError):
        fold_ranges(INDEX, fold, purge=25, inner_val_bars=5)
