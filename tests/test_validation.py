"""Tests for bakuml.validation: walk-forward splits and spatial block folds.

Pure combinatorial invariants - no listings or model fits are needed here,
so the file runs in well under a second.
"""

from __future__ import annotations

from collections import Counter

import h3
import pytest

from bakuml import config
from bakuml.data.schema import month_range
from bakuml.validation.spatial_cv import spatial_block_folds
from bakuml.validation.temporal import walk_forward_splits

MONTHS = month_range(config.PANEL_START, config.PANEL_END)  # 44 months

# A compact study area of real H3 ids: 127 res-8 cells around the centre.
BASE_CELL = h3.latlng_to_cell(40.37, 49.84, 8)
CELLS = sorted(h3.grid_disk(BASE_CELL, 6))
# The compact disk spans only ~3 parents at the citywide default block
# resolution (5); res 7 yields ~24 blocks - enough to exercise balancing.
BLOCK_RES = 7


# ---------------------------------------------------------------------------
# walk_forward_splits
# ---------------------------------------------------------------------------


def test_walk_forward_default_invariants():
    splits = walk_forward_splits(MONTHS)
    # defaults: min_train=18, test_size=1, step=1 -> one split per test month
    assert len(splits) == len(MONTHS) - config.WALK_FORWARD_MIN_TRAIN_MONTHS
    for train, test in splits:
        assert len(train) >= config.WALK_FORWARD_MIN_TRAIN_MONTHS
        assert len(test) == 1
        # expanding window: train always starts at the panel start
        assert train[0] == MONTHS[0]
        # train strictly precedes test, no overlap
        assert max(train) < min(test)
        assert not set(train) & set(test)
    # the train window strictly expands and test months strictly increase
    for (tr_a, te_a), (tr_b, te_b) in zip(splits, splits[1:]):
        assert len(tr_b) > len(tr_a)
        assert te_b[0] > te_a[-1]
    # every post-warm-up month is tested exactly once; last month covered
    tested = [m for _, test in splits for m in test]
    assert tested == MONTHS[config.WALK_FORWARD_MIN_TRAIN_MONTHS:]
    assert splits[-1][1][-1] == MONTHS[-1]


def test_walk_forward_step_and_test_size():
    months = month_range("2023-01", "2025-06")  # 30 months
    splits = walk_forward_splits(months, min_train=12, test_size=3, step=3)
    starts = [months.index(test[0]) for _, test in splits]
    assert starts == [12, 15, 18, 21, 24, 27]
    for train, test in splits:
        assert max(train) < min(test)
        assert len(test) == 3
    assert splits[-1][1][-1] == months[-1]


def test_walk_forward_anchors_final_month():
    # step > test_size would skip the final month; an anchored split
    # ending exactly at the last month must be appended.
    months = month_range("2023-01", "2025-08")  # 32 months
    splits = walk_forward_splits(months, min_train=12, test_size=2, step=5)
    assert splits[-1][1][-1] == months[-1]
    for train, test in splits:
        assert max(train) < min(test)
        assert not set(train) & set(test)


def test_walk_forward_handles_unsorted_input():
    shuffled = list(reversed(MONTHS))
    assert walk_forward_splits(shuffled) == walk_forward_splits(MONTHS)


def test_walk_forward_rejects_short_panels_and_bad_params():
    with pytest.raises(ValueError):
        walk_forward_splits(MONTHS[:10], min_train=18)
    with pytest.raises(ValueError):
        walk_forward_splits(MONTHS, min_train=0)
    with pytest.raises(ValueError):
        walk_forward_splits(MONTHS, step=0)


# ---------------------------------------------------------------------------
# spatial_block_folds
# ---------------------------------------------------------------------------


def test_spatial_folds_every_cell_assigned_once():
    folds = spatial_block_folds(CELLS, n_folds=5, block_res=BLOCK_RES, seed=0)
    assert set(folds) == set(CELLS)  # exactly the input cells, each once
    assert set(folds.values()) <= set(range(5))


def test_spatial_folds_keep_parent_blocks_together():
    folds = spatial_block_folds(CELLS, n_folds=5, block_res=BLOCK_RES, seed=0)
    for cell in CELLS:
        parent = h3.cell_to_parent(cell, BLOCK_RES)
        siblings = [c for c in CELLS if h3.cell_to_parent(c, BLOCK_RES) == parent]
        assert len({folds[c] for c in siblings}) == 1


def test_spatial_folds_are_balanced():
    folds = spatial_block_folds(CELLS, n_folds=5, block_res=BLOCK_RES, seed=0)
    sizes = Counter(folds.values())
    assert len(sizes) == 5  # no empty fold with 24 blocks over 5 folds
    assert max(sizes.values()) <= 2 * min(sizes.values())


def test_spatial_folds_deterministic_given_seed():
    a = spatial_block_folds(CELLS, n_folds=5, block_res=BLOCK_RES, seed=3)
    b = spatial_block_folds(CELLS, n_folds=5, block_res=BLOCK_RES, seed=3)
    assert a == b


def test_spatial_folds_default_block_res_still_total():
    # At the citywide default (res 5) this compact disk collapses into ~3
    # blocks; the mapping must still cover every cell with valid fold ids.
    folds = spatial_block_folds(CELLS, seed=0)
    assert set(folds) == set(CELLS)
    assert set(folds.values()) <= set(range(config.SPATIAL_CV_FOLDS))


def test_spatial_folds_rejects_bad_input():
    with pytest.raises(ValueError):
        spatial_block_folds([], n_folds=5)
    with pytest.raises(ValueError):
        spatial_block_folds(CELLS, n_folds=0)
    with pytest.raises(ValueError):
        spatial_block_folds(CELLS, block_res=12)  # finer than the cells
