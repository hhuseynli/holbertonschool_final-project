"""Temporal validation: expanding-window walk-forward splits.

Why not plain K-fold?  House prices are strongly autocorrelated in time
(citywide trend + slow-moving cell-level dynamics).  A random K-fold would
put future months in the training set and past months in the test set, so
the model could "predict" the past from the future - an optimistic,
leakage-contaminated score.  Walk-forward validation mirrors how the model
is actually deployed: at each evaluation point we train only on months that
strictly precede the test window, then roll the window forward.

The window is *expanding* (train always starts at the first month) rather
than sliding, because the panel is short (~44 months) and discarding early
history would starve the model; this matches the DESIGN.md contract.

Months are "YYYY-MM" strings throughout the project, which sort
lexicographically in chronological order - no date parsing is needed here.
"""

from __future__ import annotations

from bakuml import config


def walk_forward_splits(
    months: list[str],
    *,
    min_train: int = config.WALK_FORWARD_MIN_TRAIN_MONTHS,
    test_size: int = config.WALK_FORWARD_TEST_MONTHS,
    step: int = 1,
) -> list[tuple[list[str], list[str]]]:
    """Expanding-window walk-forward splits over ``months``.

    Parameters
    ----------
    months : list of "YYYY-MM" strings (any order, duplicates ignored).
    min_train : minimum number of leading months in the first training set.
    test_size : number of months per test window (the last window may be
        shorter if the panel ends mid-window).
    step : how many months the test window advances between splits.

    Returns
    -------
    List of ``(train_months, test_months)`` tuples where every train month
    strictly precedes every test month and the train window expands from
    the first month.  The final month of the panel is always covered: if
    the regular stepping would skip past it (``step > 1``), one anchored
    split ending exactly at the final month is appended.
    """
    if min_train < 1 or test_size < 1 or step < 1:
        raise ValueError("min_train, test_size and step must all be >= 1")
    months = sorted(set(months))
    n = len(months)
    if n < min_train + 1:
        raise ValueError(
            f"need at least min_train+1={min_train + 1} distinct months, got {n}"
        )

    splits: list[tuple[list[str], list[str]]] = []
    for test_start in range(min_train, n, step):
        train = months[:test_start]
        test = months[test_start : test_start + test_size]
        splits.append((train, test))

    # Anchor the last split at the final month when stepping skipped it.
    if splits[-1][1][-1] != months[-1]:
        anchor = max(min_train, n - test_size)
        splits.append((months[:anchor], months[anchor:]))
    return splits
