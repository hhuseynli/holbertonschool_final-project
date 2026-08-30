"""Feature-matrix assembly: the single table every model trains on.

`build_feature_matrix` starts from the *observed* (h3, month) panel produced
by `bakuml.spatial.grid.build_cell_month_panel` (NOT the completed
cells x months rectangle) and attaches three families of features:

1. **Static geometry** — `masterplan.masterplan_features`, merged once per
   cell. Exempt from the lag rule (constant over time).
2. **Metro accessibility** — `infrastructure.metro_features`, recomputed
   once per unique panel month so station openings switch on over time.
   Deterministic ex ante (opening dates, not price data), hence leakage-free.
3. **Temporal lags** — everything below is a function of months <= t-1 only,
   satisfying the project's headline leakage rule (DESIGN.md, Ground rules):

   * ``month_ix`` — integer position of the row's month within the sorted
     unique panel months (a plain time trend; knows nothing about prices).
   * ``lag_own_1m`` / ``lag_own_3m`` — the cell's own median price at t-1 /
     t-3, computed by a **calendar-aware** shift: the panel is pivoted to a
     (cells x months) matrix reindexed over the complete month grid between
     the first and last panel month, and columns are shifted by k positions.
     A cell unobserved at t-1 therefore gets NaN — never the value from t-2,
     which a naive positional `groupby.shift` on the sparse panel would
     silently substitute.
   * ``mom_3m`` — past-only momentum: the percentage change of the cell's
     own median price over the three months ending at t-1, i.e.
     ``(P[t-1] - P[t-4]) / P[t-4] * 100``. Both endpoints are strictly
     before t, so the feature never touches the current month.
   * ``nbr_price_prev_month`` — mean median-price of the six adjacent cells
     at t-1, via `spatial.lags.neighbour_target_encoding(months_lag=1)`
     (leakage-safe by construction).
   * ``new_share_prev_month`` / ``n_listings_prev_month`` — the panel's
     supply-composition columns shifted with the same calendar-aware rule.
     The *current-month* ``new_share`` / ``n_listings`` are outcome-adjacent
     (they are aggregates of the very listings that define the target) and
     are deliberately **excluded** from `FEATURE_COLS`.

Missing-value policy
--------------------
Rows whose ``lag_own_1m`` is NaN are dropped: without a persistence anchor
the model cannot be compared to the naive lag-1 baseline, and the first
observed month of every cell is unusable anyway. All remaining NaNs
(``mom_3m`` / ``lag_own_3m`` in a cell's early months, sparse neighbourhoods
in ``nbr_price_prev_month``, thin previous months) are left in place:
XGBoost handles missing values natively by learning default split
directions, and imputing them would smuggle cross-sectional information
between cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakuml import config
from bakuml.data.schema import month_range, validate_panel
from bakuml.features.infrastructure import METRO_FEATURE_COLS, metro_features
from bakuml.features.masterplan import MASTERPLAN_FEATURE_COLS, masterplan_features
from bakuml.spatial import lags

#: Temporal (lagged / trend) feature columns added by `build_feature_matrix`.
TEMPORAL_FEATURE_COLS: list[str] = [
    "month_ix",
    "lag_own_1m",
    "lag_own_3m",
    "mom_3m",
    "nbr_price_prev_month",
    "new_share_prev_month",
    "n_listings_prev_month",
]

#: The canonical model feature list: every column a model may train on.
#: Excludes the keys (h3, month), the target price columns and the
#: current-month n_listings / new_share (outcome-adjacent aggregates).
FEATURE_COLS: list[str] = [
    *MASTERPLAN_FEATURE_COLS,
    *METRO_FEATURE_COLS,
    *TEMPORAL_FEATURE_COLS,
]


def _value_matrix(panel: pd.DataFrame, col: str, full_months: list[str]) -> pd.DataFrame:
    """(cells x full_months) matrix of `col`; NaN where (cell, month) is unobserved."""
    return panel.pivot(index="h3", columns="month", values=col).reindex(
        columns=full_months
    )


def _shift_months(mat: pd.DataFrame, k: int) -> pd.DataFrame:
    """Shift a (cells x months) matrix k columns to the right (calendar lag k).

    Because the columns cover the *complete* month grid, entry [cell, t] of
    the result is exactly the value at calendar month t-k (NaN if
    unobserved) — a positional shift can never skip months.
    """
    arr = mat.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan)
    if k < arr.shape[1]:
        out[:, k:] = arr[:, : arr.shape[1] - k]
    return pd.DataFrame(out, index=mat.index, columns=mat.columns)


def _lookup(mat: pd.DataFrame, cells: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Read mat[cell, month] per row; NaN for keys absent from the matrix."""
    ri = mat.index.get_indexer(cells)
    ci = mat.columns.get_indexer(months)
    out = np.full(len(cells), np.nan)
    hit = (ri >= 0) & (ci >= 0)
    if hit.any():
        out[hit] = mat.to_numpy()[ri[hit], ci[hit]]
    return out


def build_feature_matrix(
    panel: pd.DataFrame,
    *,
    include_planned_metro: bool = False,
) -> pd.DataFrame:
    """Attach all model features to the observed (h3, month) panel.

    Parameters
    ----------
    panel : the observed panel (`schema.PANEL_BASE_COLUMNS`); one row per
        observed (h3, month), as produced by `grid.build_cell_month_panel`.
    include_planned_metro : forwarded to `infrastructure.metro_features`;
        True makes planned stations (B-04) count from their `simulated_open`
        (the "transit" scenario).

    Returns
    -------
    DataFrame sorted by (h3, month): the input columns plus every
    `FEATURE_COLS` column, with rows lacking ``lag_own_1m`` dropped
    (see the module docstring for the full leakage / NaN policy).
    """
    validate_panel(panel)
    work = (
        panel.copy()
        .sort_values(["h3", "month"], kind="stable")
        .reset_index(drop=True)
    )
    cells = sorted(work["h3"].unique())
    months = sorted(work["month"].unique())
    full_months = month_range(months[0], months[-1])

    # 1. Static Master Plan geometry (one row per cell).
    work = work.merge(masterplan_features(cells), on="h3", how="left",
                      validate="many_to_one")

    # 2. Metro accessibility, computed once per unique panel month so that
    #    station openings switch on at the right time.
    metro = pd.concat(
        [
            metro_features(cells, m, include_planned=include_planned_metro).assign(month=m)
            for m in months
        ],
        ignore_index=True,
    )
    work = work.merge(metro, on=["h3", "month"], how="left", validate="one_to_one")

    # 3a. Time trend.
    work["month_ix"] = (
        work["month"].map({m: i for i, m in enumerate(months)}).astype("int64")
    )

    # 3b. Calendar-aware own-price lags and past-only momentum.
    row_cells = work["h3"].to_numpy()
    row_months = work["month"].to_numpy()
    price = _value_matrix(work, config.TARGET_COL, full_months)
    work["lag_own_1m"] = _lookup(_shift_months(price, 1), row_cells, row_months)
    work["lag_own_3m"] = _lookup(_shift_months(price, 3), row_cells, row_months)
    lag_own_4m = _lookup(_shift_months(price, 4), row_cells, row_months)
    work["mom_3m"] = (work["lag_own_1m"] - lag_own_4m) / lag_own_4m * 100.0

    # 3c. Neighbour price at t-1 (leakage-safe by construction).
    work["nbr_price_prev_month"] = lags.neighbour_target_encoding(
        work, config.TARGET_COL, months_lag=1
    )

    # 3d. Previous-month supply composition (same calendar-aware shift).
    for src, name in [
        ("new_share", "new_share_prev_month"),
        ("n_listings", "n_listings_prev_month"),
    ]:
        mat = _value_matrix(work, src, full_months)
        work[name] = _lookup(_shift_months(mat, 1), row_cells, row_months)

    out = work[work["lag_own_1m"].notna()].reset_index(drop=True)
    return out[[*panel.columns, *FEATURE_COLS]]
