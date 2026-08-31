"""Spatial lags and leakage-safe neighbour target encoding on the H3 panel.

Methodology
-----------
House prices are spatially autocorrelated: what neighbouring cells trade at
is one of the strongest predictors of a cell's own price. Two operators
expose that signal to the models:

* `spatial_lag` — the same-month mean of a value over the k-ring
  neighbourhood `grid_disk(cell, k)` **excluding the cell itself**. This is
  the classic spatial-lag term (W·y of spatial econometrics with a row-
  binary contiguity matrix). Because it uses contemporaneous neighbour
  values it is a *descriptive* quantity (spatial smoothing, diagnostics,
  maps) — it must NOT be fed to a model predicting the same month's price.

* `neighbour_target_encoding` — the same neighbour mean, but read from month
  `t - months_lag` (lag >= 1 enforced). This is the version the feature
  matrix uses: it is leakage-safe *by construction*, because the encoding
  attached to (cell, t) is computed exclusively from observations dated
  <= t-1; no information from month t or later can flow into it, satisfying
  the project's headline leakage rule (DESIGN.md, Ground rules).

Efficiency
----------
Naively calling `grid_disk` per panel row is O(rows) h3 calls and repeats
work every month. Instead the panel is pivoted once into a (cells x months)
value matrix; the k-ring neighbour sets are resolved once per *cell* (and
memoised process-wide), encoded as a binary neighbour matrix B (no self
loops), and the neighbour means for *all* months are produced with two
matrix products: `(B @ values) / (B @ observed)`. Cell-months where no
neighbour has data yield NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakuml.spatial import tessellation as tess_mod
from bakuml.spatial.tessellation import Tessellation


def _neighbours(cell: str, k: int, tess: Tessellation | None = None) -> tuple[str, ...]:
    """k-step neighbourhood of `cell`, excluding the cell itself.

    Delegated to the active tessellation, which memoises its own adjacency:
    H3 computes k-rings analytically, while the fitted tessellations look
    them up in a contiguity map built once at fit time.
    """
    return tess_mod.resolve(tess).neighbours(cell, k)


def _neighbour_mean_matrix(
    panel: pd.DataFrame,
    value_col: str,
    k: int,
    tess: Tessellation | None = None,
) -> pd.DataFrame:
    """(cells x months) matrix of neighbour means of `value_col`.

    Entry [cell, month] is the mean of `value_col` over the cells of
    `grid_disk(cell, k)` minus the cell itself, restricted to cells present
    in the panel and observed (non-NaN) in that month; NaN when no
    neighbour is observed.
    """
    mat = panel.pivot(index="h3", columns="month", values=value_col)
    cells = list(mat.index)
    pos = {c: i for i, c in enumerate(cells)}

    b = np.zeros((len(cells), len(cells)), dtype=float)
    for c in cells:
        i = pos[c]
        for nb in _neighbours(c, k, tess):
            j = pos.get(nb)
            if j is not None:
                b[i, j] = 1.0

    vals = mat.to_numpy(dtype=float)
    obs = np.isfinite(vals)
    sums = b @ np.where(obs, vals, 0.0)
    counts = b @ obs.astype(float)
    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return pd.DataFrame(means, index=mat.index, columns=mat.columns)


def _lookup(nbr: pd.DataFrame, cells: pd.Series, months: np.ndarray) -> np.ndarray:
    """Read nbr[cell, month] per row; NaN for keys absent from the matrix."""
    ri = nbr.index.get_indexer(cells)
    ci = nbr.columns.get_indexer(months)
    out = np.full(len(cells), np.nan)
    hit = (ri >= 0) & (ci >= 0)
    if hit.any():
        out[hit] = nbr.to_numpy()[ri[hit], ci[hit]]
    return out


def spatial_lag(
    panel: pd.DataFrame,
    value_col: str,
    k: int = 1,
    *,
    tess: Tessellation | None = None,
) -> pd.Series:
    """Same-month neighbour mean of `value_col`, excluding the cell itself.

    Returns a float Series aligned to `panel.index`; NaN where no cell of
    `grid_disk(h3, k)` (self excluded) is observed in that month. Uses
    contemporaneous values — for model features use
    `neighbour_target_encoding` instead.
    """
    nbr = _neighbour_mean_matrix(panel, value_col, k, tess)
    vals = _lookup(nbr, panel["h3"], panel["month"].to_numpy())
    return pd.Series(vals, index=panel.index, name=f"{value_col}_nbr_k{k}")


def neighbour_target_encoding(
    panel: pd.DataFrame,
    target_col: str,
    months_lag: int = 1,
    *,
    tess: Tessellation | None = None,
) -> pd.Series:
    """Neighbour mean of `target_col` taken from month `t - months_lag`.

    Leakage-safe by construction: the value attached to (cell, t) is a
    function only of neighbour observations dated t - months_lag (<= t-1
    since `months_lag >= 1` is enforced), so a model trained on months
    <= T never sees information from its own month or the future.

    Returns a float Series aligned to `panel.index`; NaN when the lagged
    month precedes the panel or no neighbour was observed then.
    """
    if months_lag < 1:
        raise ValueError(
            f"months_lag must be >= 1 to stay leakage-safe, got {months_lag}"
        )
    nbr = _neighbour_mean_matrix(panel, target_col, k=1, tess=tess)
    shifted = (
        (pd.PeriodIndex(panel["month"], freq="M") - months_lag).astype(str).to_numpy()
    )
    vals = _lookup(nbr, panel["h3"], shifted)
    return pd.Series(
        vals, index=panel.index, name=f"{target_col}_nbr_lag{months_lag}m"
    )
