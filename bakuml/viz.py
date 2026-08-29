"""Pure visualization helpers for the Streamlit demo (no streamlit import).

This module is the testable half of the presentation layer: it turns pipeline
artifacts (``schema.ARTIFACT_FILES``) into map-ready GeoJSON + colormaps and
into tidy per-cell frames for the history / forecast charts. Keeping it free
of any ``streamlit`` import means every piece of the app's data plumbing can
be unit-tested headlessly with plain pytest, while ``app/streamlit_app.py``
stays a thin, declarative UI shell.

Design choices worth defending
------------------------------
* **Robust colour bounds.** Choropleth colour scales are anchored at the 5th
  and 95th percentiles of the plotted values, not min/max: a single outlier
  hexagon (one luxury tower, one data glitch) would otherwise wash out the
  contrast for the whole city. ``branca.colormap.LinearColormap`` clamps
  values outside [vmin, vmax] to the end colours, so outliers stay visible,
  just saturated.
* **Semantically matched palettes.** Appreciation is a signed quantity, so it
  gets a red-yellow-green diverging scale *symmetric around zero* (equal
  appreciation and depreciation get equally strong colours); prices get a
  sequential yellow-orange-red heat ramp; hotspot probability uses a
  viridis-like perceptually uniform ramp pinned to the absolute [0, 1] scale
  (a probability of 0.7 must look the same in every scenario); interval width
  (uncertainty) uses sequential blues.
* **Graceful degradation.** ``load_artifacts`` returns only the artifacts
  that exist on disk; the app renders whatever subset is available instead of
  crashing, and points the user at ``make demo`` when the essentials are
  missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import branca.colormap as bcm
import numpy as np
import pandas as pd

from bakuml import config
from bakuml.data.schema import ARTIFACT_FILES
from bakuml.spatial.grid import cells_to_geojson

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


def _fmt_signed_pct(v: float) -> str:
    return f"{v:+.1f} %"


def _fmt_price(v: float) -> str:
    return f"{v:,.0f} AZN/m²"


def _fmt_prob(v: float) -> str:
    return f"{100.0 * v:.0f} %"


def _fmt_interval_width(v: float) -> str:
    return f"±{v / 2.0:,.0f} AZN/m² (q90−q10 = {v:,.0f})"


_RDYLGN_DIVERGING = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
_YLORRD_SEQUENTIAL = ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]
_VIRIDIS_LIKE = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"]
_BLUES_SEQUENTIAL = ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"]

#: metric key -> (human label, palette, pretty-printer). The keys are the
#: valid ``metric`` arguments of :func:`hex_layer_geojson`; the app's metric
#: selectbox is built from this mapping so UI and logic cannot drift apart.
METRIC_SPECS: dict[str, tuple[str, list[str], object]] = {
    "appreciation_pct": ("Forecast appreciation (%)", _RDYLGN_DIVERGING, _fmt_signed_pct),
    "q50": ("Forecast price q50 (AZN/m²)", _YLORRD_SEQUENTIAL, _fmt_price),
    "hotspot_prob": ("Hotspot probability", _VIRIDIS_LIKE, _fmt_prob),
    "uncertainty": ("Forecast uncertainty (q90−q10)", _BLUES_SEQUENTIAL, _fmt_interval_width),
    "current_price": ("Current price (AZN/m², last observed)", _YLORRD_SEQUENTIAL, _fmt_price),
}

#: Convenience mapping metric key -> label for UI widgets.
METRIC_LABELS: dict[str, str] = {k: spec[0] for k, spec in METRIC_SPECS.items()}


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_artifacts(artifacts_dir: Path = config.ARTIFACTS_DIR) -> dict:
    """Load every pipeline artifact that exists under ``artifacts_dir``.

    Iterates ``schema.ARTIFACT_FILES``; ``*.parquet`` files load as
    DataFrames, ``*.json`` files as plain dicts. Files absent on disk are
    simply absent from the returned dict — no exception — so the app can
    render partial pipelines (e.g. a run that crashed before the DiD stage).
    """
    artifacts_dir = Path(artifacts_dir)
    out: dict = {}
    for key, fname in ARTIFACT_FILES.items():
        path = artifacts_dir / fname
        if not path.is_file():
            continue
        if fname.endswith(".parquet"):
            out[key] = pd.read_parquet(path)
        elif fname.endswith(".json"):
            out[key] = json.loads(path.read_text())
    return out


# ---------------------------------------------------------------------------
# Hexagon choropleth layer
# ---------------------------------------------------------------------------


def _robust_bounds(values: np.ndarray, metric: str) -> tuple[float, float]:
    """Colour-scale bounds for `metric` (see module docstring).

    5th-95th percentile of the plotted values; symmetric around 0 for the
    diverging appreciation scale; absolute [0, 1] for probabilities. Always
    returns finite ``vmin < vmax`` (degenerate/constant inputs are widened),
    which branca requires to build a valid LinearColormap.
    """
    if metric == "hotspot_prob":
        return 0.0, 1.0
    finite = values[np.isfinite(values)]
    if finite.size:
        vmin, vmax = (float(q) for q in np.percentile(finite, [5.0, 95.0]))
    else:
        vmin, vmax = 0.0, 1.0
    if metric == "appreciation_pct":
        half_span = max(abs(vmin), abs(vmax))
        vmin, vmax = -half_span, half_span
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        centre = float(finite.mean()) if finite.size else 0.0
        pad = max(0.05 * abs(centre), 1.0)
        vmin, vmax = centre - pad, centre + pad
    return float(vmin), float(vmax)


def hex_layer_geojson(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    scenario: str,
    horizon: int,
    metric: str,
) -> tuple[dict, bcm.LinearColormap]:
    """Build the map layer for one (scenario, horizon, metric) combination.

    Filters `predictions` to the requested scenario and horizon, derives the
    metric value per cell, and returns

    * a GeoJSON FeatureCollection (via :func:`spatial.grid.cells_to_geojson`)
      where every feature carries ``properties["value"]`` (float, drives the
      fill colour) and ``properties["label"]`` (pretty string for tooltips);
    * a ``branca.colormap.LinearColormap`` with robust bounds and a palette
      matched to the metric's semantics (see :data:`METRIC_SPECS`).

    Metric definitions:

    * ``appreciation_pct`` / ``q50`` / ``hotspot_prob`` — taken directly from
      the predictions table.
    * ``uncertainty`` — the conformal interval width ``q90 - q10``.
    * ``current_price`` — each cell's *last observed* ``price_azn_m2_median``
      from the panel (the map still uses the prediction table's cell set so
      the layer geometry is stable when the user toggles metrics).

    Cells whose metric value is missing/non-finite are dropped from the layer.
    """
    if metric not in METRIC_SPECS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRIC_SPECS)}")
    sel = predictions[
        (predictions["scenario"] == scenario)
        & (predictions["horizon_months"] == int(horizon))
    ]
    if sel.empty:
        raise ValueError(f"no predictions for scenario={scenario!r}, horizon={horizon}")
    sel = sel.drop_duplicates(subset="h3", keep="last").set_index("h3")

    if metric == "uncertainty":
        values = sel["q90"] - sel["q10"]
    elif metric == "current_price":
        last_obs = (
            panel.sort_values("month", kind="stable")
            .groupby("h3")["price_azn_m2_median"]
            .last()
        )
        values = last_obs.reindex(sel.index)
    else:
        values = sel[metric]

    values = values.astype(float)
    values = values[np.isfinite(values)]

    label, palette, fmt = METRIC_SPECS[metric]
    vmin, vmax = _robust_bounds(values.to_numpy(), metric)
    colormap = bcm.LinearColormap(palette, vmin=vmin, vmax=vmax, caption=label)

    properties = {
        cell: {"value": float(v), "label": fmt(float(v))} for cell, v in values.items()
    }
    geojson = cells_to_geojson(list(values.index), properties=properties)
    return geojson, colormap


# ---------------------------------------------------------------------------
# Per-cell detail frames
# ---------------------------------------------------------------------------


def cell_history(panel: pd.DataFrame, h3_id: str) -> pd.DataFrame:
    """Observed price history of one cell.

    Returns ``[month, price_azn_m2_median, n_listings]`` sorted by month
    (empty frame with those columns if the cell never appears in the panel).
    """
    hist = panel.loc[
        panel["h3"] == h3_id, ["month", "price_azn_m2_median", "n_listings"]
    ]
    return hist.sort_values("month", kind="stable").reset_index(drop=True)


def cell_forecast(predictions: pd.DataFrame, h3_id: str, scenario: str) -> pd.DataFrame:
    """Forecast quantiles of one cell under one scenario.

    Returns ``[horizon_months, q10, q50, q90]`` with one row per horizon,
    sorted by ``horizon_months`` (empty frame if the cell/scenario pair is
    absent from the predictions table).
    """
    sel = predictions.loc[
        (predictions["h3"] == h3_id) & (predictions["scenario"] == scenario),
        ["horizon_months", "q10", "q50", "q90"],
    ]
    return sel.sort_values("horizon_months", kind="stable").reset_index(drop=True)
