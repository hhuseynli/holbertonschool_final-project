"""Tests for `bakuml.viz` — the framework-free half of the app.

The fixtures build a miniature but structurally faithful artifact set
(3 real H3 cells, 4 panel months, 2 scenarios x 2 horizons of predictions,
plus small JSON artifacts) in a temp directory. No network needed: viz
consumes pipeline *outputs*, so the tests plant known outputs and assert
exact pass-through.
"""

from __future__ import annotations

import json

import h3
import numpy as np
import pandas as pd
import pytest

from bakuml import config, viz
from bakuml.data.schema import ARTIFACT_FILES, PREDICTION_COLUMNS

RES = config.H3_RESOLUTION
CENTRE_CELL = h3.latlng_to_cell(*config.CITY_CENTRE, RES)
CELLS = sorted(h3.grid_disk(CENTRE_CELL, 1))[:3]  # 3 real, adjacent res-8 cells
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04"]
SCENARIOS = ["baseline", "transit"]
HORIZONS = [12, 24]
BASE_MONTH = MONTHS[-1]


def _price(cell_ix: int, month_ix: int) -> float:
    """Deterministic planted median: varies across cells and months."""
    return 2000.0 + 250.0 * cell_ix + 20.0 * month_ix


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rows = [
        {
            "h3": cell,
            "month": month,
            "price_azn_m2_median": _price(i, j),
            "price_azn_m2_mean": _price(i, j) + 15.0,
            "n_listings": 4 + i + j,
            "new_share": 0.25 + 0.1 * i,
        }
        for i, cell in enumerate(CELLS)
        for j, month in enumerate(MONTHS)
    ]
    # shuffled on purpose: viz must sort, never rely on input order
    return pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)


@pytest.fixture(scope="module")
def predictions() -> pd.DataFrame:
    rows = []
    for s, scen in enumerate(SCENARIOS):
        for horizon in HORIZONS:
            for i, cell in enumerate(CELLS):
                last = _price(i, len(MONTHS) - 1)
                q50 = last * (1.0 + 0.01 * horizon + 0.05 * s + 0.02 * i)
                rows.append(
                    {
                        "h3": cell,
                        "base_month": BASE_MONTH,
                        "horizon_months": horizon,
                        "scenario": scen,
                        "q10": q50 * 0.90,
                        "q50": q50,
                        "q90": q50 * (1.12 + 0.01 * i),
                        "appreciation_pct": (q50 / last - 1.0) * 100.0,
                        "hotspot_prob": 0.2 + 0.25 * i,
                    }
                )
    # shuffled on purpose (see panel fixture)
    df = pd.DataFrame(rows)[list(PREDICTION_COLUMNS)]
    return df.sample(frac=1.0, random_state=11).reset_index(drop=True)


@pytest.fixture(scope="module")
def artifacts_dir(tmp_path_factory, panel, predictions):
    """A partial artifact directory: parquet + json present, some missing."""
    d = tmp_path_factory.mktemp("artifacts")
    panel.to_parquet(d / ARTIFACT_FILES["panel"], index=False)
    predictions.to_parquet(d / ARTIFACT_FILES["predictions"], index=False)
    (d / ARTIFACT_FILES["metrics"]).write_text(
        json.dumps(
            {
                "walk_forward": {
                    "aggregate": {"mae": 50.0, "mape": 2.5, "r2": 0.9},
                    "naive": {"mae": 80.0, "mape": 4.0, "r2": 0.8},
                },
                "conformal": {"coverage_q10_q90": 0.81},
            }
        )
    )
    (d / ARTIFACT_FILES["shap"]).write_text(
        json.dumps({"lag_own_1m": 310.0, "dist_centre_km": 120.0, "month_ix": 40.0})
    )
    (d / ARTIFACT_FILES["did"]).write_text(
        json.dumps(
            {
                "att_log": 0.075,
                "att_pct": 7.8,
                "se": 0.01,
                "p_value": 0.0001,
                "ci_low": 0.055,
                "ci_high": 0.095,
                "n_treated_listings": 500,
                "n_control_listings": 4000,
                "open_month": "2025-06",
                "spec": "OLS with FE",
                "event_study": [
                    {"rel_month": -2, "coef": 0.0, "se": 0.01,
                     "ci_low": -0.02, "ci_high": 0.02, "n": 30},
                    {"rel_month": 3, "coef": 0.08, "se": 0.01,
                     "ci_low": 0.06, "ci_high": 0.10, "n": 35},
                ],
            }
        )
    )
    # "listings", "cell_features" and "truth" deliberately NOT written
    return d


# ---------------------------------------------------------------------------
# load_artifacts
# ---------------------------------------------------------------------------


def test_load_artifacts_loads_present_and_skips_missing(artifacts_dir):
    loaded = viz.load_artifacts(artifacts_dir)
    assert set(loaded) == {"panel", "predictions", "metrics", "shap", "did"}
    assert isinstance(loaded["panel"], pd.DataFrame)
    assert isinstance(loaded["predictions"], pd.DataFrame)
    assert isinstance(loaded["metrics"], dict)
    assert loaded["did"]["att_pct"] == 7.8
    assert len(loaded["predictions"]) == len(SCENARIOS) * len(HORIZONS) * len(CELLS)


def test_load_artifacts_empty_dir(tmp_path):
    assert viz.load_artifacts(tmp_path) == {}


# ---------------------------------------------------------------------------
# hex_layer_geojson
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", sorted(viz.METRIC_SPECS))
def test_hex_layer_geojson_valid_for_every_metric(predictions, panel, metric):
    geojson, colormap = viz.hex_layer_geojson(
        predictions, panel, scenario="baseline", horizon=12, metric=metric
    )
    assert geojson["type"] == "FeatureCollection"
    assert {f["id"] for f in geojson["features"]} == set(CELLS)
    for feat in geojson["features"]:
        props = feat["properties"]
        assert np.isfinite(props["value"]) and isinstance(props["value"], float)
        assert isinstance(props["label"], str) and props["label"]
        assert props["h3"] == feat["id"]
        ring = feat["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1], "linear ring must be closed"
        for lon, lat in ring:  # GeoJSON order: [lon, lat] — Baku, not Guinea
            assert 49.0 < lon < 51.0
            assert 40.0 < lat < 41.0
    # colormap usable: finite, ordered bounds; callable to a colour
    assert np.isfinite(colormap.vmin) and np.isfinite(colormap.vmax)
    assert colormap.vmin < colormap.vmax
    assert colormap((colormap.vmin + colormap.vmax) / 2.0).startswith("#")


def test_hex_layer_filters_scenario_and_horizon(predictions, panel):
    geojson, _ = viz.hex_layer_geojson(
        predictions, panel, scenario="transit", horizon=24, metric="q50"
    )
    expected = predictions[
        (predictions["scenario"] == "transit") & (predictions["horizon_months"] == 24)
    ].set_index("h3")["q50"]
    got = {f["id"]: f["properties"]["value"] for f in geojson["features"]}
    for cell in CELLS:
        assert got[cell] == pytest.approx(expected[cell])


def test_uncertainty_is_interval_width(predictions, panel):
    geojson, _ = viz.hex_layer_geojson(
        predictions, panel, scenario="baseline", horizon=12, metric="uncertainty"
    )
    sel = predictions[
        (predictions["scenario"] == "baseline") & (predictions["horizon_months"] == 12)
    ].set_index("h3")
    for feat in geojson["features"]:
        cell = feat["id"]
        assert feat["properties"]["value"] == pytest.approx(
            sel.loc[cell, "q90"] - sel.loc[cell, "q10"]
        )


def test_current_price_is_last_observed_median(predictions, panel):
    geojson, _ = viz.hex_layer_geojson(
        predictions, panel, scenario="baseline", horizon=12, metric="current_price"
    )
    got = {f["id"]: f["properties"]["value"] for f in geojson["features"]}
    for i, cell in enumerate(CELLS):
        assert got[cell] == pytest.approx(_price(i, len(MONTHS) - 1))


def test_colormap_bounds_semantics(predictions, panel):
    # probability scale is absolute [0, 1]
    _, cm_prob = viz.hex_layer_geojson(
        predictions, panel, scenario="baseline", horizon=12, metric="hotspot_prob"
    )
    assert (cm_prob.vmin, cm_prob.vmax) == (0.0, 1.0)
    # diverging appreciation scale is symmetric around 0
    _, cm_app = viz.hex_layer_geojson(
        predictions, panel, scenario="baseline", horizon=12, metric="appreciation_pct"
    )
    assert cm_app.vmin == pytest.approx(-cm_app.vmax)
    assert cm_app.vmax > 0


def test_hex_layer_rejects_bad_inputs(predictions, panel):
    with pytest.raises(ValueError, match="unknown metric"):
        viz.hex_layer_geojson(
            predictions, panel, scenario="baseline", horizon=12, metric="nope"
        )
    with pytest.raises(ValueError, match="no predictions"):
        viz.hex_layer_geojson(
            predictions, panel, scenario="no_such_scenario", horizon=12, metric="q50"
        )


# ---------------------------------------------------------------------------
# cell_history / cell_forecast
# ---------------------------------------------------------------------------


def test_cell_history_rows_and_order(panel):
    hist = viz.cell_history(panel, CELLS[1])
    assert list(hist.columns) == ["month", "price_azn_m2_median", "n_listings"]
    assert len(hist) == len(MONTHS)
    assert hist["month"].tolist() == MONTHS  # sorted despite shuffled input
    assert hist["price_azn_m2_median"].tolist() == [
        _price(1, j) for j in range(len(MONTHS))
    ]
    assert viz.cell_history(panel, "not_a_cell").empty


def test_cell_forecast_rows_and_order(predictions):
    fc = viz.cell_forecast(predictions, CELLS[0], "baseline")
    assert list(fc.columns) == ["horizon_months", "q10", "q50", "q90"]
    assert fc["horizon_months"].tolist() == sorted(HORIZONS)
    assert (fc["q10"] <= fc["q50"]).all() and (fc["q50"] <= fc["q90"]).all()
    assert viz.cell_forecast(predictions, CELLS[0], "no_such_scenario").empty


# ---------------------------------------------------------------------------
# the app file itself
# ---------------------------------------------------------------------------


def test_flask_app_compiles():
    """The Flask app module must be syntactically valid."""
    app_path = config.REPO_ROOT / "app" / "flask_app.py"
    assert app_path.is_file()
    compile(app_path.read_text(), str(app_path), "exec")


def test_viz_module_is_framework_free():
    """The contract: bakuml.viz stays importable/testable without flask/streamlit."""
    import pathlib
    import re

    source = pathlib.Path(viz.__file__).read_text()
    assert not re.search(r"^\s*(import|from)\s+(streamlit|flask)", source, re.MULTILINE)
