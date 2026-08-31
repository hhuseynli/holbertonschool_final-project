"""Tests for the H3 spatial backbone: grid, lags, graph.

Correctness of the lag/encoding operators is asserted *exactly* on a tiny
hand-built panel of three mutually adjacent cells x three months, where the
expected neighbour means can be computed by hand. Panel-construction tests
run on a reduced synthetic sample (250 listings/month instead of 800): they
only need broad cell/month coverage, not the statistical power that the
DiD / dedup recovery tests require, so the reduction is safe and keeps the
file fast.
"""

from __future__ import annotations

import dataclasses

import h3
import numpy as np
import pandas as pd
import pytest

from bakuml import config
from bakuml.config import SYNTHETIC_TRUTH
from bakuml.data.schema import PANEL_BASE_COLUMNS, month_range, validate_panel
from bakuml.data.synthetic import generate_listings
from bakuml.spatial import graph, grid, lags
from bakuml.spatial.zones import build_zones, zones_to_geojson

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def listings() -> pd.DataFrame:
    """Reduced synthetic sample (see module docstring for the justification)."""
    truth = dataclasses.replace(SYNTHETIC_TRUTH, n_listings_per_month=250)
    frame, _ = generate_listings(truth)
    return frame


@pytest.fixture(scope="module")
def panel(listings: pd.DataFrame) -> pd.DataFrame:
    return grid.build_cell_month_panel(listings)


@pytest.fixture(scope="module")
def triangle() -> tuple[str, str, str]:
    """Three mutually adjacent res-8 cells around the city centre."""
    c0 = h3.latlng_to_cell(*config.CITY_CENTRE, config.H3_RESOLUTION)
    ring = h3.grid_ring(c0, 1)
    n1 = ring[0]
    n2 = next(c for c in ring if c != n1 and c in h3.grid_disk(n1, 1))
    # sanity: pairwise adjacency (the exact-mean assertions rely on it)
    for a, b in [(c0, n1), (c0, n2), (n1, n2)]:
        assert b in h3.grid_disk(a, 1) and a in h3.grid_disk(b, 1)
    return c0, n1, n2


@pytest.fixture()
def tiny_panel(triangle: tuple[str, str, str]) -> pd.DataFrame:
    """Hand-built panel: 3 mutually adjacent cells x 3 months.

    Missingness is exercised both ways: (n1, m3) has NO row at all, while
    (n2, m2) has a row with a NaN value.
    """
    c0, n1, n2 = triangle
    m1, m2, m3 = "2024-01", "2024-02", "2024-03"
    return pd.DataFrame(
        {
            "h3": [c0, c0, c0, n1, n1, n2, n2, n2],
            "month": [m1, m2, m3, m1, m2, m1, m2, m3],
            "price_azn_m2_median": [100.0, 110.0, 120.0, 200.0, 210.0, 300.0, np.nan, 330.0],
        }
    )


# --------------------------------------------------------------------------
# h3 invariants
# --------------------------------------------------------------------------


def test_h3_disk_invariants() -> None:
    c = h3.latlng_to_cell(*config.CITY_CENTRE, config.H3_RESOLUTION)
    assert not h3.is_pentagon(c)
    disk = h3.grid_disk(c, 1)
    assert len(disk) == 7  # a non-pentagon hexagon has exactly 6 neighbours
    assert c in disk
    assert len(h3.grid_ring(c, 1)) == 6


# --------------------------------------------------------------------------
# grid.py
# --------------------------------------------------------------------------


def test_assign_cells(listings: pd.DataFrame) -> None:
    assigned = grid.assign_cells(listings)
    assert "h3" not in listings.columns  # input not mutated
    assert "h3" in assigned.columns
    sample = assigned["h3"].head(50)
    assert all(h3.get_resolution(c) == config.H3_RESOLUTION for c in sample)
    # cell centroid must be near the listing itself (< ~1 km at res 8)
    row = assigned.iloc[0]
    clat, clon = h3.cell_to_latlng(row["h3"])
    assert abs(clat - row["lat"]) < 0.01 and abs(clon - row["lon"]) < 0.015


def test_build_cell_month_panel_schema_and_filter(panel: pd.DataFrame) -> None:
    assert list(panel.columns) == list(PANEL_BASE_COLUMNS)
    validate_panel(panel)
    assert (panel["n_listings"] >= config.MIN_LISTINGS_PER_CELL_MONTH).all()
    assert panel["n_listings"].dtype == np.int64
    # sorted by (h3, month)
    keys = list(zip(panel["h3"], panel["month"]))
    assert keys == sorted(keys)
    assert panel["new_share"].between(0.0, 1.0).all()
    assert len(panel) > 100  # broad coverage even on the reduced sample


def test_build_cell_month_panel_aggregates_exactly(listings: pd.DataFrame, panel: pd.DataFrame) -> None:
    assigned = grid.assign_cells(listings)
    top = panel.sort_values("n_listings", ascending=False).iloc[0]
    group = assigned[
        (assigned["h3"] == top["h3"]) & (assigned["listed_month"] == top["month"])
    ]
    assert len(group) == top["n_listings"]
    assert top["price_azn_m2_median"] == pytest.approx(group["price_azn_m2"].median())
    assert top["price_azn_m2_mean"] == pytest.approx(group["price_azn_m2"].mean())
    assert top["new_share"] == pytest.approx(group["building_type"].eq("new").mean())


def test_complete_panel_rectangle(panel: pd.DataFrame) -> None:
    months = month_range(config.PANEL_START, config.PANEL_END)
    dense = grid.complete_panel(panel, months)
    n_cells = panel["h3"].nunique()
    assert len(dense) == n_cells * len(months)
    assert dense["month"].nunique() == len(months)
    assert list(dense.columns) == list(PANEL_BASE_COLUMNS)
    # filled-in rows: n_listings == 0 and NaN prices
    filled = dense[dense["n_listings"] == 0]
    assert len(filled) == len(dense) - len(panel)
    assert filled["price_azn_m2_median"].isna().all()
    assert filled["price_azn_m2_mean"].isna().all()
    # observed rows survive untouched
    merged = dense.merge(panel, on=["h3", "month"], suffixes=("", "_orig"))
    assert len(merged) == len(panel)
    assert np.allclose(
        merged["price_azn_m2_median"], merged["price_azn_m2_median_orig"]
    )
    assert (merged["n_listings"] == merged["n_listings_orig"]).all()


def test_cell_centroids(triangle: tuple[str, str, str]) -> None:
    cells = list(triangle)
    cent = grid.cell_centroids(cells)
    assert list(cent.columns) == ["h3", "lat", "lon"]
    assert cent["h3"].tolist() == cells
    for _, row in cent.iterrows():
        lat, lon = h3.cell_to_latlng(row["h3"])
        assert row["lat"] == pytest.approx(lat)
        assert row["lon"] == pytest.approx(lon)


def test_cells_to_geojson(triangle: tuple[str, str, str]) -> None:
    c0, n1, n2 = triangle
    props = {c0: {"metric": 3.5}}
    gj = grid.cells_to_geojson([c0, n1, n2], properties=props)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 3

    feat = gj["features"][0]
    assert feat["id"] == c0
    assert feat["properties"]["h3"] == c0
    assert feat["properties"]["metric"] == 3.5
    assert gj["features"][1]["properties"] == {"h3": n1}

    ring = feat["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]  # closed linear ring
    assert len(ring) == 7  # 6 hexagon vertices + closing point
    # GeoJSON is [lon, lat]: Baku sits at lon ~49.8, lat ~40.4
    boundary = h3.cell_to_boundary(c0)  # (lat, lng) tuples
    for (blat, blng), (glon, glat) in zip(boundary, ring):
        assert glon == pytest.approx(blng)
        assert glat == pytest.approx(blat)
    assert all(45 < lon < 55 and 39 < lat < 42 for lon, lat in ring)


# --------------------------------------------------------------------------
# lags.py — exact means on the hand-built panel
# --------------------------------------------------------------------------


def test_spatial_lag_exact(tiny_panel: pd.DataFrame, triangle: tuple[str, str, str]) -> None:
    c0, n1, n2 = triangle
    lag = lags.spatial_lag(tiny_panel, "price_azn_m2_median", k=1)
    assert lag.index.equals(tiny_panel.index)

    expected = {
        (c0, "2024-01"): 250.0,  # mean(200, 300)
        (c0, "2024-02"): 210.0,  # n2 is NaN that month -> only n1
        (c0, "2024-03"): 330.0,  # n1 has no row that month -> only n2
        (n1, "2024-01"): 200.0,  # mean(100, 300)
        (n1, "2024-02"): 110.0,  # only c0 observed
        (n2, "2024-01"): 150.0,  # mean(100, 200)
        (n2, "2024-02"): 160.0,  # mean(110, 210)
        (n2, "2024-03"): 120.0,  # only c0 observed
    }
    for i, row in tiny_panel.iterrows():
        assert lag.loc[i] == pytest.approx(expected[(row["h3"], row["month"])])


def test_spatial_lag_excludes_self(tiny_panel: pd.DataFrame, triangle: tuple[str, str, str]) -> None:
    c0, _, _ = triangle
    # keep only c0's rows: no neighbour has data -> all NaN (self excluded)
    solo = tiny_panel[tiny_panel["h3"] == c0].reset_index(drop=True)
    lag = lags.spatial_lag(solo, "price_azn_m2_median", k=1)
    assert lag.isna().all()


def test_neighbour_target_encoding_exact(tiny_panel: pd.DataFrame, triangle: tuple[str, str, str]) -> None:
    c0, n1, n2 = triangle
    enc = lags.neighbour_target_encoding(tiny_panel, "price_azn_m2_median", months_lag=1)
    assert enc.index.equals(tiny_panel.index)

    expected = {
        (c0, "2024-01"): np.nan,  # 2023-12 precedes the panel
        (c0, "2024-02"): 250.0,   # neighbours at 2024-01: mean(200, 300)
        (c0, "2024-03"): 210.0,   # neighbours at 2024-02: only n1
        (n1, "2024-01"): np.nan,
        (n1, "2024-02"): 200.0,   # mean(100, 300)
        (n2, "2024-01"): np.nan,
        (n2, "2024-02"): 150.0,   # mean(100, 200)
        (n2, "2024-03"): 160.0,   # mean(110, 210)
    }
    for i, row in tiny_panel.iterrows():
        want = expected[(row["h3"], row["month"])]
        if np.isnan(want):
            assert np.isnan(enc.loc[i])
        else:
            assert enc.loc[i] == pytest.approx(want)


def test_neighbour_target_encoding_two_month_lag(tiny_panel: pd.DataFrame, triangle: tuple[str, str, str]) -> None:
    c0, _, _ = triangle
    enc = lags.neighbour_target_encoding(tiny_panel, "price_azn_m2_median", months_lag=2)
    row = tiny_panel[(tiny_panel["h3"] == c0) & (tiny_panel["month"] == "2024-03")]
    assert enc.loc[row.index[0]] == pytest.approx(250.0)  # neighbours at 2024-01
    first_month = tiny_panel["month"].isin(["2024-01", "2024-02"])
    assert enc[first_month.to_numpy()].isna().all()


def test_neighbour_target_encoding_rejects_leaky_lag(tiny_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        lags.neighbour_target_encoding(tiny_panel, "price_azn_m2_median", months_lag=0)


def test_neighbour_target_encoding_leakage_canary(tiny_panel: pd.DataFrame) -> None:
    """Perturbing month-t values must not move any encoding at months <= t."""
    base = lags.neighbour_target_encoding(tiny_panel, "price_azn_m2_median", months_lag=1)
    poisoned = tiny_panel.copy()
    last = poisoned["month"] == "2024-03"
    poisoned.loc[last, "price_azn_m2_median"] = 9_999.0
    enc = lags.neighbour_target_encoding(poisoned, "price_azn_m2_median", months_lag=1)
    # encodings at every month <= 2024-03 read only months <= t-1 <= 2024-02
    pd.testing.assert_series_equal(base, enc)


def test_spatial_lag_on_synthetic_panel(panel: pd.DataFrame) -> None:
    lag = lags.spatial_lag(panel, "price_azn_m2_median", k=1)
    assert len(lag) == len(panel)
    observed = lag.dropna()
    assert len(observed) > 0.5 * len(panel)  # dense core has neighbours
    assert (observed > 0).all() and np.isfinite(observed).all()


# --------------------------------------------------------------------------
# graph.py
# --------------------------------------------------------------------------


def test_adjacency_triangle_exact(triangle: tuple[str, str, str]) -> None:
    cells = list(triangle)
    a_hat, order = graph.build_adjacency(cells)
    assert order == cells
    # 3 mutually adjacent cells: A + I is all-ones, degrees all 3
    assert np.allclose(a_hat, np.full((3, 3), 1.0 / 3.0))


def test_adjacency_isolated_cell(triangle: tuple[str, str, str]) -> None:
    c0 = triangle[0]
    far = h3.latlng_to_cell(*config.POLYCENTRIC_NODES["sumgait"], config.H3_RESOLUTION)
    assert far not in h3.grid_disk(c0, 1)
    a_hat, order = graph.build_adjacency([c0, far])
    assert order == [c0, far]
    assert np.allclose(a_hat, np.eye(2))  # self loops only


def test_adjacency_invariants_on_panel_cells(panel: pd.DataFrame) -> None:
    cells = sorted(panel["h3"].unique())[:200]
    a_hat, order = graph.build_adjacency(cells)
    assert order == cells
    assert a_hat.shape == (len(cells), len(cells))
    assert a_hat.dtype == np.float64
    assert np.isfinite(a_hat).all()
    assert np.allclose(a_hat, a_hat.T)  # symmetric normalisation
    assert (np.diag(a_hat) > 0).all()  # self loop survives normalisation
    # normalisation bounds: entries in [0, 1], rows never sum above sqrt(deg)
    assert (a_hat >= 0).all() and (a_hat <= 1).all()
    # D^-1/2 (A+I) D^-1/2 of a graph with self loops has spectral radius 1
    eigvals = np.linalg.eigvalsh(a_hat)
    assert eigvals.max() == pytest.approx(1.0, abs=1e-8)


def test_adjacency_deduplicates_input(triangle: tuple[str, str, str]) -> None:
    c0, n1, _ = triangle
    a_hat, order = graph.build_adjacency([c0, n1, c0])
    assert order == [c0, n1]
    assert a_hat.shape == (2, 2)


def test_edge_index(triangle: tuple[str, str, str]) -> None:
    cells = list(triangle)
    ei = graph.edge_index(cells)
    assert ei.shape == (2, 6)  # 3 undirected edges, both directions
    assert ei.dtype == np.int64
    pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert len(pairs) == 6
    assert all((j, i) in pairs for i, j in pairs)  # both directions
    assert all(i != j for i, j in pairs)  # no self loops
    # matches the off-diagonal support of the normalised adjacency
    a_hat, _ = graph.build_adjacency(cells)
    off = {(i, j) for i in range(3) for j in range(3) if i != j and a_hat[i, j] > 0}
    assert pairs == off


def test_edge_index_empty_graph(triangle: tuple[str, str, str]) -> None:
    c0 = triangle[0]
    far = h3.latlng_to_cell(*config.POLYCENTRIC_NODES["sumgait"], config.H3_RESOLUTION)
    ei = graph.edge_index([c0, far])
    assert ei.shape == (2, 0)
    assert ei.dtype == np.int64


# --------------------------------------------------------------------------
# grid.py — sea filtering
# --------------------------------------------------------------------------


def test_filter_land_cells_rejects_sea(listings: pd.DataFrame) -> None:
    """A synthetic listing placed in the Caspian must be dropped."""
    assigned = grid.assign_cells(listings)
    # Inject a sea cell: place a listing well into the Caspian
    sea_row = assigned.iloc[[0]].copy()
    sea_row["lat"] = 40.40
    sea_row["lon"] = 50.50  # deep in the Caspian
    sea_row["h3"] = h3.latlng_to_cell(40.40, 50.50, config.H3_RESOLUTION)
    combined = pd.concat([assigned, sea_row], ignore_index=True)
    filtered = grid.filter_land_cells(combined)
    assert sea_row["h3"].iloc[0] not in filtered["h3"].values
    # All original land cells survive
    assert filtered["h3"].nunique() == assigned["h3"].nunique()


def test_filter_land_cells_keeps_all_synthetic(listings: pd.DataFrame) -> None:
    """Synthetic listings are all on land: no rows should be dropped."""
    assigned = grid.assign_cells(listings)
    filtered = grid.filter_land_cells(assigned)
    assert len(filtered) == len(assigned)


# --------------------------------------------------------------------------
# zones.py — market micro-zones
# --------------------------------------------------------------------------


def test_build_zones_covers_all_cells(panel: pd.DataFrame) -> None:
    zones = build_zones(panel, max_zones=15)
    assert set(zones["h3"]) == set(panel["h3"].unique())
    assert "zone" in zones.columns
    assert "zone_name" in zones.columns
    assert zones["zone"].nunique() <= 15


def test_zones_to_geojson_structure(panel: pd.DataFrame) -> None:
    zones = build_zones(panel, max_zones=10)
    gj = zones_to_geojson(zones, zone_values={0: {"value": 42.0}})
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == zones["zone"].nunique()
    for feat in gj["features"]:
        assert feat["geometry"]["type"] in ("Polygon", "MultiPolygon")
        assert "zone_name" in feat["properties"]
        assert "n_cells" in feat["properties"]


def test_zones_few_cells() -> None:
    """When cells < max_zones, each cell is its own zone."""
    mini = pd.DataFrame({
        "h3": ["a", "b", "c"],
        "month": ["2024-01"] * 3,
        "n_listings": [10, 20, 30],
    })
    zones = build_zones(mini, max_zones=10)
    assert len(zones) == 3
    assert zones["zone"].nunique() == 3
