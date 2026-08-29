"""Tests for `bakuml.features`: masterplan, infrastructure, build.

Exactness strategy
------------------
Geometry features are asserted against independent brute-force computations
from `config` (haversine over `h3.cell_to_latlng` centroids). Temporal
features are asserted *exactly* on a tiny hand-built panel of three mutually
adjacent cells, where every lag / neighbour mean can be computed by hand —
including a deliberately missing cell-month that must produce NaN (and a
dropped row), not a stale value. The leakage canary required by DESIGN.md
perturbs all prices after a cutoff and asserts the pre-cutoff feature rows
are bit-identical.

Sample-size note: this file asserts structural and exact-arithmetic
properties, not statistical recovery of planted effects, so a reduced
synthetic sample (300 listings/month instead of 800) is sufficient and keeps
the file fast. The recovery tests that genuinely need statistical power live
in test_did.py / test_dedup.py.
"""

from __future__ import annotations

import dataclasses

import h3
import numpy as np
import pandas as pd
import pytest

from bakuml import config
from bakuml.config import SYNTHETIC_TRUTH
from bakuml.data.synthetic import generate_listings
from bakuml.features import build, infrastructure, masterplan
from bakuml.geo import haversine_km
from bakuml.spatial import grid

B04 = next(s for s in config.METRO_STATIONS if s.name == config.DID_TREATMENT_STATION)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def listings() -> pd.DataFrame:
    """Reduced synthetic sample (see module docstring for the justification)."""
    truth = dataclasses.replace(SYNTHETIC_TRUTH, n_listings_per_month=300)
    frame, _ = generate_listings(truth)
    return frame


@pytest.fixture(scope="module")
def panel(listings: pd.DataFrame) -> pd.DataFrame:
    return grid.build_cell_month_panel(grid.assign_cells(listings))


@pytest.fixture(scope="module")
def fm(panel: pd.DataFrame) -> pd.DataFrame:
    return build.build_feature_matrix(panel)


@pytest.fixture(scope="module")
def triangle() -> tuple[str, str, str]:
    """Three mutually adjacent res-8 cells around the city centre."""
    c0 = h3.latlng_to_cell(*config.CITY_CENTRE, config.H3_RESOLUTION)
    ring = h3.grid_ring(c0, 1)
    n1 = ring[0]
    n2 = next(c for c in ring if c != n1 and c in h3.grid_disk(n1, 1))
    for a, b in [(c0, n1), (c0, n2), (n1, n2)]:
        assert b in h3.grid_disk(a, 1)
    return c0, n1, n2


@pytest.fixture()
def tiny_panel(triangle: tuple[str, str, str]) -> pd.DataFrame:
    """Hand-built observed panel: 3 mutually adjacent cells, 4 months.

    n2 deliberately skips 2024-02: the row (n2, 2024-03) must then be
    dropped (its calendar t-1 is unobserved), and (n2, 2024-04) must get
    lag_own_3m from 2024-01 — a naive positional shift would get both wrong.
    """
    c0, n1, n2 = triangle
    rows = [
        # h3, month, median, n_listings, new_share
        (c0, "2024-01", 100.0, 3, 0.00),
        (c0, "2024-02", 110.0, 4, 0.25),
        (c0, "2024-03", 120.0, 5, 0.40),
        (n1, "2024-01", 200.0, 3, 1.00),
        (n1, "2024-02", 210.0, 6, 0.50),
        (n1, "2024-03", 220.0, 3, 0.00),
        (n2, "2024-01", 300.0, 4, 0.75),
        (n2, "2024-03", 330.0, 3, 1.00),
        (n2, "2024-04", 340.0, 5, 0.20),
    ]
    frame = pd.DataFrame(
        rows, columns=["h3", "month", "price_azn_m2_median", "n_listings", "new_share"]
    )
    frame["price_azn_m2_mean"] = frame["price_azn_m2_median"]
    return frame[
        ["h3", "month", "price_azn_m2_median", "price_azn_m2_mean",
         "n_listings", "new_share"]
    ]


def _centroid(cell: str) -> tuple[float, float]:
    return h3.cell_to_latlng(cell)


# --------------------------------------------------------------------------
# masterplan.py
# --------------------------------------------------------------------------


def test_masterplan_schema_and_dedup(triangle: tuple[str, str, str]) -> None:
    c0, n1, _ = triangle
    out = masterplan.masterplan_features([c0, n1, c0])  # duplicate collapses
    assert list(out.columns) == ["h3", *masterplan.MASTERPLAN_FEATURE_COLS]
    assert out["h3"].tolist() == [c0, n1]  # one row per cell, order kept
    node_cols = [f"dist_node_{n}_km" for n in config.POLYCENTRIC_NODES]
    assert all(c in out.columns for c in node_cols)


def test_masterplan_distances_exact(triangle: tuple[str, str, str]) -> None:
    c0 = triangle[0]
    out = masterplan.masterplan_features([c0]).iloc[0]
    lat, lon = _centroid(c0)
    assert out["dist_centre_km"] == pytest.approx(
        float(haversine_km(lat, lon, *config.CITY_CENTRE))
    )
    node_d = {}
    for name, (nlat, nlon) in config.POLYCENTRIC_NODES.items():
        d = float(haversine_km(lat, lon, nlat, nlon))
        node_d[name] = d
        assert out[f"dist_node_{name}_km"] == pytest.approx(d)
    assert out["dist_nearest_node_km"] == pytest.approx(min(node_d.values()))
    # max over nodes of 1/(1+d) equals 1/(1+min d)
    assert out["node_gravity"] == pytest.approx(1.0 / (1.0 + min(node_d.values())))
    assert 0.0 < out["node_gravity"] <= 1.0


def test_redev_zone_flags() -> None:
    zlat, zlon, _ = config.REDEVELOPMENT_ZONES["sovetski"]
    inside = h3.latlng_to_cell(zlat, zlon, config.H3_RESOLUTION)
    far = h3.latlng_to_cell(
        *config.POLYCENTRIC_NODES["sumgait"], config.H3_RESOLUTION
    )
    out = masterplan.masterplan_features([inside, far]).set_index("h3")
    assert out.loc[inside, "in_redev_zone"] == 1
    assert out.loc[far, "in_redev_zone"] == 0
    assert out["in_redev_zone"].dtype == np.int64


# --------------------------------------------------------------------------
# infrastructure.py
# --------------------------------------------------------------------------


def _brute_metro(cell: str, as_of: str, include_planned: bool) -> tuple[float, int]:
    """Independent brute-force (dist_metro_km, n_stations_2km) from config."""
    lat, lon = _centroid(cell)
    active = [
        s
        for s in config.METRO_STATIONS
        if (s.opened is not None and s.opened <= as_of)
        or (
            include_planned
            and s.opened is None
            and s.simulated_open is not None
            and s.simulated_open <= as_of
        )
    ]
    dists = [float(haversine_km(lat, lon, s.lat, s.lon)) for s in active]
    if not dists:
        return float("inf"), 0
    return min(dists), sum(d <= infrastructure.STATION_COUNT_RADIUS_KM for d in dists)


def test_metro_features_match_bruteforce() -> None:
    cells = [
        h3.latlng_to_cell(40.3797, 49.8489, config.H3_RESOLUTION),  # 28 May
        h3.latlng_to_cell(*config.POLYCENTRIC_NODES["sumgait"], config.H3_RESOLUTION),
        h3.latlng_to_cell(B04.lat, B04.lon, config.H3_RESOLUTION),
    ]
    for as_of in ["1967-11", "2023-01", "2026-08"]:
        for planned in (False, True):
            out = infrastructure.metro_features(
                cells, as_of, include_planned=planned
            )
            assert list(out.columns) == ["h3", *infrastructure.METRO_FEATURE_COLS]
            assert out["n_stations_2km"].dtype == np.int64
            for _, row in out.iterrows():
                d, n = _brute_metro(row["h3"], as_of, planned)
                assert row["dist_metro_km"] == pytest.approx(d)
                assert row["n_stations_2km"] == n


def test_metro_no_station_open_yet(triangle: tuple[str, str, str]) -> None:
    out = infrastructure.metro_features([triangle[0]], "1960-01").iloc[0]
    assert np.isinf(out["dist_metro_km"])
    assert out["n_stations_2km"] == 0
    # dist_treatment_km is computed even when nothing is open
    assert np.isfinite(out["dist_treatment_km"])


def test_metro_planned_b04_switches_on() -> None:
    """dist_metro_km drops for the B-04 cell once simulated_open passes."""
    cell = h3.latlng_to_cell(B04.lat, B04.lon, config.H3_RESOLUTION)

    def dist(as_of: str, planned: bool) -> float:
        return infrastructure.metro_features(
            [cell], as_of, include_planned=planned
        )["dist_metro_km"].iloc[0]

    # Before simulated_open (2025-06) the planned flag changes nothing.
    assert dist("2025-05", True) == pytest.approx(dist("2025-05", False))
    # From simulated_open onward, planned strictly beats baseline here
    # (the cell contains B-04, whose nearest open station is > 1 km away).
    for as_of in ["2025-06", "2026-08"]:
        assert dist(as_of, True) < dist(as_of, False)
        # opened=None stations never count without include_planned
        assert dist(as_of, False) == pytest.approx(dist("2025-05", False))


def test_dist_treatment_always_to_b04(triangle: tuple[str, str, str]) -> None:
    cell = triangle[0]
    lat, lon = _centroid(cell)
    expected = float(haversine_km(lat, lon, B04.lat, B04.lon))
    for as_of in ["2023-01", "2025-06", "2026-08"]:
        for planned in (False, True):
            out = infrastructure.metro_features([cell], as_of, include_planned=planned)
            assert out["dist_treatment_km"].iloc[0] == pytest.approx(expected)


def test_metro_station_count_monotone_over_time() -> None:
    cell = h3.latlng_to_cell(40.3797, 49.8489, config.H3_RESOLUTION)  # 28 May
    counts = [
        infrastructure.metro_features([cell], m)["n_stations_2km"].iloc[0]
        for m in ["1967-11", "1980-01", "2000-01", "2026-08"]
    ]
    assert counts == sorted(counts)
    assert counts[0] >= 1  # the 1967 stations are within 2 km of 28 May


# --------------------------------------------------------------------------
# build.py — schema and lag correctness on the synthetic panel
# --------------------------------------------------------------------------


def test_feature_cols_exist_and_are_numeric(fm: pd.DataFrame) -> None:
    for col in build.FEATURE_COLS:
        assert col in fm.columns, col
        assert pd.api.types.is_numeric_dtype(fm[col]), col
    # outcome-adjacent / key columns must NOT be model inputs
    forbidden = {
        "h3", "month", "price_azn_m2_median", "price_azn_m2_mean",
        "n_listings", "new_share",
    }
    assert not forbidden & set(build.FEATURE_COLS)
    # but the panel columns are still carried in the frame (models need y)
    assert config.TARGET_COL in fm.columns
    assert not fm.duplicated(subset=["h3", "month"]).any()
    assert fm["lag_own_1m"].notna().all()  # rows without lag_own_1m dropped
    assert fm["month_ix"].dtype == np.int64


def test_row_set_and_lags_match_panel_lookup(panel: pd.DataFrame, fm: pd.DataFrame) -> None:
    """Every lag equals a direct (h3, t-k) panel lookup; rows whose calendar
    t-1 is unobserved are exactly the ones dropped."""
    key = {
        (h, m): v
        for h, m, v in zip(panel["h3"], panel["month"], panel[config.TARGET_COL])
    }

    def lagged(k: int) -> np.ndarray:
        shifted = (pd.PeriodIndex(fm["month"], freq="M") - k).astype(str)
        return np.array(
            [key.get((h, m), np.nan) for h, m in zip(fm["h3"], shifted)]
        )

    assert np.allclose(fm["lag_own_1m"].to_numpy(), lagged(1))
    assert np.allclose(fm["lag_own_3m"].to_numpy(), lagged(3), equal_nan=True)
    exp1, exp4 = lagged(1), lagged(4)
    with np.errstate(invalid="ignore"):
        exp_mom = (exp1 - exp4) / exp4 * 100.0
    assert np.allclose(fm["mom_3m"].to_numpy(), exp_mom, equal_nan=True)

    prev = (pd.PeriodIndex(panel["month"], freq="M") - 1).astype(str)
    expected_rows = {
        (h, m)
        for h, m, p in zip(panel["h3"], panel["month"], prev)
        if (h, p) in key
    }
    assert set(zip(fm["h3"], fm["month"])) == expected_rows


def test_month_ix_positions(panel: pd.DataFrame, fm: pd.DataFrame) -> None:
    order = {m: i for i, m in enumerate(sorted(panel["month"].unique()))}
    assert (fm["month_ix"] == fm["month"].map(order)).all()


def test_prev_month_supply_columns(panel: pd.DataFrame, fm: pd.DataFrame) -> None:
    n_key = {(h, m): v for h, m, v in zip(panel["h3"], panel["month"], panel["n_listings"])}
    prev = (pd.PeriodIndex(fm["month"], freq="M") - 1).astype(str)
    expected = np.array(
        [float(n_key.get((h, m), np.nan)) for h, m in zip(fm["h3"], prev)]
    )
    assert np.allclose(fm["n_listings_prev_month"].to_numpy(), expected, equal_nan=True)
    ok = fm["new_share_prev_month"].dropna()
    assert ok.between(0.0, 1.0).all()


# --------------------------------------------------------------------------
# build.py — exact temporal arithmetic on the tiny hand-built panel
# --------------------------------------------------------------------------


def test_tiny_panel_exact(tiny_panel: pd.DataFrame, triangle: tuple[str, str, str]) -> None:
    c0, n1, n2 = triangle
    out = build.build_feature_matrix(tiny_panel)
    got = out.set_index(["h3", "month"])

    # survivors: calendar t-1 observed; (n2, 2024-03) dropped, all first months dropped
    assert set(got.index) == {
        (c0, "2024-02"), (c0, "2024-03"),
        (n1, "2024-02"), (n1, "2024-03"),
        (n2, "2024-04"),
    }

    # own lags (calendar-aware)
    assert got.loc[(c0, "2024-02"), "lag_own_1m"] == pytest.approx(100.0)
    assert got.loc[(c0, "2024-03"), "lag_own_1m"] == pytest.approx(110.0)
    assert got.loc[(n1, "2024-03"), "lag_own_1m"] == pytest.approx(210.0)
    assert got.loc[(n2, "2024-04"), "lag_own_1m"] == pytest.approx(330.0)
    # lag_own_3m: only (n2, 2024-04) reaches back to 2024-01
    assert got.loc[(n2, "2024-04"), "lag_own_3m"] == pytest.approx(300.0)
    lag3_others = got.drop(index=(n2, "2024-04"))["lag_own_3m"]
    assert lag3_others.isna().all()
    # mom_3m needs t-4: out of range everywhere here — stays NaN (not dropped)
    assert got["mom_3m"].isna().all()

    # hand-computed neighbour means at t-1 (3 mutually adjacent cells)
    expected_nbr = {
        (c0, "2024-02"): (200.0 + 300.0) / 2,
        (n1, "2024-02"): (100.0 + 300.0) / 2,
        (c0, "2024-03"): 210.0,           # n2 unobserved in 2024-02
        (n1, "2024-03"): 110.0,           # n2 unobserved in 2024-02
        (n2, "2024-04"): (120.0 + 220.0) / 2,
    }
    for key_, want in expected_nbr.items():
        assert got.loc[key_, "nbr_price_prev_month"] == pytest.approx(want)

    # previous-month supply composition
    assert got.loc[(c0, "2024-02"), "n_listings_prev_month"] == pytest.approx(3.0)
    assert got.loc[(n1, "2024-03"), "n_listings_prev_month"] == pytest.approx(6.0)
    assert got.loc[(n2, "2024-04"), "n_listings_prev_month"] == pytest.approx(3.0)
    assert got.loc[(c0, "2024-02"), "new_share_prev_month"] == pytest.approx(0.00)
    assert got.loc[(n2, "2024-04"), "new_share_prev_month"] == pytest.approx(1.00)

    # month_ix over the 4 sorted panel months
    assert got.loc[(c0, "2024-02"), "month_ix"] == 1
    assert got.loc[(n2, "2024-04"), "month_ix"] == 3

    # static geometry rides along correctly
    lat, lon = _centroid(c0)
    assert got.loc[(c0, "2024-02"), "dist_treatment_km"] == pytest.approx(
        float(haversine_km(lat, lon, B04.lat, B04.lon))
    )


# --------------------------------------------------------------------------
# build.py — the leakage canary (DESIGN.md requirement)
# --------------------------------------------------------------------------


def test_leakage_canary(panel: pd.DataFrame, fm: pd.DataFrame) -> None:
    """Tripling every price after a cutoff must leave all feature rows at
    months <= cutoff bit-identical — proof that no feature reads the future."""
    months = sorted(panel["month"].unique())
    t0 = months[len(months) // 2]

    poisoned = panel.copy()
    after = poisoned["month"] > t0
    assert after.any() and (~after).any()
    for col in ["price_azn_m2_median", "price_azn_m2_mean"]:
        poisoned.loc[after, col] = poisoned.loc[after, col] * 3.0

    fm_poisoned = build.build_feature_matrix(poisoned)

    base_slice = fm[fm["month"] <= t0]
    pois_slice = fm_poisoned[fm_poisoned["month"] <= t0]
    assert len(base_slice) > 0
    pd.testing.assert_frame_equal(base_slice, pois_slice, check_exact=True)


# --------------------------------------------------------------------------
# build.py — planned-metro scenario switch
# --------------------------------------------------------------------------


def test_include_planned_metro_flag(panel: pd.DataFrame, fm: pd.DataFrame) -> None:
    open_month = B04.simulated_open
    fm_planned = build.build_feature_matrix(panel, include_planned_metro=True)

    # identical row set; only metro accessibility columns may differ
    assert list(fm_planned.columns) == list(fm.columns)
    assert (fm_planned["h3"] == fm["h3"]).all()
    assert (fm_planned["month"] == fm["month"]).all()
    pd.testing.assert_frame_equal(
        fm.drop(columns=["dist_metro_km", "n_stations_2km"]),
        fm_planned.drop(columns=["dist_metro_km", "n_stations_2km"]),
    )

    pre = (fm["month"] < open_month).to_numpy()
    post = ~pre
    assert post.any()
    # before simulated_open the planned flag is a no-op...
    assert np.array_equal(
        fm.loc[pre, "dist_metro_km"].to_numpy(),
        fm_planned.loc[pre, "dist_metro_km"].to_numpy(),
    )
    # ...afterwards the planned station can only bring the metro closer
    assert (
        fm_planned.loc[post, "dist_metro_km"].to_numpy()
        <= fm.loc[post, "dist_metro_km"].to_numpy() + 1e-12
    ).all()
    assert (
        fm_planned.loc[post, "n_stations_2km"].to_numpy()
        >= fm.loc[post, "n_stations_2km"].to_numpy()
    ).all()
