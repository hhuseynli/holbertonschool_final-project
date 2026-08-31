"""Tessellations must satisfy one contract, whatever their geometry.

The pipeline is agnostic to how space is carved up, so the invariants that
matter are shared: a total partition (every point lands in exactly one
cell), symmetric contiguity, closed polygons, and contiguous CV blocks.
These tests assert that contract for all three implementations, plus the
properties each one exists for (equal counts for the KD-tree, price
homogeneity and leakage-free fitting for market regions).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import shapely.geometry as sgeom

from bakuml import config
from bakuml.config import SYNTHETIC_TRUTH
from bakuml.data.synthetic import generate_listings
from bakuml.spatial import tessellation as tess_mod
from bakuml.spatial.tessellation import (
    AdaptiveKDTessellation,
    H3Tessellation,
    MarketRegionTessellation,
    Tessellation,
)
from bakuml.spatial.grid import assign_cells, build_cell_month_panel, cells_to_geojson

CUTOFF = "2026-01"


@pytest.fixture(scope="module")
def listings():
    truth = dataclasses.replace(SYNTHETIC_TRUTH, n_listings_per_month=250)
    df, _ = generate_listings(truth=truth)
    return df


@pytest.fixture(scope="module")
def fitted(listings):
    """One fitted instance of each tessellation."""
    out = {
        "h3": H3Tessellation(8, 6),
        "kdtree": AdaptiveKDTessellation(target_per_cell=150),
        "market": MarketRegionTessellation(n_regions=60),
    }
    for t in out.values():
        t.fit(listings, price_cutoff_month=CUTOFF)
    return out


@pytest.fixture(autouse=True)
def _restore_active():
    """No test may leak an active tessellation into another."""
    yield
    tess_mod.reset_active()


# --------------------------------------------------------------------------
# Shared contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["h3", "kdtree", "market"])
def test_satisfies_protocol(fitted, kind):
    assert isinstance(fitted[kind], Tessellation)


@pytest.mark.parametrize("kind", ["h3", "kdtree", "market"])
def test_assign_is_total_and_deterministic(fitted, listings, kind):
    tess = fitted[kind]
    lat = listings["lat"].to_numpy()[:2000]
    lon = listings["lon"].to_numpy()[:2000]
    a = tess.assign(lat, lon)
    b = tess.assign(lat, lon)
    assert len(a) == len(lat)
    assert all(x is not None and isinstance(x, str) and x for x in a)
    assert list(a) == list(b)


@pytest.mark.parametrize("kind", ["h3", "kdtree", "market"])
def test_contiguity_is_symmetric(fitted, listings, kind):
    tess = fitted[kind]
    cells = sorted(set(tess.assign(
        listings["lat"].to_numpy()[:3000], listings["lon"].to_numpy()[:3000]
    )))
    present = set(cells)
    for c in cells[:60]:
        for nb in tess.neighbours(c, 1):
            if nb in present:
                assert c in tess.neighbours(nb, 1), f"{kind}: {c}~{nb} asymmetric"
            assert nb != c, "a cell must not be its own neighbour"


@pytest.mark.parametrize("kind", ["h3", "kdtree", "market"])
def test_k_ring_grows_monotonically(fitted, listings, kind):
    tess = fitted[kind]
    cells = sorted(set(tess.assign(
        listings["lat"].to_numpy()[:3000], listings["lon"].to_numpy()[:3000]
    )))
    # A cell with at least two neighbours, so k=2 can actually expand.
    cell = next((c for c in cells if len(tess.neighbours(c, 1)) >= 2), cells[0])
    k1 = set(tess.neighbours(cell, 1))
    k2 = set(tess.neighbours(cell, 2))
    assert k1 <= k2
    assert cell not in k2
    with pytest.raises(ValueError):
        tess.neighbours(cell, 0)


@pytest.mark.parametrize("kind", ["h3", "kdtree", "market"])
def test_centroid_inside_bbox(fitted, listings, kind):
    tess = fitted[kind]
    lat_min, lat_max, lon_min, lon_max = config.BBOX
    cells = sorted(set(tess.assign(
        listings["lat"].to_numpy()[:2000], listings["lon"].to_numpy()[:2000]
    )))
    for c in cells[:40]:
        la, lo = tess.centroid(c)
        assert lat_min - 0.2 <= la <= lat_max + 0.2
        assert lon_min - 0.2 <= lo <= lon_max + 0.2


@pytest.mark.parametrize("kind", ["h3", "kdtree", "market"])
def test_geojson_rings_are_valid_and_lonlat(fitted, listings, kind):
    """Rings must close and use [lon, lat] - the classic map-in-the-wrong-
    ocean bug the grid module warns about."""
    tess = fitted[kind]
    cells = sorted(set(tess.assign(
        listings["lat"].to_numpy()[:2000], listings["lon"].to_numpy()[:2000]
    )))[:25]
    gj = cells_to_geojson(cells, tess=tess)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == len(cells)
    for feat in gj["features"]:
        ring = feat["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1], "ring must be closed"
        assert len(ring) >= 4
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        # Baku: lon ~49-50, lat ~40. Swapped order would put lat first.
        assert min(lons) > 48.0 and max(lons) < 51.0
        assert min(lats) > 39.0 and max(lats) < 41.5
        assert sgeom.Polygon(ring).is_valid
        assert feat["id"] == feat["properties"]["h3"]


@pytest.mark.parametrize("kind", ["h3", "kdtree", "market"])
def test_blocks_partition_cells(fitted, listings, kind):
    tess = fitted[kind]
    cells = sorted(set(tess.assign(
        listings["lat"].to_numpy(), listings["lon"].to_numpy()
    )))
    blocks = {c: tess.block(c) for c in cells}
    assert all(isinstance(b, str) and b for b in blocks.values())
    # More than one block, else spatially blocked CV is meaningless.
    assert len(set(blocks.values())) >= 2
    # Blocks must be coarser than cells.
    assert len(set(blocks.values())) < len(cells)


@pytest.mark.parametrize("kind", ["h3", "kdtree", "market"])
def test_panel_builds_and_validates(fitted, listings, kind):
    """The downstream contract: any tessellation yields a valid panel."""
    tess = fitted[kind]
    panel = build_cell_month_panel(assign_cells(listings, tess=tess))
    assert not panel.empty
    assert not panel.duplicated(subset=["h3", "month"]).any()
    assert (panel["n_listings"] >= config.MIN_LISTINGS_PER_CELL_MONTH).all()


# --------------------------------------------------------------------------
# Why each alternative exists
# --------------------------------------------------------------------------


def test_kdtree_cells_hold_equal_counts(fitted, listings):
    """The KD-tree's whole point: homogeneous sample size per cell, so
    estimation variance does not swing by an order of magnitude."""
    tess = fitted["kdtree"]
    ids = tess.assign(listings["lat"].to_numpy(), listings["lon"].to_numpy())
    counts = np.bincount(np.unique(ids, return_inverse=True)[1])
    spread = counts.max() / counts.min()
    assert spread < 1.6, f"KD leaf counts should be near-equal, got {spread:.2f}x"
    # Contrast: the fixed grid's counts span orders of magnitude, which is
    # exactly the estimation-variance heterogeneity the KD-tree removes.
    h3_ids = H3Tessellation(8, 6).assign(
        listings["lat"].to_numpy(), listings["lon"].to_numpy()
    )
    h3_counts = np.bincount(np.unique(h3_ids, return_inverse=True)[1])
    assert h3_counts.max() / h3_counts.min() > 10 * spread


def test_kdtree_retains_more_data_than_h3(fitted, listings):
    """H3 discards ~37 % of listings to the thinness filter; the adaptive
    grid must do materially better - that is the reason it exists."""
    kept = {}
    for kind in ("h3", "kdtree"):
        tess = fitted[kind]
        assigned = assign_cells(listings, tess=tess)
        panel = build_cell_month_panel(assigned)
        kept[kind] = panel["n_listings"].sum() / len(assigned)
    assert kept["kdtree"] > kept["h3"] + 0.10, kept


def test_kdtree_gives_longer_price_histories(fitted, listings):
    """Autoregressive forecasting needs history; fixed cells starve the
    periphery of it."""
    months = {}
    for kind in ("h3", "kdtree"):
        panel = build_cell_month_panel(assign_cells(listings, tess=fitted[kind]))
        months[kind] = panel.groupby("h3")["month"].nunique().median()
    assert months["kdtree"] > months["h3"], months


def test_kdtree_blocks_are_spatially_contiguous(fitted):
    """Blocks come from KD path prefixes, and every prefix is a rectangle -
    so a block is a contiguous area by construction."""
    tess = fitted["kdtree"]
    by_block: dict[str, list[str]] = {}
    for c in tess.cells():
        by_block.setdefault(tess.block(c), []).append(c)
    for block, members in by_block.items():
        if len(members) < 2:
            continue
        # Every member reachable from any other through in-block neighbours.
        seen = {members[0]}
        stack = [members[0]]
        member_set = set(members)
        while stack:
            node = stack.pop()
            for nb in tess.neighbours(node, 1):
                if nb in member_set and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        assert seen == member_set, f"block {block} is not contiguous"


def _within_var_share(tess, listings) -> tuple[float, int]:
    """Mean within-cell log-price variance / total, and the cell count."""
    import pandas as pd

    logp = np.log(listings["price_azn_m2"].to_numpy(dtype=float))
    ids = tess.assign(listings["lat"].to_numpy(), listings["lon"].to_numpy())
    d = pd.DataFrame({"h3": ids, "logp": logp})
    share = float(d.groupby("h3")["logp"].var().mean()) / float(np.var(logp))
    return share, len(set(ids))


def test_market_regions_are_more_homogeneous_than_h3(listings):
    """Market regions exist to cut the map where the market changes.

    The comparison must hold granularity fixed to be meaningful - any
    tessellation looks homogeneous if you make the cells small enough. So
    market regions are pitted against H3 res-7, which has *more* cells:
    beating a finer arbitrary grid with fewer, better-placed boundaries is
    the actual MAUP claim.
    """
    # 120 regions: the budget is shared across the ~20 disconnected
    # components of the observed grid (Sumgait, Alat, the Absheron
    # villages), so too small a budget leaves the dense core unsubdivided.
    market = MarketRegionTessellation(n_regions=120).fit(
        listings, price_cutoff_month=CUTOFF
    )
    hexes = H3Tessellation(7, 5)
    m_share, m_cells = _within_var_share(market, listings)
    h_share, h_cells = _within_var_share(hexes, listings)
    assert m_cells <= h_cells, (
        f"market ({m_cells}) must not be finer than h3_res7 ({h_cells})"
    )
    assert m_share < h_share, {"market": m_share, "h3_res7": h_share}


def test_finer_h3_is_not_automatically_better(listings):
    """Guards the reasoning above: H3 homogeneity improves monotonically
    with resolution, so a raw within-variance comparison across different
    cell counts proves nothing on its own."""
    coarse, n_coarse = _within_var_share(H3Tessellation(6, 5), listings)
    fine, n_fine = _within_var_share(H3Tessellation(8, 6), listings)
    assert n_fine > n_coarse
    assert fine < coarse


def test_market_regions_ignore_post_cutoff_prices(listings):
    """The only price-driven tessellation must be leakage-free: perturbing
    prices after the cutoff cannot move a single region boundary."""
    tampered = listings.copy()
    post = tampered["listed_month"] >= CUTOFF
    assert post.any(), "fixture must span the cutoff"
    tampered.loc[post, "price_azn_m2"] *= 5.0

    a = MarketRegionTessellation(n_regions=60).fit(
        listings, price_cutoff_month=CUTOFF
    )
    b = MarketRegionTessellation(n_regions=60).fit(
        tampered, price_cutoff_month=CUTOFF
    )
    lat = listings["lat"].to_numpy()[:1500]
    lon = listings["lon"].to_numpy()[:1500]
    assert list(a.assign(lat, lon)) == list(b.assign(lat, lon))


def _region_pieces(tess, members: list[str]) -> int:
    """How many disconnected pieces a set of base cells falls into."""
    from bakuml.spatial.tessellation import connected_components

    base = tess._base
    member_set = set(members)
    comps = connected_components(
        members, lambda c: [n for n in base.neighbours(c, 1) if n in member_set]
    )
    return len(comps)


def test_market_regions_are_contiguous(fitted):
    """Ward under a connectivity constraint must never split a region.

    Contiguity is measured against the *base grid's own* components: an
    island cannot be joined to the mainland, so a region is well-formed
    when its pieces are pieces of the observed grid, not of the region.
    """
    tess = fitted["market"]
    base = tess._base
    all_cells = list(tess._member_of)
    from bakuml.spatial.tessellation import connected_components

    grid_comp = {}
    for i, comp in enumerate(
        connected_components(all_cells, lambda c: base.neighbours(c, 1))
    ):
        for c in comp:
            grid_comp[c] = i

    for rid, members in tess._members.items():
        # every member must sit in one component of the observed grid...
        assert len({grid_comp[c] for c in members}) == 1, (
            f"region {rid} spans disconnected parts of the grid"
        )
        # ...and be a single connected piece within it
        assert _region_pieces(tess, members) == 1, f"region {rid} is split"


def test_contiguity_test_can_actually_fail(fitted):
    """Guard the guard: the assertions above must reject a broken region.

    The obvious formulation of this test (BFS from one member and compare
    against the member set) is tautological - unreachable members are by
    definition non-adjacent to the reached set - so it passes even for a
    region deliberately built from two separate places. This asserts the
    real check rejects exactly that.
    """
    tess = fitted["market"]
    regions = sorted(tess._members)
    # Find two regions that do not touch, and pretend they are one.
    for a in regions:
        far = [
            b for b in regions
            if b != a and b not in tess.neighbours(a, 2) and b != a
        ]
        if far:
            merged = tess._members[a] + tess._members[far[0]]
            assert _region_pieces(tess, merged) > 1, (
                "merging two non-adjacent regions must register as split"
            )
            return
    pytest.skip("fixture has no pair of non-adjacent regions")


# --------------------------------------------------------------------------
# Registry / plumbing
# --------------------------------------------------------------------------


def test_build_factory_and_unknown_kind():
    assert isinstance(tess_mod.build("h3"), H3Tessellation)
    assert isinstance(tess_mod.build("kdtree", target_per_cell=90),
                      AdaptiveKDTessellation)
    assert isinstance(tess_mod.build("market", n_regions=42),
                      MarketRegionTessellation)
    with pytest.raises(ValueError, match="unknown tessellation"):
        tess_mod.build("voronoi")


def test_set_active_round_trips():
    original = tess_mod.get_active()
    kd = AdaptiveKDTessellation(100)
    previous = tess_mod.set_active(kd)
    assert previous is original
    assert tess_mod.get_active() is kd
    tess_mod.set_active(previous)
    assert tess_mod.get_active() is original


def test_unfitted_tessellations_refuse_to_answer():
    for tess in (AdaptiveKDTessellation(100), MarketRegionTessellation(20)):
        with pytest.raises(RuntimeError, match="fit"):
            tess.centroid("kd:0")


def test_assign_cells_rejects_conflicting_arguments(listings):
    with pytest.raises(ValueError, match="either res=|not both"):
        assign_cells(listings, res=8, tess=H3Tessellation(8, 6))


def test_h3_rejects_finer_block_than_cells():
    with pytest.raises(ValueError, match="coarser"):
        H3Tessellation(6, 8)
