"""Spatial backbone: from point listings to a (cell, month) panel.

Methodology
-----------
The study discretises Baku & Absheron into cells and aggregates listings
within them. Uber H3 hexagons are the documented baseline: near-uniform
area, six equidistant neighbours, and a parent hierarchy that yields
contiguous blocks for spatial cross-validation - all of which beat
administrative rayons, whose polygons blend elite blocks with industrial
outskirts (the Modifiable Areal Unit Problem, MAUP).

Hexagons nonetheless impose a *uniform* grid on a radically non-uniform
city, which on this panel discards a large share of the data to the
thinness filter below. So the geometry is not hard-coded here: every
function delegates to a `bakuml.spatial.tessellation.Tessellation` (H3, an
adaptive KD-tree of equal-count cells, or contiguity-constrained market
regions), defaulting to the module-level active one. Pass `tess=` to
override per call; `scripts/maup_study.py` compares the options under the
project's leakage-proof protocols.

The `h3` column keeps its name for schema stability: it holds whatever cell
id the active tessellation produces (an H3 index, a KD path like
`kd:0110`, or a region id like `mr:0042`).

Listings are aggregated to one row per (cell, month) with the median and
mean of `price_azn_m2`, the listing count, and the share of new-construction
listings. Cell-months with fewer than `config.MIN_LISTINGS_PER_CELL_MONTH`
listings are dropped: a median of one or two adverts is broker noise, not a
market price. For tensor-shaped consumers (STGCN) the sparse panel is
re-inflated to the full cells x months rectangle by `complete_panel`, with
explicit `n_listings = 0` and NaN prices so downstream code can mask
unobserved cell-months instead of silently treating them as zeros.

Where this module does touch h3 directly it uses the v4 API exclusively.
"""

from __future__ import annotations

import pandas as pd
from shapely.geometry import Point, Polygon, mapping, shape

from bakuml import config
from bakuml.data.schema import PANEL_BASE_COLUMNS, validate_panel
from bakuml.spatial import tessellation as tess_mod
from bakuml.spatial.tessellation import Tessellation

# Built once at import time from config; used by filter_land_cells.
_LAND_POLY = Polygon(config.LAND_POLYGON_LONLAT)


def assign_cells(
    listings: pd.DataFrame,
    res: int | None = None,
    *,
    tess: Tessellation | None = None,
) -> pd.DataFrame:
    """Return a copy of `listings` with an added `h3` cell-id column.

    Cells come from `tess` (default: the active tessellation). `res` is a
    convenience shortcut selecting an H3 grid at that resolution, kept
    because the documented baseline is resolution-parameterised.
    The input frame is never mutated.
    """
    if res is not None and tess is not None:
        raise ValueError("pass either res= or tess=, not both")
    if res is not None:
        tess = tess_mod.H3Tessellation(res, min(res, config.H3_BLOCK_RESOLUTION))
    tess = tess_mod.resolve(tess)
    out = listings.copy()
    out["h3"] = tess.assign(out["lat"].to_numpy(), out["lon"].to_numpy())
    return out


def filter_land_cells(
    listings: pd.DataFrame, *, tess: Tessellation | None = None
) -> pd.DataFrame:
    """Drop listings whose cell centroid falls in the Caspian Sea.

    Requires an ``h3`` column (call :func:`assign_cells` first). Checks each
    unique cell's centroid - via the active tessellation, so this works for
    hexagons, KD-tree rectangles and market regions alike - against the
    simplified Absheron land polygon (:data:`config.LAND_POLYGON_LONLAT`).
    Returns the subset of rows whose cell is on land.

    Note that an adaptive tessellation is already largely self-limiting
    here: its cells are drawn from where listings actually are, so it
    generates few sea cells to begin with. The filter still matters for the
    map, where a cell's *polygon* can reach offshore.
    """
    tess = tess_mod.resolve(tess)
    cells = listings["h3"].unique()
    land_cells = set()
    for c in cells:
        lat, lon = tess.centroid(c)
        if _LAND_POLY.contains(Point(lon, lat)):
            land_cells.add(c)
    return listings[listings["h3"].isin(land_cells)].reset_index(drop=True)


def build_cell_month_panel(
    listings: pd.DataFrame,
    min_listings: int = config.MIN_LISTINGS_PER_CELL_MONTH,
) -> pd.DataFrame:
    """Aggregate listings into the (h3, month) panel.

    For every (cell, listed_month) group the panel records the median and
    mean `price_azn_m2`, the number of listings, and the share of
    `building_type == "new"` listings. Groups with fewer than `min_listings`
    adverts are dropped (thin cell-months carry no reliable price signal).

    If `listings` lacks an `h3` column it is assigned at the default
    resolution first. Returns exactly `schema.PANEL_BASE_COLUMNS`, sorted by
    (h3, month).
    """
    if "h3" not in listings.columns:
        listings = assign_cells(listings)

    work = listings[["h3", "listed_month", "price_azn_m2"]].copy()
    work["is_new"] = listings["building_type"].eq("new")

    panel = (
        work.groupby(["h3", "listed_month"])
        .agg(
            price_azn_m2_median=("price_azn_m2", "median"),
            price_azn_m2_mean=("price_azn_m2", "mean"),
            n_listings=("price_azn_m2", "size"),
            new_share=("is_new", "mean"),
        )
        .reset_index()
        .rename(columns={"listed_month": "month"})
    )
    panel = panel[panel["n_listings"] >= min_listings]
    panel = (
        panel[list(PANEL_BASE_COLUMNS)]
        .sort_values(["h3", "month"], kind="stable")
        .reset_index(drop=True)
    )
    panel["n_listings"] = panel["n_listings"].astype("int64")
    return validate_panel(panel)


def complete_panel(panel: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Reindex `panel` to the full cells x months rectangle.

    Every cell present in `panel` gets one row for every month in `months`.
    Cell-months absent from the input become explicit missing observations:
    `n_listings = 0` and NaN prices / new_share. This dense rectangle is what
    the forecasting tensors (`models.stgcn.panel_tensor`) are built from —
    the NaNs drive the observation mask, so unobserved cell-months are never
    mistaken for zero-price markets.

    Extra (feature) columns present in `panel` are carried through and are
    NaN on the filled-in rows. Rows of `panel` whose month falls outside
    `months` are excluded from the rectangle.
    """
    cells = sorted(panel["h3"].unique())
    full = pd.MultiIndex.from_product([cells, list(months)], names=["h3", "month"])
    out = panel.set_index(["h3", "month"]).reindex(full).reset_index()
    out["n_listings"] = out["n_listings"].fillna(0).astype("int64")
    ordered = list(PANEL_BASE_COLUMNS) + [
        c for c in out.columns if c not in PANEL_BASE_COLUMNS
    ]
    return out[ordered]


def cell_centroids(
    cells: list[str], *, tess: Tessellation | None = None
) -> pd.DataFrame:
    """DataFrame [h3, lat, lon] with the centroid of each cell."""
    tess = tess_mod.resolve(tess)
    latlng = [tess.centroid(c) for c in cells]
    return pd.DataFrame(
        {
            "h3": list(cells),
            "lat": [ll[0] for ll in latlng],
            "lon": [ll[1] for ll in latlng],
        }
    )


def cells_to_geojson(
    cells: list[str],
    properties: dict[str, dict] | None = None,
    *,
    tess: Tessellation | None = None,
) -> dict:
    """GeoJSON FeatureCollection of the cells' hexagon polygons.

    GeoJSON (RFC 7946) mandates `[lon, lat]` coordinate order and a closed
    linear ring (first vertex repeated last). `Tessellation.boundary` returns
    `(lat, lng)` tuples, so each vertex is swapped here — getting this wrong
    draws the cells in the Gulf of Guinea instead of the Caspian.

    Each feature carries `id = <cell>` (used by folium/plotly choropleth
    joins) and `properties["h3"] = <cell>`; per-cell entries from
    `properties` (a `{cell: {name: value}}` mapping) are merged in.
    """
    tess = tess_mod.resolve(tess)
    features = []
    for cell in cells:
        ring = [[lng, lat] for lat, lng in tess.boundary(cell)]
        ring.append(list(ring[0]))  # close the ring
        cell_poly = Polygon(ring)
        # Clip to land so coastal cells don't extend into the Caspian
        clipped = cell_poly.intersection(_LAND_POLY)
        if clipped.is_empty:
            continue
        props = {"h3": cell}
        if properties and cell in properties:
            props.update(properties[cell])
        features.append(
            {
                "type": "Feature",
                "id": cell,
                "geometry": mapping(clipped),
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}
