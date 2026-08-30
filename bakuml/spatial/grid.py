"""H3 spatial backbone: from point listings to a (cell, month) panel.

Methodology
-----------
The study discretises Baku & Absheron into Uber H3 hexagons at resolution 8
(~0.73 km^2 per cell, `config.H3_RESOLUTION`). Hexagons are preferred over
administrative districts because (a) they have near-uniform area, so cell
medians are comparable across the city, (b) each cell has exactly six
equidistant neighbours (pentagons never occur in our bounding box at res 8),
which gives clean spatial-lag and graph-convolution operators, and (c) the
hierarchy (`cell_to_parent`) yields contiguous blocks for spatial
cross-validation.

Listings are aggregated to one row per (cell, month) with the median and
mean of `price_azn_m2`, the listing count, and the share of new-construction
listings. Cell-months with fewer than `config.MIN_LISTINGS_PER_CELL_MONTH`
listings are dropped: a median of one or two adverts is broker noise, not a
market price. For tensor-shaped consumers (STGCN) the sparse panel is
re-inflated to the full cells x months rectangle by `complete_panel`, with
explicit `n_listings = 0` and NaN prices so downstream code can mask
unobserved cell-months instead of silently treating them as zeros.

All functions use the h3 v4 API exclusively.
"""

from __future__ import annotations

import h3
import pandas as pd

from bakuml import config
from bakuml.data.schema import PANEL_BASE_COLUMNS, validate_panel


def assign_cells(listings: pd.DataFrame, res: int = config.H3_RESOLUTION) -> pd.DataFrame:
    """Return a copy of `listings` with an added `h3` cell-id column.

    Cells are computed from the `lat` / `lon` columns at resolution `res`.
    The input frame is never mutated.
    """
    out = listings.copy()
    out["h3"] = [
        h3.latlng_to_cell(la, lo, res)
        for la, lo in zip(out["lat"].to_numpy(), out["lon"].to_numpy())
    ]
    return out


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


def cell_centroids(cells: list[str]) -> pd.DataFrame:
    """DataFrame [h3, lat, lon] with the centroid of each cell."""
    latlng = [h3.cell_to_latlng(c) for c in cells]
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
) -> dict:
    """GeoJSON FeatureCollection of the cells' hexagon polygons.

    GeoJSON (RFC 7946) mandates `[lon, lat]` coordinate order and a closed
    linear ring (first vertex repeated last). `h3.cell_to_boundary` returns
    `(lat, lng)` tuples, so each vertex is swapped here — getting this wrong
    draws hexagons in the Gulf of Guinea instead of the Caspian.

    Each feature carries `id = <cell>` (used by folium/plotly choropleth
    joins) and `properties["h3"] = <cell>`; per-cell entries from
    `properties` (a `{cell: {name: value}}` mapping) are merged in.
    """
    features = []
    for cell in cells:
        ring = [[lng, lat] for lat, lng in h3.cell_to_boundary(cell)]
        ring.append(list(ring[0]))  # close the ring
        props = {"h3": cell}
        if properties and cell in properties:
            props.update(properties[cell])
        features.append(
            {
                "type": "Feature",
                "id": cell,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}
