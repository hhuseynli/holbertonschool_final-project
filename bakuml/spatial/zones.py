"""Market micro-zones: density-aware clustering of H3 cells.

The analytical backbone stays on H3 hexagons (uniform area, clean spatial
lags, hierarchical CV blocks). This module adds a *presentation layer* that
groups hexagons into visually meaningful market micro-zones using
agglomerative clustering with spatial connectivity constraints.

Clusters are formed using cell centroids + normalised listing density so that
high-activity urban cores get finer zones while sparse periphery cells are
grouped into larger zones. The resulting zone polygons (shapely
``unary_union`` of constituent hex polygons) give the app an organic,
non-hexagonal map appearance.
"""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from bakuml import config


def _hex_polygon(cell: str) -> ShapelyPolygon:
    """Shapely polygon for one H3 cell (lon/lat order)."""
    ring = [(lng, lat) for lat, lng in h3.cell_to_boundary(cell)]
    ring.append(ring[0])
    return ShapelyPolygon(ring)


def build_zones(
    panel: pd.DataFrame,
    *,
    max_zones: int = 40,
    density_weight: float = 0.5,
) -> pd.DataFrame:
    """Cluster panel cells into market micro-zones.

    Parameters
    ----------
    panel : DataFrame
        The (h3, month) panel with ``n_listings`` and ``price_azn_m2_median``.
    max_zones : int
        Target number of zones. The actual count may be slightly lower when
        cells form disconnected components.
    density_weight : float
        Relative importance of listing density vs. spatial proximity in the
        clustering feature space (0 = pure spatial, 1 = density-dominated).

    Returns
    -------
    DataFrame with columns ``[h3, zone, zone_name]`` — one row per cell.
    ``zone`` is an int label, ``zone_name`` is a human-readable string
    (``"Zone 1"`` .. ``"Zone N"``).
    """
    cells = sorted(panel["h3"].unique())
    n = len(cells)
    if n <= max_zones:
        # Fewer cells than zones: each cell is its own zone
        return pd.DataFrame({
            "h3": cells,
            "zone": range(n),
            "zone_name": [f"Zone {i + 1}" for i in range(n)],
        })

    # --- features: centroid lat/lon + log listing density ---
    centroids = np.array([h3.cell_to_latlng(c) for c in cells])  # (n, 2)
    density = (
        panel.groupby("h3")["n_listings"]
        .sum()
        .reindex(cells)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    log_density = np.log1p(density).reshape(-1, 1)

    features = np.hstack([centroids, log_density * density_weight])
    features = StandardScaler().fit_transform(features)

    # --- spatial connectivity from H3 neighbours ---
    cell_idx = {c: i for i, c in enumerate(cells)}
    connectivity = np.zeros((n, n), dtype=bool)
    for c in cells:
        i = cell_idx[c]
        for nb in h3.grid_disk(c, 1):
            j = cell_idx.get(nb)
            if j is not None and j != i:
                connectivity[i, j] = True
                connectivity[j, i] = True

    model = AgglomerativeClustering(
        n_clusters=max_zones,
        connectivity=connectivity,
        linkage="ward",
    )
    labels = model.fit_predict(features)

    return pd.DataFrame({
        "h3": cells,
        "zone": labels,
        "zone_name": [f"Zone {lab + 1}" for lab in labels],
    })


def zones_to_geojson(
    zone_df: pd.DataFrame,
    zone_values: dict[int, dict] | None = None,
) -> dict:
    """GeoJSON FeatureCollection of merged zone polygons.

    Each zone's geometry is the ``unary_union`` of its constituent H3 hex
    polygons, producing organic (non-hexagonal) shapes.

    Parameters
    ----------
    zone_df : DataFrame
        Output of :func:`build_zones` (columns ``h3, zone, zone_name``).
    zone_values : dict, optional
        ``{zone_int: {prop_name: value, ...}}`` merged into each feature's
        properties (for choropleth colouring / tooltips).
    """
    features = []
    for zone_id, grp in zone_df.groupby("zone"):
        hexes = [_hex_polygon(c) for c in grp["h3"]]
        merged = unary_union(hexes)
        # unary_union may produce MultiPolygon for disconnected zones
        if merged.geom_type == "Polygon":
            coords = [list(merged.exterior.coords)]
        elif merged.geom_type == "MultiPolygon":
            coords = [list(p.exterior.coords) for p in merged.geoms]
        else:
            continue

        # GeoJSON coordinates: list of rings, each ring is list of [lon, lat]
        geom_type = "Polygon" if merged.geom_type == "Polygon" else "MultiPolygon"
        if geom_type == "Polygon":
            geometry = {"type": "Polygon", "coordinates": [
                [[x, y] for x, y in coords[0]]
            ]}
        else:
            geometry = {"type": "MultiPolygon", "coordinates": [
                [[[x, y] for x, y in ring]] for ring in coords
            ]}

        props = {
            "zone": int(zone_id),
            "zone_name": grp["zone_name"].iloc[0],
            "n_cells": len(grp),
        }
        if zone_values and int(zone_id) in zone_values:
            props.update(zone_values[int(zone_id)])

        features.append({
            "type": "Feature",
            "id": int(zone_id),
            "geometry": geometry,
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}
