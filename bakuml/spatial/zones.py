"""Market micro-zones: density-aware clustering of panel cells.

This is a *presentation layer*: whatever cells the analytical backbone uses
(H3 hexagons by default, or an adaptive KD-tree / market regions - see
:mod:`bakuml.spatial.tessellation`), this module groups them into visually
meaningful market micro-zones using agglomerative clustering with spatial
connectivity constraints. The panel, features and models are unaffected.

Clusters are formed using cell centroids + normalised listing density so that
high-activity urban cores get finer zones while sparse periphery cells are
grouped into larger zones. The resulting zone polygons (shapely
``unary_union`` of constituent cell polygons) give the app an organic,
non-hexagonal map appearance.

Geometry and contiguity come from the active tessellation rather than from
h3 directly, so zones work for every unit of analysis. Clustering runs
separately per connected component of the contiguity graph: the observed
grid is not one blob (Sumgait, Alat and the Absheron villages are islands),
and handing sklearn a disconnected connectivity matrix makes it silently
"complete" the graph, which merges geographically separate places into a
single "zone".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from bakuml import config
from bakuml.spatial import tessellation as tess_mod
from bakuml.spatial.tessellation import Tessellation


def _cell_polygon(cell: str, tess: Tessellation) -> ShapelyPolygon:
    """Shapely polygon for one cell (lon/lat order)."""
    ring = [(lng, lat) for lat, lng in tess.boundary(cell)]
    ring.append(ring[0])
    return ShapelyPolygon(ring)


def _stored_polygons(geometry: dict) -> dict[str, ShapelyPolygon]:
    """Cell id -> polygon, from a stored per-cell FeatureCollection."""
    out: dict[str, ShapelyPolygon] = {}
    for feat in geometry.get("features", []):
        cell = feat.get("id") or (feat.get("properties") or {}).get("h3")
        geom = feat.get("geometry") or {}
        if cell is None or geom.get("type") != "Polygon":
            continue
        ring = [(x, y) for x, y in geom["coordinates"][0]]
        if len(ring) >= 4:
            out[cell] = ShapelyPolygon(ring)
    return out


def _components(cells: list[str], tess: Tessellation) -> list[list[str]]:
    """Connected components of the cell contiguity graph, largest first."""
    remaining = set(cells)
    out: list[list[str]] = []
    while remaining:
        seed = min(remaining)
        remaining.discard(seed)
        comp, stack = [seed], [seed]
        while stack:
            node = stack.pop()
            for nb in tess.neighbours(node, 1):
                if nb in remaining:
                    remaining.discard(nb)
                    comp.append(nb)
                    stack.append(nb)
        out.append(sorted(comp))
    return sorted(out, key=len, reverse=True)


def build_zones(
    panel: pd.DataFrame,
    *,
    max_zones: int = 40,
    density_weight: float = 0.5,
    tess: Tessellation | None = None,
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
    tess = tess_mod.resolve(tess)
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
    centroids = np.array([tess.centroid(c) for c in cells])  # (n, 2)
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

    # --- cluster within each connected component ---
    # A zone must be one place. Clustering the whole set against a
    # disconnected connectivity matrix lets sklearn bridge the gaps, so
    # each component gets its own run and its own share of the zone budget.
    cell_idx = {c: i for i, c in enumerate(cells)}
    comps = _components(cells, tess)
    budget = _allocate(comps, max_zones)

    labels = np.empty(n, dtype=int)
    next_label = 0
    for comp in comps:
        idx = np.array([cell_idx[c] for c in comp], dtype=int)
        k = budget[id(comp)]
        if k >= len(comp):
            labels[idx] = np.arange(next_label, next_label + len(comp))
            next_label += len(comp)
            continue
        sub_idx = {c: i for i, c in enumerate(comp)}
        m = len(comp)
        connectivity = np.zeros((m, m), dtype=bool)
        for c in comp:
            i = sub_idx[c]
            for nb in tess.neighbours(c, 1):
                j = sub_idx.get(nb)
                if j is not None and j != i:
                    connectivity[i, j] = True
                    connectivity[j, i] = True
        model = AgglomerativeClustering(
            n_clusters=k, connectivity=connectivity, linkage="ward",
        )
        labels[idx] = model.fit_predict(features[idx]) + next_label
        next_label += k

    return pd.DataFrame({
        "h3": cells,
        "zone": labels,
        "zone_name": [f"Zone {lab + 1}" for lab in labels],
    })


def _allocate(components: list[list[str]], total: int) -> dict[int, int]:
    """Share `total` zones across components by size (>= 1 each).

    Largest-remainder apportionment, capped so no component is asked for
    more zones than it has cells.
    """
    n_cells = sum(len(c) for c in components)
    total = max(total, len(components))
    exact = {
        id(c): 1 + (total - len(components)) * len(c) / n_cells for c in components
    }
    floors = {k: int(np.floor(v)) for k, v in exact.items()}
    short = total - sum(floors.values())
    order = sorted(components, key=lambda c: -(exact[id(c)] - floors[id(c)]))
    for c in order[:max(0, short)]:
        floors[id(c)] += 1
    for c in components:
        floors[id(c)] = min(floors[id(c)], len(c))
    return floors


def zones_to_geojson(
    zone_df: pd.DataFrame,
    zone_values: dict[int, dict] | None = None,
    *,
    tess: Tessellation | None = None,
    geometry: dict | None = None,
) -> dict:
    """GeoJSON FeatureCollection of merged zone polygons.

    Each zone's geometry is the ``unary_union`` of its constituent cell
    polygons, producing organic (non-hexagonal) shapes.

    Parameters
    ----------
    zone_df : DataFrame
        Output of :func:`build_zones` (columns ``h3, zone, zone_name``).
    zone_values : dict, optional
        ``{zone_int: {prop_name: value, ...}}`` merged into each feature's
        properties (for choropleth colouring / tooltips).
    geometry : dict, optional
        A per-cell GeoJSON FeatureCollection (the run's ``cell_geometry``
        artifact). When given, cell polygons are read from it instead of
        from a tessellation - which is what lets the app draw a run whose
        (fitted) tessellation it cannot reconstruct.
    """
    stored = _stored_polygons(geometry) if geometry is not None else None
    if stored is None:
        tess = tess_mod.resolve(tess)
    features = []
    for zone_id, grp in zone_df.groupby("zone"):
        if stored is not None:
            polys = [stored[c] for c in grp["h3"] if c in stored]
            if not polys:
                continue
        else:
            polys = [_cell_polygon(c, tess) for c in grp["h3"]]
        merged = unary_union(polys)
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
