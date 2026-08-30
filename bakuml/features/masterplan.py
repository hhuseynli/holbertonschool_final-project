"""Master Plan 2040 static geometry features.

Methodology
-----------
Baku's Master Plan 2040 commits the city to a *polycentric* structure: growth
is to be redirected from the saturated historic centre toward designated
secondary nodes (Alat, Mardakan, Sumgait and five local centres,
`config.POLYCENTRIC_NODES`), while several inner-city districts are slated
for demolition and redevelopment (`config.REDEVELOPMENT_ZONES`, approximated
as circles). The plan is therefore a *map of announced future demand* — this
module digitises it into per-cell features so the models can learn how the
market prices in that announced future.

All features here are purely static geometry: distances from the H3 cell
centroid to fixed points, and membership in fixed zones. Under the project's
leakage rule (DESIGN.md, Ground rules) static geometry is the one class of
features exempt from the "months <= t-1" requirement, because it is constant
over time and cannot encode information about future prices.

Features per cell (one row per unique cell, first-occurrence order kept):

* ``dist_centre_km`` — haversine distance to the elite centre
  (`config.CITY_CENTRE`, Fountain Square), the axis of the price gradient.
* ``dist_node_<name>_km`` — distance to each polycentric node individually,
  so tree models can localise node-specific appreciation.
* ``dist_nearest_node_km`` — distance to the closest node.
* ``node_gravity`` — max over nodes of ``1 / (1 + dist_km)``: a bounded
  (0, 1] accessibility score that the scenario engine can scale
  (`config.Scenario.polycentric_pull`).
* ``in_redev_zone`` — 1 if the centroid falls inside any redevelopment
  circle (demolition uncertainty trades at a discount), else 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakuml import config
from bakuml.geo import haversine_km
from bakuml.spatial.grid import cell_centroids

#: Column names produced by `masterplan_features` (besides the ``h3`` key),
#: in output order. `build.FEATURE_COLS` is assembled from this list.
MASTERPLAN_FEATURE_COLS: list[str] = [
    "dist_centre_km",
    *(f"dist_node_{name}_km" for name in config.POLYCENTRIC_NODES),
    "dist_nearest_node_km",
    "node_gravity",
    "in_redev_zone",
]


def masterplan_features(cells: list[str]) -> pd.DataFrame:
    """Static Master Plan 2040 features for each H3 cell.

    Distances are computed from the cell centroid (`h3.cell_to_latlng` via
    `spatial.grid.cell_centroids`) with the shared vectorised haversine.
    Duplicate input cells are collapsed to one row (first occurrence wins).

    Returns a DataFrame with columns ``["h3", *MASTERPLAN_FEATURE_COLS]``.
    """
    unique_cells = list(dict.fromkeys(cells))
    cents = cell_centroids(unique_cells)
    lat = cents["lat"].to_numpy()
    lon = cents["lon"].to_numpy()

    out = pd.DataFrame({"h3": unique_cells})
    out["dist_centre_km"] = haversine_km(lat, lon, *config.CITY_CENTRE)

    node_dists = []
    for name, (nlat, nlon) in config.POLYCENTRIC_NODES.items():
        d = haversine_km(lat, lon, nlat, nlon)
        out[f"dist_node_{name}_km"] = d
        node_dists.append(d)
    node_mat = np.stack(node_dists)  # (n_nodes, n_cells)
    out["dist_nearest_node_km"] = node_mat.min(axis=0)
    out["node_gravity"] = (1.0 / (1.0 + node_mat)).max(axis=0)

    in_zone = np.zeros(len(unique_cells), dtype=bool)
    for zlat, zlon, zradius_km in config.REDEVELOPMENT_ZONES.values():
        in_zone |= haversine_km(lat, lon, zlat, zlon) <= zradius_km
    out["in_redev_zone"] = in_zone.astype("int64")

    return out[["h3", *MASTERPLAN_FEATURE_COLS]]
