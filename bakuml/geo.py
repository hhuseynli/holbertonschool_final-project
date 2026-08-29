"""Small shared geodesy helpers (vectorised haversine distances)."""

from __future__ import annotations

import numpy as np

from bakuml.config import EARTH_RADIUS_KM


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or numpy arrays and
    broadcasts like numpy ufuncs.
    """
    lat1, lon1, lat2, lon2 = (np.asarray(x, dtype=float) for x in (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def min_distance_km(lat, lon, points):
    """Distance from (lat, lon) arrays to the nearest of `points`
    (iterable of (lat, lon)). Returns an array shaped like `lat`;
    infinity when `points` is empty.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    points = list(points)
    if not points:
        return np.full(lat.shape, np.inf)
    dists = np.stack([haversine_km(lat, lon, plat, plon) for plat, plon in points])
    return dists.min(axis=0)
