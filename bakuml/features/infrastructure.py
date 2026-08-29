"""Time-varying metro accessibility features.

Methodology
-----------
Metro proximity is the strongest infrastructure amenity in the Baku market
(the synthetic generator plants a premium near operational stations, and the
staggered purple-line expansion is the project's DiD natural experiment).
Unlike the static Master Plan geometry, metro accessibility *changes over
the panel*: a station only contributes to a cell's accessibility from the
month it opens. `metro_features` therefore takes an ``as_of_month`` and
counts exactly the stations operational at that month.

Leakage note: these features are time-varying but *deterministic ex ante* —
the station set at month t is defined by opening dates fixed in
`config.METRO_STATIONS`, not by anything derived from price data, so
attaching them to (cell, month=t) rows cannot leak future market outcomes.

Station eligibility at ``as_of_month``:

* a station with ``opened is not None`` counts iff ``opened <= as_of_month``;
* when ``include_planned=True``, a station with ``opened is None`` also
  counts from its ``simulated_open`` (iff ``simulated_open is not None and
  simulated_open <= as_of_month``). This switch powers the "transit"
  scenario (`config.SCENARIOS`) and lets the DiD/forecast modules flip the
  planned B-04 station on and off.

Month comparisons use zero-padded ``"YYYY-MM"`` strings, whose
lexicographic order equals chronological order.

Features per cell (computed at the H3 cell centroid, vectorised haversine):

* ``dist_metro_km`` — distance to the nearest eligible station
  (``inf`` if none is eligible yet);
* ``dist_treatment_km`` — distance to `config.DID_TREATMENT_STATION`
  (B-04), **always** computed regardless of opening status, because the DiD
  design needs the treatment geometry in every month;
* ``n_stations_2km`` — number of eligible stations within
  `STATION_COUNT_RADIUS_KM` (2 km), a network-density complement to the
  nearest-station distance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakuml import config
from bakuml.config import MetroStation
from bakuml.geo import haversine_km
from bakuml.spatial.grid import cell_centroids

#: Radius (km) for the station-count feature ``n_stations_2km``.
STATION_COUNT_RADIUS_KM = 2.0

#: Column names produced by `metro_features` (besides the ``h3`` key).
METRO_FEATURE_COLS: list[str] = [
    "dist_metro_km",
    "dist_treatment_km",
    "n_stations_2km",
]


def _active_stations(as_of_month: str, include_planned: bool) -> list[MetroStation]:
    """Stations operational at `as_of_month` under the eligibility rule above."""
    active = []
    for s in config.METRO_STATIONS:
        if s.opened is not None:
            if s.opened <= as_of_month:
                active.append(s)
        elif include_planned and s.simulated_open is not None:
            if s.simulated_open <= as_of_month:
                active.append(s)
    return active


def metro_features(
    cells: list[str],
    as_of_month: str,
    *,
    include_planned: bool = False,
) -> pd.DataFrame:
    """Metro accessibility of each H3 cell as of a given month.

    Duplicate input cells are collapsed to one row (first occurrence wins).
    Returns a DataFrame with columns ``["h3", *METRO_FEATURE_COLS]``.
    """
    as_of = str(pd.Period(as_of_month, freq="M"))  # normalise to "YYYY-MM"
    unique_cells = list(dict.fromkeys(cells))
    cents = cell_centroids(unique_cells)
    lat = cents["lat"].to_numpy()
    lon = cents["lon"].to_numpy()

    active = _active_stations(as_of, include_planned)
    if active:
        dists = np.stack(
            [haversine_km(lat, lon, s.lat, s.lon) for s in active]
        )  # (n_stations, n_cells)
        dist_metro = dists.min(axis=0)
        n_within = (dists <= STATION_COUNT_RADIUS_KM).sum(axis=0)
    else:
        dist_metro = np.full(len(unique_cells), np.inf)
        n_within = np.zeros(len(unique_cells), dtype=int)

    treatment = next(
        s for s in config.METRO_STATIONS if s.name == config.DID_TREATMENT_STATION
    )
    dist_treatment = haversine_km(lat, lon, treatment.lat, treatment.lon)

    return pd.DataFrame(
        {
            "h3": unique_cells,
            "dist_metro_km": dist_metro,
            "dist_treatment_km": dist_treatment,
            "n_stations_2km": np.asarray(n_within, dtype="int64"),
        }
    )
