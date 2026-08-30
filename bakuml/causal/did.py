"""Spatial Difference-in-Differences around the B-04 purple-line station.

Natural experiment
------------------
The Baku Metro purple line is being extended in stages. Station B-04
(``config.DID_TREATMENT_STATION``) opens mid-panel (its real ``opened`` is
still ``None``, so the study uses the simulation assumption
``simulated_open = 2025-06`` planted in the synthetic data). Flats within
walking distance of the new station receive a transit-access shock; flats a
few kilometres away do not. Comparing the *change* in log prices near the
station with the *change* farther away identifies the causal effect of the
opening under the parallel-trends assumption -- classic
difference-in-differences (DiD), made spatial by defining treatment through
distance rings.

Study design (all radii from ``bakuml.config``)
-----------------------------------------------
* **Treated**: listings within ``DID_TREATMENT_RADIUS_KM`` (1 km, a proxy for
  the walking catchment) of the station.
* **Fuzzy ring** (1, 2] km: *dropped*. The treatment boundary is not sharp --
  walking access decays gradually and spillovers are likely just outside the
  catchment -- so keeping these listings would attenuate the contrast in both
  directions.
* **Controls** (2, 6] km: close enough to share neighbourhood-level shocks
  (same sub-market, same citywide trend), far enough to be unaffected by the
  station itself.

Identification
--------------
The estimating equation is

    log_price ~ treated:post + C(month) + C(h3 cell) + area_m2 + rooms
                + floor + is_new

* **Cell fixed effects** (H3 resolution ``config.H3_RESOLUTION``) absorb every
  time-constant spatial premium: the centre-distance gradient, the *static*
  premium around long-open metro stations, redevelopment-zone discounts.
  Only a premium that *switches on* at the opening month, *only* near B-04,
  can load onto ``treated:post``.
* **Month fixed effects** absorb the citywide trend and any common monthly
  shock, so the ``post`` main effect is redundant; the ``treated`` main
  effect is (essentially) constant within cells and is absorbed by the cell
  fixed effects. Hence only the interaction enters.
* **Hedonic covariates** (area, rooms, floor, new-building flag) soak up
  composition changes in what happens to be listed each month.

Fixed effects are implemented as **explicit dummy variables via the patsy
formula** (not within-demeaning): with ~44 months and a few hundred cells the
dummy design matrix is small, and dummies keep degrees of freedom and
standard errors exact without an iterative demeaning step. Standard errors
are **cluster-robust by H3 cell** (``cov_type="cluster"``), because listings
in the same hexagon share unobserved micro-location shocks.

Because the planted effect ramps in linearly over
``SYNTHETIC_TRUTH.did_ramp_months = 3`` months (1/3, 2/3, then the full
``did_effect_log = 0.08`` from the second post month on), the single
``treated:post`` coefficient is the ramp-*averaged* ATT over the post window
-- slightly below 0.08. The :func:`event_study` traces the dynamic path and
should show flat pre-coefficients (no anticipation, parallel trends) and the
3-month ramp up to ~0.08.

This module consumes **deduplicated listings** (the pipeline runs
``bakuml.data.dedup.dedupe`` first), never the cell-month panel: DiD wants
raw hedonic observations, not aggregates.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import h3
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from bakuml import config
from bakuml.geo import haversine_km

#: Hedonic covariates included in every specification.
COVARIATES = ("area_m2", "rooms", "floor", "is_new")

#: Event-study reference period (the month before opening); its coefficient
#: is normalised to zero.
BASE_PERIOD = -1

_REQUIRED_INPUT_COLUMNS = (
    "lat", "lon", "price_azn_m2", "listed_month",
    "area_m2", "rooms", "floor", "building_type",
)


@dataclass
class DiDResult:
    """Outcome of the spatial DiD regression (log-price scale + % effect)."""

    att_log: float            # treated:post coefficient (log points)
    att_pct: float            # 100 * (exp(att_log) - 1)
    se: float                 # cluster-robust standard error (by h3 cell)
    p_value: float
    ci_low: float             # 95% confidence interval, log points
    ci_high: float
    n_treated_listings: int
    n_control_listings: int
    open_month: str           # "YYYY-MM" the treatment switches on
    spec: str                 # human-readable model specification


def _lookup_station(name: str) -> config.MetroStation:
    """Return the metro station named `name` from config.METRO_STATIONS."""
    for station in config.METRO_STATIONS:
        if station.name == name:
            return station
    raise ValueError(f"unknown metro station: {name!r}")


def _station_open_month(station: config.MetroStation) -> str:
    """Opening month used for the experiment: real `opened` when available,
    otherwise the planted `simulated_open` scenario assumption."""
    open_month = station.opened if station.opened is not None else station.simulated_open
    if open_month is None:
        raise ValueError(
            f"station {station.name!r} has neither `opened` nor `simulated_open`"
        )
    return open_month


def _months_since(months: pd.Series, origin: str) -> np.ndarray:
    """Vectorised whole-month difference `months - origin` for 'YYYY-MM' strings."""
    per = pd.PeriodIndex(months, freq="M")
    origin_p = pd.Period(origin, freq="M")
    return ((per.year - origin_p.year) * 12 + (per.month - origin_p.month)).to_numpy()


def prepare_did_frame(
    listings: pd.DataFrame,
    *,
    treatment: str = config.DID_TREATMENT_STATION,
    radius_km: float = config.DID_TREATMENT_RADIUS_KM,
    buffer_km: float = config.DID_BUFFER_RADIUS_KM,
    control_max_km: float = config.DID_CONTROL_MAX_RADIUS_KM,
) -> pd.DataFrame:
    """Build the DiD estimation sample from (deduplicated) listings.

    Adds ``dist_treatment_km``, ``treated`` (distance <= `radius_km`),
    ``post`` (listed at/after the opening month), ``rel_month`` (whole months
    since opening, negative pre-opening), ``log_price`` (natural log of
    price_azn_m2), ``is_new`` (0/1) and the H3 cell id ``h3``.

    Keeps treated listings and controls in ``(buffer_km, control_max_km]``;
    the fuzzy ring ``(radius_km, buffer_km]`` and everything beyond
    `control_max_km` are dropped (see module docstring).
    """
    missing = [c for c in _REQUIRED_INPUT_COLUMNS if c not in listings.columns]
    if missing:
        raise ValueError(f"listings missing columns required for DiD: {missing}")

    station = _lookup_station(treatment)
    open_month = _station_open_month(station)

    dist = haversine_km(
        listings["lat"].to_numpy(), listings["lon"].to_numpy(),
        station.lat, station.lon,
    )
    is_treated = dist <= radius_km
    is_control = (dist > buffer_km) & (dist <= control_max_km)

    frame = listings.loc[is_treated | is_control].copy()
    frame["dist_treatment_km"] = dist[is_treated | is_control]
    frame["treated"] = frame["dist_treatment_km"] <= radius_km

    rel = _months_since(frame["listed_month"], open_month)
    frame["rel_month"] = rel.astype("int64")
    frame["post"] = frame["rel_month"] >= 0

    frame["log_price"] = np.log(frame["price_azn_m2"].to_numpy())
    frame["is_new"] = (frame["building_type"] == "new").astype("int64")
    frame["h3"] = [
        h3.latlng_to_cell(la, lo, config.H3_RESOLUTION)
        for la, lo in zip(frame["lat"], frame["lon"])
    ]

    frame = frame.reset_index(drop=True)
    frame.attrs["open_month"] = open_month
    return frame


def _recover_open_month(frame: pd.DataFrame) -> str:
    """Opening month implied by the frame itself (listed_month - rel_month).

    Derived from the columns rather than `frame.attrs` so that slicing /
    serialisation of the frame cannot lose it.
    """
    row = frame.iloc[0]
    return str(pd.Period(row["listed_month"], freq="M") - int(row["rel_month"]))


def _fit_clustered_ols(formula: str, frame: pd.DataFrame):
    """OLS with cluster-robust (by H3 cell) covariance."""
    model = smf.ols(formula, data=frame)
    return model.fit(cov_type="cluster", cov_kwds={"groups": frame["h3"]})


def run_spatial_did(frame: pd.DataFrame) -> DiDResult:
    """Estimate the average treatment effect on the treated (ATT).

    OLS of ``log_price`` on the ``treated x post`` interaction with month and
    H3-cell fixed effects (patsy dummies -- see module docstring for why not
    within-demeaning) plus hedonic covariates; SEs clustered by cell. The
    main effects of `treated` and `post` are intentionally absent: they are
    absorbed by the cell and month fixed effects respectively.
    """
    if frame.empty or frame["treated"].nunique() < 2:
        raise ValueError("DiD frame needs both treated and control listings")

    est = frame.copy()
    est["treated_post"] = (est["treated"] & est["post"]).astype("int64")

    formula = (
        "log_price ~ treated_post + C(listed_month) + C(h3) + "
        + " + ".join(COVARIATES)
    )
    res = _fit_clustered_ols(formula, est)

    att_log = float(res.params["treated_post"])
    ci_low, ci_high = (float(x) for x in res.conf_int().loc["treated_post"])
    spec = (
        "OLS: log_price ~ treated:post + C(month) + C(h3) + "
        + " + ".join(COVARIATES)
        + " | month & cell fixed effects as dummies, "
        "cluster-robust SEs by h3 cell"
    )
    return DiDResult(
        att_log=att_log,
        att_pct=float(100.0 * (np.exp(att_log) - 1.0)),
        se=float(res.bse["treated_post"]),
        p_value=float(res.pvalues["treated_post"]),
        ci_low=ci_low,
        ci_high=ci_high,
        n_treated_listings=int(est["treated"].sum()),
        n_control_listings=int((~est["treated"]).sum()),
        open_month=_recover_open_month(frame),
        spec=spec,
    )


def _event_term(rel_month: int) -> str:
    """Patsy-safe dummy name for one event time, e.g. -3 -> 'ev_m03'."""
    sign = "m" if rel_month < 0 else "p"
    return f"ev_{sign}{abs(rel_month):02d}"


def event_study(
    frame: pd.DataFrame,
    window: tuple[int, int] = (-12, 14),
) -> pd.DataFrame:
    """Dynamic DiD: one ``treated x 1[rel_month = k]`` coefficient per event
    time k in `window`, normalised to the month before opening
    (``rel_month = -1``, coefficient fixed at 0).

    Window handling: listings with `rel_month` **outside** the window are
    **dropped** from the estimation sample (rather than binned into the
    window edges). Dropping keeps every reported coefficient a clean
    single-month contrast against the base period; the discarded far
    pre-period months carry no treatment dynamics by construction.

    Same controls as :func:`run_spatial_did` (month FE, cell FE, hedonics),
    SEs clustered by cell; 95% confidence intervals.

    Returns a DataFrame ``[rel_month, coef, se, ci_low, ci_high, n]`` sorted
    by `rel_month`, where `n` counts treated listings at that event time.
    The base period appears as a row with coef/se/ci of 0 so plots can show
    the normalisation point; event times with no treated listings are omitted.
    """
    lo, hi = int(window[0]), int(window[1])
    if not lo <= BASE_PERIOD <= hi:
        raise ValueError(f"window {window} must contain base period {BASE_PERIOD}")

    est = frame.loc[frame["rel_month"].between(lo, hi)].copy()
    treated_n = (
        est.loc[est["treated"]].groupby("rel_month").size()
    )
    if treated_n.get(BASE_PERIOD, 0) == 0:
        raise ValueError("no treated listings in the base period rel_month=-1")

    terms = []
    for k in range(lo, hi + 1):
        if k == BASE_PERIOD or treated_n.get(k, 0) == 0:
            continue
        name = _event_term(k)
        est[name] = (est["treated"] & (est["rel_month"] == k)).astype("int64")
        terms.append((k, name))

    formula = (
        "log_price ~ "
        + " + ".join(name for _, name in terms)
        + " + C(listed_month) + C(h3) + "
        + " + ".join(COVARIATES)
    )
    res = _fit_clustered_ols(formula, est)
    conf = res.conf_int()

    rows = [
        {
            "rel_month": BASE_PERIOD, "coef": 0.0, "se": 0.0,
            "ci_low": 0.0, "ci_high": 0.0,
            "n": int(treated_n[BASE_PERIOD]),
        }
    ]
    for k, name in terms:
        rows.append(
            {
                "rel_month": k,
                "coef": float(res.params[name]),
                "se": float(res.bse[name]),
                "ci_low": float(conf.loc[name, 0]),
                "ci_high": float(conf.loc[name, 1]),
                "n": int(treated_n[k]),
            }
        )
    events = pd.DataFrame(rows).sort_values("rel_month").reset_index(drop=True)
    return events.astype(
        {"rel_month": "int64", "coef": "float64", "se": "float64",
         "ci_low": "float64", "ci_high": "float64", "n": "int64"}
    )


def did_artifact(result: DiDResult, events: pd.DataFrame) -> dict:
    """JSON-safe dict for ``schema.ARTIFACT_FILES["did"]``.

    Flattens the :class:`DiDResult` and attaches the event study as a list of
    per-event-time records; every value is a plain Python float/int/str so
    the dict round-trips through ``json.dumps``.
    """
    artifact = dataclasses.asdict(result)
    # Defensive coercion: a DiDResult built elsewhere may carry numpy scalars,
    # and np.int64 is not JSON-serialisable.
    for key, value in artifact.items():
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            artifact[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            artifact[key] = float(value)
    artifact["event_study"] = [
        {
            "rel_month": int(row.rel_month),
            "coef": float(row.coef),
            "se": float(row.se),
            "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high),
            "n": int(row.n),
        }
        for row in events.itertuples(index=False)
    ]
    return artifact
