"""Central configuration for the Baku & Absheron spatiotemporal ML project.

Every module reads its constants from here so that spatial resolution,
geography, and study-design parameters stay consistent across the pipeline.

Coordinates below are approximate (rounded to ~100 m). They are good enough
for feature engineering at H3 resolution 8 (~0.73 km^2 cells); a production
run against live bina.az data would refresh them from OpenStreetMap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Spatial backbone
# ---------------------------------------------------------------------------

# H3 resolution 8 -> average hexagon area ~0.737 km^2 ("~0.73 km^2 cells").
H3_RESOLUTION = 8
#: Default unit of analysis. "h3" is the documented hexagonal baseline;
#: "kdtree" adapts cell size to listing density (equal sample per cell);
#: "market" grows contiguous, price-homogeneous regions.
#:
#: The default is empirical, not doctrinal: `scripts/maup_study.py` scores
#: every candidate under the project's own leakage-proof protocols, and the
#: adaptive grid wins on spatial transfer while retaining far more data and
#: giving cells the price history a 12-month rollout needs. It is also
#: *stable*: its skill barely moves across its tuning knob, where H3's
#: swings by 0.48 and changes sign. Set to "h3" to reproduce the baseline.
TESSELLATION = "kdtree"

# Coarser parent resolution used to form contiguous spatial CV blocks.
# Res 6 parents (~36 km^2) give ~19 contiguous blocks over the study area;
# res 5 collapses to so few parents that folds degenerate.
H3_BLOCK_RESOLUTION = 6

# Elite centre of the price gradient (Fountain Square area, Sabail).
CITY_CENTRE = (40.3703, 49.8403)

# Bounding box that contains effectively all Baku + Absheron + Sumgait
# listings (lat_min, lat_max, lon_min, lon_max).
BBOX = (40.28, 40.66, 49.55, 50.25)

EARTH_RADIUS_KM = 6371.0088

# Simplified Absheron + Baku + Sumgait land polygon (lon, lat vertices,
# clockwise). Traced from the OpenStreetMap coastline with a ~300 m
# buffer on the western/northern edges and a tighter trace on the
# eastern/southern Absheron coast, where the peninsula narrows and
# axis-aligned KD-tree cells would otherwise extend visibly into the
# Caspian. Used by `spatial.grid.filter_land_cells` to drop cells
# whose centroids fall at sea, and by `cells_to_geojson` to clip cell
# polygons to the coastline.
LAND_POLYGON_LONLAT: list[tuple[float, float]] = [
    # North coast, west to east (Sumgait → Bilgah). The coastline
    # curves south between Sumgait and Bilgah; coordinates are ~300 m
    # seaward of the northernmost listings at each longitude.
    (49.35, 40.65), (49.50, 40.65), (49.55, 40.65), (49.60, 40.65),
    (49.65, 40.65), (49.72, 40.64), (49.75, 40.62), (49.80, 40.59),
    (49.85, 40.58), (49.90, 40.57), (49.95, 40.57), (50.00, 40.56),
    (50.05, 40.55), (50.10, 40.55), (50.15, 40.54), (50.18, 40.52),
    # Peninsula tip (Mardakan → Sea Breeze)
    (50.20, 40.51), (50.21, 40.50), (50.22, 40.49),
    # East coast, south along the narrowing peninsula
    (50.21, 40.48), (50.19, 40.47), (50.17, 40.46),
    (50.16, 40.45), (50.14, 40.44), (50.13, 40.43),
    (50.12, 40.42), (50.11, 40.41), (50.11, 40.40),
    # South coast (Hovsan → Sangachal) — stays east because the mainland
    # coast runs roughly north-south here before curving southwest.
    (50.11, 40.39), (50.10, 40.38), (50.08, 40.37),
    (50.07, 40.36), (50.07, 40.35), (50.06, 40.34),
    (50.06, 40.33), (50.04, 40.32), (50.00, 40.31),
    # Southwest coast
    (49.95, 40.30), (49.90, 40.29), (49.80, 40.28),
    (49.75, 40.27), (49.70, 40.26), (49.60, 40.25),
    (49.50, 40.25), (49.40, 40.25), (49.35, 40.25),
    # West edge back to start
    (49.35, 40.40), (49.35, 40.55), (49.35, 40.68),
]

# ---------------------------------------------------------------------------
# Master Plan 2040 - the city's official future, digitized
# ---------------------------------------------------------------------------

# Designated polycentric nodes: Alat, Mardakan, Sumgait + 5 local centres.
POLYCENTRIC_NODES: dict[str, tuple[float, float]] = {
    "alat": (39.9490, 49.4110),
    "mardakan": (40.4920, 50.1460),
    "sumgait": (40.5897, 49.6686),
    "khirdalan": (40.4489, 49.7550),
    "binagadi": (40.4653, 49.8286),
    "zigh_hovsan": (40.3744, 50.0400),
    "lokbatan": (40.3242, 49.7301),
    "bilgah": (40.5644, 49.9944),
}

# Official demolition / redevelopment zones, approximated as circles
# (centre lat, centre lon, radius km). A production run would replace these
# with the ARXKOM shapefiles.
REDEVELOPMENT_ZONES: dict[str, tuple[float, float, float]] = {
    "sovetski": (40.3830, 49.8280, 0.9),        # Sovetski demolition zone, Yasamal
    "white_city": (40.3720, 49.8760, 1.2),      # Black City -> White City redevelopment
    "hovsan_coastal": (40.3650, 50.0250, 1.5),  # Hovsan coastal development zone
}

# Master Plan 2040 demand anchors.
MASTERPLAN_POPULATION_2040 = 3_170_000     # residents by 2040
MASTERPLAN_NEW_HOUSING_M2_PER_YEAR = 4_900_000  # ~4.9M m^2 new housing / year

# ---------------------------------------------------------------------------
# Baku Metro - staggered expansion used as the natural experiment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetroStation:
    name: str
    line: str                     # "red" | "green" | "purple"
    lat: float
    lon: float
    opened: str | None            # "YYYY-MM" or None if not yet operational
    announced: str | None = None  # "YYYY-MM" construction/announcement date
    simulated_open: str | None = None  # used by the synthetic scenario only


METRO_STATIONS: list[MetroStation] = [
    # Red line
    MetroStation("Icherisheher", "red", 40.3660, 49.8317, "1967-11"),
    MetroStation("Sahil", "red", 40.3717, 49.8442, "1967-11"),
    MetroStation("28 May", "red", 40.3797, 49.8489, "1967-11"),
    MetroStation("Ganjlik", "red", 40.4005, 49.8517, "1967-11"),
    MetroStation("Nariman Narimanov", "red", 40.4028, 49.8702, "1967-11"),
    MetroStation("Bakmil", "red", 40.4189, 49.8564, "1979-01"),
    MetroStation("Ulduz", "red", 40.4153, 49.8925, "1970-11"),
    MetroStation("Koroghlu", "red", 40.4211, 49.9187, "1972-11"),
    MetroStation("Qara Qarayev", "red", 40.4175, 49.9333, "1972-11"),
    MetroStation("Neftchilar", "red", 40.4106, 49.9439, "1972-11"),
    MetroStation("Khalglar Dostlugu", "red", 40.3975, 49.9519, "1989-04"),
    MetroStation("Ahmedli", "red", 40.3853, 49.9539, "1989-04"),
    MetroStation("Hazi Aslanov", "red", 40.3722, 49.9542, "2002-12"),
    # Green line
    MetroStation("Nizami", "green", 40.3794, 49.8297, "1976-05"),
    MetroStation("Elmler Akademiyasi", "green", 40.3753, 49.8147, "1985-12"),
    MetroStation("Inshaatchilar", "green", 40.3908, 49.8033, "1985-12"),
    MetroStation("20 Yanvar", "green", 40.4042, 49.8078, "1985-12"),
    MetroStation("Memar Ajami", "green", 40.4108, 49.8144, "1985-12"),
    MetroStation("Nasimi", "green", 40.4242, 49.8250, "2008-10"),
    MetroStation("Azadlig Prospekti", "green", 40.4258, 49.8428, "2009-12"),
    MetroStation("Darnagul", "green", 40.4253, 49.8619, "2011-06"),
    MetroStation("Jafar Jabbarli", "green", 40.3794, 49.8480, "1993-12"),
    MetroStation("Khojasan", "green", 40.4145, 49.7825, "2021-11"),
    # Purple line - the staggered expansion at the heart of the DiD design
    MetroStation("Avtovagzal", "purple", 40.4211, 49.7947, "2016-04"),
    MetroStation("Memar Ajami-2", "purple", 40.4108, 49.8155, "2016-04"),
    MetroStation("8 Noyabr", "purple", 40.4000, 49.8200, "2021-05", announced="2019-01"),
    # B-04: under construction. `simulated_open` is a SIMULATION ASSUMPTION
    # used only by the synthetic data generator / scenario engine so the
    # Spatial DiD module can be demonstrated and unit-tested end to end.
    MetroStation(
        "B-04", "purple", 40.3925, 49.8090,
        opened=None, announced="2022-06", simulated_open="2025-06",
    ),
]

# Spatial DiD study design (distances in km, straight-line haversine as a
# proxy for the 1 km walking radius).
DID_TREATMENT_STATION = "B-04"
DID_TREATMENT_RADIUS_KM = 1.0
# Listings between the treatment radius and the buffer are dropped (fuzzy
# treatment edge); controls live between buffer and control max radius.
DID_BUFFER_RADIUS_KM = 2.0
DID_CONTROL_MAX_RADIUS_KM = 6.0

# ---------------------------------------------------------------------------
# Study window / panel
# ---------------------------------------------------------------------------

# Monthly panel window (inclusive), "YYYY-MM".
PANEL_START = "2023-01"
PANEL_END = "2026-08"
# Forecast horizon in months (12-24 month forward forecasts).
FORECAST_HORIZONS = (12, 24)

# ---------------------------------------------------------------------------
# Market anchors (used by the synthetic generator and sanity checks)
# ---------------------------------------------------------------------------

PRICE_CENTRE_AZN_M2 = 3500.0     # elite centre: Nizami/28 May/Sahil ~3,500
PRICE_PERIPHERY_AZN_M2 = 1300.0  # periphery: Absheron/outer suburbs ~1,300
PRICE_FLOOR_AZN_M2 = 580.0       # bottom of the observed market
PRICE_CAP_AZN_M2 = 9500.0        # luxury Sea Breeze / Nardaran penthouses

# ---------------------------------------------------------------------------
# Administrative districts (approximate centroids) - used to attach a
# district label to listings; the model itself works on H3 cells.
# ---------------------------------------------------------------------------

DISTRICT_CENTRES: dict[str, tuple[float, float]] = {
    # Central Baku
    "28 May": (40.3795, 49.8490),
    "Sahil": (40.3690, 49.8440),
    "Səbail": (40.3610, 49.8370),
    "İçəri Şəhər": (40.3660, 49.8340),
    "Nəsimi": (40.3840, 49.8300),
    "Nizami": (40.3780, 49.8210),
    "Yasamal": (40.3820, 49.8120),
    "Yeni Yasamal": (40.3950, 49.8020),
    # Inner suburbs
    "Nəriman Nərimanov": (40.4030, 49.8700),
    "Nərimanov": (40.4080, 49.8650),
    "Gənclik": (40.4020, 49.8520),
    "Şah İsmayıl Xətai": (40.3830, 49.9490),
    "Xətai": (40.3900, 49.9350),
    "Elmlər Akademiyası": (40.3720, 49.8630),
    "Koroğlu": (40.4100, 49.8820),
    "İnşaatçılar": (40.4000, 49.8950),
    "Memar Əcəmi": (40.4090, 49.8120),
    "Bakmil": (40.4150, 49.8250),
    "Dərnəgül": (40.4200, 49.8150),
    # Outer suburbs
    "Həzi Aslanov": (40.3730, 49.9520),
    "Əhmədli": (40.3850, 49.9700),
    "Neftçilər": (40.3980, 49.9400),
    "Qara Qarayev": (40.4100, 49.9300),
    "20 Yanvar": (40.4050, 49.9100),
    "Azadlıq Prospekti": (40.4200, 49.9500),
    "Binəqədi": (40.4520, 49.8080),
    "Bakıxanov": (40.4350, 49.9300),
    "Sabunçu": (40.4430, 49.9470),
    "Suraxanı": (40.4120, 50.0050),
    "Hövsan": (40.4250, 50.0300),
    "Yeni Günəşli": (40.3990, 49.9650),
    "Köhnə Günəşli": (40.3970, 49.9550),
    # Periphery & Absheron
    "Badamdar": (40.3550, 49.8100),
    "Bayıl": (40.3500, 49.8250),
    "Şıxov": (40.3350, 49.8350),
    "Abşeron": (40.4489, 49.7550),
    "Masazır": (40.4700, 49.7400),
    "Biləcəri": (40.4600, 49.7900),
    "Qaradağ": (40.3242, 49.7301),
    "Lökbatan": (40.3800, 49.7350),
    "Qaraçuxur": (40.4300, 49.9800),
    "Zığ": (40.4400, 50.0100),
    "Xəzər": (40.4720, 50.1060),
    "Nardaran": (40.5580, 50.0100),
    # Premium
    "Sea Breeze": (40.4950, 50.1800),
    "Ağ şəhər": (40.3600, 49.8600),
    # Sumgait
    "Sumqayıt": (40.5897, 49.6686),
}

# ---------------------------------------------------------------------------
# Scenario engine (advanced scope)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """Multipliers applied to forward-looking features before forecasting."""

    name: str
    description: str
    # Scales the appreciation pull of polycentric nodes (1.0 = as observed).
    polycentric_pull: float = 1.0
    # Whether planned metro stations (B-04, ...) are treated as open in the
    # forecast horizon.
    transit_buildout: bool = False


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario(
        "baseline",
        "Current trends continue; no new transit beyond what is already open.",
    ),
    "polycentric": Scenario(
        "polycentric",
        "Master Plan 2040 polycentric shift succeeds; node gravity doubled.",
        polycentric_pull=2.0,
    ),
    "transit": Scenario(
        "transit",
        "Full transit build-out: planned purple-line stations come online.",
        polycentric_pull=1.5,
        transit_buildout=True,
    ),
}

# ---------------------------------------------------------------------------
# Synthetic data generator - planted ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticTruth:
    """Effects deliberately planted in the synthetic listings so downstream
    modules (DiD, forecasting, SHAP) can be validated against known answers.
    """

    seed: int = 20240 + 3  # Group 3
    n_listings_per_month: int = 800
    # log-price decay: price = centre * exp(-decay * dist_km) floored at periphery
    centre_decay_per_km: float = 0.065
    # Static premium (log points) within 0.8 km of an operational metro station.
    metro_premium_log: float = 0.06
    metro_premium_radius_km: float = 0.8
    # DiD ground truth: extra log-premium within DID_TREATMENT_RADIUS_KM of the
    # treatment station after its (simulated) opening, ramping in over 3 months.
    did_effect_log: float = 0.08
    did_ramp_months: int = 3
    # Citywide monthly appreciation (log points / month).
    monthly_trend_log: float = 0.004
    # Extra monthly appreciation per (1 / (1 + dist_km)) of polycentric node
    # proximity, phased in over the second half of the panel.
    polycentric_trend_log: float = 0.006
    # New-construction premium (log points).
    new_building_premium_log: float = 0.25
    # Redevelopment-zone discount (log points; demolition uncertainty).
    redev_zone_discount_log: float = -0.05
    # Idiosyncratic noise (std of log price).
    noise_log_std: float = 0.18
    # Share of listings that are broker duplicates of another listing.
    duplicate_share: float = 0.06


SYNTHETIC_TRUTH = SyntheticTruth()

# ---------------------------------------------------------------------------
# Modelling defaults
# ---------------------------------------------------------------------------

TARGET_COL = "price_azn_m2_median"
MIN_LISTINGS_PER_CELL_MONTH = 3   # cells-months thinner than this are dropped
CONFORMAL_QUANTILES = (0.10, 0.50, 0.90)
WALK_FORWARD_MIN_TRAIN_MONTHS = 18
WALK_FORWARD_TEST_MONTHS = 1
SPATIAL_CV_FOLDS = 5

_ALL_DIRS = (DATA_DIR, RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR)


def ensure_dirs() -> None:
    """Create the data/artifact directories if missing."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
