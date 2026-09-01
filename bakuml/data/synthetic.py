"""Synthetic Baku & Absheron listings with *planted* ground truth.

bina.az sits behind Cloudflare and the State Registry is closed, so the
repository ships a fully offline stand-in: a generator that draws listings
whose spatial and temporal structure mirrors the real market

  * an exponential price gradient from the elite centre (~3,200 AZN/m^2)
    to the periphery (~1,500 AZN/m^2),
  * a static premium near operational metro stations,
  * a *causal* premium that switches on when the B-04 purple-line station
    (simulated) opens - the Spatial DiD module must recover this number,
  * a citywide monthly trend plus late-panel appreciation that pulls toward
    the Master Plan 2040 polycentric nodes,
  * broker duplicates (re-posts of the same flat with jittered price, text
    and near-identical image hashes) for the dedup module to find.

Every planted parameter comes from `config.SyntheticTruth` and is written to
`synthetic_truth.json`, so tests can assert that the pipeline recovers what
was planted rather than merely "runs".
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bakuml import config
from bakuml.config import SYNTHETIC_TRUTH, SyntheticTruth
from bakuml.data.schema import LISTING_COLUMNS, month_range, validate_listings
from bakuml.geo import haversine_km, min_distance_km

# (lat, lon, lat_std, lon_std, weight) - where listings concentrate
_CLUSTERS = [
    (40.3780, 49.8450, 0.020, 0.026, 0.42),   # Baku core
    (40.4150, 49.8900, 0.030, 0.040, 0.20),   # inner suburbs / east
    (40.4489, 49.7550, 0.016, 0.020, 0.09),   # Khirdalan / Absheron
    (40.5897, 49.6686, 0.018, 0.022, 0.09),   # Sumgait
    (40.4000, 50.0100, 0.025, 0.035, 0.08),   # Surakhani / Hovsan
    (40.4050, 49.8100, 0.014, 0.016, 0.12),   # Yasamal / purple-line corridor
]

_ROOM_PROBS = {1: 0.18, 2: 0.34, 3: 0.30, 4: 0.13, 5: 0.05}

_TITLE_TMPL = "{rooms}-otaqlı mənzil, {area} m², {district}"
_DESC_TMPL = (
    "{rooms} otaqlı {btype} tikili mənzil, {area} m², {floor}/{bfloors} mərtəbə, "
    "{district} rayonu. Təmirli, sənədlər qaydasındadır. Metro yaxınlığında."
)
_DUP_DESC_TMPL = (
    "TƏCİLİ! {rooms} otaqlı {btype} tikili, {area} m², mərtəbə {floor}/{bfloors}, "
    "{district} rayonu. Təmirli, sənədlər qaydasındadır. Metro yaxınlığında."
)


def _months_between(a: str, b: str) -> int:
    """Whole months from a to b (positive when b is later)."""
    return (pd.Period(b, freq="M") - pd.Period(a, freq="M")).n


def _phash_hex(rng: np.random.Generator) -> str:
    return format(int(rng.integers(0, 2**63)), "016x")


def _flip_phash_bits(phash: str, rng: np.random.Generator, n_bits: int = 2) -> str:
    val = int(phash, 16)
    for bit in rng.choice(64, size=n_bits, replace=False):
        val ^= 1 << int(bit)
    return format(val, "016x")


def _nearest_district(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    names = list(config.DISTRICT_CENTRES)
    dists = np.stack(
        [haversine_km(lat, lon, *config.DISTRICT_CENTRES[n]) for n in names]
    )
    return np.asarray(names, dtype=object)[dists.argmin(axis=0)]


def generate_listings(
    truth: SyntheticTruth = SYNTHETIC_TRUTH,
    start: str = config.PANEL_START,
    end: str = config.PANEL_END,
) -> tuple[pd.DataFrame, dict]:
    """Generate the synthetic listings table.

    Returns
    -------
    listings : DataFrame matching `schema.LISTING_COLUMNS`
    truth_info : dict with the planted parameters, the duplicate map
        (dup listing_id -> original listing_id) and bookkeeping columns.
    """
    rng = np.random.default_rng(truth.seed)
    months = month_range(start, end)
    n_months = len(months)
    lat_min, lat_max, lon_min, lon_max = config.BBOX

    # ------------------------------------------------------------------ #
    # 1. Draw locations and static attributes for all base listings
    # ------------------------------------------------------------------ #
    n_total = truth.n_listings_per_month * n_months
    weights = np.array([c[4] for c in _CLUSTERS])
    weights = weights / weights.sum()
    cluster_ix = rng.choice(len(_CLUSTERS), size=n_total, p=weights)
    lat = np.empty(n_total)
    lon = np.empty(n_total)
    for i, (clat, clon, slat, slon, _) in enumerate(_CLUSTERS):
        mask = cluster_ix == i
        k = int(mask.sum())
        lat[mask] = rng.normal(clat, slat, size=k)
        lon[mask] = rng.normal(clon, slon, size=k)
    lat = np.clip(lat, lat_min, lat_max)
    lon = np.clip(lon, lon_min, lon_max)

    month_arr = np.repeat(np.arange(n_months), truth.n_listings_per_month)
    rng.shuffle(month_arr)
    month_str = np.asarray(months, dtype=object)[month_arr]

    rooms = rng.choice(
        list(_ROOM_PROBS), size=n_total, p=list(_ROOM_PROBS.values())
    ).astype(int)
    area = np.clip(rng.normal(38 + 24 * rooms, 9), 25, 350).round(1)
    building_floors = rng.choice(
        [5, 9, 12, 16, 20], size=n_total, p=[0.25, 0.30, 0.20, 0.15, 0.10]
    )
    floor = (rng.uniform(size=n_total) * building_floors).astype(int) + 1

    dist_centre = haversine_km(lat, lon, *config.CITY_CENTRE)
    # New construction is likelier away from the historic centre.
    p_new = np.clip(0.18 + 0.03 * dist_centre, 0.18, 0.65)
    is_new = rng.uniform(size=n_total) < p_new
    building_type = np.where(is_new, "new", "old")

    # ------------------------------------------------------------------ #
    # 2. Planted log-price surface
    # ------------------------------------------------------------------ #
    log_p = np.log(config.PRICE_CENTRE_AZN_M2) - truth.centre_decay_per_km * dist_centre
    log_p = np.maximum(log_p, np.log(config.PRICE_PERIPHERY_AZN_M2))

    # Static premium near stations operational at the listing month.
    treatment = next(
        s for s in config.METRO_STATIONS if s.name == config.DID_TREATMENT_STATION
    )
    dist_treatment = haversine_km(lat, lon, treatment.lat, treatment.lon)
    open_stations = [
        (s.lat, s.lon) for s in config.METRO_STATIONS if s.opened is not None
    ]
    dist_open_metro = min_distance_km(lat, lon, open_stations)
    log_p += truth.metro_premium_log * (
        dist_open_metro <= truth.metro_premium_radius_km
    )

    # DiD ground truth: premium near B-04 after simulated opening, with ramp.
    open_m = _months_between(start, treatment.simulated_open)
    rel = month_arr - open_m  # months since opening (negative = pre)
    ramp = np.clip((rel + 1) / truth.did_ramp_months, 0.0, 1.0)
    ramp[rel < 0] = 0.0
    treated = dist_treatment <= config.DID_TREATMENT_RADIUS_KM
    log_p += truth.did_effect_log * ramp * treated

    # Citywide trend + late-panel polycentric pull.
    log_p += truth.monthly_trend_log * month_arr
    node_prox = np.zeros(n_total)
    for nlat, nlon in config.POLYCENTRIC_NODES.values():
        node_prox = np.maximum(
            node_prox, 1.0 / (1.0 + haversine_km(lat, lon, nlat, nlon))
        )
    phase_start = n_months // 2
    log_p += truth.polycentric_trend_log * node_prox * np.maximum(
        0, month_arr - phase_start
    )

    # Structure effects + redevelopment discount.
    log_p += truth.new_building_premium_log * is_new
    log_p += -0.0012 * (area - 85.0)          # small flats trade richer per m^2
    log_p += -0.03 * (floor == 1)             # ground floor discount
    for zlat, zlon, zr in config.REDEVELOPMENT_ZONES.values():
        in_zone = haversine_km(lat, lon, zlat, zlon) <= zr
        log_p += truth.redev_zone_discount_log * in_zone

    log_p += rng.normal(0.0, truth.noise_log_std, size=n_total)

    price_m2 = np.clip(
        np.exp(log_p), config.PRICE_FLOOR_AZN_M2, config.PRICE_CAP_AZN_M2
    ).round(1)
    price = (price_m2 * area).round(0)

    district = _nearest_district(lat, lon)
    phash = np.array([_phash_hex(rng) for _ in range(n_total)], dtype=object)

    base = pd.DataFrame(
        {
            "listing_id": [f"SYN-{i:07d}" for i in range(n_total)],
            "source": "synthetic",
            "lat": lat.round(6),
            "lon": lon.round(6),
            "price_azn": price,
            "area_m2": area,
            "price_azn_m2": price_m2,
            "rooms": rooms,
            "floor": floor.astype(int),
            "building_floors": building_floors.astype(int),
            "building_type": building_type,
            "listed_month": month_str,
            "image_phash": phash,
            "photo_urls": "[]",
            "district": district,
        }
    )
    base["title"] = [
        _TITLE_TMPL.format(rooms=r, area=a, district=d)
        for r, a, d in zip(base.rooms, base.area_m2, base.district)
    ]
    base["description"] = [
        _DESC_TMPL.format(
            rooms=r, area=a, btype="yeni" if b == "new" else "köhnə",
            floor=f, bfloors=bf, district=d,
        )
        for r, a, b, f, bf, d in zip(
            base.rooms, base.area_m2, base.building_type,
            base.floor, base.building_floors, base.district,
        )
    ]

    # ------------------------------------------------------------------ #
    # 3. Broker duplicates - re-posts of the same flat
    # ------------------------------------------------------------------ #
    n_dup = int(round(truth.duplicate_share * n_total))
    dup_src = rng.choice(n_total, size=n_dup, replace=False)
    dups = base.iloc[dup_src].copy().reset_index(drop=True)
    dup_ids = [f"SYN-DUP-{i:06d}" for i in range(n_dup)]
    dups["listing_id"] = dup_ids
    dups["price_azn_m2"] = (
        dups["price_azn_m2"] * rng.uniform(0.98, 1.02, size=n_dup)
    ).round(1)
    dups["price_azn"] = (dups["price_azn_m2"] * dups["area_m2"]).round(0)
    dups["lat"] = (dups["lat"] + rng.normal(0, 2e-4, size=n_dup)).round(6)
    dups["lon"] = (dups["lon"] + rng.normal(0, 2e-4, size=n_dup)).round(6)
    dups["image_phash"] = [
        _flip_phash_bits(h, rng) for h in dups["image_phash"]
    ]
    dups["description"] = [
        _DUP_DESC_TMPL.format(
            rooms=r, area=a, btype="yeni" if b == "new" else "köhnə",
            floor=f, bfloors=bf, district=d,
        )
        for r, a, b, f, bf, d in zip(
            dups.rooms, dups.area_m2, dups.building_type,
            dups.floor, dups.building_floors, dups.district,
        )
    ]

    listings = pd.concat([base, dups], ignore_index=True)
    listings = listings.sample(frac=1.0, random_state=truth.seed).reset_index(drop=True)
    listings = listings[list(LISTING_COLUMNS)]
    validate_listings(listings)

    truth_info = {
        "planted": dataclasses.asdict(truth),
        "panel_start": start,
        "panel_end": end,
        "treatment_station": treatment.name,
        "treatment_open_month": treatment.simulated_open,
        "duplicate_map": dict(
            zip(dup_ids, base.iloc[dup_src]["listing_id"].tolist())
        ),
        "n_base_listings": int(n_total),
        "n_duplicates": int(n_dup),
    }
    return listings, truth_info


def save_synthetic(
    outdir: Path | None = None,
    truth: SyntheticTruth = SYNTHETIC_TRUTH,
) -> tuple[Path, Path]:
    """Generate and persist the synthetic dataset.

    Writes `listings.parquet` and `synthetic_truth.json` into `outdir`
    (default: config.RAW_DIR). Returns both paths.
    """
    outdir = Path(outdir) if outdir is not None else config.RAW_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    listings, truth_info = generate_listings(truth=truth)
    listings_path = outdir / "listings.parquet"
    truth_path = outdir / "synthetic_truth.json"
    listings.to_parquet(listings_path, index=False)
    truth_path.write_text(json.dumps(truth_info, indent=2))
    return listings_path, truth_path
