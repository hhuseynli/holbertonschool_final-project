"""Upgrade listing addresses to real coordinates via OpenStreetMap Nominatim.

Why this module exists
----------------------
bina.az publishes no per-listing coordinates: the only points on an item
page are district centroids and the agency's office. Without real
coordinates every listing in a rayon collapses onto one point, and the
project's whole spatial layer - tessellation, spatial lags, the metro DiD,
spatially blocked CV - becomes meaningless on real data.

Each listing does carry a street address ("Məhəmməd Xiyabani küç., 196.")
plus a district and a named micro-location. Nominatim resolves a useful
share of those to street-level points.

Nominatim's usage policy is respected literally
-----------------------------------------------
* **at most one request per second** (``MIN_INTERVAL_S``), serialised;
* an honest ``User-Agent`` identifying the project;
* **results cached on disk**, so an address is never looked up twice;
* a hard cap per run (``max_lookups``), because Nominatim asks bulk users
  to run their own instance rather than hammer the public one.

Every geocoded row records ``geo_precision`` and ``geo_confidence`` so a
street-level match is never silently mixed with a district fallback.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import pandas as pd
import requests

from bakuml import config
from bakuml.data.sources.base import USER_AGENT

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
MIN_INTERVAL_S = 1.1  # policy is 1/s; leave headroom
CACHE_PATH = config.DATA_DIR / "cache" / "geocode.json"

#: Nominatim place types we accept as street-level. A match on a whole
#: suburb or city is not better than the district centroid we already have.
_STREET_CLASSES = {"building", "place", "highway", "amenity", "shop", "office", "tourism"}

_last_call = 0.0


@dataclass
class GeocodeResult:
    listings: pd.DataFrame
    attempted: int = 0
    matched: int = 0
    from_cache: int = 0
    failed: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.matched / self.attempted if self.attempted else 0.0

    def report(self) -> str:
        return (
            f"geocoding: {self.matched}/{self.attempted} addresses resolved "
            f"to street level ({100 * self.match_rate:.1f}%), "
            f"{self.from_cache} from cache, {self.failed} unresolved"
        )


def _load_cache() -> dict:
    if CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False))


def _strip_type_suffix(name: str) -> str:
    """'20 Yanvar m.' -> '20 Yanvar'; 'Yasamal r.' -> 'Yasamal'."""
    name = name.strip()
    for suffix in (" m.", " q.", " r.", " ş.", " k."):
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def query_variants(row: pd.Series) -> list[str]:
    """Nominatim queries for one listing, most precise first.

    Measured against the real feed, a full bina.az address with a house
    number ("Məhəmməd Xiyabani küç., 196.") usually does *not* match, while
    the bare street name does, and the named micro-location ("20 Yanvar")
    almost always does. So rather than one query, try a cascade and stop at
    the first street-level hit - which is the difference between a ~10% and
    a usable match rate.
    """
    address = str(row.get("address") or "").strip().rstrip(".")
    district = _strip_type_suffix(str(row.get("district") or ""))
    loc = _strip_type_suffix(str(row.get("location_name") or ""))
    tail = [p for p in (district, "Bakı", "Azərbaycan") if p]

    # Street name without the house number: drop trailing digits/punctuation.
    street = re.sub(r"[,\s]*\d+[a-zA-Z]?\.?$", "", address).strip().rstrip(",")

    out: list[str] = []
    for head in (address, street, loc):
        if not head:
            continue
        q = ", ".join([head] + [t for t in tail if t not in head])
        if q not in out:
            out.append(q)
    return out


def query_string(row: pd.Series) -> str:
    """The most precise query for a listing (first cascade variant)."""
    variants = query_variants(row)
    return variants[0] if variants else ""


def _lookup(query: str, *, timeout: float = 20.0) -> dict | None:
    """One rate-limited Nominatim call."""
    global _last_call
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "az",
                "addressdetails": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
    except requests.RequestException:
        _last_call = time.monotonic()
        return None
    _last_call = time.monotonic()
    if resp.status_code != 200:
        return None
    try:
        hits = resp.json()
    except ValueError:
        return None
    return hits[0] if hits else None


def geocode_listings(
    listings: pd.DataFrame,
    *,
    max_lookups: int = 400,
    require_bbox: bool = True,
    progress_every: int = 25,
) -> GeocodeResult:
    """Attach street-level coordinates where Nominatim can resolve them.

    Distinct addresses are looked up once each (many adverts share a
    building), so the request count is far below the row count. Rows that
    resolve get ``lat``/``lon`` replaced and ``geo_precision="street"``;
    rows that do not keep the district centroid they arrived with.
    """
    out = listings.copy()
    if out.empty:
        return GeocodeResult(listings=out)
    for col in ("geo_precision", "geo_confidence", "geo_query"):
        if col not in out.columns:
            out[col] = "" if col != "geo_confidence" else 0.0

    cache = _load_cache()
    res = GeocodeResult(listings=out)
    lat_min, lat_max, lon_min, lon_max = config.BBOX

    variants = out.apply(query_variants, axis=1)
    out["geo_query"] = variants.map(lambda v: v[0] if v else "")

    # One cascade per distinct address, not per advert: many adverts share a
    # building, and Nominatim's policy is about request volume.
    keys = variants.map(tuple)
    unique_cascades = list(dict.fromkeys(keys.tolist()))
    budget = max_lookups

    def accept(hit: dict | None) -> tuple[float, float, str] | None:
        if not hit:
            return None
        try:
            la, lo = float(hit["lat"]), float(hit["lon"])
        except (KeyError, TypeError, ValueError):
            return None
        if require_bbox and not (
            lat_min <= la <= lat_max and lon_min <= lo <= lon_max
        ):
            return None  # a match outside Absheron is a mis-parse
        if (
            hit.get("category") not in _STREET_CLASSES
            and hit.get("type") not in _STREET_CLASSES
        ):
            return None  # only a suburb/city: no better than the centroid
        return (la, lo, str(hit.get("type") or hit.get("category") or ""))

    resolved: dict[tuple, tuple[float, float, str] | None] = {}
    try:
        for i, cascade in enumerate(unique_cascades, 1):
            res.attempted += 1
            found = None
            for q in cascade:
                if q in cache:
                    res.from_cache += 1
                    hit = cache[q]
                else:
                    if budget <= 0:
                        break
                    budget -= 1
                    hit = _lookup(q)
                    cache[q] = hit  # cache misses too: never retry a dead one
                found = accept(hit)
                if found:
                    break
            resolved[cascade] = found
            res.matched += int(found is not None)
            res.failed += int(found is None)
            if progress_every and i % progress_every == 0:
                print(
                    f"  geocode {i}/{len(unique_cascades)} addresses, "
                    f"{res.matched} matched",
                    flush=True,
                )
            if budget <= 0:
                break
    finally:
        _save_cache(cache)

    hit_mask = keys.map(lambda k: resolved.get(k) is not None)
    queries = keys
    if hit_mask.any():
        # Guarded: assigning an empty string-backed series into a float
        # column raises under pandas 3's stricter dtype rules.
        out.loc[hit_mask, "lat"] = (
            keys[hit_mask].map(lambda k: resolved[k][0]).astype(float)
        )
        out.loc[hit_mask, "lon"] = (
            keys[hit_mask].map(lambda k: resolved[k][1]).astype(float)
        )
        out.loc[hit_mask, "geo_precision"] = "street"
        out.loc[hit_mask, "geo_confidence"] = 1.0
    miss = ~hit_mask
    if miss.any():
        out.loc[miss, "geo_precision"] = (
            out.loc[miss, "geo_precision"].replace("", "district_centroid")
        )
    res.listings = out
    res.notes.append(
        f"{len(unique_cascades)} distinct addresses for {len(out)} adverts"
    )
    if budget <= 0:
        res.notes.append(
            f"max_lookups={max_lookups} reached; remaining addresses kept at "
            "district precision"
        )
    return res
