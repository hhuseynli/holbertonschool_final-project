"""bina.az adapter - the real listing feed.

How the site actually works (verified 2026-08)
---------------------------------------------
bina.az is a Next.js + Apollo application. The listing *grid* is fetched
client-side, so scraping the catalogue HTML yields nothing, and its CSS
classes are build-hashed (``sc-eb5192be-3 dJBjvE``) and so change on every
deploy - selectors written against them are worthless.

Two stable, machine-intended sources are used instead:

1. **The item sitemap** (advertised in ``robots.txt``) enumerates item URLs.
2. **Each item page** embeds two structured payloads:
   * a ``schema.org/Product`` JSON-LD block - price, currency, rooms,
     floorSize, street address, locality, images;
   * the inline Apollo cache - an ``"Item:<id>"`` object with rooms, floor,
     floors, ``updatedAt``, description, ``contactTypeName``
     ("sahibi" = owner, "vasitəçi (agent)" = broker) and breadcrumbs that
     say whether the flat is ``yeni tikili`` (new) or ``köhnə tikili`` (old).

Both are contracts the site maintains for machines, so this is far more
robust than HTML scraping - and it is why no Cloudflare bypass is needed.

The coordinate problem
----------------------
**bina.az does not publish per-listing coordinates.** The only latitudes and
longitudes on an item page belong to (a) the *district* (``Location``, e.g.
Yasamal r. at 40.379962, 49.808995) and (b) the listing agency's office.
What every listing does carry is a free-text street address and a named
micro-location - usually a metro station or settlement ("20 Yanvar m.").

So ``lat``/``lon`` here are **district centroids**, and every row records
``geo_precision`` so no downstream consumer can mistake them for a
measured point. A grid-based unit of analysis is not meaningful at this
precision - every listing in a district lands on one coordinate - which is
why :mod:`bakuml.data.sources.geocode` exists to upgrade addresses to real
points where the licence permits.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import pandas as pd

from bakuml.data.schema import LISTING_COLUMNS
from bakuml.data.sources.base import HarvestResult, polite_get

SITEMAP_INDEX = "https://bina.azstatic.com/uploads/sitemaps/sitemap_items.xml"
ITEM_URL_RE = re.compile(r"https://bina\.az/items/(\d+)")

_LD_JSON_RE = re.compile(
    r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.S
)
_ITEM_OBJ_RE = re.compile(r'"Item:(\d+)":\{')
#: District entries in the Apollo cache, which are the only objects on the
#: page carrying coordinates (the item itself has none).
_LOC_RE = re.compile(
    r'"Location:(\d+)":\{"__typename":"Location","id":"\d+","name":"([^"]+)",'
    r'"latitude":([\d.]+),"longitude":([\d.]+)'
)
#: The listing's own geography: nearest district / metro / settlement, most
#: specific last, each referencing a Location id.
_NEAREST_RE = re.compile(r'"nearestLocations":\[(.*?)\]', re.S)
_NEAREST_ITEM_RE = re.compile(r'"fullName":"([^"]+)","id":"(\d+)"')

#: Sale categories only: rentals price on a different basis entirely.
_SALE_BREADCRUMB = "Alqı-satqı"


def iter_item_urls(limit: int | None = None, *, shards: int = 1) -> list[str]:
    """Item URLs from the advertised sitemap (newest shards first)."""
    index, _ = polite_get(SITEMAP_INDEX)
    if index is None:
        return []
    shard_urls = re.findall(r"<loc>([^<]+)</loc>", index)[:shards]
    out: list[str] = []
    for shard in shard_urls:
        body, _ = polite_get(shard)
        if body is None:
            continue
        out.extend(re.findall(r"<loc>(https://bina\.az/items/\d+)</loc>", body))
        if limit is not None and len(out) >= limit:
            break
    return out[:limit] if limit is not None else out


def _extract_json_object(text: str, start: int) -> dict | None:
    """Read one balanced JSON object starting at `start` (a '{')."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _district_centroids(html: str) -> dict[str, tuple[str, float, float]]:
    """Location id -> (name, lat, lon) for every district on the page."""
    return {
        loc_id: (name, float(lat), float(lon))
        for loc_id, name, lat, lon in _LOC_RE.findall(html)
    }


def _nearest_locations(html: str) -> list[tuple[str, str]]:
    """The listing's own (fullName, id) locations, broadest first.

    bina.az lists a district ("Yasamal r."), usually a metro area
    ("20 Yanvar m.") and sometimes a settlement ("Yeni Yasamal q."). Only
    the district resolves to coordinates, but the finer names are the most
    useful geography the platform publishes, so they are kept as labels.
    """
    block = _NEAREST_RE.search(html)
    if not block:
        return []
    return _NEAREST_ITEM_RE.findall(block.group(1))


def parse_item(html: str, url: str) -> dict | None:
    """Canonical listing row from one item page, or None if unusable.

    Returns the columns of :data:`bakuml.data.schema.LISTING_COLUMNS` plus
    real-data extras (``address``, ``location_name``, ``district``,
    ``geo_precision``, ``contact_type``, ``image_url``, ``url``).
    """
    m = ITEM_URL_RE.search(url)
    if not m:
        return None
    item_id = m.group(1)

    # ---- JSON-LD Product: price, area, rooms, address ----
    product: dict = {}
    for block in _LD_JSON_RE.findall(html):
        try:
            d = json.loads(block)
        except json.JSONDecodeError:
            continue
        if d.get("@type") == "Product":
            product = d
            break
    if not product:
        return None

    offer = product.get("offers") or {}
    if offer.get("priceCurrency") not in (None, "AZN"):
        return None
    offered = offer.get("itemOffered") or {}
    address = offered.get("address") or {}
    floor_size = offered.get("floorSize") or {}

    try:
        price = float(offer.get("price"))
        area = float(floor_size.get("value"))
    except (TypeError, ValueError):
        return None
    if price <= 0 or area <= 0:
        return None

    # ---- inline Apollo item: floor, dates, description, contact type ----
    item: dict = {}
    for om in _ITEM_OBJ_RE.finditer(html):
        if om.group(1) != item_id:
            continue
        obj = _extract_json_object(html, om.end() - 1)
        if obj and "rooms" in obj:
            item = obj
            break

    crumbs = [c.get("name", "") for c in (item.get("breadcrumbs") or [])]
    if crumbs and _SALE_BREADCRUMB not in crumbs:
        return None  # rental or other category

    rooms = item.get("rooms") or offered.get("numberOfRooms") or 0
    floor = item.get("floor") or 0
    floors = item.get("floors") or 0
    is_new = any("yeni tikili" in c.lower() for c in crumbs)
    if not is_new and not any("köhnə tikili" in c.lower() for c in crumbs):
        # Fall back to the headline, which always states one or the other.
        is_new = "yeni tikili" in (product.get("name") or "").lower()

    updated = item.get("updatedAt") or ""
    try:
        month = datetime.fromisoformat(updated).strftime("%Y-%m")
    except ValueError:
        return None

    # Geography: the listing's nearestLocations reference Location ids, and
    # those are the only page objects with coordinates.
    centroids = _district_centroids(html)
    nearest = _nearest_locations(html)
    addr_text = item.get("address") or address.get("streetAddress") or ""

    district, lat, lon = None, None, None
    for full_name, loc_id in nearest:
        if loc_id in centroids:
            district, lat, lon = centroids[loc_id][0], centroids[loc_id][1], centroids[loc_id][2]
            break
    if lat is None:
        # Fall back to matching a district name in the address text.
        for _loc_id, (name, clat, clon) in centroids.items():
            if name and name in addr_text:
                district, lat, lon = name, clat, clon
                break
    if lat is None:
        return None  # no resolvable geography at all

    # Most specific published label (metro area / settlement), for a
    # location-based unit of analysis that does not need coordinates.
    location_name = nearest[-1][0] if nearest else ""

    images = product.get("image") or []
    return {
        "listing_id": f"BINA-{item_id}",
        "source": "bina.az",
        "lat": lat,
        "lon": lon,
        "price_azn": price,
        "area_m2": area,
        "price_azn_m2": round(price / area, 2),
        "rooms": int(rooms),
        "floor": int(floor),
        "building_floors": int(floors),
        "building_type": "new" if is_new else "old",
        "listed_month": month,
        "title": (product.get("name") or "").strip(),
        "description": (item.get("description") or product.get("description") or "").strip(),
        "image_phash": "",           # filled by sources.images when enabled
        "district": district or "",
        # ---- real-data extras (not in LISTING_COLUMNS) ----
        "address": addr_text,
        "location_name": location_name,
        "location_path": " / ".join(n for n, _ in nearest),
        "geo_precision": "district_centroid",
        "contact_type": item.get("contactTypeName") or "",
        "image_url": images[0] if images else "",
        "url": url,
        "updated_at": updated,
    }


def harvest(
    limit: int = 500,
    *,
    delay: float = 1.0,
    use_cache: bool = True,
    progress_every: int = 50,
) -> HarvestResult:
    """Fetch and parse up to `limit` sale listings from bina.az."""
    urls = iter_item_urls(limit=limit)
    rows: list[dict] = []
    res = HarvestResult(listings=pd.DataFrame(), source="bina.az")
    if not urls:
        res.notes.append("sitemap unavailable")
        return res

    for i, url in enumerate(urls, 1):
        html, cached = polite_get(url, use_cache=use_cache, delay=delay)
        if html is None:
            res.failed += 1
            continue
        res.from_cache += int(cached)
        res.fetched += int(not cached)
        row = parse_item(html, url)
        if row is None:
            continue
        rows.append(row)
        if progress_every and i % progress_every == 0:
            print(f"  bina.az {i}/{len(urls)} urls -> {len(rows)} listings", flush=True)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="listing_id", keep="first").reset_index(drop=True)
        ordered = [c for c in LISTING_COLUMNS if c in df.columns]
        extras = [c for c in df.columns if c not in ordered]
        df = df[ordered + extras]
    res.listings = df
    return res
