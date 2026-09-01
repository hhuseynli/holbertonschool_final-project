"""Parsing helpers and normalisation for bina.az listings.

All text parsing lives here as **pure functions** so the tricky Azerbaijani
formats can be unit-tested offline:

* prices with grouped thousands: ``"150 000 AZN"`` (regular, no-break or thin
  spaces), occasionally ``"150.000 AZN"``;
* areas with dot or comma decimals: ``"85.5 m²"`` / ``"85,5 m²"``;
* floors as ``"4/9 mərtəbə"``;
* dates with Azerbaijani month names (yanvar ... dekabr) or the relative
  words ``bugün`` ("today") / ``dünən`` ("yesterday"), which are resolved
  against a caller-supplied ``today`` date so parsing stays deterministic.

A locale subtlety worth defending: Azerbaijani has *four* letters i/İ/ı/I and
Python's ``str.lower`` maps ``"İ"`` to ``"i" + COMBINING DOT ABOVE``, which
breaks naive substring matching ("İyun" would not contain "iyun"). ``_az_lower``
performs the Azerbaijani-correct case fold first.

The :func:`normalize_graphql_node` function maps a flattened GraphQL node
(from :mod:`bakuml.data.scraping.client`) onto the canonical
``schema.LISTING_COLUMNS`` row.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re

from bakuml.data.schema import LISTING_COLUMNS

# ---------------------------------------------------------------------------
# Azerbaijani text helpers
# ---------------------------------------------------------------------------

#: Azerbaijani month names -> month number.
AZ_MONTHS: dict[str, int] = {
    "yanvar": 1, "fevral": 2, "mart": 3, "aprel": 4,
    "may": 5, "iyun": 6, "iyul": 7, "avqust": 8,
    "sentyabr": 9, "oktyabr": 10, "noyabr": 11, "dekabr": 12,
}

# Unicode spaces seen in scraped price strings (NBSP, thin/narrow spaces).
_EXOTIC_SPACES = ("\u00a0", "\u2009", "\u202f", "\u2007")


def _az_lower(text: str) -> str:
    """Azerbaijani-correct lowercase (İ -> i, I -> ı), then str.lower()."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def _normalise_spaces(text: str) -> str:
    """Replace exotic Unicode spaces with plain spaces."""
    for ch in _EXOTIC_SPACES:
        text = text.replace(ch, " ")
    return text


def _extract_number(text) -> float | None:
    """First numeric token in ``text`` as float, handling Azerbaijani price
    formatting: spaces as thousands separators ("150 000"), occasionally
    dots/commas as thousands separators ("150.000"), and comma decimals
    ("85,5"). Returns None when nothing numeric is found.
    """
    if text is None:
        return None
    s = _normalise_spaces(str(text))
    m = re.search(r"\d[\d\s.,]*", s)
    if not m:
        return None
    num = re.sub(r"\s+", "", m.group()).rstrip(".,")
    if not num:
        return None
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", num):
        # "150.000" / "1,500,000": groups of exactly 3 -> thousands separators.
        num = re.sub(r"[.,]", "", num)
    else:
        # "85,5" -> "85.5"; a plain "85.5" is untouched.
        num = num.replace(",", ".")
    try:
        value = float(num)
    except ValueError:
        return None
    return value


# ---------------------------------------------------------------------------
# Pure parse functions (the unit-tested public API of this module)
# ---------------------------------------------------------------------------

def parse_price(text) -> float | None:
    """``"150 000 AZN"`` -> 150000.0. None when missing/non-positive."""
    value = _extract_number(text)
    if value is None or value <= 0:
        return None
    return value


def parse_area(text) -> float | None:
    """``"85.5 m²"`` / ``"85,5 m²"`` -> 85.5. None when missing/non-positive."""
    value = _extract_number(text)
    if value is None or value <= 0:
        return None
    return value


def parse_rooms(text) -> int | None:
    """``"3 otaqlı"`` / ``"Otaq sayı: 3"`` / ``"3"`` -> 3. None on miss."""
    if text is None:
        return None
    m = re.search(r"\d+", str(text))
    return int(m.group()) if m else None


def parse_floor(text) -> tuple[int | None, int | None]:
    """``"4/9 mərtəbə"`` -> (4, 9); ``"5 mərtəbə"`` -> (5, None);
    unparseable -> (None, None).
    """
    if text is None:
        return (None, None)
    s = str(text)
    m = re.search(r"(\d+)\s*/\s*(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"\d+", s)
    if m:
        return (int(m.group()), None)
    return (None, None)


def parse_listed_month(text, today: dt.date) -> str | None:
    """Resolve a bina.az listing date to a ``"YYYY-MM"`` month string.

    Handles absolute dates ("28 Avqust 2026", year optional -> ``today``'s
    year) and the relative words "bugün" / "dünən" (any capitalisation,
    optional extra text such as "Bugün, 14:32"), resolved against ``today``.
    Returns None when nothing date-like is found.
    """
    if text is None:
        return None
    lowered = _az_lower(_normalise_spaces(str(text)))
    compact = re.sub(r"\s+", "", lowered)
    if "bugün" in compact:  # also matches "bu gün" via compact form
        return f"{today:%Y-%m}"
    if "dünən" in compact:
        yesterday = today - dt.timedelta(days=1)
        return f"{yesterday:%Y-%m}"
    for name, number in AZ_MONTHS.items():
        if name in lowered:
            year_match = re.search(r"\b(?:19|20)\d{2}\b", lowered)
            year = int(year_match.group()) if year_match else today.year
            return f"{year:04d}-{number:02d}"
    return None


def phash_hex(image_bytes: bytes | None) -> str:
    """64-bit perceptual hash of an image as a 16-char hex string.

    Degrades gracefully: any failure (no bytes, undecodable image, missing
    optional deps) returns ``""`` so an offline crawl still produces valid
    rows — the dedup module treats "" as "no photo evidence".
    """
    if not image_bytes:
        return ""
    try:
        import imagehash
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            return str(imagehash.phash(img))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# GraphQL node normalisation
# ---------------------------------------------------------------------------

def normalize_graphql_node(node: dict, today: dt.date | None = None) -> dict | None:
    """Map a flattened GraphQL listing node to a canonical LISTING_COLUMNS row.

    Returns None (skip) when critical fields (price, area, coordinates) are
    missing or invalid.
    """
    today = today or dt.date.today()

    price = node.get("price")
    area = node.get("area")
    lat = node.get("lat")
    lon = node.get("lon")

    if price is None or price <= 0:
        return None
    if area is None or area <= 0:
        return None
    if lat is None or lon is None:
        return None

    raw_id = node.get("listing_id")
    if not raw_id:
        path = node.get("path") or ""
        raw_id = path.rstrip("/").rsplit("/", 1)[-1] or "unknown"

    rooms = node.get("rooms")
    floor = node.get("floor")
    floors = node.get("floors")

    # Derive listed_month from updatedAt ISO timestamp
    updated_at = node.get("updated_at") or ""
    if updated_at and len(updated_at) >= 7:
        listed_month = updated_at[:7]  # "YYYY-MM" from ISO string
    else:
        listed_month = f"{today:%Y-%m}"

    has_repair = node.get("has_repair")
    building_type = "new" if has_repair else "old"

    row = {
        "listing_id": f"BINA-{raw_id}",
        "source": "bina.az",
        "lat": float(lat),
        "lon": float(lon),
        "price_azn": float(price),
        "area_m2": float(area),
        "price_azn_m2": round(float(price) / float(area), 1),
        "rooms": int(rooms) if rooms is not None else 0,
        "floor": int(floor) if floor is not None else 0,
        "building_floors": int(floors) if floors is not None else 0,
        "building_type": building_type,
        "listed_month": listed_month,
        "title": "",
        "description": "",
        "image_phash": "",
        "photo_urls": json.dumps(node.get("photos") or []),
        "district": node.get("location_name") or "",
    }
    return {col: row[col] for col in LISTING_COLUMNS}
