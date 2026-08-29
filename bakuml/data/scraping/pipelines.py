"""Item pipelines + pure parsing helpers for bina.az adverts.

All text parsing lives here as **pure functions** (no scrapy imports needed
to use them) so the tricky Azerbaijani formats can be unit-tested offline:

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

Pipelines:

``NormalizePipeline``
    Maps a raw :class:`~bakuml.data.scraping.items.ListingItem` onto the
    canonical ``schema.LISTING_COLUMNS`` row: ``listing_id`` prefixed
    ``"BINA-"``, ``source="bina.az"``, parsed numeric fields, derived
    ``price_azn_m2``. Rows without a usable price, area or coordinates are
    dropped (``scrapy.exceptions.DropItem``) — everything else degrades to
    documented sentinels (0 for unknown rooms/floors, scrape month for an
    unparseable listing date).

``PhashPipeline``
    Computes the perceptual hash of the cover photo *if* image bytes were
    attached to the row (key ``image_bytes``). The polite offline-friendly
    default crawl never downloads photos, so the hash degrades gracefully to
    ``""`` — the dedup module treats an empty hash as "no photo evidence".
"""

from __future__ import annotations

import datetime as dt
import io
import re

from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem

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
_EXOTIC_SPACES = (" ", " ", " ", " ")


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
# Pipelines
# ---------------------------------------------------------------------------

def _to_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # reject NaN


class NormalizePipeline:
    """ListingItem -> canonical ``schema.LISTING_COLUMNS`` dict.

    ``today`` is injectable so tests (and re-runs over archived HTML) resolve
    "bugün"/"dünən" deterministically; the live crawl uses the scrape date.
    """

    def __init__(self, today: dt.date | None = None) -> None:
        self.today = today or dt.date.today()

    def process_item(self, item, spider=None):
        raw = ItemAdapter(item) if not isinstance(item, dict) else item

        price = parse_price(raw.get("price_raw"))
        area = parse_area(raw.get("area_raw"))
        lat = _to_float(raw.get("lat"))
        lon = _to_float(raw.get("lon"))
        if price is None or area is None or lat is None or lon is None:
            raise DropItem(
                f"listing {raw.get('listing_id') or raw.get('url')}: "
                "missing price, area or coordinates"
            )

        rooms = parse_rooms(raw.get("rooms_raw"))
        floor, building_floors = parse_floor(raw.get("floor_raw"))
        listed_month = parse_listed_month(raw.get("listed_date_raw"), self.today)

        raw_id = raw.get("listing_id")
        if not raw_id:
            # deterministic fallback: last path segment of the advert URL
            raw_id = str(raw.get("url") or "unknown").rstrip("/").rsplit("/", 1)[-1]

        building_type_raw = raw.get("building_type_raw") or ""
        building_type = "new" if "yeni" in _az_lower(str(building_type_raw)) else "old"

        row = {
            "listing_id": f"BINA-{raw_id}",
            "source": "bina.az",
            "lat": lat,
            "lon": lon,
            "price_azn": price,
            "area_m2": area,
            "price_azn_m2": round(price / area, 1),
            # 0 = unknown, documented sentinel (schema requires int64).
            "rooms": rooms if rooms is not None else 0,
            "floor": floor if floor is not None else 0,
            "building_floors": building_floors if building_floors is not None else 0,
            "building_type": building_type,
            # Unparseable dates fall back to the scrape month: listings on the
            # live board without an explicit date were (re)posted recently.
            "listed_month": listed_month or f"{self.today:%Y-%m}",
            "title": raw.get("title") or "",
            "description": raw.get("description") or "",
            "image_phash": "",  # PhashPipeline fills this in when photo bytes exist
            "district": raw.get("district") or "",
        }
        # Keep column order identical to the canonical schema.
        return {col: row[col] for col in LISTING_COLUMNS}


class PhashPipeline:
    """Fill ``image_phash`` from attached cover-photo bytes, if any.

    Runs after :class:`NormalizePipeline` (see ``ITEM_PIPELINES`` ordering).
    A crawl that also fetched the cover photo attaches its bytes under the
    transient key ``image_bytes``; the key is always popped so the final row
    matches the canonical schema exactly. Offline (no bytes), the hash stays
    ``""``.
    """

    def process_item(self, item, spider=None):
        if isinstance(item, dict):
            blob = item.pop("image_bytes", None)
            if blob:
                item["image_phash"] = phash_hex(blob)
            else:
                item.setdefault("image_phash", "")
        return item
