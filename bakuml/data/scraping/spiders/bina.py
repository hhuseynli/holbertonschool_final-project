"""Spider for bina.az apartment sale listings.

Crawl shape
-----------
``https://bina.az/alqi-satqi/menziller`` lists advert *cards* (price, a short
"3 otaqlı / 85.5 m² / 4/9 mərtəbə" summary, location, date) with pagination.
Each card links to a *detail page* (``/items/<id>``) carrying the full
properties table, description, map coordinates and photos.

Defensive parsing
-----------------
bina.az markup changes without notice, so every extraction goes through
:func:`safe_css`, which returns ``None`` instead of raising when a selector
misses, and most fields try several selector variants (documented as of
2026). Card-level values travel to the detail callback via ``cb_kwargs`` and
serve as fallbacks when the detail page lacks a field; nothing here raises on
malformed markup — missing critical fields are dropped later by
``NormalizePipeline``.

Politeness: the project settings keep ``ROBOTSTXT_OBEY=True``, autothrottle
on, 2 concurrent requests and a multi-second download delay; the spider also
caps pagination at ``max_pages`` (default 5).
"""

from __future__ import annotations

import re

import scrapy

from bakuml.data.scraping.items import ListingItem
from bakuml.data.scraping.pipelines import AZ_MONTHS, _az_lower

_ITEM_ID_RE = re.compile(r"/items/(\d+)")
_RELATIVE_DATE_WORDS = ("bugün", "dünən")


def safe_css(selector, query: str) -> str | None:
    """First match of a CSS query as a stripped string; None on any miss.

    Central guard for a site whose markup drifts: a selector that no longer
    matches (or matches an empty node) degrades to None instead of raising.
    """
    try:
        value = selector.css(query).get()
    except Exception:
        return None
    if value is None:
        return None
    value = value.strip()
    return value or None


def _looks_like_date(text: str) -> bool:
    """True when a snippet contains an Azerbaijani month or relative day."""
    lowered = _az_lower(text)
    compact = re.sub(r"\s+", "", lowered)
    if any(word in compact for word in _RELATIVE_DATE_WORDS):
        return True
    return any(name in lowered for name in AZ_MONTHS)


class BinaSpider(scrapy.Spider):
    """Crawl bina.az apartment sale cards, follow details, emit ListingItem."""

    name = "bina"
    allowed_domains = ["bina.az"]
    start_urls = ["https://bina.az/alqi-satqi/menziller"]

    def __init__(self, max_pages: int = 5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # scrapy CLI passes -a arguments as strings
        self.max_pages = int(max_pages)

    # ------------------------------------------------------------------ #
    # Listing page: cards + pagination
    # ------------------------------------------------------------------ #
    def parse(self, response):
        """Parse a listing page: follow each card's detail link, paginate."""
        cards = response.css("div.items-i, article.items-i")
        for card in cards:
            href = (
                safe_css(card, "a.item_link::attr(href)")
                or safe_css(card, "a::attr(href)")
            )
            if not href:
                continue  # card without a detail link is unusable
            yield response.follow(
                href,
                callback=self.parse_detail,
                cb_kwargs={"card": self._parse_card(card)},
            )

        page = response.meta.get("page", 1)
        if page < self.max_pages:
            next_href = (
                safe_css(response, "a.next::attr(href)")
                or safe_css(response, "li.next a::attr(href)")
                or safe_css(response, 'a[rel="next"]::attr(href)')
            )
            if next_href:
                yield response.follow(
                    next_href, callback=self.parse, meta={"page": page + 1}
                )

    def _parse_card(self, card) -> dict:
        """Extract card-level fields (fallbacks for the detail page)."""
        price_val = safe_css(card, ".price-val::text")
        price_cur = safe_css(card, ".price-cur::text") or "AZN"
        price_raw = f"{price_val} {price_cur}" if price_val else None

        # The card summary is a <ul class="name"> with items like
        # "3 otaqlı" / "85.5 m²" / "4/9 mərtəbə" — order not guaranteed,
        # so classify each entry by its unit token.
        rooms_raw = area_raw = floor_raw = None
        for entry in card.css("ul.name li::text").getall():
            entry = entry.strip()
            if not entry:
                continue
            lowered = _az_lower(entry)
            if "otaq" in lowered and rooms_raw is None:
                rooms_raw = entry
            elif ("m²" in lowered or "m2" in lowered) and area_raw is None:
                area_raw = entry
            elif "mərtəbə" in lowered and floor_raw is None:
                floor_raw = entry

        return {
            "price_raw": price_raw,
            "rooms_raw": rooms_raw,
            "area_raw": area_raw,
            "floor_raw": floor_raw,
            "district": safe_css(card, ".location::text"),
            "listed_date_raw": safe_css(card, ".city_when::text"),
        }

    # ------------------------------------------------------------------ #
    # Detail page
    # ------------------------------------------------------------------ #
    def parse_detail(self, response, card: dict | None = None):
        """Parse an advert detail page into a raw ListingItem.

        Detail-page values win; card values fill any gaps. Everything stays
        raw text — parsing/typing happens in NormalizePipeline.
        """
        card = card or {}
        item = ListingItem()
        item["url"] = response.url

        id_match = _ITEM_ID_RE.search(response.url)
        item["listing_id"] = (
            id_match.group(1)
            if id_match
            else response.url.rstrip("/").rsplit("/", 1)[-1]
        )

        item["title"] = safe_css(response, "h1.product-title::text")
        desc_parts = response.css(".product-description__content ::text").getall()
        item["description"] = (
            " ".join(p.strip() for p in desc_parts if p.strip()) or None
        )

        price_val = (
            safe_css(response, ".product-price__i .price-val::text")
            or safe_css(response, ".product-price .price-val::text")
        )
        price_cur = (
            safe_css(response, ".product-price__i .price-cur::text") or "AZN"
        )
        item["price_raw"] = (
            f"{price_val} {price_cur}" if price_val else card.get("price_raw")
        )

        # Properties table: label/value pairs, keyed by Azerbaijani label.
        props: dict[str, str] = {}
        for prop in response.css(".product-properties__i"):
            name = safe_css(prop, ".product-properties__i-name::text")
            value = safe_css(prop, ".product-properties__i-value::text")
            if name and value:
                props[_az_lower(name)] = value
        item["area_raw"] = props.get("sahə") or card.get("area_raw")
        item["rooms_raw"] = props.get("otaq sayı") or card.get("rooms_raw")
        item["floor_raw"] = props.get("mərtəbə") or card.get("floor_raw")
        item["building_type_raw"] = (
            props.get("kateqoriya") or props.get("binanın tipi")
        )

        item["lat"] = safe_css(response, "#item_map::attr(data-lat)")
        item["lon"] = safe_css(response, "#item_map::attr(data-lng)")
        item["district"] = (
            safe_css(response, ".product-map__left__location::text")
            or card.get("district")
        )

        # The statistics strip mixes view counts with the posting date; keep
        # the first entry that actually looks like a date.
        stats = [
            t.strip()
            for t in response.css(".product-statistics__i-text::text").getall()
            if t.strip()
        ]
        item["listed_date_raw"] = next(
            (t for t in stats if _looks_like_date(t)),
            card.get("listed_date_raw"),
        )

        item["image_urls"] = response.css(
            ".product-photos img::attr(src)"
        ).getall()
        yield item
