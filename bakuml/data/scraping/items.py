"""Scrapy item definition for bina.az apartment adverts.

The item deliberately carries *raw* strings straight off the page
(``price_raw``, ``area_raw``, ...) rather than parsed values: parsing is
concentrated in :mod:`bakuml.data.scraping.pipelines` as pure functions so it
can be unit-tested offline and reused by any future spider (tap.az, emlak.az).
Only the coordinates come pre-typed because they are read from ``data-lat`` /
``data-lng`` map attributes, not free text.
"""

from __future__ import annotations

import scrapy


class ListingItem(scrapy.Item):
    """One scraped advert, pre-normalisation (see DESIGN.md section 1)."""

    url = scrapy.Field()
    listing_id = scrapy.Field()        # bina.az numeric id (string)
    title = scrapy.Field()
    description = scrapy.Field()
    price_raw = scrapy.Field()         # e.g. "150 000 AZN"
    area_raw = scrapy.Field()          # e.g. "85.5 m²"
    rooms_raw = scrapy.Field()         # e.g. "3 otaqlı" / "3"
    floor_raw = scrapy.Field()         # e.g. "4/9 mərtəbə"
    building_type_raw = scrapy.Field()  # e.g. "Yeni tikili" / "Köhnə tikili"
    lat = scrapy.Field()
    lon = scrapy.Field()
    district = scrapy.Field()
    listed_date_raw = scrapy.Field()   # e.g. "28 Avqust 2026" / "bugün"
    image_urls = scrapy.Field()        # list[str]
