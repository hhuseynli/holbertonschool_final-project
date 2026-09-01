"""Offline tests for the bina.az scraping package.

No network is touched anywhere: the parse helpers are pure functions, the
GraphQL client's ``_flatten_node`` is tested with hand-built dicts, and
``normalize_graphql_node`` is tested against the canonical schema.
``runner.scrape_to_parquet`` is never invoked (it hits the network); only
its import/signature is checked.
"""

from __future__ import annotations

import datetime as dt
import inspect
import io
import re

import pytest

from bakuml.data.schema import LISTING_COLUMNS
from bakuml.data.scraping.client import _flatten_node
from bakuml.data.scraping.pipelines import (
    AZ_MONTHS,
    normalize_graphql_node,
    parse_area,
    parse_floor,
    parse_listed_month,
    parse_price,
    parse_rooms,
    phash_hex,
)

TODAY = dt.date(2026, 8, 29)


# =========================================================================
# Pure parse functions
# =========================================================================

class TestParsePrice:
    def test_grouped_thousands(self):
        assert parse_price("150 000 AZN") == 150000.0

    def test_no_break_space_and_weird_whitespace(self):
        assert parse_price("98\u00a0500 AZN") == 98500.0
        assert parse_price("  150 000   AZN ") == 150000.0
        assert parse_price("1 250 000 AZN") == 1250000.0

    def test_dot_as_thousands_separator(self):
        assert parse_price("150.000 AZN") == 150000.0

    def test_decimal_values(self):
        assert parse_price("1 500.5 AZN") == 1500.5
        assert parse_price("1500,5") == 1500.5

    def test_missing_or_invalid(self):
        assert parse_price(None) is None
        assert parse_price("") is None
        assert parse_price("Qiymət yoxdur") is None
        assert parse_price("0 AZN") is None


class TestParseArea:
    def test_dot_decimal(self):
        assert parse_area("85.5 m²") == 85.5

    def test_comma_decimal(self):
        assert parse_area("85,5 m²") == 85.5

    def test_integer_and_bare_number(self):
        assert parse_area("120 m²") == 120.0
        assert parse_area("62") == 62.0

    def test_missing(self):
        assert parse_area(None) is None
        assert parse_area("m²") is None


class TestParseRooms:
    def test_forms(self):
        assert parse_rooms("3 otaqlı") == 3
        assert parse_rooms("Otaq sayı: 3") == 3
        assert parse_rooms("4") == 4

    def test_missing(self):
        assert parse_rooms(None) is None
        assert parse_rooms("otaqlı") is None


class TestParseFloor:
    def test_pair(self):
        assert parse_floor("4/9") == (4, 9)
        assert parse_floor("4/9 mərtəbə") == (4, 9)
        assert parse_floor("4 / 9") == (4, 9)
        assert parse_floor("16/20 mərtəbə") == (16, 20)

    def test_single_number(self):
        assert parse_floor("5 mərtəbə") == (5, None)

    def test_missing(self):
        assert parse_floor(None) == (None, None)
        assert parse_floor("mərtəbə") == (None, None)


class TestParseListedMonth:
    def test_absolute_dates(self):
        assert parse_listed_month("28 Avqust 2026", TODAY) == "2026-08"
        assert parse_listed_month("1 yanvar 2024", TODAY) == "2024-01"
        assert parse_listed_month("Dekabr 2023", TODAY) == "2023-12"

    def test_dotted_capital_i_months(self):
        assert parse_listed_month("15 İyun 2025", TODAY) == "2025-06"
        assert parse_listed_month("3 İyul 2025", TODAY) == "2025-07"

    def test_missing_year_falls_back_to_today(self):
        assert parse_listed_month("28 avqust", TODAY) == "2026-08"

    def test_relative_today(self):
        assert parse_listed_month("bugün", TODAY) == "2026-08"
        assert parse_listed_month("Bugün, 14:32", TODAY) == "2026-08"

    def test_relative_yesterday_crosses_month_boundary(self):
        assert parse_listed_month("dünən", dt.date(2026, 3, 1)) == "2026-02"
        assert parse_listed_month("Dünən", TODAY) == "2026-08"

    def test_unparseable(self):
        assert parse_listed_month(None, TODAY) is None
        assert parse_listed_month("Baxışların sayı: 154", TODAY) is None


class TestPhashHex:
    def _png_bytes(self) -> bytes:
        from PIL import Image

        img = Image.new("RGB", (32, 32))
        for x in range(32):
            for y in range(32):
                img.putpixel((x, y), (x * 8 % 256, y * 8 % 256, (x * y) % 256))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_valid_image(self):
        digest = phash_hex(self._png_bytes())
        assert re.fullmatch(r"[0-9a-f]{16}", digest)

    def test_deterministic(self):
        blob = self._png_bytes()
        assert phash_hex(blob) == phash_hex(blob)

    def test_degrades_gracefully(self):
        assert phash_hex(None) == ""
        assert phash_hex(b"") == ""
        assert phash_hex(b"definitely not an image") == ""


# =========================================================================
# GraphQL client: _flatten_node
# =========================================================================

class TestFlattenNode:
    def test_full_node(self):
        node = {
            "id": "6390275",
            "path": "/items/6390275",
            "rooms": 3,
            "floor": 13,
            "floors": 16,
            "area": {"value": 161.0},
            "price": {"total": 499000, "currency": "AZN"},
            "location": {"id": "8", "name": "28 May", "latitude": 40.38, "longitude": 49.85},
            "city": {"id": "1", "name": "Bakı"},
            "photos": [{"thumbnail": "https://t.jpg", "large": "https://l.jpg"}],
            "hasRepair": True,
            "hasMortgage": False,
            "isFeatured": False,
            "updatedAt": "2026-09-01T14:11:17+04:00",
        }
        flat = _flatten_node(node)
        assert flat["listing_id"] == "6390275"
        assert flat["price"] == 499000
        assert flat["area"] == 161.0
        assert flat["lat"] == 40.38
        assert flat["lon"] == 49.85
        assert flat["location_name"] == "28 May"
        assert flat["rooms"] == 3
        assert flat["floor"] == 13
        assert flat["floors"] == 16
        assert flat["has_repair"] is True
        assert flat["updated_at"] == "2026-09-01T14:11:17+04:00"
        assert flat["photos"] == ["https://l.jpg"]

    def test_missing_nested_fields(self):
        node = {"id": "1", "path": "/items/1"}
        flat = _flatten_node(node)
        assert flat["price"] is None
        assert flat["area"] is None
        assert flat["lat"] is None
        assert flat["photos"] == []


# =========================================================================
# Normalisation: GraphQL node -> canonical schema
# =========================================================================

def _minimal_node(**overrides) -> dict:
    base = {
        "listing_id": "12345",
        "path": "/items/12345",
        "price": 100000,
        "currency": "AZN",
        "area": 50.0,
        "lat": 40.40,
        "lon": 49.80,
        "rooms": 3,
        "floor": 4,
        "floors": 9,
        "has_repair": True,
        "location_name": "Yasamal",
        "updated_at": "2026-08-29T10:00:00+04:00",
        "photos": ["https://example.com/photo1.jpg", "https://example.com/photo2.jpg"],
    }
    base.update(overrides)
    return base


class TestNormalizeGraphqlNode:
    def test_full_normalisation(self):
        row = normalize_graphql_node(_minimal_node(), today=TODAY)
        assert row is not None
        assert list(row) == list(LISTING_COLUMNS)
        assert row["listing_id"] == "BINA-12345"
        assert row["source"] == "bina.az"
        assert row["price_azn"] == 100000.0
        assert row["area_m2"] == 50.0
        assert row["price_azn_m2"] == pytest.approx(2000.0, abs=0.05)
        assert row["rooms"] == 3
        assert row["floor"] == 4
        assert row["building_floors"] == 9
        assert row["building_type"] == "new"
        assert row["lat"] == pytest.approx(40.40)
        assert row["lon"] == pytest.approx(49.80)
        assert row["listed_month"] == "2026-08"
        assert row["district"] == "Yasamal"
        assert row["image_phash"] == ""

    def test_drops_when_critical_fields_missing(self):
        assert normalize_graphql_node(_minimal_node(price=None), today=TODAY) is None
        assert normalize_graphql_node(_minimal_node(price=0), today=TODAY) is None
        assert normalize_graphql_node(_minimal_node(area=None), today=TODAY) is None
        assert normalize_graphql_node(_minimal_node(lat=None), today=TODAY) is None
        assert normalize_graphql_node(_minimal_node(lon=None), today=TODAY) is None

    def test_sentinels_for_optional_fields(self):
        row = normalize_graphql_node(
            _minimal_node(rooms=None, floor=None, floors=None, location_name=None),
            today=TODAY,
        )
        assert row["rooms"] == 0
        assert row["floor"] == 0
        assert row["building_floors"] == 0
        assert row["district"] == ""

    def test_no_repair_maps_to_old(self):
        row = normalize_graphql_node(_minimal_node(has_repair=False), today=TODAY)
        assert row["building_type"] == "old"

    def test_missing_updated_at_uses_today(self):
        row = normalize_graphql_node(_minimal_node(updated_at=None), today=TODAY)
        assert row["listed_month"] == "2026-08"

    def test_id_from_path_fallback(self):
        row = normalize_graphql_node(
            _minimal_node(listing_id=None, path="/items/99999"),
            today=TODAY,
        )
        assert row["listing_id"] == "BINA-99999"


# =========================================================================
# Runner surface (never actually fetched)
# =========================================================================

def test_runner_signature_only():
    from bakuml.data.scraping.runner import scrape_to_parquet

    params = inspect.signature(scrape_to_parquet).parameters
    assert list(params) == ["out_path", "max_pages"]
    assert params["max_pages"].default == 5
