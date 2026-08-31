"""Real-data adapters: parsing, politeness, and duplicate-once accounting.

The fixture is a *trimmed real* bina.az item page - the same JSON-LD block,
inline Apollo ``Item`` object, ``nearestLocations`` and ``Location``
entries the live site serves, with the bulk stripped out. Tests are fully
offline; nothing here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bakuml.data.schema import LISTING_COLUMNS
from bakuml.data.sources import bina
from bakuml.data.sources.combine import build_real_dataset, canonical_only
from bakuml.data.sources.geocode import query_variants

FIXTURE = Path(__file__).parent / "fixtures" / "bina_item_real.html"
ITEM_URL = "https://bina.az/items/6412411"


@pytest.fixture(scope="module")
def item_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Parsing the real payload
# --------------------------------------------------------------------------


def test_parses_real_item_page(item_html):
    row = bina.parse_item(item_html, ITEM_URL)
    assert row is not None
    # Values transcribed from the live advert this fixture was cut from.
    assert row["listing_id"] == "BINA-6412411"
    assert row["source"] == "bina.az"
    assert row["price_azn"] == 195000.0
    assert row["area_m2"] == 90.0
    assert row["price_azn_m2"] == pytest.approx(2166.67, abs=0.01)
    assert row["rooms"] == 3
    assert row["floor"] == 15
    assert row["building_floors"] == 15
    assert row["building_type"] == "new"
    assert row["listed_month"] == "2026-08"
    assert row["district"] == "Yasamal"


def test_parsed_row_covers_the_canonical_schema(item_html):
    row = bina.parse_item(item_html, ITEM_URL)
    missing = [c for c in LISTING_COLUMNS if c not in row]
    assert not missing, f"parsed row missing canonical columns: {missing}"


def test_coordinates_are_labelled_as_district_precision(item_html):
    """bina.az publishes no per-listing point, so the row must say so.

    This is the guard against the worst silent error available here:
    treating a rayon centroid as a measured location.
    """
    row = bina.parse_item(item_html, ITEM_URL)
    assert row["geo_precision"] == "district_centroid"
    # It is the district's centroid, not the agency office that also appears
    # on the page.
    assert row["lat"] == pytest.approx(40.379962, abs=1e-6)
    assert row["lon"] == pytest.approx(49.808995, abs=1e-6)


def test_keeps_the_platform_location_hierarchy(item_html):
    """The finest geography bina.az publishes is a location path."""
    row = bina.parse_item(item_html, ITEM_URL)
    assert "Yasamal r." in row["location_path"]
    assert "20 Yanvar m." in row["location_path"]
    assert row["location_name"]


def test_rejects_pages_without_a_product_block():
    assert bina.parse_item("<html><body>nothing</body></html>", ITEM_URL) is None


def test_rejects_non_item_urls(item_html):
    assert bina.parse_item(item_html, "https://bina.az/agentlikler/veli-emlak") is None


def test_balanced_json_extraction_handles_nested_braces_and_strings():
    text = 'x{"a":{"b":"}{"},"c":[1,2]}y'
    obj = bina._extract_json_object(text, text.index("{"))
    assert obj == {"a": {"b": "}{"}, "c": [1, 2]}


# --------------------------------------------------------------------------
# Politeness
# --------------------------------------------------------------------------


def test_robots_rules_are_parsed_from_a_real_robots_txt(monkeypatch):
    """bina.az allows its catalogue and denies session endpoints.

    Enforcement must come from the file, not from a hard-coded list - and it
    must not fail closed on every URL, which is what urllib's own fetch does
    behind this environment's proxy.
    """
    from bakuml.data.sources import base

    robots = (
        "User-Agent: *\n"
        "Disallow: /company_session/new\n"
        "Disallow: /items/new\n"
        "Disallow: /bookmarks\n"
    )

    class _Resp:
        status_code = 200
        text = robots

    monkeypatch.setattr(base.requests, "get", lambda *a, **k: _Resp())
    base._ROBOTS.clear()
    assert base.robots_allows("https://bina.az/items/6412411")
    assert base.robots_allows("https://bina.az/alqi-satqi/menziller")
    assert not base.robots_allows("https://bina.az/items/new")
    assert not base.robots_allows("https://bina.az/bookmarks")
    base._ROBOTS.clear()


def test_disallowed_urls_are_never_fetched(monkeypatch):
    from bakuml.data.sources import base

    calls = []

    class _Robots:
        def can_fetch(self, *_a):
            return False

    monkeypatch.setitem(base._ROBOTS, "bina.az", _Robots())
    monkeypatch.setattr(
        base.requests, "get", lambda *a, **k: calls.append(a) or None
    )
    text, cached = base.polite_get("https://bina.az/bookmarks")
    assert text is None and not cached
    assert not calls, "a robots-disallowed URL must not be requested"
    base._ROBOTS.clear()


def test_user_agent_identifies_the_project():
    from bakuml.data.sources.base import USER_AGENT

    assert "BakuML" in USER_AGENT
    # An honest crawler says who it is rather than posing as a browser.
    assert "Mozilla" not in USER_AGENT


# --------------------------------------------------------------------------
# Geocoding query cascade
# --------------------------------------------------------------------------


def test_query_cascade_drops_the_house_number_then_falls_back():
    """Measured behaviour: a bina.az address with a house number usually
    misses in Nominatim, the bare street hits, the micro-location hits."""
    row = pd.Series(
        {
            "address": "Məhəmməd Xiyabani küç., 196.",
            "location_name": "Yeni Yasamal q.",
            "district": "Yasamal",
        }
    )
    variants = query_variants(row)
    queries = [q for q, _p in variants]
    precisions = [p for _q, p in variants]
    assert len(variants) >= 3
    assert queries[0].startswith("Məhəmməd Xiyabani küç., 196")
    assert "196" not in queries[1], "second variant must drop the house number"
    assert queries[1].startswith("Məhəmməd Xiyabani küç.")
    assert queries[2].startswith("Yeni Yasamal")
    assert all(q.endswith("Azərbaycan") for q in queries)
    # Each variant declares what it is, so a fallback cannot claim precision
    # it does not have.
    assert precisions[:3] == ["address", "street", "locality"]
    # The street abbreviation "küç." keeps its period; only the house
    # number's trailing period is stripped.
    assert "küç." in queries[1]


def test_query_cascade_handles_a_missing_address():
    row = pd.Series({"address": "", "location_name": "20 Yanvar m.", "district": "Yasamal"})
    variants = query_variants(row)
    assert variants
    assert variants[0][0].startswith("20 Yanvar")
    assert variants[0][1] == "locality"


def test_address_without_a_house_number_is_street_not_address_precision():
    """Cascade position is not precision: a street name tried first is still
    a street, and calling it 'address' would overstate the point."""
    row = pd.Series(
        {"address": "Əsəd Əhmədov küç.", "location_name": "", "district": "Yasamal"}
    )
    variants = query_variants(row)
    assert variants[0][1] == "street"


# --------------------------------------------------------------------------
# Counting every flat once
# --------------------------------------------------------------------------


def _advert(listing_id: str, source: str, **over) -> dict:
    base = {
        "listing_id": listing_id,
        "source": source,
        "lat": 40.379962,
        "lon": 49.808995,
        "price_azn": 195000.0,
        "area_m2": 90.0,
        "price_azn_m2": 2166.67,
        "rooms": 3,
        "floor": 15,
        "building_floors": 15,
        "building_type": "new",
        "listed_month": "2026-08",
        "title": "3 otaqlı yeni tikili 90 m², 20 Yanvar m.",
        "description": "Yasamal rayonu, Mehemmed Xiyabani kucesi, super temirli menzil satilir.",
        "image_phash": "f80eeac35e5cc292",
        "district": "Yasamal",
    }
    base.update(over)
    return base


def test_the_same_flat_on_two_platforms_is_counted_once():
    """Cross-platform duplicates share photographs, not ids or wording."""
    frames = {
        "bina.az": pd.DataFrame([_advert("BINA-1", "bina.az")]),
        "tap.az": pd.DataFrame(
            [
                _advert(
                    "TAP-1",
                    "tap.az",
                    listed_month="2026-08",
                    description="Tecili satilir! Yasamal r., Xiyabani kuc., temirli, senedler qaydasinda.",
                    # same photo, one bit of JPEG noise apart
                    image_phash="f80eeac35e5cc293",
                    price_azn=196000.0,
                    price_azn_m2=2177.78,
                )
            ]
        ),
    }
    ds = build_real_dataset(frames)
    assert ds.n_raw == 2
    assert ds.n_unique == 1, "one flat advertised twice must count once"
    assert ds.cross_source_pairs == 1
    assert ds.within_source_pairs == 0
    assert len(ds.duplicate_map) == 1


def test_broker_repost_within_one_platform_is_counted_once():
    frames = {
        "bina.az": pd.DataFrame(
            [
                _advert("BINA-1", "bina.az", listed_month="2026-07"),
                _advert(
                    "BINA-2",
                    "bina.az",
                    description="TECILI! Yasamal rayonu Mehemmed Xiyabani kucesi super temirli menzil satilir.",
                    image_phash="f80eeac35e5cc296",
                ),
            ]
        )
    }
    ds = build_real_dataset(frames)
    assert ds.n_unique == 1
    assert ds.within_source_pairs == 1
    assert ds.cross_source_pairs == 0
    # The earliest advert survives, so a re-post cannot reset a flat's history.
    assert ds.listings.iloc[0]["listed_month"] == "2026-07"


def test_two_genuinely_different_flats_are_both_kept():
    frames = {
        "bina.az": pd.DataFrame(
            [
                _advert("BINA-1", "bina.az"),
                _advert(
                    "BINA-9",
                    "bina.az",
                    lat=40.3358719,
                    lon=49.8223201,
                    district="Səbail",
                    rooms=2,
                    area_m2=55.0,
                    price_azn=250000.0,
                    price_azn_m2=4545.45,
                    floor=4,
                    building_floors=9,
                    description="Sebail rayonu, Neftchilar prospekti, deniz manzereli menzil.",
                    image_phash="0123456789abcdef",
                ),
            ]
        )
    }
    ds = build_real_dataset(frames)
    assert ds.n_unique == 2, "distinct flats must not be collapsed"
    assert ds.within_source_pairs == 0


def test_report_accounts_for_every_advert():
    frames = {
        "bina.az": pd.DataFrame([_advert("BINA-1", "bina.az")]),
        "tap.az": pd.DataFrame([_advert("TAP-1", "tap.az")]),
    }
    ds = build_real_dataset(frames)
    text = ds.report()
    assert "unique flats" in text and "duplicates removed" in text
    assert ds.n_raw - ds.n_unique == len(ds.duplicate_map)


def test_canonical_only_is_pipeline_ready(item_html):
    row = bina.parse_item(item_html, ITEM_URL)
    df = canonical_only(pd.DataFrame([row]))
    assert list(df.columns) == list(LISTING_COLUMNS)
    from bakuml.data.schema import validate_listings

    validate_listings(df)  # must satisfy the same contract as synthetic data


# --------------------------------------------------------------------------
# Geocoding precision must reflect which query matched
# --------------------------------------------------------------------------


def test_precision_reflects_which_cascade_variant_matched(monkeypatch):
    """A hit on the micro-location fallback is not a street-level point.

    Measured on the real feed, labelling every accepted hit "street" made two
    different streets in one quarter share coordinates while both claimed
    street precision. Precision must come from the cascade position.
    """
    from bakuml.data.sources import geocode

    monkeypatch.setattr(geocode, "_load_cache", lambda: {})
    monkeypatch.setattr(geocode, "_save_cache", lambda cache: None)

    # Only the third variant (the micro-location) resolves.
    def fake_lookup(query: str, **_):
        if query.startswith("Yeni Yasamal"):
            return {
                "lat": "40.392724",
                "lon": "49.791028",
                "category": "place",
                "type": "suburb",
            }
        return None

    monkeypatch.setattr(geocode, "_lookup", fake_lookup)

    df = pd.DataFrame(
        [
            {
                "address": "Məhəmməd Xiyabani küç., 196.",
                "location_name": "Yeni Yasamal q.",
                "district": "Yasamal",
                "lat": 40.379962,
                "lon": 49.808995,
            }
        ]
    )
    res = geocode.geocode_listings(df, max_lookups=10, progress_every=0)
    assert res.matched == 1
    row = res.listings.iloc[0]
    assert row["geo_precision"] == "locality", "fallback must not claim 'street'"
    assert row["geo_confidence"] < 0.5
    assert row["lat"] == pytest.approx(40.392724)


def test_full_address_match_is_labelled_address_precision(monkeypatch):
    from bakuml.data.sources import geocode

    monkeypatch.setattr(geocode, "_load_cache", lambda: {})
    monkeypatch.setattr(geocode, "_save_cache", lambda cache: None)
    monkeypatch.setattr(
        geocode,
        "_lookup",
        lambda q, **_: {
            "lat": "40.3901",
            "lon": "49.7911",
            "category": "building",
            "type": "apartments",
        },
    )
    df = pd.DataFrame(
        [{"address": "Əsəd Əhmədov küç., 12.", "location_name": "", "district": "Yasamal",
          "lat": 40.379962, "lon": 49.808995}]
    )
    res = geocode.geocode_listings(df, max_lookups=5, progress_every=0)
    assert res.listings.iloc[0]["geo_precision"] == "address"
    assert res.listings.iloc[0]["geo_confidence"] == 1.0


def test_unresolved_rows_keep_the_district_centroid(monkeypatch):
    from bakuml.data.sources import geocode

    monkeypatch.setattr(geocode, "_load_cache", lambda: {})
    monkeypatch.setattr(geocode, "_save_cache", lambda cache: None)
    monkeypatch.setattr(geocode, "_lookup", lambda q, **_: None)
    df = pd.DataFrame(
        [{"address": "Nowhere küç., 1.", "location_name": "", "district": "Yasamal",
          "lat": 40.379962, "lon": 49.808995}]
    )
    res = geocode.geocode_listings(df, max_lookups=5, progress_every=0)
    row = res.listings.iloc[0]
    assert res.matched == 0
    assert row["geo_precision"] == "district_centroid"
    assert row["lat"] == pytest.approx(40.379962), "centroid must be preserved"


def test_matches_outside_absheron_are_rejected(monkeypatch):
    """Nominatim mis-parses can land in another country; a point outside the
    study bbox is a parse failure, not a location."""
    from bakuml.data.sources import geocode

    monkeypatch.setattr(geocode, "_load_cache", lambda: {})
    monkeypatch.setattr(geocode, "_save_cache", lambda cache: None)
    monkeypatch.setattr(
        geocode,
        "_lookup",
        lambda q, **_: {"lat": "41.7151", "lon": "44.8271",  # Tbilisi
                        "category": "highway", "type": "residential"},
    )
    df = pd.DataFrame(
        [{"address": "Somewhere küç.", "location_name": "", "district": "Yasamal",
          "lat": 40.379962, "lon": 49.808995}]
    )
    res = geocode.geocode_listings(df, max_lookups=5, progress_every=0)
    assert res.matched == 0
    assert res.listings.iloc[0]["geo_precision"] == "district_centroid"


def test_cached_addresses_resolve_even_with_no_network_budget(monkeypatch):
    """max_lookups caps *network* calls; it must not stop cached resolution."""
    from bakuml.data.sources import geocode

    cached = {
        "Əsəd Əhmədov küç., Yasamal, Bakı, Azərbaycan": {
            "lat": "40.3941", "lon": "49.7981",
            "category": "highway", "type": "residential",
        }
    }
    monkeypatch.setattr(geocode, "_load_cache", lambda: dict(cached))
    monkeypatch.setattr(geocode, "_save_cache", lambda cache: None)

    def no_network(*_a, **_k):
        raise AssertionError("must not hit the network when budget is 0")

    monkeypatch.setattr(geocode, "_lookup", no_network)
    df = pd.DataFrame(
        [{"address": "Əsəd Əhmədov küç.", "location_name": "", "district": "Yasamal",
          "lat": 40.379962, "lon": 49.808995}]
    )
    res = geocode.geocode_listings(df, max_lookups=0, progress_every=0)
    assert res.matched == 1
    assert res.listings.iloc[0]["geo_precision"] == "street"
