"""Offline tests for the bina.az scraping package (DESIGN.md section 1).

No network is touched anywhere: the parse helpers are pure functions, the
middlewares are exercised with hand-built ``Request``/``HtmlResponse``
objects, and the spider runs against ``tests/fixtures/bina_sample.html`` — a
hand-authored, structurally plausible snapshot of bina.az markup containing
listing cards (including deliberately broken ones), pagination and one
embedded detail-page section. ``runner.scrape_to_parquet`` is never invoked
(it starts a real crawl); only its import/signature is checked.
"""

from __future__ import annotations

import datetime as dt
import inspect
import io
import re
from pathlib import Path

import pytest
from scrapy.exceptions import DropItem
from scrapy.http import HtmlResponse, Request
from scrapy.utils.test import get_crawler

from bakuml.data.schema import LISTING_COLUMNS
from bakuml.data.scraping import settings as scraping_settings
from bakuml.data.scraping.items import ListingItem
from bakuml.data.scraping.middlewares import (
    USER_AGENTS,
    CloudflareRetryMiddleware,
    ProxyRotationMiddleware,
    is_cloudflare_challenge,
)
from bakuml.data.scraping.pipelines import (
    NormalizePipeline,
    PhashPipeline,
    parse_area,
    parse_floor,
    parse_listed_month,
    parse_price,
    parse_rooms,
    phash_hex,
)
from bakuml.data.scraping.spiders.bina import BinaSpider, safe_css

FIXTURE = Path(__file__).parent / "fixtures" / "bina_sample.html"
LIST_URL = "https://bina.az/alqi-satqi/menziller"
DETAIL_URL = "https://bina.az/items/4661288"
TODAY = dt.date(2026, 8, 29)


def fake_response(url=LIST_URL, body=None, status=200, headers=None):
    """HtmlResponse tied to a Request so response.meta / follow() work."""
    if body is None:
        body = FIXTURE.read_bytes()
    if isinstance(body, str):
        body = body.encode("utf-8")
    return HtmlResponse(
        url=url,
        body=body,
        status=status,
        headers=headers,
        request=Request(url=url),
        encoding="utf-8",
    )


# =========================================================================
# Pure parse functions
# =========================================================================

class TestParsePrice:
    def test_grouped_thousands(self):
        assert parse_price("150 000 AZN") == 150000.0

    def test_no_break_space_and_weird_whitespace(self):
        assert parse_price("98 500 AZN") == 98500.0
        assert parse_price("  150 000   AZN ") == 150000.0
        assert parse_price("1 250 000 AZN") == 1250000.0

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
        # Azerbaijani İ: "İyun".lower() is NOT "iyun" without special casing.
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
# Middlewares
# =========================================================================

def _cf_response(request, status=403, body=b"<html>Just a moment...</html>",
                 headers=None):
    return HtmlResponse(
        url=request.url, status=status, body=body, headers=headers,
        request=request, encoding="utf-8",
    )


class TestCloudflareDetection:
    def test_403_with_body_marker(self):
        req = Request(LIST_URL)
        assert is_cloudflare_challenge(_cf_response(req, status=403))

    def test_503_with_cf_mitigated_header(self):
        req = Request(LIST_URL)
        resp = _cf_response(
            req, status=503, body=b"<html></html>",
            headers={"cf-mitigated": "challenge"},
        )
        assert is_cloudflare_challenge(resp)

    def test_403_from_cloudflare_edge(self):
        req = Request(LIST_URL)
        resp = _cf_response(
            req, status=403, body=b"<html>error</html>",
            headers={"Server": "cloudflare"},
        )
        assert is_cloudflare_challenge(resp)

    def test_200_with_marker_is_not_a_challenge(self):
        req = Request(LIST_URL)
        assert not is_cloudflare_challenge(_cf_response(req, status=200))

    def test_plain_403_is_not_a_challenge(self):
        req = Request(LIST_URL)
        resp = _cf_response(req, status=403, body=b"<html>forbidden</html>")
        assert not is_cloudflare_challenge(resp)


class TestCloudflareRetryMiddleware:
    def _mw(self, **kwargs):
        kwargs.setdefault("max_retries", 4)
        kwargs.setdefault("backoff_base", 0.0)  # never sleep in tests
        return CloudflareRetryMiddleware(**kwargs)

    def test_retries_with_rotated_user_agent(self):
        mw = self._mw()
        spider = BinaSpider()
        req = Request(LIST_URL, headers={"User-Agent": USER_AGENTS[0]})
        out = mw.process_response(req, _cf_response(req), spider)
        assert isinstance(out, Request)
        assert out.meta["cf_retries"] == 1
        assert out.dont_filter is True
        ua1 = out.headers["User-Agent"].decode()
        assert ua1 == USER_AGENTS[1] and ua1 != USER_AGENTS[0]

        out2 = mw.process_response(out, _cf_response(out), spider)
        assert out2.meta["cf_retries"] == 2
        assert out2.headers["User-Agent"].decode() == USER_AGENTS[2]

    def test_gives_up_after_max_retries(self):
        mw = self._mw(max_retries=4)
        spider = BinaSpider()
        req = Request(LIST_URL, meta={"cf_retries": 4})
        resp = _cf_response(req)
        assert mw.process_response(req, resp, spider) is resp

    def test_normal_response_passes_through(self):
        mw = self._mw()
        req = Request(LIST_URL)
        resp = fake_response(body=b"<html>ok</html>")
        assert mw.process_response(req, resp, BinaSpider()) is resp

    def test_backoff_is_exponential(self):
        mw = CloudflareRetryMiddleware(backoff_base=2.0)
        delays = [mw.backoff_seconds(k) for k in range(4)]
        assert delays == [2.0, 4.0, 8.0, 16.0]

    def test_from_crawler_reads_settings(self):
        crawler = get_crawler(settings_dict={
            "CF_MAX_RETRIES": 2, "CF_BACKOFF_BASE_SECONDS": 0.5,
        })
        mw = CloudflareRetryMiddleware.from_crawler(crawler)
        assert mw.max_retries == 2
        assert mw.backoff_base == 0.5


class TestProxyRotationMiddleware:
    def test_empty_list_is_a_noop(self):
        mw = ProxyRotationMiddleware([])
        req = Request(LIST_URL)
        assert mw.process_request(req, BinaSpider()) is None
        assert "proxy" not in req.meta

    def test_round_robin(self):
        proxies = ["http://p1:8080", "http://p2:8080"]
        mw = ProxyRotationMiddleware(proxies)
        spider = BinaSpider()
        seen = []
        for _ in range(4):
            req = Request(LIST_URL)
            mw.process_request(req, spider)
            seen.append(req.meta["proxy"])
        assert seen == ["http://p1:8080", "http://p2:8080"] * 2

    def test_existing_proxy_not_overwritten(self):
        mw = ProxyRotationMiddleware(["http://p1:8080"])
        req = Request(LIST_URL, meta={"proxy": "http://pinned:1"})
        mw.process_request(req, BinaSpider())
        assert req.meta["proxy"] == "http://pinned:1"

    def test_from_crawler_reads_settings(self):
        crawler = get_crawler(settings_dict={"PROXY_LIST": ["http://p9:1"]})
        mw = ProxyRotationMiddleware.from_crawler(crawler)
        assert mw.proxies == ["http://p9:1"]


# =========================================================================
# Spider: cards, pagination, detail page — all against the local fixture
# =========================================================================

class TestSafeCss:
    def test_hit_miss_and_whitespace(self):
        resp = fake_response(
            body="<html><b>  x  </b><i>   </i></html>"
        )
        assert safe_css(resp, "b::text") == "x"
        assert safe_css(resp, "i::text") is None        # whitespace-only
        assert safe_css(resp, ".missing::text") is None  # no match


class TestBinaSpiderParse:
    @pytest.fixture()
    def results(self):
        spider = BinaSpider(max_pages=3)
        return spider, list(spider.parse(fake_response()))

    def test_follows_only_cards_with_links(self, results):
        spider, out = results
        detail = [r for r in out if r.callback == spider.parse_detail]
        # 4 cards in the fixture; the one without an <a> must be skipped
        assert sorted(r.url for r in detail) == [
            "https://bina.az/items/4661288",
            "https://bina.az/items/4661302",
            "https://bina.az/items/4661417",
        ]

    def test_card_fields_travel_via_cb_kwargs(self, results):
        spider, out = results
        detail = [r for r in out if r.callback == spider.parse_detail]
        by_url = {r.url: r.cb_kwargs["card"] for r in detail}

        card1 = by_url["https://bina.az/items/4661288"]
        assert parse_price(card1["price_raw"]) == 150000.0
        assert card1["rooms_raw"] == "3 otaqlı"
        assert card1["area_raw"] == "85.5 m²"
        assert card1["floor_raw"] == "4/9 mərtəbə"
        assert card1["district"] == "Yasamal r."
        assert parse_listed_month(card1["listed_date_raw"], TODAY) == "2026-08"

        # nbsp-separated price on card 2
        card2 = by_url["https://bina.az/items/4661302"]
        assert parse_price(card2["price_raw"]) == 98500.0

        # card 3 has no price block: the guard yields None, never raises
        card3 = by_url["https://bina.az/items/4661417"]
        assert card3["price_raw"] is None
        assert card3["area_raw"] == "38 m²"
        assert card3["floor_raw"] is None

    def test_pagination_respects_max_pages(self, results):
        spider, out = results
        pages = [r for r in out if r.callback == spider.parse]
        assert len(pages) == 1
        assert pages[0].url == "https://bina.az/alqi-satqi/menziller?page=2"
        assert pages[0].meta["page"] == 2

        single = BinaSpider(max_pages=1)
        out1 = list(single.parse(fake_response()))
        assert [r for r in out1 if r.callback == single.parse] == []


class TestBinaSpiderParseDetail:
    @pytest.fixture()
    def item(self):
        spider = BinaSpider()
        items = list(spider.parse_detail(fake_response(url=DETAIL_URL), card={}))
        assert len(items) == 1
        return items[0]

    def test_identity_and_price(self, item):
        assert isinstance(item, ListingItem)
        assert item["url"] == DETAIL_URL
        assert item["listing_id"] == "4661288"
        assert item["price_raw"] == "150 000 AZN"

    def test_properties_table(self, item):
        assert item["area_raw"] == "85.5 m²"
        assert item["rooms_raw"] == "3"
        assert item["floor_raw"] == "4/9"
        assert item["building_type_raw"] == "Yeni tikili"

    def test_geo_text_and_photos(self, item):
        assert item["lat"] == "40.3892"
        assert item["lon"] == "49.8156"
        assert item["district"] == "Yasamal r."
        assert item["title"].startswith("3 otaqlı")
        assert "mənzil satılır" in item["description"]
        assert item["listed_date_raw"] == "Yerləşdirilib: 28 Avqust 2026"
        assert len(item["image_urls"]) == 2

    def test_card_values_fill_detail_gaps(self):
        spider = BinaSpider()
        bare = fake_response(url="https://bina.az/items/999",
                             body=b"<html><body>empty</body></html>")
        card = {"price_raw": "75 000 AZN", "area_raw": "55 m²",
                "district": "Binəqədi r.", "rooms_raw": "2 otaqlı",
                "floor_raw": None, "listed_date_raw": "dünən"}
        item = next(iter(spider.parse_detail(bare, card=card)))
        assert item["price_raw"] == "75 000 AZN"
        assert item["area_raw"] == "55 m²"
        assert item["district"] == "Binəqədi r."
        assert item["listed_date_raw"] == "dünən"
        assert item["listing_id"] == "999"


# =========================================================================
# Pipelines: normalisation to the canonical schema
# =========================================================================

def _minimal_item(**overrides) -> ListingItem:
    base = {
        "listing_id": "1",
        "url": "https://bina.az/items/1",
        "price_raw": "100 000 AZN",
        "area_raw": "50 m²",
        "lat": "40.40",
        "lon": "49.80",
        "listed_date_raw": "bugün",
    }
    base.update(overrides)
    item = ListingItem()
    for key, value in base.items():
        if value is not None:
            item[key] = value
    return item


class TestNormalizePipeline:
    def test_full_roundtrip_from_fixture(self):
        spider = BinaSpider()
        raw = next(iter(spider.parse_detail(fake_response(url=DETAIL_URL), card={})))
        row = NormalizePipeline(today=TODAY).process_item(raw, spider)
        row = PhashPipeline().process_item(row, spider)

        assert list(row) == list(LISTING_COLUMNS)
        assert row["listing_id"] == "BINA-4661288"
        assert row["source"] == "bina.az"
        assert row["price_azn"] == 150000.0
        assert row["area_m2"] == 85.5
        assert row["price_azn_m2"] == pytest.approx(150000.0 / 85.5, abs=0.05)
        assert row["rooms"] == 3
        assert (row["floor"], row["building_floors"]) == (4, 9)
        assert row["building_type"] == "new"
        assert row["lat"] == pytest.approx(40.3892)
        assert row["lon"] == pytest.approx(49.8156)
        assert row["listed_month"] == "2026-08"
        assert row["image_phash"] == ""  # offline: no photo bytes
        assert row["district"] == "Yasamal r."

    def test_drops_when_critical_fields_missing(self):
        pipeline = NormalizePipeline(today=TODAY)
        for overrides in (
            {"price_raw": None},
            {"price_raw": "Qiymət yoxdur"},
            {"area_raw": None},
            {"lat": None},
            {"lon": "not-a-number"},
        ):
            with pytest.raises(DropItem):
                pipeline.process_item(_minimal_item(**overrides))

    def test_sentinels_for_optional_fields(self):
        row = NormalizePipeline(today=TODAY).process_item(_minimal_item())
        assert row["rooms"] == 0
        assert row["floor"] == 0
        assert row["building_floors"] == 0
        assert row["building_type"] == "old"
        assert row["title"] == ""
        assert row["description"] == ""
        assert row["district"] == ""
        assert row["listed_month"] == "2026-08"  # "bugün" vs injected today

    def test_unparseable_date_falls_back_to_scrape_month(self):
        row = NormalizePipeline(today=TODAY).process_item(
            _minimal_item(listed_date_raw=None)
        )
        assert row["listed_month"] == "2026-08"

    def test_building_type_mapping(self):
        pipeline = NormalizePipeline(today=TODAY)
        new = pipeline.process_item(_minimal_item(building_type_raw="Yeni tikili"))
        old = pipeline.process_item(_minimal_item(building_type_raw="Köhnə tikili"))
        assert new["building_type"] == "new"
        assert old["building_type"] == "old"


class TestPhashPipeline:
    def test_fills_hash_from_attached_bytes(self):
        from PIL import Image

        img = Image.new("RGB", (16, 16), (200, 30, 90))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        row = {"image_phash": "", "image_bytes": buf.getvalue()}
        out = PhashPipeline().process_item(row)
        assert re.fullmatch(r"[0-9a-f]{16}", out["image_phash"])
        assert "image_bytes" not in out  # transient key removed

    def test_offline_degrades_to_empty_hash(self):
        out = PhashPipeline().process_item({"image_phash": ""})
        assert out["image_phash"] == ""

    def test_bad_bytes_degrade_to_empty_hash(self):
        row = {"image_phash": "", "image_bytes": b"junk"}
        assert PhashPipeline().process_item(row)["image_phash"] == ""


# =========================================================================
# Settings contract + runner surface (never actually crawled)
# =========================================================================

class TestSettingsContract:
    def test_politeness(self):
        assert scraping_settings.ROBOTSTXT_OBEY is True
        assert scraping_settings.AUTOTHROTTLE_ENABLED is True
        assert scraping_settings.CONCURRENT_REQUESTS == 2
        assert scraping_settings.DOWNLOAD_DELAY >= 1.0

    def test_middlewares_and_pipelines_registered(self):
        mws = scraping_settings.DOWNLOADER_MIDDLEWARES
        assert "bakuml.data.scraping.middlewares.CloudflareRetryMiddleware" in mws
        assert "bakuml.data.scraping.middlewares.ProxyRotationMiddleware" in mws
        pipes = scraping_settings.ITEM_PIPELINES
        norm = "bakuml.data.scraping.pipelines.NormalizePipeline"
        ph = "bakuml.data.scraping.pipelines.PhashPipeline"
        assert pipes[norm] < pipes[ph]  # normalise first, then hash

    def test_cf_and_proxy_defaults(self):
        assert scraping_settings.CF_MAX_RETRIES == 4
        assert scraping_settings.PROXY_LIST == []
        assert 503 not in scraping_settings.RETRY_HTTP_CODES


def test_runner_signature_only():
    from bakuml.data.scraping.runner import scrape_to_parquet

    params = inspect.signature(scrape_to_parquet).parameters
    assert list(params) == ["out_path", "max_pages"]
    assert params["max_pages"].default == 5
