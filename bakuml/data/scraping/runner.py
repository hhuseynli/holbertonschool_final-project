"""Blocking entry point that runs the bina.az spider and writes parquet.

``scrape_to_parquet`` is the only piece of the scraping package that touches
the network, and it is **never invoked by tests** (DESIGN.md section 1) — the
offline test-suite exercises the parse functions, middlewares and spider
against a local HTML fixture instead. Typical use::

    from bakuml.data.scraping.runner import scrape_to_parquet
    scrape_to_parquet(config.RAW_DIR / "bina_listings.parquet", max_pages=5)

Items reaching the ``item_scraped`` signal have already passed through
``NormalizePipeline`` + ``PhashPipeline``, i.e. they are canonical
``schema.LISTING_COLUMNS`` dicts; this module only stacks them into a typed
DataFrame and writes it out.

Note: ``CrawlerProcess.start()`` runs the Twisted reactor, which can only be
started once per OS process — call this from a fresh process (e.g. the
``scripts/`` CLI), not from a long-lived notebook kernel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.settings import Settings

from bakuml.data.schema import LISTING_COLUMNS
from bakuml.data.scraping.spiders.bina import BinaSpider


def _project_settings() -> Settings:
    """Scrapy Settings populated from bakuml.data.scraping.settings."""
    settings = Settings()
    settings.setmodule("bakuml.data.scraping.settings", priority="project")
    return settings


def scrape_to_parquet(out_path: Path, max_pages: int = 5) -> Path:
    """Crawl bina.az (up to ``max_pages`` listing pages) into a parquet file.

    Returns ``out_path``. The file always has the full canonical column set,
    even when the crawl yields nothing (e.g. Cloudflare never lets us in).
    """
    out_path = Path(out_path)
    rows: list[dict] = []

    process = CrawlerProcess(_project_settings())
    crawler = process.create_crawler(BinaSpider)

    def _collect(item, response, spider):  # noqa: ARG001 - scrapy signal signature
        rows.append(dict(item))

    crawler.signals.connect(_collect, signal=signals.item_scraped)
    process.crawl(crawler, max_pages=max_pages)
    process.start()  # blocks until the crawl finishes

    frame = pd.DataFrame(rows, columns=list(LISTING_COLUMNS))
    frame = frame.astype(LISTING_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return out_path
