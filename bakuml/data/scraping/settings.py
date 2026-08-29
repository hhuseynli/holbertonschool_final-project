"""Scrapy project settings for the bina.az crawl.

The crawl is deliberately conservative — this is an academic project and
bina.az is a production site behind Cloudflare:

* ``ROBOTSTXT_OBEY = True`` — robots.txt is honoured unconditionally;
* ``AUTOTHROTTLE_ENABLED = True`` with a target concurrency of 1 — request
  rate adapts to observed server latency;
* ``CONCURRENT_REQUESTS = 2`` and a randomised multi-second
  ``DOWNLOAD_DELAY`` — at most a couple of in-flight requests, ever.

Cloudflare handling is delegated to
:class:`bakuml.data.scraping.middlewares.CloudflareRetryMiddleware`
(``CF_MAX_RETRIES`` / ``CF_BACKOFF_BASE_SECONDS``); 503 is therefore removed
from scrapy's stock ``RETRY_HTTP_CODES`` so the two retry layers never fight
over the same response. ``PROXY_LIST`` is empty by default — direct polite
crawling; operators may supply proxies at runtime.
"""

from bakuml.data.scraping.middlewares import USER_AGENTS

BOT_NAME = "bakuml_bina"

SPIDER_MODULES = ["bakuml.data.scraping.spiders"]
NEWSPIDER_MODULE = "bakuml.data.scraping.spiders"

# --------------------------------------------------------------------------
# Politeness
# --------------------------------------------------------------------------
ROBOTSTXT_OBEY = True
CONCURRENT_REQUESTS = 2
CONCURRENT_REQUESTS_PER_DOMAIN = 2
DOWNLOAD_DELAY = 3.0
RANDOMIZE_DOWNLOAD_DELAY = True

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 3.0
AUTOTHROTTLE_MAX_DELAY = 60.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

# --------------------------------------------------------------------------
# Identity / headers
# --------------------------------------------------------------------------
USER_AGENT = USER_AGENTS[0]
DEFAULT_REQUEST_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "az,ru;q=0.8,en;q=0.6",
}

# --------------------------------------------------------------------------
# Cloudflare retry + proxy rotation (see middlewares.py)
# --------------------------------------------------------------------------
CF_MAX_RETRIES = 4
CF_BACKOFF_BASE_SECONDS = 2.0
PROXY_LIST: list[str] = []

DOWNLOADER_MIDDLEWARES = {
    "bakuml.data.scraping.middlewares.ProxyRotationMiddleware": 350,
    "bakuml.data.scraping.middlewares.CloudflareRetryMiddleware": 543,
}

# 503 is owned by CloudflareRetryMiddleware; keep scrapy's RetryMiddleware
# for genuine transient errors only.
RETRY_HTTP_CODES = [500, 502, 504, 522, 524, 408, 429]

# --------------------------------------------------------------------------
# Item pipelines: normalise to the canonical schema, then hash photos.
# --------------------------------------------------------------------------
ITEM_PIPELINES = {
    "bakuml.data.scraping.pipelines.NormalizePipeline": 300,
    "bakuml.data.scraping.pipelines.PhashPipeline": 400,
}

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
FEED_EXPORT_ENCODING = "utf-8"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
TELNETCONSOLE_ENABLED = False
COOKIES_ENABLED = True  # Cloudflare clearance cookies must persist
