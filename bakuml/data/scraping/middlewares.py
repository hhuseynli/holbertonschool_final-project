"""Downloader middlewares for the bina.az crawl.

bina.az sits behind Cloudflare. When Cloudflare decides a client looks like a
bot it answers with an HTTP 403/503 interstitial ("Just a moment...") instead
of the real page. Two middlewares make the crawl resilient while staying
polite (the project also keeps ``ROBOTSTXT_OBEY=True``, autothrottle on and
2 concurrent requests — see ``settings.py``):

``CloudflareRetryMiddleware``
    Detects challenge responses (status 403/503 plus Cloudflare markers such
    as the ``cf-mitigated`` header or "Just a moment" in the body), waits an
    exponentially growing backoff, rotates the ``User-Agent`` header through a
    bundled list of realistic desktop UAs, and re-issues the request. After
    ``CF_MAX_RETRIES`` (default 4) failed attempts it gives up and lets the
    challenge response through, so the spider's guarded selectors simply
    yield nothing for that URL.

``ProxyRotationMiddleware``
    Round-robins ``request.meta["proxy"]`` over the ``PROXY_LIST`` setting.
    With an empty list (the default) it is a clean no-op.

Both middlewares are plain classes with injectable parameters so tests can
exercise them offline with fake ``Request``/``HtmlResponse`` objects.
"""

from __future__ import annotations

import itertools
import logging
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A small pool of realistic desktop user agents (Chrome / Firefox / Safari /
# Edge on Windows / macOS / Linux), current as of mid-2026. Rotating the UA on
# retry is often enough to pass a transient Cloudflare "managed challenge".
# ---------------------------------------------------------------------------
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:141.0) "
    "Gecko/20100101 Firefox/141.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:140.0) "
    "Gecko/20100101 Firefox/140.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:141.0) "
    "Gecko/20100101 Firefox/141.0",
]

# Byte markers (lower-cased) that identify a Cloudflare challenge page body.
CF_BODY_MARKERS: tuple[bytes, ...] = (
    b"just a moment",
    b"cf-browser-verification",
    b"_cf_chl_opt",
    b"challenge-platform",
    b"cf-chl",
)

#: Statuses Cloudflare uses for challenges / blocks.
CF_STATUSES = (403, 503)


def is_cloudflare_challenge(response) -> bool:
    """True when ``response`` looks like a Cloudflare challenge/block page.

    Requires a 403/503 status AND at least one Cloudflare marker: the
    ``cf-mitigated`` header, a known challenge string in the first 4 KiB of
    the body, or a ``Server: cloudflare`` header (a 403 served *by*
    Cloudflare's edge is a block, not the origin's answer).
    """
    if response.status not in CF_STATUSES:
        return False
    mitigated = (response.headers.get("cf-mitigated") or b"").lower()
    if b"challenge" in mitigated or b"block" in mitigated:
        return True
    body = (response.body or b"")[:4096].lower()
    if any(marker in body for marker in CF_BODY_MARKERS):
        return True
    server = (response.headers.get("Server") or b"").lower()
    return b"cloudflare" in server


class CloudflareRetryMiddleware:
    """Retry Cloudflare-challenged responses with backoff + UA rotation.

    Notes for the defense: the backoff uses ``time.sleep``, which blocks the
    Twisted reactor. That is a deliberate trade-off — this crawl runs at
    2 concurrent requests with multi-second polite delays anyway, and pausing
    *everything* while Cloudflare is rate-limiting us is exactly the desired
    behaviour. Tests pass ``backoff_base=0`` so no test ever sleeps.
    """

    def __init__(
        self,
        max_retries: int = 4,
        backoff_base: float = 2.0,
        user_agents: list[str] | None = None,
    ) -> None:
        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.user_agents = list(user_agents) if user_agents else list(USER_AGENTS)

    @classmethod
    def from_crawler(cls, crawler):
        s = crawler.settings
        return cls(
            max_retries=s.getint("CF_MAX_RETRIES", 4),
            backoff_base=s.getfloat("CF_BACKOFF_BASE_SECONDS", 2.0),
        )

    def backoff_seconds(self, retry_no: int) -> float:
        """Exponential backoff before retry number ``retry_no`` (0-based)."""
        return self.backoff_base * (2.0 ** retry_no)

    def process_response(self, request, response, spider):
        if not is_cloudflare_challenge(response):
            return response

        retries = request.meta.get("cf_retries", 0)
        log = spider.logger if spider is not None else logger
        if retries >= self.max_retries:
            log.warning(
                "Giving up on %s after %d Cloudflare retries (status %d)",
                request.url, retries, response.status,
            )
            return response

        delay = self.backoff_seconds(retries)
        log.info(
            "Cloudflare challenge on %s (status %d); retry %d/%d after %.1fs",
            request.url, response.status, retries + 1, self.max_retries, delay,
        )
        if delay > 0:
            time.sleep(delay)

        retryreq = request.copy()  # meta is shallow-copied by scrapy
        retryreq.meta["cf_retries"] = retries + 1
        # Deterministic rotation: retry k gets user_agents[k % len].
        new_ua = self.user_agents[(retries + 1) % len(self.user_agents)]
        retryreq.headers["User-Agent"] = new_ua
        retryreq.dont_filter = True
        return retryreq


class ProxyRotationMiddleware:
    """Round-robin outbound proxies from the ``PROXY_LIST`` setting.

    With an empty list the middleware never touches the request. A proxy
    already pinned in ``request.meta["proxy"]`` (e.g. by a retry) is
    respected, never overwritten.
    """

    def __init__(self, proxies: list[str] | None = None) -> None:
        self.proxies = [p for p in (proxies or []) if p]
        self._cycle = itertools.cycle(self.proxies) if self.proxies else None

    @classmethod
    def from_crawler(cls, crawler):
        return cls(proxies=crawler.settings.getlist("PROXY_LIST", []))

    def process_request(self, request, spider):
        if self._cycle is not None and "proxy" not in request.meta:
            request.meta["proxy"] = next(self._cycle)
        return None
