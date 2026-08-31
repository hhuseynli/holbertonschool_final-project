"""Shared HTTP plumbing for the real-data adapters.

Scraping someone else's site is a courtesy relationship, so the defaults
here are deliberately conservative:

* one request at a time, with a delay between them (``REQUEST_DELAY_S``);
* an honest ``User-Agent`` naming the project and its purpose, rather than
  a browser string pretending to be someone else;
* a disk cache, so re-running an experiment costs the site nothing;
* ``robots.txt`` consulted through :mod:`urllib.robotparser` and obeyed.

bina.az and arenda.az both allow their listing catalogues in robots.txt
(only login / session / saved-search endpoints are disallowed), and bina.az
publishes a sitemap of items specifically for crawlers.
"""

from __future__ import annotations

import hashlib
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from bakuml import config

#: Identifies the crawler honestly. A contact URL is the norm for academic
#: crawlers and lets a site operator ask us to stop.
USER_AGENT = (
    "BakuML/0.1 (Holberton School ML graduation project; "
    "spatiotemporal research; +https://github.com/hhuseynli/holbertonschool_final-project)"
)

#: Seconds between requests to the same host.
REQUEST_DELAY_S = 1.0

#: Where fetched pages are cached.
CACHE_DIR = config.DATA_DIR / "cache"

_ROBOTS: dict[str, urllib.robotparser.RobotFileParser] = {}
_LAST_REQUEST: dict[str, float] = {}


@dataclass
class HarvestResult:
    """What one adapter run produced."""

    listings: pd.DataFrame
    source: str
    fetched: int = 0
    from_cache: int = 0
    failed: int = 0
    skipped_by_robots: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.source}: {len(self.listings)} listings "
            f"({self.fetched} fetched, {self.from_cache} cached, "
            f"{self.failed} failed, {self.skipped_by_robots} robots-blocked)"
        )


def _robots_for(url: str) -> urllib.robotparser.RobotFileParser:
    """Fetch and parse a host's robots.txt.

    ``RobotFileParser.read()`` uses urllib, which does not honour this
    environment's HTTPS proxy and treats any fetch error as
    "disallow everything" - which would silently block every source. So the
    file is fetched with `requests` (proxy-aware) and handed to the parser.
    A genuinely missing robots.txt means no restrictions; an unreachable one
    is treated as restrictive, because we cannot show we are permitted.
    """
    host = urlparse(url).netloc
    if host not in _ROBOTS:
        rp = urllib.robotparser.RobotFileParser()
        try:
            resp = requests.get(
                f"https://{host}/robots.txt",
                headers={"User-Agent": USER_AGENT},
                timeout=15.0,
            )
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            elif resp.status_code in (401, 403):
                rp.disallow_all = True
            else:  # 404 and friends: nothing is disallowed
                rp.allow_all = True
        except requests.RequestException:
            rp.disallow_all = True
        _ROBOTS[host] = rp
    return _ROBOTS[host]


def robots_allows(url: str) -> bool:
    """Whether robots.txt permits fetching `url` with our user agent."""
    try:
        return _robots_for(url).can_fetch(USER_AGENT, url)
    except Exception:
        return True


def _cache_path(url: str) -> Path:
    host = urlparse(url).netloc.replace(":", "_")
    digest = hashlib.sha256(url.encode()).hexdigest()[:20]
    return CACHE_DIR / host / f"{digest}.html"


def polite_get(
    url: str,
    *,
    use_cache: bool = True,
    delay: float = REQUEST_DELAY_S,
    timeout: float = 25.0,
    check_robots: bool = True,
) -> tuple[str | None, bool]:
    """Fetch `url` politely.

    Returns ``(text, from_cache)``; ``text`` is None when the fetch was
    disallowed or failed.
    """
    if check_robots and not robots_allows(url):
        return None, False

    path = _cache_path(url)
    if use_cache and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace"), True

    host = urlparse(url).netloc
    elapsed = time.monotonic() - _LAST_REQUEST.get(host, 0.0)
    if elapsed < delay:
        time.sleep(delay - elapsed)
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "az,en;q=0.8",
            },
            timeout=timeout,
        )
    except requests.RequestException:
        _LAST_REQUEST[host] = time.monotonic()
        return None, False
    _LAST_REQUEST[host] = time.monotonic()
    if resp.status_code != 200:
        return None, False

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
    return resp.text, False
