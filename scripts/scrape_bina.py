#!/usr/bin/env python3
"""Live bina.az scrape (requires network; respects robots.txt + autothrottle).

Set the PROXY_LIST environment variable (comma-separated http proxies) to
enable proxy rotation. Cloudflare challenges are retried with rotating
user agents; heavy challenges may still require a residential proxy pool.
"""

import argparse

from bakuml import config
from bakuml.data.scraping.runner import scrape_to_parquet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pages", type=int, default=5)
    ap.add_argument("--out", default=str(config.RAW_DIR / "bina_listings.parquet"))
    args = ap.parse_args()

    config.ensure_dirs()
    out = scrape_to_parquet(args.out, max_pages=args.max_pages)
    print(f"scraped listings -> {out}")


if __name__ == "__main__":
    main()
