#!/usr/bin/env python3
"""Fetch bina.az listings via GraphQL and write to parquet."""

import argparse
import logging

from bakuml import config
from bakuml.data.scraping.runner import scrape_to_parquet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pages", type=int, default=2400,
                    help="Max pages to fetch (25 listings/page, 2400 = ~55K)")
    ap.add_argument("--out", default=str(config.ARTIFACTS_DIR / "bina_full.parquet"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.ensure_dirs()
    out = scrape_to_parquet(args.out, max_pages=args.max_pages)
    print(f"scraped listings -> {out}")


if __name__ == "__main__":
    main()
