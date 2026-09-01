"""Fetch bina.az listings via GraphQL and write parquet.

``scrape_to_parquet`` is the only piece of the scraping package that touches
the network, and it is **never invoked by tests** — the offline test-suite
exercises the parse functions and normalisation against fixtures. Typical use::

    from bakuml.data.scraping.runner import scrape_to_parquet
    scrape_to_parquet(config.RAW_DIR / "bina_listings.parquet", max_pages=5)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from bakuml.data.schema import LISTING_COLUMNS
from bakuml.data.scraping.client import fetch_listings
from bakuml.data.scraping.pipelines import normalize_graphql_node

logger = logging.getLogger(__name__)


def scrape_to_parquet(out_path: Path, max_pages: int = 5) -> Path:
    """Fetch bina.az listings (up to ``max_pages`` pages) into a parquet file.

    Returns ``out_path``. The file always has the full canonical column set,
    even when the fetch yields nothing.
    """
    out_path = Path(out_path)

    raw_nodes = fetch_listings(max_pages=max_pages)
    logger.info("Fetched %d raw listings from bina.az", len(raw_nodes))

    rows: list[dict] = []
    skipped = 0
    for node in raw_nodes:
        row = normalize_graphql_node(node)
        if row is not None:
            rows.append(row)
        else:
            skipped += 1

    if skipped:
        logger.info("Skipped %d listings with missing critical fields", skipped)

    frame = pd.DataFrame(rows, columns=list(LISTING_COLUMNS))
    frame = frame.astype(LISTING_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    logger.info("Wrote %d listings to %s", len(frame), out_path)
    return out_path
