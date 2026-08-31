#!/usr/bin/env python3
"""Harvest real listings from Azerbaijani platforms, counting each flat once.

Fetches public sale listings, normalises them into the canonical schema,
optionally hashes cover photos, then removes duplicate adverts - both broker
re-posts within one platform and the same flat listed on two - so every
real-world flat is counted exactly once.

Politeness: one request at a time with a delay, an honest User-Agent, a disk
cache so re-runs cost the sites nothing, and robots.txt obeyed (bina.az
allows its item catalogue and publishes a sitemap for crawlers).

Examples
--------
python scripts/harvest_real.py --limit 500
python scripts/harvest_real.py --limit 2000 --images     # + photo hashing
python scripts/harvest_real.py --limit 500 --no-cache    # force refetch
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from bakuml import config
from bakuml.data.sources import bina
from bakuml.data.sources.combine import build_real_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=500,
                    help="max adverts to fetch per source")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between requests to one host")
    ap.add_argument("--images", action="store_true",
                    help="download cover photos and compute perceptual "
                         "hashes (much better duplicate detection, one extra "
                         "request per advert)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the local page cache")
    ap.add_argument("--out", default=None,
                    help="output parquet (default: data/raw/real_listings.parquet)")
    args = ap.parse_args()

    config.ensure_dirs()
    frames: dict[str, pd.DataFrame] = {}

    print("=== bina.az ===", flush=True)
    res = bina.harvest(
        limit=args.limit, delay=args.delay, use_cache=not args.no_cache
    )
    print(res.summary(), flush=True)
    for note in res.notes:
        print(f"  note: {note}")
    frames["bina.az"] = res.listings

    if args.images and not res.listings.empty:
        from bakuml.data.sources.images import attach_phashes

        print("=== cover photo hashes ===", flush=True)
        frames["bina.az"] = attach_phashes(frames["bina.az"], delay=0.3)

    dataset = build_real_dataset(frames)
    print()
    print(dataset.report())

    if dataset.listings.empty:
        raise SystemExit("no listings harvested")

    out = args.out or str(config.RAW_DIR / "real_listings.parquet")
    dataset.listings.to_parquet(out, index=False)
    meta = {
        "per_source_raw": dataset.per_source_raw,
        "n_raw": dataset.n_raw,
        "n_unique": dataset.n_unique,
        "duplicates_removed": dataset.n_raw - dataset.n_unique,
        "within_source_pairs": dataset.within_source_pairs,
        "cross_source_pairs": dataset.cross_source_pairs,
        "image_hashes": bool(args.images),
        "geo_precision": sorted(
            dataset.listings.get("geo_precision", pd.Series(dtype=str)).unique().tolist()
        ),
    }
    meta_path = config.RAW_DIR / "real_listings_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}\nwrote {meta_path}")

    df = dataset.listings
    print("\n=== price distribution (AZN/m^2) ===")
    print(df["price_azn_m2"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_string())
    print("\n=== adverts per district ===")
    print(df["district"].value_counts().to_string())


if __name__ == "__main__":
    main()
