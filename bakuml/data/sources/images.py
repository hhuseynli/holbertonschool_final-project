"""Perceptual hashes for listing cover photos.

Broker re-posts are the dominant duplicate mechanism on bina.az: the same
flat is advertised repeatedly, often by several agencies, with reworded text
and a shuffled photo set. Text similarity alone cannot separate a genuine
re-post from two similar flats in the same building; the *photographs* can,
because re-posts reuse the same source images.

The hash is `imagehash.phash` (DCT-based, 64-bit), which survives the
recompression and resizing that platforms apply. Hashes are cached on disk
keyed by image URL, so a re-run costs the CDN nothing.

Fetching images is opt-in: it is the expensive part of ingestion (one
request per listing) and the pipeline works without it, falling back to text
and attribute matching.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from bakuml import config
from bakuml.data.sources.base import USER_AGENT, REQUEST_DELAY_S

CACHE_PATH = config.DATA_DIR / "cache" / "phash.json"


def _load_cache() -> dict[str, str]:
    if CACHE_PATH.is_file():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def phash_url(url: str, *, cache: dict[str, str], delay: float = 0.3) -> str:
    """Perceptual hash of one image URL ("" on any failure)."""
    if not url:
        return ""
    if url in cache:
        return cache[url]
    import time

    import imagehash
    import requests
    from PIL import Image

    try:
        time.sleep(delay)
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=20.0
        )
        if resp.status_code != 200:
            cache[url] = ""
            return ""
        img = Image.open(io.BytesIO(resp.content))
        value = str(imagehash.phash(img))
    except Exception:
        value = ""
    cache[url] = value
    return value


def attach_phashes(
    listings: pd.DataFrame,
    *,
    column: str = "image_url",
    delay: float = 0.3,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Return a copy of `listings` with ``image_phash`` filled in.

    Rows whose image could not be fetched keep ``""``, which
    :func:`bakuml.data.dedup.phash_hamming` treats as "no information"
    rather than as a match.
    """
    if column not in listings.columns:
        return listings.copy()
    cache = _load_cache()
    out = listings.copy()
    hashes: list[str] = []
    try:
        for i, url in enumerate(out[column].fillna("").tolist(), 1):
            hashes.append(phash_url(url, cache=cache, delay=delay))
            if progress_every and i % progress_every == 0:
                got = sum(1 for h in hashes if h)
                print(f"  phash {i}/{len(out)} ({got} hashed)", flush=True)
    finally:
        _save_cache(cache)
    out["image_phash"] = hashes
    return out
