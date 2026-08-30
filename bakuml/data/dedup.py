"""Broker re-post (duplicate) detection for real-estate listings.

Motivation
----------
On bina.az the same flat is routinely advertised several times by competing
brokers: the cover photos are re-used (re-cropped / re-encoded), the ad text
is lightly reworded, and the price is nudged by a percent or two. Left in the
data, these re-posts overweight broker-heavy market segments in every
downstream estimate — cell-month medians, the Spatial DiD, the forecasts — so
deduplication runs immediately after ingestion, on the raw listings table.

Method
------
Two complementary similarity signals are evaluated on a *blocked* candidate
set; a candidate pair is declared a duplicate when **either** one fires
(brokers may swap the photo set or rewrite the text, but rarely both):

1. **Perceptual-hash Hamming distance** (`phash_hamming`). Listings carry a
   64-bit pHash of the cover photo as a 16-char hex string. Re-encoding,
   resizing or mild re-cropping of the same photo flips only a few of the 64
   bits, while unrelated photos differ in ~32 bits on average, so a small
   Hamming threshold (default 6) separates the two regimes cleanly. Missing
   or malformed hashes are assigned the maximally-distant value 64, which
   simply disables the photo signal for that pair.

2. **Character n-gram TF-IDF cosine.** Reworded Azerbaijani ad texts still
   share most of their character 3–5-grams. ``TfidfVectorizer`` with
   ``analyzer="char_wb"`` is robust to word reordering, agglutinative
   morphology and typos, and the IDF weighting emphasises the listing-specific
   tokens (area, floor, price digits) over boilerplate phrases, which keeps
   distinct-but-formulaic ads below the threshold (default cosine >= 0.80).
   The vectorizer is fit only on the descriptions of listings that appear in
   candidate pairs, and cosine similarity is computed pair-by-pair via
   ``linear_kernel`` on small row batches — never as a full n x n matrix.

Blocking — avoiding O(n^2)
--------------------------
Comparing all listing pairs is quadratic and unnecessary: a genuine re-post
advertises the *same physical flat*. Candidates are therefore generated with
cheap, high-recall blocking keys:

* a square planar grid with ``max_dist_m``-sized cells over (lat, lon); each
  listing is compared only against listings in its own and the 8 surrounding
  cells. Because the cell edge equals ``max_dist_m``, any two points within
  ``max_dist_m`` of each other necessarily land in the same or adjacent
  cells, so the grid loses no true pair;
* identical ``rooms``, ``floor`` and ``building_floors`` — physical
  attributes of the flat that a re-post cannot change (brokers reword text
  and nudge price, but the flat stays on its floor). Blocking on them is
  essential precision-wise: ad texts are highly formulaic, so two *different*
  same-sized flats in the same building can produce n-gram cosines above any
  usable threshold; requiring the immutable attributes to agree removes those
  collisions while costing no recall;
* ``area_m2`` within 5 % (symmetric relative difference);
* ``price_azn_m2`` within ``price_rel_tol`` (re-posts nudge price by ~2 %);
* finally, exact haversine distance <= ``max_dist_m``.

Consolidation
-------------
`dedupe` treats duplicate pairs as edges of an undirected graph and collapses
each connected component with union-find onto one canonical listing: the one
with the earliest ``listed_month`` ("YYYY-MM" strings order correctly
lexicographically), ties broken by the lexicographically smallest
``listing_id``. All other members are dropped and reported in a
``dup_id -> canonical_id`` map so provenance survives the cleaning step.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from bakuml.config import EARTH_RADIUS_KM
from bakuml.geo import haversine_km

#: Columns of the frame returned by :func:`find_duplicate_pairs`.
PAIR_COLUMNS = ["listing_id_a", "listing_id_b", "phash_hamming", "text_sim"]

#: pHash width in bits; also the sentinel distance for missing/invalid hashes.
PHASH_BITS = 64

#: Symmetric relative tolerance on area_m2 used for blocking ("within 5 %").
AREA_REL_TOL = 0.05

# Metres per degree of latitude on the WGS84-ish sphere used by bakuml.geo.
_M_PER_DEG_LAT = np.pi / 180.0 * EARTH_RADIUS_KM * 1000.0

# Row batch used when computing pairwise cosines through linear_kernel.
_COSINE_BATCH = 512


# --------------------------------------------------------------------------
# Perceptual hash distance
# --------------------------------------------------------------------------


def _phash_to_int(h: object) -> int:
    """Parse a hex pHash string to an int; return -1 when empty or invalid.

    Anything that is not a non-empty string encoding a non-negative 64-bit
    hex value counts as invalid (None, NaN, "", non-hex characters, values
    outside [0, 2^64)).
    """
    if not isinstance(h, str) or not h:
        return -1
    try:
        val = int(h, 16)
    except ValueError:
        return -1
    if not 0 <= val < 2**PHASH_BITS:
        return -1
    return val


def phash_hamming(a: str, b: str) -> int:
    """Hamming distance between two 64-bit hex perceptual hashes.

    Returns ``64`` (the maximum possible distance) when either hash is
    empty or invalid, so pairs without usable photos never match on the
    photo signal.
    """
    va, vb = _phash_to_int(a), _phash_to_int(b)
    if va < 0 or vb < 0:
        return PHASH_BITS
    return (va ^ vb).bit_count()


# --------------------------------------------------------------------------
# Candidate generation (blocking)
# --------------------------------------------------------------------------


def _grid_candidate_pairs(
    lat: np.ndarray, lon: np.ndarray, max_dist_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """All index pairs (i, j), i != j, whose grid cells are identical or
    adjacent on a square planar grid with ``max_dist_m``-sized cells.

    Each unordered pair is produced exactly once: within-cell pairs come
    from the upper triangle, cross-cell pairs from a half stencil of the
    8-neighbourhood ((+1,0), (0,+1), (+1,+1), (+1,-1)) whose reflections
    cover the remaining four offsets.
    """
    finite = np.isfinite(lat) & np.isfinite(lon)
    # Use the maximum |latitude| so metres-per-degree-longitude is a
    # guaranteed UNDER-estimate for every row: planar dx then never exceeds
    # the true east-west distance, so no pair within max_dist_m can land in
    # non-adjacent buckets (the documented no-loss guarantee).
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(np.abs(lat[finite]).max()))
    gx = np.full(lat.shape, np.iinfo(np.int64).min, dtype=np.int64)
    gy = gx.copy()
    gx[finite] = np.floor(lon[finite] * m_per_deg_lon / max_dist_m).astype(np.int64)
    gy[finite] = np.floor(lat[finite] * _M_PER_DEG_LAT / max_dist_m).astype(np.int64)

    buckets: dict[tuple[int, int], list[int]] = {}
    for i in np.flatnonzero(finite).tolist():
        buckets.setdefault((int(gx[i]), int(gy[i])), []).append(i)
    arrays = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}

    ia_parts: list[np.ndarray] = []
    ib_parts: list[np.ndarray] = []
    for (bx, by), idx in arrays.items():
        if len(idx) > 1:  # within-bucket combinations
            r, c = np.triu_indices(len(idx), k=1)
            ia_parts.append(idx[r])
            ib_parts.append(idx[c])
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):  # half stencil
            nb = arrays.get((bx + dx, by + dy))
            if nb is not None:
                ia_parts.append(np.repeat(idx, len(nb)))
                ib_parts.append(np.tile(nb, len(idx)))

    if not ia_parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(ia_parts), np.concatenate(ib_parts)


def _pairwise_cosine(
    tfidf, rows_a: np.ndarray, rows_b: np.ndarray, batch: int = _COSINE_BATCH
) -> np.ndarray:
    """Cosine similarity for the given row pairs of an L2-normalised TF-IDF
    matrix, via ``linear_kernel`` on small aligned batches (the diagonal of
    each batch kernel), never materialising the full similarity matrix.
    """
    sims = np.empty(len(rows_a), dtype=np.float64)
    for s in range(0, len(rows_a), batch):
        e = min(s + batch, len(rows_a))
        sims[s:e] = np.diagonal(linear_kernel(tfidf[rows_a[s:e]], tfidf[rows_b[s:e]]))
    return sims


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id_a": pd.Series(dtype=str),
            "listing_id_b": pd.Series(dtype=str),
            "phash_hamming": pd.Series(dtype="int64"),
            "text_sim": pd.Series(dtype="float64"),
        }
    )


def find_duplicate_pairs(
    listings: pd.DataFrame,
    *,
    phash_max_hamming: int = 6,
    text_sim_min: float = 0.80,
    max_dist_m: float = 300.0,
    price_rel_tol: float = 0.10,
) -> pd.DataFrame:
    """Detect broker re-post pairs among listings.

    Candidates are blocked spatially (same/adjacent ``max_dist_m`` grid
    cells, then exact haversine <= ``max_dist_m``) and on attributes (equal
    ``rooms``, ``floor`` and ``building_floors``; ``area_m2`` within 5 %;
    ``price_azn_m2`` within ``price_rel_tol``). A candidate is a duplicate when
    ``phash_hamming <= phash_max_hamming`` **or** the TF-IDF char-(3,5)-gram
    cosine of the descriptions is ``>= text_sim_min``.

    Returns a frame with columns ``[listing_id_a, listing_id_b,
    phash_hamming, text_sim]``, each unordered pair reported once with
    ``listing_id_a < listing_id_b``, sorted for determinism.
    """
    if len(listings) < 2:
        return _empty_pairs()

    ids = np.asarray(listings["listing_id"].astype(object).to_numpy(), dtype=object)
    lat = listings["lat"].to_numpy(dtype=np.float64)
    lon = listings["lon"].to_numpy(dtype=np.float64)
    rooms = listings["rooms"].to_numpy()
    floor = listings["floor"].to_numpy()
    bfloors = listings["building_floors"].to_numpy()
    area = listings["area_m2"].to_numpy(dtype=np.float64)
    price = listings["price_azn_m2"].to_numpy(dtype=np.float64)

    # ---- spatial blocking on the rounded grid --------------------------- #
    ia, ib = _grid_candidate_pairs(lat, lon, max_dist_m)

    # ---- attribute blocking (cheapest filters first) --------------------- #
    keep = (
        (rooms[ia] == rooms[ib])
        & (floor[ia] == floor[ib])
        & (bfloors[ia] == bfloors[ib])
    )
    ia, ib = ia[keep], ib[keep]

    pa, pb = price[ia], price[ib]
    keep = np.abs(pa - pb) <= price_rel_tol * np.maximum(pa, pb)
    ia, ib = ia[keep], ib[keep]

    aa, ab = area[ia], area[ib]
    keep = np.abs(aa - ab) <= AREA_REL_TOL * np.maximum(aa, ab)
    ia, ib = ia[keep], ib[keep]

    keep = haversine_km(lat[ia], lon[ia], lat[ib], lon[ib]) * 1000.0 <= max_dist_m
    ia, ib = ia[keep], ib[keep]

    if len(ia) == 0:
        return _empty_pairs()

    # ---- signal 1: pHash Hamming distance (vectorised popcount) ---------- #
    parsed = [_phash_to_int(h) for h in listings["image_phash"]]
    ph_ok = np.array([v >= 0 for v in parsed], dtype=bool)
    ph_val = np.array([max(v, 0) for v in parsed], dtype=np.uint64)
    ham = np.bitwise_count(ph_val[ia] ^ ph_val[ib]).astype(np.int64)
    ham[~(ph_ok[ia] & ph_ok[ib])] = PHASH_BITS

    # ---- signal 2: TF-IDF char-ngram cosine on candidate descriptions ---- #
    cand = np.unique(np.concatenate([ia, ib]))
    pos = np.full(len(listings), -1, dtype=np.int64)
    pos[cand] = np.arange(len(cand))
    docs = listings["description"].fillna("").astype(str).to_numpy()[cand].tolist()
    try:
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(docs)
    except ValueError:  # empty vocabulary: every candidate description blank
        sims = np.zeros(len(ia), dtype=np.float64)
    else:
        sims = _pairwise_cosine(tfidf, pos[ia], pos[ib])

    # ---- decision: either signal fires ----------------------------------- #
    is_dup = (ham <= phash_max_hamming) | (sims >= text_sim_min)
    ia, ib, ham, sims = ia[is_dup], ib[is_dup], ham[is_dup], sims[is_dup]

    id_a, id_b = ids[ia], ids[ib]
    swap = id_a > id_b
    id_a, id_b = np.where(swap, id_b, id_a), np.where(swap, id_a, id_b)
    out = pd.DataFrame(
        {
            "listing_id_a": id_a,
            "listing_id_b": id_b,
            "phash_hamming": ham,
            "text_sim": sims,
        }
    )
    return out.sort_values(["listing_id_a", "listing_id_b"], ignore_index=True)


# --------------------------------------------------------------------------
# Consolidation
# --------------------------------------------------------------------------


def dedupe(
    listings: pd.DataFrame,
    *,
    phash_max_hamming: int = 6,
    text_sim_min: float = 0.80,
    max_dist_m: float = 300.0,
    price_rel_tol: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Remove broker re-posts, keeping one canonical listing per flat.

    Duplicate pairs (from :func:`find_duplicate_pairs`) are edges of an
    undirected graph; union-find groups them into connected components. The
    canonical member of each component is the listing with the earliest
    ``listed_month``, ties broken by lexicographically smallest
    ``listing_id``. Every other member is dropped.

    Returns ``(clean_frame, mapping)`` where ``clean_frame`` is ``listings``
    without the duplicate rows (original order, index reset) and ``mapping``
    maps each removed ``dup_id`` to its ``canonical_id``.
    """
    pairs = find_duplicate_pairs(
        listings,
        phash_max_hamming=phash_max_hamming,
        text_sim_min=text_sim_min,
        max_dist_m=max_dist_m,
        price_rel_tol=price_rel_tol,
    )

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(pairs["listing_id_a"], pairs["listing_id_b"]):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    components: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        components[find(node)].append(node)

    month = dict(zip(listings["listing_id"], listings["listed_month"]))
    mapping: dict[str, str] = {}
    for members in components.values():
        canonical = min(members, key=lambda i: (month[i], i))
        for m in members:
            if m != canonical:
                mapping[m] = canonical

    clean = listings.loc[~listings["listing_id"].isin(list(mapping))]
    return clean.reset_index(drop=True), mapping
