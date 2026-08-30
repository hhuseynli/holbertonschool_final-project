"""Tests for `bakuml.data.dedup` — broker re-post detection.

The headline test regenerates the full synthetic dataset (module-scoped
fixture, a few seconds) and checks that the detector *recovers the planted
ground truth*: >= 90 % of the pairs in ``truth_info["duplicate_map"]`` — which
were planted with 2 flipped pHash bits, +/-2 % price, ~20 m coordinate jitter
and a reworded description — while keeping false positives (reported pairs
whose two ids both belong to the non-duplicate base population) at <= 2 % of
all reported pairs. The full generator defaults are kept on purpose: the
recall/false-positive thresholds are statements about the whole planted
duplicate population, and shrinking the sample would also shrink the hard
negatives the FP bound is meant to exercise.

Unit tests cover the pHash Hamming distance (known bit flips, empty/invalid
hashes), the spatial and attribute blocking (far-apart or attribute-mismatched
twins are never compared), the either-signal decision rule, and the
union-find canonicalisation (earliest ``listed_month``, then lexicographic
``listing_id``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from bakuml.data.dedup import (
    PAIR_COLUMNS,
    PHASH_BITS,
    dedupe,
    find_duplicate_pairs,
    phash_hamming,
)
from bakuml.data.synthetic import generate_listings

# ~0.045 deg of latitude is ~5 km; ~0.0002 deg is ~22 m.
_FIVE_KM_LAT = 0.045
_TWENTY_M_LAT = 0.0002

_BASE_ROW = {
    "source": "synthetic",
    "lat": 40.3800,
    "lon": 49.8400,
    "price_azn": 170_000.0,
    "area_m2": 85.0,
    "price_azn_m2": 2000.0,
    "rooms": 3,
    "floor": 4,
    "building_floors": 9,
    "building_type": "old",
    "listed_month": "2024-05",
    "title": "3-otaqlı mənzil, 85.0 m², Nasimi",
    "description": (
        "3 otaqlı köhnə tikili mənzil, 85.0 m², 4/9 mərtəbə, Nasimi rayonu. "
        "Təmirli, sənədlər qaydasındadır."
    ),
    "image_phash": "a5a5a5a5a5a5a5a5",
    "district": "Nasimi",
}


def make_listings(overrides: list[dict]) -> pd.DataFrame:
    """Small hand-crafted listings frame; row i = _BASE_ROW + overrides[i]."""
    rows = []
    for i, ov in enumerate(overrides):
        row = {**_BASE_ROW, "listing_id": f"T-{i:03d}"}
        row.update(ov)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synth():
    """Full-size synthetic listings + planted truth (generated once)."""
    return generate_listings()


@pytest.fixture(scope="module")
def reported(synth):
    """Duplicate pairs found on the full synthetic dataset with defaults."""
    listings, _ = synth
    return find_duplicate_pairs(listings)


# --------------------------------------------------------------------------
# phash_hamming
# --------------------------------------------------------------------------


def test_phash_hamming_counts_flipped_bits():
    h = _BASE_ROW["image_phash"]
    assert phash_hamming(h, h) == 0
    v = int(h, 16)
    for bits in ([0], [3, 17], [1, 8, 63], [2, 9, 33, 47, 58]):
        w = v
        for b in bits:
            w ^= 1 << b
        assert phash_hamming(h, format(w, "016x")) == len(bits)


def test_phash_hamming_empty_or_invalid_is_max():
    h = _BASE_ROW["image_phash"]
    assert PHASH_BITS == 64
    assert phash_hamming("", h) == 64
    assert phash_hamming(h, "") == 64
    assert phash_hamming("", "") == 64
    assert phash_hamming("not-hex-zz", h) == 64
    assert phash_hamming(h, "-4") == 64
    assert phash_hamming(None, h) == 64  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Blocking behaviour
# --------------------------------------------------------------------------


def test_far_apart_identical_listings_are_not_compared():
    """Spatial blocking: perfect twins 5 km apart never become a pair."""
    frame = make_listings([{}, {"lat": _BASE_ROW["lat"] + _FIVE_KM_LAT}])
    pairs = find_duplicate_pairs(frame)
    assert list(pairs.columns) == PAIR_COLUMNS
    assert pairs.empty


def test_nearby_identical_listings_are_paired():
    """Positive control for the blocking test: same twins ~20 m apart."""
    frame = make_listings([{}, {"lat": _BASE_ROW["lat"] + _TWENTY_M_LAT}])
    pairs = find_duplicate_pairs(frame)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert {row["listing_id_a"], row["listing_id_b"]} == {"T-000", "T-001"}
    assert row["phash_hamming"] == 0
    assert row["text_sim"] == pytest.approx(1.0)


def test_attribute_blocking_vetoes_mismatched_flats():
    """Co-located identical text/photo but a hard attribute differs."""
    for ov in (
        {"rooms": 4},
        {"floor": 7},
        {"building_floors": 16},
        {"area_m2": 95.0},          # 10.5 % away: outside the 5 % area block
        {"price_azn_m2": 3000.0},   # 33 % away: outside price_rel_tol
    ):
        frame = make_listings([{}, ov])
        assert find_duplicate_pairs(frame).empty, ov


# --------------------------------------------------------------------------
# Decision rule: either signal suffices
# --------------------------------------------------------------------------


def test_phash_signal_alone_detects_fully_rewritten_repost():
    v = int(_BASE_ROW["image_phash"], 16) ^ (1 << 5) ^ (1 << 40) ^ (1 << 63)
    frame = make_listings(
        [
            {},
            {
                "image_phash": format(v, "016x"),
                "description": (
                    "Bu tamamilə başqa sözlərlə yazılmış elan mətnidir, "
                    "heç bir oxşarlıq yoxdur."
                ),
            },
        ]
    )
    pairs = find_duplicate_pairs(frame)
    assert len(pairs) == 1
    assert pairs.loc[0, "phash_hamming"] == 3
    assert pairs.loc[0, "text_sim"] < 0.80  # detected by the photo signal only


def test_text_signal_alone_detects_repost_without_photos():
    frame = make_listings([{"image_phash": ""}, {"image_phash": ""}])
    pairs = find_duplicate_pairs(frame)
    assert len(pairs) == 1
    assert pairs.loc[0, "phash_hamming"] == 64  # photo signal unavailable
    assert pairs.loc[0, "text_sim"] >= 0.80


def test_single_listing_yields_no_pairs_and_identity_dedupe():
    frame = make_listings([{}])
    pairs = find_duplicate_pairs(frame)
    assert list(pairs.columns) == PAIR_COLUMNS and pairs.empty
    clean, mapping = dedupe(frame)
    assert mapping == {}
    assert len(clean) == 1


# --------------------------------------------------------------------------
# Recovery of the planted ground truth (full synthetic dataset)
# --------------------------------------------------------------------------


def test_pair_frame_schema_and_decision_thresholds(reported):
    assert list(reported.columns) == PAIR_COLUMNS
    assert (reported["listing_id_a"] < reported["listing_id_b"]).all()
    assert not reported.duplicated(["listing_id_a", "listing_id_b"]).any()
    fired = (reported["phash_hamming"] <= 6) | (reported["text_sim"] >= 0.80)
    assert fired.all()


def test_recovers_planted_duplicate_pairs(synth, reported):
    listings, truth = synth
    truth_pairs = {frozenset(p) for p in truth["duplicate_map"].items()}
    got = {
        frozenset((a, b))
        for a, b in zip(reported["listing_id_a"], reported["listing_id_b"])
    }
    assert len(truth_pairs) == truth["n_duplicates"] > 1000

    recovered = len(truth_pairs & got)
    assert recovered >= 0.90 * len(truth_pairs)

    # False positives: reported pairs entirely inside the base (non-duplicate)
    # population — planted pairs always contain one SYN-DUP id.
    dup_ids = set(truth["duplicate_map"])
    false_pos = [p for p in got if not (p & dup_ids)]
    assert len(false_pos) <= 0.02 * len(got)


def test_dedupe_collapses_planted_duplicates(synth):
    listings, truth = synth
    clean, mapping = dedupe(listings)

    assert len(clean) + len(mapping) == len(listings)
    assert not clean["listing_id"].isin(set(mapping)).any()
    clean_ids = set(clean["listing_id"])
    assert set(mapping.values()) <= clean_ids
    assert not clean["listing_id"].duplicated().any()

    # The planted duplicate shares its source's listed_month, and base ids sort
    # before SYN-DUP ids, so the recovered canonical must be the source itself.
    exact = sum(1 for d, s in truth["duplicate_map"].items() if mapping.get(d) == s)
    assert exact >= 0.90 * truth["n_duplicates"]


# --------------------------------------------------------------------------
# Canonical selection
# --------------------------------------------------------------------------


def test_dedupe_canonical_earliest_month_then_lexicographic_id():
    frame = make_listings(
        [
            {"listing_id": "B", "listed_month": "2023-03"},
            {"listing_id": "A", "listed_month": "2023-05"},
            {"listing_id": "C", "listed_month": "2023-03"},
        ]
    )
    clean, mapping = dedupe(frame)
    # Earliest month wins (B, C over A); the month tie breaks on the id (B < C).
    assert mapping == {"A": "B", "C": "B"}
    assert clean["listing_id"].tolist() == ["B"]
