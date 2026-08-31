"""Combine several real sources into one deduplicated listing table.

The requirement is that a flat advertised more than once - re-posted by the
same broker, or listed on two platforms at the same time - is *counted
once*. That is what :func:`build_real_dataset` guarantees, and the count is
reported so the reduction is auditable rather than implicit.

Cross-platform duplicates are the harder case: the two adverts share no id,
their text is written independently, and their coordinates come from
different geocoding conventions. What they do share is the photographs, so
image hashing carries the work and text similarity is the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from bakuml.data.dedup import dedupe, find_duplicate_pairs
from bakuml.data.schema import LISTING_COLUMNS


@dataclass
class RealDataset:
    """A deduplicated multi-source dataset plus the audit trail."""

    listings: pd.DataFrame
    per_source_raw: dict[str, int] = field(default_factory=dict)
    duplicate_map: dict[str, str] = field(default_factory=dict)
    cross_source_pairs: int = 0
    within_source_pairs: int = 0

    @property
    def n_raw(self) -> int:
        return sum(self.per_source_raw.values())

    @property
    def n_unique(self) -> int:
        return len(self.listings)

    def report(self) -> str:
        lines = ["Real dataset", "-" * 44]
        for src, n in sorted(self.per_source_raw.items()):
            lines.append(f"  {src:<18} {n:>7,} adverts")
        removed = self.n_raw - self.n_unique
        pct = (100.0 * removed / self.n_raw) if self.n_raw else 0.0
        lines += [
            f"  {'TOTAL adverts':<18} {self.n_raw:>7,}",
            f"  {'unique flats':<18} {self.n_unique:>7,}",
            f"  {'duplicates removed':<18} {removed:>7,}  ({pct:.1f}%)",
            f"      within-source pairs: {self.within_source_pairs:,}",
            f"      cross-source pairs:  {self.cross_source_pairs:,}",
        ]
        return "\n".join(lines)


def build_real_dataset(
    frames: dict[str, pd.DataFrame],
    **dedup_kwargs,
) -> RealDataset:
    """Concatenate per-source frames and remove duplicate adverts.

    `frames` maps a source label to its listings. Each frame must carry the
    canonical columns; extra per-source columns are preserved where they do
    not collide. Every real-world flat appears exactly once in the result.
    """
    per_source = {src: len(df) for src, df in frames.items() if df is not None}
    usable = [df for df in frames.values() if df is not None and not df.empty]
    if not usable:
        return RealDataset(listings=pd.DataFrame(), per_source_raw=per_source)

    combined = pd.concat(usable, ignore_index=True)
    combined = combined.drop_duplicates(subset="listing_id", keep="first")
    combined = combined.reset_index(drop=True)

    # Inspect pairs before collapsing them, so the report can distinguish a
    # broker re-posting on one platform from the same flat on two.
    pairs = find_duplicate_pairs(combined, **dedup_kwargs)
    cross = within = 0
    if not pairs.empty:
        src_of = combined.set_index("listing_id")["source"]
        sa = pairs["listing_id_a"].map(src_of)
        sb = pairs["listing_id_b"].map(src_of)
        cross = int((sa != sb).sum())
        within = int((sa == sb).sum())

    clean, dup_map = dedupe(combined, **dedup_kwargs)
    return RealDataset(
        listings=clean.reset_index(drop=True),
        per_source_raw=per_source,
        duplicate_map=dup_map,
        cross_source_pairs=cross,
        within_source_pairs=within,
    )


def canonical_only(listings: pd.DataFrame) -> pd.DataFrame:
    """Reduce to exactly the canonical listing columns (pipeline input)."""
    missing = [c for c in LISTING_COLUMNS if c not in listings.columns]
    if missing:
        raise ValueError(f"real dataset missing canonical columns: {missing}")
    return listings[list(LISTING_COLUMNS)].copy()
