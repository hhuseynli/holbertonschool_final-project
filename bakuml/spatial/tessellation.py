"""Pluggable spatial tessellations - the unit of analysis is a choice, not a given.

The project's spec argues for Uber's H3 hexagons over administrative rayons,
and the argument is sound as far as it goes: equal-area cells with
equidistant neighbours beat polygons that blend elite blocks with industrial
outskirts (the Modifiable Areal Unit Problem, MAUP).

But hexagons only make the arbitrariness *uniform* - they do not remove it.
Measured on this project's own panel, a res-8 grid over Baku & Absheron:

* discards **37 %** of deduplicated listings, because 70 % of cell-months
  fall below ``MIN_LISTINGS_PER_CELL_MONTH``;
* leaves **68 %** of cells with under 12 months of price history, which is
  what makes autoregressive forecasting from those cells so unstable;
* spreads estimation variance unevenly - the per-cell-month median is built
  from 1 listing in the periphery and 18 in the centre.

A uniform grid over a radically non-uniform city is a poor fit. The fix is
not to swap one hard-coded geometry for another, but to make the unit of
analysis a *parameter* and choose it with the same leakage-proof validation
used everywhere else in the project (see ``scripts/maup_study.py``).

Three tessellations are provided:

``H3Tessellation``
    The spec baseline, at any resolution. No fitting required.

``AdaptiveKDTessellation``
    Recursive median splits of the listing point cloud, so every leaf holds
    roughly the same number of listings. Equal *sample size* instead of
    equal *area*: homogeneous estimation variance, far less data thrown
    away, and long price histories even in the periphery (where one big
    cell replaces many empty small ones). Splits use coordinates only -
    never prices - so no target information enters the geography.

``MarketRegionTessellation``
    Contiguity-constrained Ward clustering of fine H3 cells on price level
    and building mix, so boundaries follow *actual market discontinuities*
    (the coastline, a highway, the elite/industrial divide) rather than
    arbitrary geometry. This is the textbook answer to MAUP - data-driven
    regionalisation - and the only tessellation here that uses prices, so
    it is fitted strictly on a pre-cutoff training window to stay
    leakage-free.

Every tessellation exposes the same five operations the rest of the
pipeline needs - assign, centroid, neighbours, boundary, block - so
``bakuml.spatial``, ``bakuml.features``, ``bakuml.validation`` and the app
are agnostic to which one is active.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

import h3
import numpy as np
import pandas as pd

from bakuml import config
from bakuml.geo import haversine_km

# Metres per degree of latitude; longitude is scaled by cos(lat) so that
# "the wider dimension" is measured in comparable units.
_M_PER_DEG_LAT = np.pi / 180.0 * 6_371_008.8


def connected_components(
    cells: list[str], neighbours: "Callable[[str], Iterable[str]]"
) -> list[list[str]]:
    """Components of a contiguity graph, largest first.

    Shared by the market tessellation and the zone layer: the observed grid
    is not one blob (Sumgait, Alat and the Absheron villages are islands),
    and any clustering under a connectivity constraint must respect that.
    """
    remaining = set(cells)
    out: list[list[str]] = []
    while remaining:
        seed = min(remaining)
        remaining.discard(seed)
        comp, stack = [seed], [seed]
        while stack:
            node = stack.pop()
            for nb in neighbours(node):
                if nb in remaining:
                    remaining.discard(nb)
                    comp.append(nb)
                    stack.append(nb)
        out.append(sorted(comp))
    return sorted(out, key=len, reverse=True)


def apportion(sizes: list[int], total: int) -> list[int]:
    """Share `total` units across groups by size, at least 1 each.

    Largest-remainder apportionment, capped so no group is asked for more
    units than it has members. Returned by *index*, not keyed by object
    identity. When there are more groups than units the result necessarily
    sums above `total` - every disconnected group needs at least one unit,
    or it would be merged into a discontiguous one - so callers should treat
    the request as a target and report what was achieved.
    """
    n_groups = len(sizes)
    n_members = sum(sizes) or 1
    budget = max(total, n_groups)
    exact = [1 + (budget - n_groups) * sz / n_members for sz in sizes]
    floors = [int(np.floor(v)) for v in exact]
    short = budget - sum(floors)
    order = sorted(range(n_groups), key=lambda i: -(exact[i] - floors[i]))
    for i in order[: max(0, short)]:
        floors[i] += 1
    return [min(floors[i], sizes[i]) for i in range(n_groups)]


@runtime_checkable
class Tessellation(Protocol):
    """The five operations the pipeline needs from a spatial unit of analysis."""

    name: str

    def assign(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Cell id per coordinate pair (object array of str)."""

    def centroid(self, cell: str) -> tuple[float, float]:
        """(lat, lon) representative point of the cell."""

    def neighbours(self, cell: str, k: int = 1) -> tuple[str, ...]:
        """Cells within `k` contiguity steps, excluding `cell` itself."""

    def boundary(self, cell: str) -> list[tuple[float, float]]:
        """Polygon ring as [(lat, lon), ...] (unclosed; callers close it)."""

    def block(self, cell: str) -> str:
        """Coarse contiguous block id, for spatially blocked CV."""


class _BaseTessellation:
    """Shared helpers: k-ring expansion by BFS and cell bookkeeping."""

    name = "base"

    def __init__(self) -> None:
        self._nbr_cache: dict[tuple[str, int], tuple[str, ...]] = {}

    def neighbours(self, cell: str, k: int = 1) -> tuple[str, ...]:
        """BFS to depth `k` over the 1-step contiguity graph (memoised).

        Callers hit this once per panel cell per feature, so the result is
        cached: an instance-level dict rather than ``lru_cache``, which would
        key on ``self`` and keep every fitted tessellation alive.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        cache = getattr(self, "_nbr_cache", None)
        if cache is None:  # subclass skipped __init__
            cache = self._nbr_cache = {}
        hit = cache.get((cell, k))
        if hit is not None:
            return hit
        seen = {cell}
        frontier = deque([(cell, 0)])
        out: list[str] = []
        while frontier:
            node, depth = frontier.popleft()
            if depth >= k:
                continue
            for nb in self._neighbours1(node):
                if nb not in seen:
                    seen.add(nb)
                    out.append(nb)
                    frontier.append((nb, depth + 1))
        result = tuple(out)
        cache[(cell, k)] = result
        return result

    def _neighbours1(self, cell: str) -> tuple[str, ...]:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


# ---------------------------------------------------------------------------
# 1. H3 - the spec baseline
# ---------------------------------------------------------------------------


class H3Tessellation(_BaseTessellation):
    """Uber H3 hexagons at a fixed resolution (the project's spec baseline)."""

    def __init__(
        self,
        resolution: int = config.H3_RESOLUTION,
        block_resolution: int = config.H3_BLOCK_RESOLUTION,
    ):
        if block_resolution > resolution:
            raise ValueError("block_resolution must be coarser than resolution")
        super().__init__()
        self.resolution = resolution
        self.block_resolution = block_resolution
        self.name = f"h3_res{resolution}"

    # Fitting is a no-op: the grid is defined a priori, which is precisely
    # its appeal (reproducible, global) and its weakness (ignores the data).
    def fit(self, listings: pd.DataFrame, **_) -> "H3Tessellation":
        return self

    def assign(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return np.array(
            [h3.latlng_to_cell(float(a), float(o), self.resolution)
             for a, o in zip(np.asarray(lat), np.asarray(lon))],
            dtype=object,
        )

    def centroid(self, cell: str) -> tuple[float, float]:
        return h3.cell_to_latlng(cell)

    def _neighbours1(self, cell: str) -> tuple[str, ...]:
        return tuple(n for n in h3.grid_disk(cell, 1) if n != cell)

    def neighbours(self, cell: str, k: int = 1) -> tuple[str, ...]:
        # h3 gives k-rings analytically - cheaper than the generic BFS -
        # but still worth caching at panel scale.
        if k < 1:
            raise ValueError("k must be >= 1")
        hit = self._nbr_cache.get((cell, k))
        if hit is None:
            hit = tuple(n for n in h3.grid_disk(cell, k) if n != cell)
            self._nbr_cache[(cell, k)] = hit
        return hit

    def boundary(self, cell: str) -> list[tuple[float, float]]:
        return [tuple(v) for v in h3.cell_to_boundary(cell)]

    def block(self, cell: str) -> str:
        return h3.cell_to_parent(cell, self.block_resolution)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": "h3",
            "resolution": self.resolution,
            "block_resolution": self.block_resolution,
            "mean_cell_area_km2": round(h3.average_hexagon_area(self.resolution, unit="km^2"), 3),
        }


# ---------------------------------------------------------------------------
# 2. Adaptive KD-tree - equal sample size instead of equal area
# ---------------------------------------------------------------------------


@dataclass
class _KDLeaf:
    path: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    n: int
    #: Mean position of the listings inside the leaf. A periphery leaf can be
    #: a huge rectangle whose geometric centre sits in empty desert or open
    #: sea, kilometres from any advert, so distance features (to the centre,
    #: to metro, to Master Plan nodes) must be measured from where the
    #: listings actually are. Falls back to the box centre for empty leaves.
    lat_mean: float = float("nan")
    lon_mean: float = float("nan")


class AdaptiveKDTessellation(_BaseTessellation):
    """Recursive median splits of the listing cloud into equal-count cells.

    At each node the wider side (measured in metres) is split at the
    *median* coordinate of the listings inside it, so the two children hold
    equal counts by construction. A node is split whenever halving it lands
    *closer* to ``target_per_cell`` than leaving it whole.

    Because every split halves a node, achievable leaf sizes are quantised
    to ``n_listings / 2**depth``: ``target_per_cell`` selects the nearest
    such partition rather than being honoured exactly. Two nearby targets
    can therefore produce the identical tessellation - ``describe()`` reports
    the sizes actually achieved, which is the number to quote.

    Only coordinates are used - never prices, never the target - so the
    geography carries no outcome information. Cell ids are the binary split
    path (``kd:0110``), which makes ``block()`` free: truncating the path to
    ``block_depth`` characters yields contiguous super-cells, since every KD
    prefix is a rectangle. The effective depth is capped below the tree's
    own depth: a shallow tree truncated at a depth it never reached would
    give one block per cell, silently turning spatially blocked CV into the
    random k-fold the project forbids.
    """

    def __init__(
        self,
        # 140 is the value scripts/maup_study.py recommends on this panel:
        # best spatial-transfer skill among the candidates that keep most of
        # the data and give the median cell >= 12 months of history.
        target_per_cell: int = 140,
        *,
        block_depth: int = 3,
        min_span_m: float = 250.0,
    ):
        super().__init__()
        self.target_per_cell = int(target_per_cell)
        self.block_depth = int(block_depth)
        self.min_span_m = float(min_span_m)
        self.name = f"kdtree_n{target_per_cell}"
        self._leaves: dict[str, _KDLeaf] = {}
        self._nbr: dict[str, tuple[str, ...]] = {}
        self._block_depth_eff: int = int(block_depth)

    # -- fitting ----------------------------------------------------------

    def fit(self, listings: pd.DataFrame, **_) -> "AdaptiveKDTessellation":
        """Build the partition from the listing coordinates."""
        lat = np.asarray(listings["lat"], dtype=float)
        lon = np.asarray(listings["lon"], dtype=float)
        ok = np.isfinite(lat) & np.isfinite(lon)
        lat, lon = lat[ok], lon[ok]
        if lat.size == 0:
            raise ValueError("no finite coordinates to fit on")

        # Pad the root box so later points on the edge still fall inside.
        pad = 1e-6
        root = _KDLeaf("", lat.min() - pad, lat.max() + pad,
                       lon.min() - pad, lon.max() + pad, lat.size)
        self._leaves = {}
        self._split(root, lat, lon)
        self._nbr = self._build_neighbours()
        # Keep blocks strictly coarser than cells: with a tree of depth d,
        # a prefix of length d would just re-label every leaf.
        max_depth = max((len(lf.path) for lf in self._leaves.values()), default=0)
        self._block_depth_eff = max(1, min(self.block_depth, max_depth - 1))
        return self

    def _split(self, node: _KDLeaf, lat: np.ndarray, lon: np.ndarray) -> None:
        """Recursively split `node` (points already restricted to its box)."""
        mid_lat = 0.5 * (node.lat_min + node.lat_max)
        span_lat_m = (node.lat_max - node.lat_min) * _M_PER_DEG_LAT
        span_lon_m = (
            (node.lon_max - node.lon_min)
            * _M_PER_DEG_LAT
            * float(np.cos(np.radians(mid_lat)))
        )
        # Split only if halving gets us closer to the target than stopping.
        halve_is_closer = (
            abs(lat.size / 2.0 - self.target_per_cell)
            < abs(lat.size - self.target_per_cell)
        )
        too_small = min(span_lat_m, span_lon_m) < self.min_span_m
        if not halve_is_closer or too_small:
            self._leaves[f"kd:{node.path or 'r'}"] = _KDLeaf(
                node.path, node.lat_min, node.lat_max,
                node.lon_min, node.lon_max, int(lat.size),
                float(lat.mean()) if lat.size else 0.5 * (node.lat_min + node.lat_max),
                float(lon.mean()) if lon.size else 0.5 * (node.lon_min + node.lon_max),
            )
            return

        if span_lat_m >= span_lon_m:
            cut = float(np.median(lat))
            # A degenerate median (all points identical) cannot separate.
            if not (node.lat_min < cut < node.lat_max):
                cut = mid_lat
            left_mask = lat <= cut
            left = _KDLeaf(node.path + "0", node.lat_min, cut,
                           node.lon_min, node.lon_max, 0)
            right = _KDLeaf(node.path + "1", cut, node.lat_max,
                            node.lon_min, node.lon_max, 0)
        else:
            cut = float(np.median(lon))
            if not (node.lon_min < cut < node.lon_max):
                cut = 0.5 * (node.lon_min + node.lon_max)
            left_mask = lon <= cut
            left = _KDLeaf(node.path + "0", node.lat_min, node.lat_max,
                           node.lon_min, cut, 0)
            right = _KDLeaf(node.path + "1", node.lat_min, node.lat_max,
                            cut, node.lon_max, 0)

        self._split(left, lat[left_mask], lon[left_mask])
        self._split(right, lat[~left_mask], lon[~left_mask])

    def _build_neighbours(self) -> dict[str, tuple[str, ...]]:
        """Two leaves are neighbours when their rectangles share a border."""
        ids = sorted(self._leaves)
        boxes = [self._leaves[i] for i in ids]
        tol_lat = 1e-9
        out: dict[str, list[str]] = {i: [] for i in ids}
        for a in range(len(ids)):
            ba = boxes[a]
            for b in range(a + 1, len(ids)):
                bb = boxes[b]
                lat_overlap = (
                    min(ba.lat_max, bb.lat_max) - max(ba.lat_min, bb.lat_min)
                )
                lon_overlap = (
                    min(ba.lon_max, bb.lon_max) - max(ba.lon_min, bb.lon_min)
                )
                # Share a segment in one axis and touch in the other.
                touch = (
                    (lat_overlap > tol_lat and abs(lon_overlap) <= tol_lat)
                    or (lon_overlap > tol_lat and abs(lat_overlap) <= tol_lat)
                )
                if touch:
                    out[ids[a]].append(ids[b])
                    out[ids[b]].append(ids[a])
        return {k: tuple(sorted(v)) for k, v in out.items()}

    # -- interface --------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._leaves:
            raise RuntimeError("call fit() before using the tessellation")

    def assign(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Locate each point by walking the split path (vectorised per leaf)."""
        self._check_fitted()
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        out = np.empty(lat.shape, dtype=object)
        out[:] = None
        for cid, leaf in self._leaves.items():
            inside = (
                (lat >= leaf.lat_min) & (lat <= leaf.lat_max)
                & (lon >= leaf.lon_min) & (lon <= leaf.lon_max)
            )
            unset = inside & (out == None)  # noqa: E711 - object-array compare
            out[unset] = cid
        # Points outside the fitted extent snap to the nearest leaf centroid.
        missing = out == None  # noqa: E711
        if missing.any():
            ids = list(self._leaves)
            cent = np.array([self.centroid(c) for c in ids])
            for idx in np.flatnonzero(missing):
                d = haversine_km(lat[idx], lon[idx], cent[:, 0], cent[:, 1])
                out[idx] = ids[int(np.argmin(d))]
        return out

    def centroid(self, cell: str) -> tuple[float, float]:
        """Mean position of the leaf's listings (see :class:`_KDLeaf`)."""
        self._check_fitted()
        leaf = self._leaves[cell]
        if np.isfinite(leaf.lat_mean) and np.isfinite(leaf.lon_mean):
            return (leaf.lat_mean, leaf.lon_mean)
        return (0.5 * (leaf.lat_min + leaf.lat_max),
                0.5 * (leaf.lon_min + leaf.lon_max))

    def _neighbours1(self, cell: str) -> tuple[str, ...]:
        self._check_fitted()
        return self._nbr.get(cell, ())

    def boundary(self, cell: str) -> list[tuple[float, float]]:
        self._check_fitted()
        lf = self._leaves[cell]
        return [
            (lf.lat_min, lf.lon_min),
            (lf.lat_min, lf.lon_max),
            (lf.lat_max, lf.lon_max),
            (lf.lat_max, lf.lon_min),
        ]

    def block(self, cell: str) -> str:
        self._check_fitted()
        path = self._leaves[cell].path
        return "blk:" + (path[: self._block_depth_eff] or "r")

    def cells(self) -> list[str]:
        return sorted(self._leaves)

    def describe(self) -> dict:
        self._check_fitted()
        counts = [lf.n for lf in self._leaves.values()]
        return {
            "name": self.name,
            "kind": "kdtree",
            "target_per_cell": self.target_per_cell,
            "n_cells": len(self._leaves),
            "listings_per_cell_min": int(min(counts)),
            "listings_per_cell_median": int(np.median(counts)),
            "listings_per_cell_max": int(max(counts)),
            "n_blocks": len({self.block(c) for c in self._leaves}),
            "block_depth_effective": self._block_depth_eff,
        }


# ---------------------------------------------------------------------------
# 3. Market regions - contiguity-constrained clustering on price structure
# ---------------------------------------------------------------------------


class MarketRegionTessellation(_BaseTessellation):
    """Agglomerate fine H3 cells into contiguous, price-homogeneous regions.

    Ward linkage under a connectivity constraint (the H3 adjacency graph)
    merges only *spatially adjacent* cells, so every region is contiguous
    while its boundaries fall where the market actually changes.

    This is the one tessellation that uses prices, so it is a leakage risk
    if fitted carelessly: ``fit`` accepts ``price_cutoff_month`` and ignores
    every listing from that month onward. The pipeline passes the end of
    the training window, so the geography never sees evaluation-period
    prices.
    """

    def __init__(
        self,
        n_regions: int = 120,
        *,
        base_resolution: int = config.H3_RESOLUTION,
        n_blocks: int = config.SPATIAL_CV_FOLDS,
        min_listings_per_cell: int = config.MIN_LISTINGS_PER_CELL_MONTH,
    ):
        super().__init__()
        self.n_regions = int(n_regions)
        self.min_listings_per_cell = int(min_listings_per_cell)
        self.base_resolution = int(base_resolution)
        self.n_blocks = int(n_blocks)
        self.name = f"market_k{n_regions}"
        self._base = H3Tessellation(base_resolution, base_resolution)
        self._member_of: dict[str, str] = {}     # base h3 cell -> region id
        self._members: dict[str, list[str]] = {}  # region id -> base cells
        self._nbr: dict[str, tuple[str, ...]] = {}
        self._block_of: dict[str, str] = {}
        self._centroid: dict[str, tuple[float, float]] = {}
        self._boundary: dict[str, list[tuple[float, float]]] = {}
        self.price_cutoff_month: str | None = None

    # -- fitting ----------------------------------------------------------

    def fit(
        self,
        listings: pd.DataFrame,
        *,
        price_cutoff_month: str | None = None,
        **_,
    ) -> "MarketRegionTessellation":
        from sklearn.cluster import AgglomerativeClustering
        from scipy.sparse import csr_matrix

        self.price_cutoff_month = price_cutoff_month
        df = listings
        if price_cutoff_month is not None:
            df = df[df["listed_month"] < price_cutoff_month]
            if df.empty:
                raise ValueError(
                    f"no listings before price_cutoff_month={price_cutoff_month}"
                )

        base = self._base.assign(df["lat"].to_numpy(), df["lon"].to_numpy())
        feat = pd.DataFrame(
            {
                "cell": base,
                "logp": np.log(np.asarray(df["price_azn_m2"], dtype=float)),
                "is_new": (df["building_type"].to_numpy() == "new").astype(float),
            }
        )
        agg = feat.groupby("cell").agg(
            logp=("logp", "mean"), new_share=("is_new", "mean"), n=("logp", "size")
        )
        cells = list(agg.index)
        if len(cells) <= self.n_regions:
            # Nothing to merge: degrade to the base grid, one region per cell.
            self._adopt_labels(cells, list(range(len(cells))))
            return self

        # Standardise so price level and building mix weigh comparably,
        # using count-weighted moments: a base cell holding one advert is a
        # single-observation mean, i.e. noise, and must not set the scale.
        # Cells below `min_listings_per_cell` are dropped outright, so
        # boundaries are not drawn to chase sampling noise in the sparse
        # periphery (the `n` column used to be computed and ignored).
        keep = agg["n"] >= self.min_listings_per_cell
        if keep.sum() > self.n_regions:
            agg = agg[keep]
            cells = list(agg.index)
        w = agg["n"].to_numpy(dtype=float)
        x = agg[["logp", "new_share"]].to_numpy(dtype=float)
        mu = np.average(x, axis=0, weights=w)
        var = np.average((x - mu) ** 2, axis=0, weights=w)
        x = (x - mu) / np.where(var > 1e-24, np.sqrt(var), 1.0)

        # The observed grid is not one connected blob: Sumgait, Alat and the
        # Absheron villages are islands separated by unobserved cells. Handing
        # sklearn a disconnected connectivity matrix makes it silently
        # "complete" the graph, which can merge geographically separate
        # islands into a single region. Instead each component is clustered on
        # its own, with the region budget shared out by component size - so
        # every region is genuinely contiguous, by construction.
        components = connected_components(
            cells, lambda c: self._base.neighbours(c, 1)
        )
        budget = apportion([len(c) for c in components], self.n_regions)

        labels = np.empty(len(cells), dtype=int)
        pos = {c: i for i, c in enumerate(cells)}
        next_label = 0
        for comp_i, comp in enumerate(components):
            idx = np.array([pos[c] for c in comp], dtype=int)
            k = budget[comp_i]
            if k >= len(comp):
                # One region per cell: nothing to merge in this component.
                labels[idx] = np.arange(next_label, next_label + len(comp))
                next_label += len(comp)
                continue
            sub_pos = {c: i for i, c in enumerate(comp)}
            rows, cols = [], []
            for c in comp:
                i = sub_pos[c]
                for nb in self._base.neighbours(c, 1):
                    j = sub_pos.get(nb)
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
            conn = csr_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(len(comp), len(comp))
            )
            model = AgglomerativeClustering(
                n_clusters=k, linkage="ward", connectivity=conn
            )
            sub = model.fit_predict(x[idx])
            labels[idx] = sub + next_label
            next_label += k

        self._adopt_labels(cells, list(labels))
        return self

    def _adopt_labels(self, cells: list[str], labels: list[int]) -> None:
        """Materialise regions, their geometry, adjacency and CV blocks."""
        self._members = {}
        self._member_of = {}
        for cell, lab in zip(cells, labels):
            rid = f"mr:{int(lab):04d}"
            self._members.setdefault(rid, []).append(cell)
            self._member_of[cell] = rid

        # Region adjacency: inherited from base-cell contiguity.
        nbr: dict[str, set[str]] = {r: set() for r in self._members}
        for cell, rid in self._member_of.items():
            for nb in self._base.neighbours(cell, 1):
                other = self._member_of.get(nb)
                if other is not None and other != rid:
                    nbr[rid].add(other)
        self._nbr = {r: tuple(sorted(v)) for r, v in nbr.items()}

        # Geometry: dissolve the member hexagons into one polygon.
        self._centroid = {}
        self._boundary = {}
        for rid, members in self._members.items():
            self._centroid[rid] = self._dissolved_centroid(members)
            self._boundary[rid] = self._dissolved_ring(members)

        self._block_of = self._build_blocks()

    def _dissolved_centroid(self, members: list[str]) -> tuple[float, float]:
        pts = np.array([self._base.centroid(c) for c in members], dtype=float)
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))

    def _dissolved_ring(self, members: list[str]) -> list[tuple[float, float]]:
        """Outer ring of the union of the member hexagons."""
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        polys = []
        for c in members:
            ring = [(lng, lat) for lat, lng in self._base.boundary(c)]
            polys.append(Polygon(ring))
        merged = unary_union(polys)
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)
        # Simplify lightly: dissolved hex unions carry many collinear vertices.
        merged = merged.simplify(1e-5, preserve_topology=True)
        return [(lat, lng) for lng, lat in merged.exterior.coords[:-1]]

    def _build_blocks(self) -> dict[str, str]:
        """Group regions into `n_blocks` contiguous, size-balanced blocks.

        Two failure modes to avoid. Growing one block per BFS seed and
        letting the last one "absorb the rest" leaves early blocks
        under-filled and dumps the majority into a final block that is
        itself several disconnected pieces - which makes blocked CV a
        lopsided two-way split with near-empty folds, and misattributes the
        resulting bad score to the tessellation rather than the folds.

        So: cut each connected component into contiguous chunks of roughly
        the target size, then bin-pack whole chunks onto blocks
        largest-first (LPT). Every chunk is contiguous by construction and
        the blocks come out balanced.
        """
        regions = sorted(self._members)
        if len(regions) <= self.n_blocks:
            return {r: f"blk:{i}" for i, r in enumerate(regions)}

        target = max(1, len(regions) // self.n_blocks)
        chunks: list[list[str]] = []
        for comp in connected_components(regions, lambda r: self._nbr.get(r, ())):
            remaining = set(comp)
            while remaining:
                seed = min(remaining)
                remaining.discard(seed)
                chunk, frontier = [seed], deque([seed])
                while frontier and len(chunk) < target:
                    node = frontier.popleft()
                    for nb in self._nbr.get(node, ()):
                        if nb in remaining and len(chunk) < target:
                            remaining.discard(nb)
                            chunk.append(nb)
                            frontier.append(nb)
                chunks.append(chunk)

        # LPT bin-packing: biggest chunk onto the currently smallest block.
        sizes = [0] * self.n_blocks
        block_of: dict[str, str] = {}
        for chunk in sorted(chunks, key=len, reverse=True):
            b = min(range(self.n_blocks), key=lambda k: (sizes[k], k))
            for r in chunk:
                block_of[r] = f"blk:{b}"
            sizes[b] += len(chunk)
        return block_of

    # -- interface --------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._members:
            raise RuntimeError("call fit() before using the tessellation")

    def assign(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        self._check_fitted()
        base = self._base.assign(lat, lon)
        out = np.empty(len(base), dtype=object)
        # Base cells unseen at fit time (sparse periphery) snap to the
        # nearest region centroid - the honest fallback for new geography.
        rids = list(self._centroid)
        cent = np.array([self._centroid[r] for r in rids])
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        for i, cell in enumerate(base):
            rid = self._member_of.get(cell)
            if rid is None:
                d = haversine_km(lat[i], lon[i], cent[:, 0], cent[:, 1])
                rid = rids[int(np.argmin(d))]
            out[i] = rid
        return out

    def centroid(self, cell: str) -> tuple[float, float]:
        self._check_fitted()
        return self._centroid[cell]

    def _neighbours1(self, cell: str) -> tuple[str, ...]:
        self._check_fitted()
        return self._nbr.get(cell, ())

    def boundary(self, cell: str) -> list[tuple[float, float]]:
        self._check_fitted()
        return self._boundary[cell]

    def block(self, cell: str) -> str:
        self._check_fitted()
        return self._block_of.get(cell, "blk:0")

    def cells(self) -> list[str]:
        return sorted(self._members)

    def describe(self) -> dict:
        self._check_fitted()
        sizes = [len(v) for v in self._members.values()]
        return {
            "name": self.name,
            "kind": "market_regions",
            "n_regions": len(self._members),
            "base_resolution": self.base_resolution,
            "price_cutoff_month": self.price_cutoff_month,
            "base_cells_per_region_min": int(min(sizes)),
            "base_cells_per_region_median": int(np.median(sizes)),
            "base_cells_per_region_max": int(max(sizes)),
            "n_blocks": len(set(self._block_of.values())),
        }


# ---------------------------------------------------------------------------
# Active tessellation (module-level, so existing signatures keep working)
# ---------------------------------------------------------------------------

_ACTIVE: Tessellation = H3Tessellation()


def get_active() -> Tessellation:
    """The tessellation used when a caller passes none."""
    return _ACTIVE


def set_active(tess: Tessellation) -> Tessellation:
    """Install `tess` as the active tessellation; returns the previous one."""
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = tess
    return previous


def resolve(tess: Tessellation | None) -> Tessellation:
    return _ACTIVE if tess is None else tess


def reset_active() -> None:
    """Restore the H3 default (used by tests to avoid cross-test bleed)."""
    global _ACTIVE
    _ACTIVE = H3Tessellation()


def build(kind: str, **kwargs) -> Tessellation:
    """Factory used by the pipeline CLI and the MAUP study.

    ``kind`` is one of ``h3``, ``kdtree``, ``market``.
    """
    kinds = {
        "h3": H3Tessellation,
        "kdtree": AdaptiveKDTessellation,
        "market": MarketRegionTessellation,
    }
    if kind not in kinds:
        raise ValueError(f"unknown tessellation {kind!r}; expected one of {sorted(kinds)}")
    return kinds[kind](**kwargs)
