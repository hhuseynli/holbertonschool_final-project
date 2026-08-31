"""Spatial validation: blocked cross-validation folds over coarse-block cells.

Why blocked folds?  Prices are spatially autocorrelated: two adjacent H3
cells share amenities, building stock and buyers.  If neighbouring cells
land in different folds of a random K-fold, the model effectively "sees"
each test cell through its neighbours at train time and the CV score
overstates how well the model generalises to genuinely unseen areas
(Roberts et al. 2017, "Cross-validation strategies for data with temporal,
spatial, hierarchical, or phylogenetic structure").

The fix used here: group resolution-8 cells by their coarser coarse-block
(``config.H3_BLOCK_RESOLUTION``), so whole contiguous neighbourhoods are
held out together, then assign whole blocks to folds.  Balancing uses the
classic greedy longest-processing-time heuristic: blocks are taken in
descending size order and each is placed on the currently smallest fold,
which keeps fold sizes close whenever no single block dominates.
Ties between equal-sized blocks are broken by a seeded random draw so the
assignment is deterministic for a given seed.
"""

from __future__ import annotations

import h3
import numpy as np

from bakuml import config
from bakuml.spatial import tessellation as tess_mod
from bakuml.spatial.tessellation import Tessellation


def spatial_block_folds(
    cells: list[str],
    *,
    n_folds: int = config.SPATIAL_CV_FOLDS,
    block_res: int | None = None,
    seed: int = 0,
    tess: Tessellation | None = None,
) -> dict[str, int]:
    """Assign every cell to one of ``n_folds`` spatially blocked folds.

    Cells sharing a coarse block (``Tessellation.block``) always land in the
    same fold, so whole contiguous regions are held out together. Returns
    ``{cell: fold}`` with folds numbered ``0..n_folds-1``; every input cell
    is mapped to exactly one fold.

    ``block_res`` is an H3-specific shortcut retained for the documented
    baseline: when given, blocks are coarse-blocks at that resolution. It is
    mutually exclusive with ``tess``.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    unique_cells = sorted(set(cells))
    if not unique_cells:
        raise ValueError("cells must be non-empty")
    if block_res is not None:
        if tess is not None:
            raise ValueError("pass either block_res= or tess=, not both")
        if any(h3.get_resolution(c) < block_res for c in unique_cells):
            raise ValueError(
                f"block_res={block_res} is finer than some input cells; "
                "it must be coarser than (or equal to) the cell resolution"
            )
        tess = tess_mod.H3Tessellation(
            max(h3.get_resolution(c) for c in unique_cells), block_res
        )
    tess = tess_mod.resolve(tess)

    # Group cells by their coarse block.
    blocks: dict[str, list[str]] = {}
    for cell in unique_cells:
        blocks.setdefault(tess.block(cell), []).append(cell)

    # Deterministic order: descending block size, seeded shuffle within ties.
    rng = np.random.default_rng(seed)
    tiebreak = {parent: rng.random() for parent in sorted(blocks)}
    order = sorted(blocks, key=lambda p: (-len(blocks[p]), tiebreak[p]))

    # Greedy LPT: each block goes onto the currently smallest fold
    # (lowest fold index wins size ties, keeping the result deterministic).
    fold_sizes = [0] * n_folds
    assignment: dict[str, int] = {}
    for parent in order:
        fold = min(range(n_folds), key=lambda k: (fold_sizes[k], k))
        for cell in blocks[parent]:
            assignment[cell] = fold
        fold_sizes[fold] += len(blocks[parent])
    return assignment
