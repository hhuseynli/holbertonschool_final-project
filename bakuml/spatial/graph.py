"""Graph operators over the H3 cell set for the Tier-2 STGCN.

Methodology
-----------
Graph convolutional layers of the Kipf & Welling (2017) family propagate
information with the symmetrically normalised adjacency

    A_hat = D^{-1/2} (A + I) D^{-1/2},

where A is the binary contiguity matrix of the H3 grid restricted to the
study cells (A[i, j] = 1 iff cells i and j share an edge, i.e. j is in
`grid_disk(i, 1)` and j != i), I adds the self loop so a cell's own signal
survives the convolution, and D is the degree matrix of A + I. The
normalisation keeps the operator's spectral radius at 1, so stacked graph
convolutions neither explode nor vanish. Cells with no in-study neighbour
degenerate gracefully to an identity row (degree 1 from the self loop).

Contiguity is symmetric, hence A_hat is symmetric and every row is finite:
D's diagonal is >= 1 by the self loop, so no division by zero can occur.

`edge_index` exports the same contiguity in the COO convention used by
PyTorch-Geometric-style code: every undirected edge appears in both
directions; self loops are NOT included (the model adds them via A_hat).
"""

from __future__ import annotations

import numpy as np

from bakuml.spatial import tessellation as tess_mod
from bakuml.spatial.tessellation import Tessellation


def _cell_order_and_edges(
    cells: list[str], tess: Tessellation | None = None
) -> tuple[list[str], list[tuple[int, int]]]:
    """Deduplicate `cells` preserving order; list directed contiguity edges.

    Both directions of every neighbour relation are returned: contiguity is
    symmetric, and adding (i, j) and (j, i) explicitly makes that symmetry
    hold even if a tessellation ever reported an asymmetric neighbourhood.
    """
    tess = tess_mod.resolve(tess)
    order = list(dict.fromkeys(cells))
    pos = {c: i for i, c in enumerate(order)}
    edges: set[tuple[int, int]] = set()
    for c in order:
        i = pos[c]
        for nb in tess.neighbours(c, 1):
            j = pos.get(nb)
            if j is not None and j != i:
                edges.add((i, j))
                edges.add((j, i))
    return order, sorted(edges)


def build_adjacency(
    cells: list[str], *, tess: Tessellation | None = None
) -> tuple[np.ndarray, list[str]]:
    """Dense symmetric normalised adjacency D^-1/2 (A + I) D^-1/2.

    A is the h3 neighbour (edge-sharing) relation restricted to `cells`.
    Returns `(matrix, cell_order)` where `matrix` is float64 of shape
    (n, n) and `cell_order` is the deduplicated cell list, in input order,
    that indexes the matrix rows/columns.
    """
    order, edges = _cell_order_and_edges(cells, tess)
    n = len(order)
    a_hat = np.eye(n, dtype=np.float64)  # A + I
    for i, j in edges:
        a_hat[i, j] = 1.0
    deg = a_hat.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(deg)  # deg >= 1 thanks to the self loop
    norm = a_hat * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return norm, order


def edge_index(cells: list[str], *, tess: Tessellation | None = None) -> np.ndarray:
    """COO edge list of the contiguity graph, shape (2, E), int64.

    Indices refer to the deduplicated input order (the same order
    `build_adjacency` returns). Every undirected edge appears in both
    directions, so E is even; self loops are excluded.
    """
    _, edges = _cell_order_and_edges(cells, tess)
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T
