"""Turn a semantic BEV entity into a point the planner can be given.

Two conventions meet here and they are not the same one.

The grids -- occupancy and semantic alike -- index by
``col = floor((x - origin_xy[0]) / res)`` and
``row = floor((y - origin_xy[1]) / res)``, so row 0 is the low-y edge and the
``.yaml`` sidecar's ``origin: [x, y, 0.0]`` is the bottom-left corner. The
planner, meanwhile, speaks ``(y, x)``: ``HM3DMap.cell_to_world`` returns
``np.stack([y, x])`` and every endpoint it is handed is read in that order.
Swapping them produces a confident plan to the wrong room rather than an error,
so the conversion lives in one place and is round-tripped in the tests.

Nothing here imports scipy. The ranking half of the goal resolver runs in the
SAM2 environment, which has the SigLIP encoder but no scipy, so the clearance
and connected-component work -- which is `HM3DMap`'s, not ours -- happens in a
separate step and hands its navigable mask back in.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def group_cell_counts(
    semantic_ids: NDArray[np.integer], groups: Iterable[dict[str, object]]
) -> dict[str, int]:
    """Grid cells won by each group in the manifest.

    `build_semantic_grid` awards every cell to a single entity, so most groups
    win nothing: on the MPL map only 1,611 of 14,555 hold even one cell. A group
    with no cells has no position, and ranking it is how the query ends up
    returning something that cannot be navigated to.
    """

    ids = np.asarray(semantic_ids)
    populated = ids[ids >= 0]
    counts = np.bincount(populated.ravel()) if populated.size else np.zeros(0, int)
    result: dict[str, int] = {}
    for group in groups:
        semantic_id = int(group["semantic_id"])
        result[str(group["group_id"])] = (
            int(counts[semantic_id]) if semantic_id < len(counts) else 0
        )
    return result


def cell_centre_xy(
    rows: NDArray[np.integer] | float,
    cols: NDArray[np.integer] | float,
    origin_xy: Sequence[float],
    resolution: float,
) -> tuple[FloatArray, FloatArray]:
    """Cell indices -> the (x, y) metres of their centres."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    x = float(origin_xy[0]) + (np.asarray(cols, dtype=np.float64) + 0.5) * resolution
    y = float(origin_xy[1]) + (np.asarray(rows, dtype=np.float64) + 0.5) * resolution
    return x, y


def world_xy_to_cell(
    x: FloatArray | float,
    y: FloatArray | float,
    origin_xy: Sequence[float],
    resolution: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """(x, y) metres -> the integer (row, col) whose cell contains them."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    col = np.floor((np.asarray(x, dtype=np.float64) - float(origin_xy[0])) / resolution)
    row = np.floor((np.asarray(y, dtype=np.float64) - float(origin_xy[1])) / resolution)
    return row.astype(np.int64), col.astype(np.int64)


def weighted_centroid_cell(
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
    weights: FloatArray,
) -> tuple[float, float]:
    """Confidence-weighted mean cell of an entity's footprint.

    Returned in fractional cell units, because the caller has to decide whether
    the cell it lands in is somewhere the robot may stand -- and for a U-shaped
    or split entity it very often is not. The centroid is a starting point for
    that search, never an answer on its own.
    """

    row_index = np.asarray(rows, dtype=np.float64)
    col_index = np.asarray(cols, dtype=np.float64)
    vote = np.asarray(weights, dtype=np.float64)
    if not len(row_index):
        raise ValueError("the entity occupies no cells")
    if len(col_index) != len(row_index) or len(vote) != len(row_index):
        raise ValueError("rows, cols, and weights must align")
    total = float(vote.sum())
    if not np.isfinite(total) or total <= 0:
        # Every vote zero or non-finite: fall back to the plain centroid rather
        # than dividing by nothing.
        return float(row_index.mean()), float(col_index.mean())
    return float((row_index * vote).sum() / total), float((col_index * vote).sum() / total)


def nearest_cell_in(
    mask: NDArray[np.bool_],
    row: float,
    col: float,
    *,
    prefer: FloatArray | None = None,
    tolerance_cells: float = 0.0,
) -> tuple[int, int, float]:
    """Closest True cell to a fractional (row, col), and its distance in cells.

    An object's own footprint is occupied by construction, so the centroid of
    the entity is never itself a legal goal; and a centroid can equally land in
    a wall or in unobserved space. What the planner can accept is the nearest
    cell the robot may actually stand in, which is what this returns. The caller
    passes a mask already restricted to the component it wants -- projecting to
    a nearer cell on the wrong side of a wall would just move the failure into
    A*.

    Strictly-nearest is the wrong tie-break on its own. Projecting an outside
    point onto a region always lands on that region's boundary, which is where
    clearance is smallest, so the goal reliably ends up at a pinch point the
    tube then has to collapse onto. `prefer` (clearance, in practice) breaks
    ties among cells that are within `tolerance_cells` of the nearest: inside
    one robot radius the goal is the same place as far as the task is
    concerned, and standing where there is room to stand is better.
    """

    cells = np.asarray(mask, dtype=bool)
    rows, cols = np.nonzero(cells)
    if not len(rows):
        raise ValueError("no navigable cell to project onto")
    distance = np.hypot(rows - float(row), cols - float(col))
    if prefer is None or tolerance_cells <= 0:
        winner = int(np.argmin(distance))
        return int(rows[winner]), int(cols[winner]), float(distance[winner])
    near = np.nonzero(distance <= distance.min() + float(tolerance_cells))[0]
    scores = np.asarray(prefer)[rows[near], cols[near]]
    winner = int(near[int(np.argmax(scores))])
    return int(rows[winner]), int(cols[winner]), float(distance[winner])
