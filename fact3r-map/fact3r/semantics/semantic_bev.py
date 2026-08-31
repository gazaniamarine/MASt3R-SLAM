"""Geometry helpers for projecting persistent semantic observations into BEV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class SemanticGrid:
    """Winning persistent entity and vote confidence in every BEV cell."""

    entity_ids: NDArray[np.int32]
    confidence: NDArray[np.float32]
    support: NDArray[np.float32]


def camera_to_body(
    pitch_degrees: float, camera_height: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return body-from-camera rotation and camera origin in the rover body."""

    pitch = np.radians(float(pitch_degrees))
    right = np.asarray([0.0, -1.0, 0.0])
    down = np.asarray([-np.sin(pitch), 0.0, -np.cos(pitch)])
    forward = np.asarray([np.cos(pitch), 0.0, -np.sin(pitch)])
    rotation = np.stack([right, down, forward], axis=1)
    translation = np.asarray([0.0, 0.0, float(camera_height)])
    return rotation, translation


def backproject_depth(
    depth: FloatArray,
    *,
    fx: float,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    pixel_stride: int = 1,
    depth_min: float = 0.3,
    depth_max: float = 4.0,
    mask: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
    """Back-project valid sampled pixels and return points plus image indices."""

    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("depth must have shape (height, width)")
    if fx <= 0 or (fy is not None and fy <= 0):
        raise ValueError("focal lengths must be positive")
    if pixel_stride <= 0:
        raise ValueError("pixel_stride must be positive")
    height, width = values.shape
    focal_y = float(fx if fy is None else fy)
    centre_x = width / 2.0 if cx is None else float(cx)
    centre_y = height / 2.0 if cy is None else float(cy)
    rows, columns = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    sampled_depth = values[::pixel_stride, ::pixel_stride]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth > depth_min)
        & (sampled_depth < depth_max)
    )
    if mask is not None:
        semantic_mask = np.asarray(mask, dtype=bool)
        if semantic_mask.shape != values.shape:
            raise ValueError("mask and depth must have the same image shape")
        valid &= semantic_mask[::pixel_stride, ::pixel_stride]
    z = sampled_depth[valid]
    selected_rows = rows[valid].astype(np.int32, copy=False)
    selected_columns = columns[valid].astype(np.int32, copy=False)
    points = np.stack(
        [
            (selected_columns - centre_x) * z / float(fx),
            (selected_rows - centre_y) * z / focal_y,
            z,
        ],
        axis=1,
    )
    return (
        np.ascontiguousarray(points, dtype=np.float32),
        selected_rows,
        selected_columns,
    )


def camera_points_to_rover_map(
    points_camera: FloatArray,
    *,
    rover_x: float,
    rover_y: float,
    rover_yaw: float,
    rotation_body_from_camera: FloatArray,
    translation_body_from_camera: FloatArray,
) -> NDArray[np.float32]:
    """Transform camera points to the z-up rover world used by depth_to_bev."""

    points = np.asarray(points_camera, dtype=np.float64)
    rotation = np.asarray(rotation_body_from_camera, dtype=np.float64)
    translation = np.asarray(translation_body_from_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera must have shape (count, 3)")
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("camera mount must contain a 3x3 rotation and 3-vector")
    body = points @ rotation.T + translation
    cosine, sine = np.cos(rover_yaw), np.sin(rover_yaw)
    world_x = rover_x + cosine * body[:, 0] - sine * body[:, 1]
    world_y = rover_y + sine * body[:, 0] + cosine * body[:, 1]
    # Match depth_to_bev.py / occupancy_grid.py: x, gravity-down, horizontal z.
    return np.ascontiguousarray(
        np.stack([world_x, -body[:, 2], world_y], axis=1), dtype=np.float32
    )


def aggregate_group_embeddings(
    embeddings: FloatArray,
    observations: Sequence[dict[str, object]],
    group_ids: Sequence[str],
) -> NDArray[np.float32]:
    """Build one quality-weighted, L2-normalized semantic prototype per group."""

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(observations):
        raise ValueError("one embedding row is required per observation")
    lookup = {group_id: index for index, group_id in enumerate(group_ids)}
    sums = np.zeros((len(group_ids), vectors.shape[1]), dtype=np.float64)
    weights = np.zeros(len(group_ids), dtype=np.float64)
    for row, observation in enumerate(observations):
        group_id = str(observation.get("group_id") or "")
        group_index = lookup.get(group_id)
        if group_index is None:
            continue
        weight = float(observation.get("proposal_score", 1.0)) * float(
            observation.get("association_confidence", 1.0)
        )
        weight = float(np.clip(weight, 1e-3, 1.0))
        sums[group_index] += weight * vectors[row]
        weights[group_index] += weight
    prototypes = np.zeros_like(sums, dtype=np.float32)
    populated = weights > 0
    prototypes[populated] = (sums[populated] / weights[populated, None]).astype(
        np.float32
    )
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    np.divide(prototypes, norms, out=prototypes, where=norms > 1e-12)
    return prototypes


def build_semantic_grid(
    points: FloatArray,
    group_indices: NDArray[np.integer],
    weights: FloatArray,
    *,
    shape: tuple[int, int],
    lower_xy: FloatArray,
    resolution: float,
    floor_origin: FloatArray,
    floor_u: FloatArray,
    floor_v: FloatArray,
) -> SemanticGrid:
    """Fuse sparse entity votes and retain the strongest group in each cell."""

    xyz = np.asarray(points, dtype=np.float64)
    groups = np.asarray(group_indices, dtype=np.int64)
    vote_weights = np.asarray(weights, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape (count, 3)")
    if len(groups) != len(xyz) or len(vote_weights) != len(xyz):
        raise ValueError("points, group_indices, and weights must align")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    height, width = shape
    entity_grid = np.full((height, width), -1, dtype=np.int32)
    confidence_grid = np.zeros((height, width), dtype=np.float32)
    support_grid = np.zeros((height, width), dtype=np.float32)
    if not len(xyz):
        return SemanticGrid(entity_grid, confidence_grid, support_grid)

    relative = xyz - np.asarray(floor_origin, dtype=np.float64)
    plane_x = relative @ np.asarray(floor_u, dtype=np.float64)
    plane_y = relative @ np.asarray(floor_v, dtype=np.float64)
    lower = np.asarray(lower_xy, dtype=np.float64)
    columns = np.floor((plane_x - lower[0]) / resolution).astype(np.int64)
    rows = np.floor((plane_y - lower[1]) / resolution).astype(np.int64)
    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(vote_weights)
        & (vote_weights > 0)
        & (groups >= 0)
        & (rows >= 0)
        & (rows < height)
        & (columns >= 0)
        & (columns < width)
    )
    if not valid.any():
        return SemanticGrid(entity_grid, confidence_grid, support_grid)
    rows, columns = rows[valid], columns[valid]
    groups, vote_weights = groups[valid], vote_weights[valid]
    cell_ids = rows * width + columns
    cell_count = height * width
    pair_ids = groups * cell_count + cell_ids
    unique_pairs, inverse = np.unique(pair_ids, return_inverse=True)
    pair_support = np.zeros(len(unique_pairs), dtype=np.float64)
    np.add.at(pair_support, inverse, vote_weights)
    pair_groups = unique_pairs // cell_count
    pair_cells = unique_pairs % cell_count

    total = np.zeros(cell_count, dtype=np.float64)
    np.add.at(total, pair_cells, pair_support)
    order = np.lexsort((pair_support, pair_cells))
    sorted_cells = pair_cells[order]
    is_last_for_cell = np.concatenate(
        [sorted_cells[1:] != sorted_cells[:-1], np.asarray([True])]
    )
    winners = order[is_last_for_cell]
    winning_support = np.zeros(cell_count, dtype=np.float64)
    winning_group = np.full(cell_count, -1, dtype=np.int64)
    winning_support[pair_cells[winners]] = pair_support[winners]
    winning_group[pair_cells[winners]] = pair_groups[winners]
    observed = winning_group >= 0
    flat_entities = entity_grid.ravel()
    flat_confidence = confidence_grid.ravel()
    flat_support = support_grid.ravel()
    flat_entities[observed] = winning_group[observed].astype(np.int32)
    flat_support[observed] = winning_support[observed].astype(np.float32)
    flat_confidence[observed] = (
        winning_support[observed] / np.maximum(total[observed], 1e-12)
    ).astype(np.float32)
    return SemanticGrid(entity_grid, confidence_grid, support_grid)
