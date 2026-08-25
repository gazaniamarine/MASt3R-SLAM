"""Predict current-view entity visibility from the exported MASt3R geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fact3r.entities.entity import Entity
from fact3r.reconstruction.keyframes import KeyframeRecord


@dataclass(frozen=True, slots=True)
class VisibilityConfig:
    """Projection and depth-test parameters for one current keyframe."""

    max_entity_points: int = 512
    depth_tolerance_m: float = 0.15
    unknown_depth_visibility: float = 0.0

    def __post_init__(self) -> None:
        if self.max_entity_points <= 0:
            raise ValueError("max_entity_points must be positive")
        if self.depth_tolerance_m < 0.0 or not np.isfinite(
            self.depth_tolerance_m
        ):
            raise ValueError("depth_tolerance_m must be finite and non-negative")
        if not 0.0 <= self.unknown_depth_visibility <= 1.0:
            raise ValueError("unknown_depth_visibility must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EntityVisibility:
    """Current-view evidence used to set one entity's transport demand."""

    entity_id: str
    score: float
    in_frustum_fraction: float
    unoccluded_fraction: float
    sampled_point_count: int
    projected_point_count: int
    visible_point_count: int
    used_intrinsics_fallback: bool = False


def _sample_finite_points(entity: Entity, maximum: int) -> np.ndarray:
    points = np.asarray(entity.surfel_or_voxel_geometry, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def estimate_entity_visibility(
    entities: Sequence[Entity],
    keyframe: KeyframeRecord,
    config: VisibilityConfig | None = None,
) -> tuple[EntityVisibility, ...]:
    """Project persistent geometry and reject points hidden by current depth.

    ``score`` is the visible fraction of the sampled entity surface, including
    the field-of-view test. It is therefore small both for an occluded entity and
    for one outside the camera view. If calibration is unavailable, the function
    returns an explicit neutral fallback instead of pretending to infer visibility.
    """

    config = VisibilityConfig() if config is None else config
    if keyframe.intrinsics is None:
        return tuple(
            EntityVisibility(
                entity_id=entity.id,
                score=1.0,
                in_frustum_fraction=1.0,
                unoccluded_fraction=1.0,
                sampled_point_count=min(
                    len(entity.surfel_or_voxel_geometry),
                    config.max_entity_points,
                ),
                projected_point_count=min(
                    len(entity.surfel_or_voxel_geometry),
                    config.max_entity_points,
                ),
                visible_point_count=min(
                    len(entity.surfel_or_voxel_geometry),
                    config.max_entity_points,
                ),
                used_intrinsics_fallback=True,
            )
            for entity in entities
        )

    try:
        camera_from_world = np.linalg.inv(
            np.asarray(keyframe.pose_world_from_camera, dtype=np.float64)
        )
    except np.linalg.LinAlgError as error:
        raise ValueError("pose_world_from_camera must be invertible") from error

    height, width = keyframe.image_shape
    intrinsics = np.asarray(keyframe.intrinsics, dtype=np.float64)
    observed_depth = np.asarray(keyframe.pointmap_camera[..., 2], dtype=np.float64)
    results: list[EntityVisibility] = []
    for entity in entities:
        points_world = _sample_finite_points(entity, config.max_entity_points)
        sampled_count = len(points_world)
        if sampled_count == 0:
            results.append(
                EntityVisibility(
                    entity_id=entity.id,
                    score=0.0,
                    in_frustum_fraction=0.0,
                    unoccluded_fraction=0.0,
                    sampled_point_count=0,
                    projected_point_count=0,
                    visible_point_count=0,
                )
            )
            continue

        points_camera = (
            points_world @ camera_from_world[:3, :3].T
            + camera_from_world[:3, 3]
        )
        depth = points_camera[:, 2]
        in_front = np.isfinite(depth) & (depth > 1e-6)
        columns_float = np.full(sampled_count, np.nan, dtype=np.float64)
        rows_float = np.full(sampled_count, np.nan, dtype=np.float64)
        columns_float[in_front] = (
            intrinsics[0, 0]
            * points_camera[in_front, 0]
            / depth[in_front]
            + intrinsics[0, 2]
        )
        rows_float[in_front] = (
            intrinsics[1, 1]
            * points_camera[in_front, 1]
            / depth[in_front]
            + intrinsics[1, 2]
        )
        inside = (
            in_front
            & np.isfinite(rows_float)
            & np.isfinite(columns_float)
            & (rows_float >= -0.5)
            & (rows_float < height - 0.5)
            & (columns_float >= -0.5)
            & (columns_float < width - 0.5)
        )
        projected_indices = np.flatnonzero(inside)
        projected_count = len(projected_indices)
        if projected_count == 0:
            visible_weight = 0.0
            visible_count = 0
        else:
            rows = np.rint(rows_float[projected_indices]).astype(np.int64)
            columns = np.rint(columns_float[projected_indices]).astype(np.int64)
            reference_depth = observed_depth[rows, columns]
            known_depth = np.isfinite(reference_depth) & (reference_depth > 1e-6)
            unoccluded = known_depth & (
                depth[projected_indices]
                <= reference_depth + config.depth_tolerance_m
            )
            visible_count = int(np.count_nonzero(unoccluded))
            unknown_count = int(np.count_nonzero(~known_depth))
            visible_weight = (
                visible_count
                + config.unknown_depth_visibility * unknown_count
            )

        results.append(
            EntityVisibility(
                entity_id=entity.id,
                score=float(np.clip(visible_weight / sampled_count, 0.0, 1.0)),
                in_frustum_fraction=projected_count / sampled_count,
                unoccluded_fraction=(
                    0.0
                    if projected_count == 0
                    else float(
                        np.clip(visible_weight / projected_count, 0.0, 1.0)
                    )
                ),
                sampled_point_count=sampled_count,
                projected_point_count=projected_count,
                visible_point_count=visible_count,
            )
        )
    return tuple(results)
