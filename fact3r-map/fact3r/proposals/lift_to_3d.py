"""Lift a pointmap-aligned 2D mask into globally aligned 3D points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fact3r.reconstruction.keyframes import KeyframeRecord


@dataclass(frozen=True, slots=True)
class LiftedProposal:
    proposal_id: str
    frame_id: int
    timestamp: float | str | None
    pixel_rc: NDArray[np.integer]
    points_world: NDArray[np.floating]
    colours_rgb: NDArray[np.generic]
    geometry_confidence: NDArray[np.floating]
    mast3r_descriptors: NDArray[np.floating] | None
    descriptor_confidence: NDArray[np.floating] | None
    source_mask_area: int
    parent_proposal_id: str | None = None

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixel_rc, dtype=np.int32)
        points = np.asarray(self.points_world, dtype=np.float32)
        colours = np.asarray(self.colours_rgb)
        geometry_confidence = np.asarray(
            self.geometry_confidence, dtype=np.float32
        ).reshape(-1)

        count = len(points)
        if pixels.shape != (count, 2):
            raise ValueError("pixel_rc must have shape (N, 2)")
        if points.shape != (count, 3):
            raise ValueError("points_world must have shape (N, 3)")
        if colours.shape != (count, 3):
            raise ValueError("colours_rgb must have shape (N, 3)")
        if geometry_confidence.shape != (count,):
            raise ValueError("geometry_confidence must have shape (N,)")
        if count == 0:
            raise ValueError("a lifted proposal must contain at least one valid point")
        if self.source_mask_area < count:
            raise ValueError("source_mask_area cannot be smaller than the point count")

        descriptors = None
        if self.mast3r_descriptors is not None:
            descriptors = np.asarray(self.mast3r_descriptors, dtype=np.float32)
            if descriptors.ndim != 2 or descriptors.shape[0] != count:
                raise ValueError("mast3r_descriptors must have shape (N, D)")

        descriptor_confidence = None
        if self.descriptor_confidence is not None:
            descriptor_confidence = np.asarray(
                self.descriptor_confidence, dtype=np.float32
            ).reshape(-1)
            if descriptor_confidence.shape != (count,):
                raise ValueError("descriptor_confidence must have shape (N,)")
            if descriptors is None:
                raise ValueError(
                    "descriptor_confidence cannot be set without descriptors"
                )

        object.__setattr__(self, "pixel_rc", np.ascontiguousarray(pixels))
        object.__setattr__(self, "points_world", np.ascontiguousarray(points))
        object.__setattr__(self, "colours_rgb", np.ascontiguousarray(colours))
        object.__setattr__(
            self,
            "geometry_confidence",
            np.ascontiguousarray(geometry_confidence),
        )
        object.__setattr__(self, "mast3r_descriptors", descriptors)
        object.__setattr__(self, "descriptor_confidence", descriptor_confidence)

    @property
    def centroid_xyz(self) -> NDArray[np.floating]:
        return self.points_world.mean(axis=0)

    @property
    def bounding_box_xyz(self) -> NDArray[np.floating]:
        return np.stack(
            (self.points_world.min(axis=0), self.points_world.max(axis=0)), axis=0
        )


def lift_mask_to_3d(
    keyframe: KeyframeRecord,
    mask: NDArray[np.generic],
    *,
    proposal_id: str,
    min_geometry_confidence: float = -np.inf,
    min_descriptor_confidence: float | None = None,
    parent_proposal_id: str | None = None,
) -> LiftedProposal:
    """Lift one mask using pointmap pixels and the keyframe world transform.

    Boundary erosion, voxel filtering, and component cleanup intentionally remain
    outside this primitive. They are proposal preprocessing operations introduced
    in Milestone 1.
    """

    mask_array = np.asarray(mask)
    if mask_array.shape != keyframe.image_shape:
        raise ValueError(
            f"mask must have shape {keyframe.image_shape}; got {mask_array.shape}"
        )
    mask_bool = mask_array.astype(bool, copy=False)
    source_mask_area = int(mask_bool.sum())
    if source_mask_area == 0:
        raise ValueError("mask must select at least one pixel")

    valid = mask_bool & (
        keyframe.geometry_confidence > float(min_geometry_confidence)
    )
    if min_descriptor_confidence is not None:
        if keyframe.descriptor_confidence is None:
            raise ValueError(
                "descriptor confidence filtering requested, but the keyframe has no Q map"
            )
        valid &= keyframe.descriptor_confidence > float(min_descriptor_confidence)

    points_world = keyframe.points_world()
    valid &= np.all(np.isfinite(points_world), axis=-1)
    pixels = np.argwhere(valid)
    if len(pixels) == 0:
        raise ValueError("no mask pixels survived confidence and finite-point filtering")

    rows, columns = pixels[:, 0], pixels[:, 1]
    descriptors = (
        None
        if keyframe.mast3r_descriptors is None
        else keyframe.mast3r_descriptors[rows, columns]
    )
    descriptor_confidence = (
        None
        if keyframe.descriptor_confidence is None
        else keyframe.descriptor_confidence[rows, columns]
    )

    return LiftedProposal(
        proposal_id=proposal_id,
        frame_id=keyframe.frame_id,
        timestamp=keyframe.timestamp,
        pixel_rc=pixels,
        points_world=points_world[rows, columns],
        colours_rgb=keyframe.rgb[rows, columns],
        geometry_confidence=keyframe.geometry_confidence[rows, columns],
        mast3r_descriptors=descriptors,
        descriptor_confidence=descriptor_confidence,
        source_mask_area=source_mask_area,
        parent_proposal_id=parent_proposal_id,
    )

