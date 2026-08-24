"""Stable keyframe contract consumed by the Fact3R mapping pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def _float_array(
    value: object, name: str, *, allow_nonfinite: bool = False
) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if not allow_nonfinite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class KeyframeRecord:
    """Geometry and appearance retained for one globally posed keyframe.

    ``pointmap_camera`` contains points in the keyframe/camera coordinate frame.
    ``pose_world_from_camera`` maps those points into the shared world frame. The
    upper-left 3x3 block may include Sim(3) scale, matching MASt3R-SLAM's pose.
    Dense downstream MASt3R descriptors are optional at the type level so the
    adapter can expose legacy keyframes, but association code can require them.
    """

    frame_id: int
    timestamp: float | str | None
    rgb: NDArray[np.generic]
    pointmap_camera: FloatArray
    geometry_confidence: FloatArray
    pose_world_from_camera: FloatArray
    mast3r_descriptors: FloatArray | None = None
    descriptor_confidence: FloatArray | None = None
    intrinsics: FloatArray | None = None

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("rgb must have shape (height, width, 3)")
        if not np.issubdtype(rgb.dtype, np.number):
            raise TypeError("rgb must have a numeric dtype")
        if not np.all(np.isfinite(rgb)):
            raise ValueError("rgb must contain only finite values")
        rgb = np.ascontiguousarray(rgb)

        pointmap = _float_array(
            self.pointmap_camera, "pointmap_camera", allow_nonfinite=True
        )
        if pointmap.shape != rgb.shape:
            raise ValueError(
                "pointmap_camera must have the same (height, width, 3) shape as rgb"
            )

        geometry_confidence = _float_array(
            self.geometry_confidence, "geometry_confidence"
        )
        if geometry_confidence.shape != rgb.shape[:2]:
            raise ValueError(
                "geometry_confidence must have shape (height, width)"
            )

        pose = _float_array(
            self.pose_world_from_camera, "pose_world_from_camera"
        )
        if pose.shape != (4, 4):
            raise ValueError("pose_world_from_camera must have shape (4, 4)")
        if not np.allclose(pose[3], np.array([0, 0, 0, 1]), atol=1e-6):
            raise ValueError("pose_world_from_camera must be a homogeneous transform")

        descriptors = None
        if self.mast3r_descriptors is not None:
            descriptors = _float_array(
                self.mast3r_descriptors, "mast3r_descriptors"
            )
            if descriptors.ndim != 3 or descriptors.shape[:2] != rgb.shape[:2]:
                raise ValueError(
                    "mast3r_descriptors must have shape (height, width, dimension)"
                )

        descriptor_confidence = None
        if self.descriptor_confidence is not None:
            descriptor_confidence = _float_array(
                self.descriptor_confidence, "descriptor_confidence"
            )
            if descriptor_confidence.shape != rgb.shape[:2]:
                raise ValueError(
                    "descriptor_confidence must have shape (height, width)"
                )
            if descriptors is None:
                raise ValueError(
                    "descriptor_confidence cannot be set without mast3r_descriptors"
                )

        intrinsics = None
        if self.intrinsics is not None:
            intrinsics = _float_array(self.intrinsics, "intrinsics")
            if intrinsics.shape != (3, 3):
                raise ValueError("intrinsics must have shape (3, 3)")

        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "pointmap_camera", pointmap)
        object.__setattr__(self, "geometry_confidence", geometry_confidence)
        object.__setattr__(self, "pose_world_from_camera", pose)
        object.__setattr__(self, "mast3r_descriptors", descriptors)
        object.__setattr__(self, "descriptor_confidence", descriptor_confidence)
        object.__setattr__(self, "intrinsics", intrinsics)

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]

    def points_world(self) -> FloatArray:
        """Return the dense pointmap transformed into the common world frame."""

        height, width = self.image_shape
        points = self.pointmap_camera.reshape(-1, 3)
        linear = self.pose_world_from_camera[:3, :3]
        translation = self.pose_world_from_camera[:3, 3]
        transformed = points @ linear.T + translation
        return transformed.reshape(height, width, 3).astype(np.float32, copy=False)
