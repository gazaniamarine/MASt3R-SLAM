"""Read-only adapter from MASt3R-SLAM frames to Fact3R keyframes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from fact3r.reconstruction.keyframes import KeyframeRecord


def _to_numpy(value: Any, name: str) -> NDArray[np.generic]:
    if value is None:
        raise ValueError(f"{name} is required")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _drop_leading_singletons(
    array: NDArray[np.generic], target_ndim: int
) -> NDArray[np.generic]:
    while array.ndim > target_ndim and array.shape[0] == 1:
        array = array[0]
    return array


def _reshape_vector_field(
    value: Any,
    height: int,
    width: int,
    name: str,
) -> NDArray[np.generic]:
    array = _drop_leading_singletons(_to_numpy(value, name), 3)
    if array.ndim == 3 and array.shape[:2] == (height, width):
        return array
    if array.ndim == 2 and array.shape[0] == height * width:
        return array.reshape(height, width, array.shape[-1])
    raise ValueError(
        f"{name} must have shape (H, W, D) or (H*W, D); got {array.shape}"
    )


def _reshape_scalar_field(
    value: Any,
    height: int,
    width: int,
    name: str,
) -> NDArray[np.generic]:
    array = _drop_leading_singletons(_to_numpy(value, name), 2)
    if array.shape == (height, width):
        return array
    if array.shape == (height * width, 1):
        return array.reshape(height, width)
    if array.shape == (height * width,):
        return array.reshape(height, width)
    raise ValueError(
        f"{name} must have shape (H, W), (H*W,), or (H*W, 1); got {array.shape}"
    )


def _pose_matrix(pose: Any) -> NDArray[np.generic]:
    matrix = pose.matrix() if hasattr(pose, "matrix") else pose
    array = _to_numpy(matrix, "T_WC")
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (4, 4):
        raise ValueError(
            "T_WC must be a 4x4 matrix or expose matrix() returning one; "
            f"got {array.shape}"
        )
    return array


def keyframe_record_from_mast3r(
    frame: Any,
    *,
    timestamp: float | str | None = None,
    descriptors: Any | None = None,
    descriptor_confidence: Any | None = None,
) -> KeyframeRecord:
    """Convert a MASt3R-SLAM ``Frame`` without mutating it.

    The current SLAM ``Frame`` does not retain downstream descriptor maps. Pass
    the dense MASt3R ``D`` and ``Q`` outputs explicitly when they are available.
    The adapter deliberately does not use ``frame.feat`` because encoder tokens
    are not equivalent to the dense downstream descriptor map.
    """

    rgb = _drop_leading_singletons(_to_numpy(frame.uimg, "frame.uimg"), 3)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"frame.uimg must have shape (H, W, 3); got {rgb.shape}")
    height, width = rgb.shape[:2]

    pointmap = _reshape_vector_field(
        frame.X_canon, height, width, "frame.X_canon"
    )
    if pointmap.shape[-1] != 3:
        raise ValueError("frame.X_canon must contain XYZ points")

    confidence_source = (
        frame.get_average_conf()
        if hasattr(frame, "get_average_conf")
        else frame.C
    )
    geometry_confidence = _reshape_scalar_field(
        confidence_source, height, width, "frame geometry confidence"
    )

    descriptor_map = None
    if descriptors is not None:
        descriptor_map = _reshape_vector_field(
            descriptors, height, width, "descriptors"
        )

    descriptor_confidence_map = None
    if descriptor_confidence is not None:
        descriptor_confidence_map = _reshape_scalar_field(
            descriptor_confidence, height, width, "descriptor_confidence"
        )

    intrinsics = getattr(frame, "K", None)
    if intrinsics is not None:
        intrinsics = _to_numpy(intrinsics, "frame.K")

    return KeyframeRecord(
        frame_id=int(frame.frame_id),
        timestamp=timestamp,
        rgb=rgb,
        pointmap_camera=pointmap,
        geometry_confidence=geometry_confidence,
        pose_world_from_camera=_pose_matrix(frame.T_WC),
        mast3r_descriptors=descriptor_map,
        descriptor_confidence=descriptor_confidence_map,
        intrinsics=intrinsics,
    )
