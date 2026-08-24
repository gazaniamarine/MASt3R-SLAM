"""Loader for the deterministic Milestone 0 regression sequence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fact3r.reconstruction.keyframes import KeyframeRecord


@dataclass(frozen=True, slots=True)
class RegressionMask:
    proposal_id: str
    frame_id: int
    mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RegressionSequence:
    keyframes: tuple[KeyframeRecord, ...]
    masks: tuple[RegressionMask, ...]


def load_regression_sequence(path: str | Path) -> RegressionSequence:
    """Load a small, JSON-encoded sequence without model or dataset dependencies."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    keyframes = tuple(
        KeyframeRecord(
            frame_id=frame["frame_id"],
            timestamp=frame.get("timestamp"),
            rgb=np.asarray(frame["rgb"], dtype=np.uint8),
            pointmap_camera=np.asarray(frame["pointmap_camera"], dtype=np.float32),
            geometry_confidence=np.asarray(
                frame["geometry_confidence"], dtype=np.float32
            ),
            pose_world_from_camera=np.asarray(
                frame["pose_world_from_camera"], dtype=np.float32
            ),
            mast3r_descriptors=np.asarray(
                frame["mast3r_descriptors"], dtype=np.float32
            ),
            descriptor_confidence=np.asarray(
                frame["descriptor_confidence"], dtype=np.float32
            ),
        )
        for frame in payload["keyframes"]
    )
    by_id = {keyframe.frame_id: keyframe for keyframe in keyframes}
    if len(by_id) != len(keyframes):
        raise ValueError("regression keyframe IDs must be unique")

    masks = tuple(
        RegressionMask(
            proposal_id=entry["proposal_id"],
            frame_id=entry["frame_id"],
            mask=np.asarray(entry["mask"], dtype=bool),
        )
        for entry in payload["masks"]
    )
    for mask in masks:
        if mask.frame_id not in by_id:
            raise ValueError(f"mask references unknown frame {mask.frame_id}")
        if mask.mask.shape != by_id[mask.frame_id].image_shape:
            raise ValueError(
                f"mask {mask.proposal_id} shape does not match its keyframe"
            )

    return RegressionSequence(keyframes=keyframes, masks=masks)

