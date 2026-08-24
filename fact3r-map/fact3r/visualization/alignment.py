"""Export pointmaps and selected mask points in one world-coordinate PLY."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from fact3r.proposals.lift_to_3d import LiftedProposal
from fact3r.reconstruction.keyframes import KeyframeRecord


_PROPOSAL_COLOURS = np.asarray(
    [
        [255, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [0, 255, 0],
    ],
    dtype=np.uint8,
)


def _rgb_uint8(rgb: NDArray[np.generic]) -> NDArray[np.uint8]:
    values = np.asarray(rgb)
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(values.max()) <= 1.0:
            values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def write_alignment_ply(
    path: str | Path,
    keyframes: Iterable[KeyframeRecord],
    proposals: Iterable[LiftedProposal],
) -> Path:
    """Write world pointmaps, recolouring proposal pixels for visual inspection."""

    keyframes = tuple(keyframes)
    proposals = tuple(proposals)
    by_frame: dict[int, list[tuple[int, LiftedProposal]]] = {}
    for index, proposal in enumerate(proposals):
        by_frame.setdefault(proposal.frame_id, []).append((index, proposal))

    point_blocks: list[NDArray[np.floating]] = []
    colour_blocks: list[NDArray[np.uint8]] = []
    for keyframe in keyframes:
        points = keyframe.points_world().reshape(-1, 3)
        colours = _rgb_uint8(keyframe.rgb).reshape(-1, 3).copy()
        height, width = keyframe.image_shape
        for index, proposal in by_frame.get(keyframe.frame_id, []):
            linear_pixels = proposal.pixel_rc[:, 0] * width + proposal.pixel_rc[:, 1]
            colours[linear_pixels] = _PROPOSAL_COLOURS[
                index % len(_PROPOSAL_COLOURS)
            ]
        valid = np.all(np.isfinite(points), axis=1)
        point_blocks.append(points[valid])
        colour_blocks.append(colours[valid])

    if not point_blocks:
        raise ValueError("at least one keyframe is required")

    points = np.concatenate(point_blocks, axis=0)
    colours = np.concatenate(colour_blocks, axis=0)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment Fact3R Milestone 0 alignment regression\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, colour in zip(points, colours, strict=True):
            handle.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{colour[0]} {colour[1]} {colour[2]}\n"
            )
    return output

