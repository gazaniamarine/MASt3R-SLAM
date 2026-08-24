"""Persist filtered 2D masks and aligned 3D proposal evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from fact3r.proposals.proposal_pipeline import GeneratedProposal
from fact3r.reconstruction.keyframes import KeyframeRecord
from fact3r.visualization.alignment import write_alignment_ply


def save_frame_proposals(
    output_directory: str | Path,
    keyframe: KeyframeRecord,
    proposals: Iterable[GeneratedProposal],
) -> dict[str, object]:
    proposals = tuple(proposals)
    frame_directory = Path(output_directory) / f"frame_{keyframe.frame_id:06d}"
    frame_directory.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for index, generated in enumerate(proposals):
        mask, lifted = generated.mask_2d, generated.lifted_3d
        filename = f"proposal_{index:04d}.npz"
        payload: dict[str, np.ndarray] = {
            "mask": mask.mask,
            "pixel_rc": lifted.pixel_rc,
            "points_world": lifted.points_world,
            "colours_rgb": lifted.colours_rgb,
            "geometry_confidence": lifted.geometry_confidence,
        }
        if lifted.mast3r_descriptors is not None:
            payload["mast3r_descriptors"] = lifted.mast3r_descriptors
        if lifted.descriptor_confidence is not None:
            payload["descriptor_confidence"] = lifted.descriptor_confidence
        np.savez_compressed(frame_directory / filename, **payload)
        entries.append(
            {
                "proposal_id": mask.proposal_id,
                "file": filename,
                "source": mask.source,
                "score": mask.score,
                "mask_area": mask.area,
                "lifted_point_count": len(lifted.points_world),
                "centroid_xyz": lifted.centroid_xyz.tolist(),
                "bounding_box_xyz": lifted.bounding_box_xyz.tolist(),
                "bounding_box_xyxy": (
                    None
                    if mask.bounding_box_xyxy is None
                    else mask.bounding_box_xyxy.tolist()
                ),
            }
        )

    visualization = frame_directory / "alignment.ply"
    write_alignment_ply(
        visualization,
        [keyframe],
        [proposal.lifted_3d for proposal in proposals],
    )
    manifest = {
        "frame_id": keyframe.frame_id,
        "timestamp": keyframe.timestamp,
        "image_shape": list(keyframe.image_shape),
        "proposal_count": len(proposals),
        "visualization": visualization.name,
        "proposals": entries,
    }
    (frame_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest

