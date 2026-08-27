"""Persist filtered 2D masks and aligned 3D proposal evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from fact3r.proposals.lift_to_3d import LiftedProposal
from fact3r.proposals.proposal_pipeline import GeneratedProposal
from fact3r.reconstruction.keyframes import KeyframeRecord
from fact3r.visualization.alignment import write_alignment_ply


@dataclass(frozen=True, slots=True)
class SavedProposalFrame:
    """All lifted proposals produced for one complete keyframe."""

    frame_id: int
    timestamp: float | str | None
    proposals: tuple[LiftedProposal, ...]


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
        if lifted.appearance_descriptor is not None:
            payload["appearance_descriptor"] = lifted.appearance_descriptor
        if lifted.appearance_reliability is not None:
            payload["appearance_reliability"] = np.asarray(
                lifted.appearance_reliability, dtype=np.float32
            )
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


def load_proposal_run_manifest(
    proposal_directory: str | Path,
) -> dict[str, object]:
    """Load and validate the top-level SAM2 proposal-run manifest."""

    directory = Path(proposal_directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "fact3r-sam2-proposals":
        raise ValueError(f"unsupported proposal run in {manifest_path}")
    if manifest.get("version") != 1:
        raise ValueError(
            f"unsupported proposal-run version {manifest.get('version')}"
        )
    return manifest


def iter_saved_proposal_frames(
    proposal_directory: str | Path,
) -> Iterator[SavedProposalFrame]:
    """Stream complete-frame proposal batches saved by the SAM2 runner."""

    directory = Path(proposal_directory)
    run_manifest = load_proposal_run_manifest(directory)
    for run_entry in run_manifest["frames"]:
        frame_manifest_path = directory / run_entry["manifest"]
        frame_manifest = json.loads(
            frame_manifest_path.read_text(encoding="utf-8")
        )
        frame_id = int(frame_manifest["frame_id"])
        proposals: list[LiftedProposal] = []
        for entry in frame_manifest["proposals"]:
            with np.load(
                frame_manifest_path.parent / entry["file"], allow_pickle=False
            ) as payload:
                descriptors = (
                    np.array(payload["mast3r_descriptors"], copy=True)
                    if "mast3r_descriptors" in payload.files
                    else None
                )
                descriptor_confidence = (
                    np.array(payload["descriptor_confidence"], copy=True)
                    if "descriptor_confidence" in payload.files
                    else None
                )
                appearance_descriptor = (
                    np.array(payload["appearance_descriptor"], copy=True)
                    if "appearance_descriptor" in payload.files
                    else None
                )
                appearance_reliability = (
                    float(payload["appearance_reliability"])
                    if "appearance_reliability" in payload.files
                    else None
                )
                proposals.append(
                    LiftedProposal(
                        proposal_id=str(entry["proposal_id"]),
                        frame_id=frame_id,
                        timestamp=frame_manifest.get("timestamp"),
                        pixel_rc=np.array(payload["pixel_rc"], copy=True),
                        points_world=np.array(payload["points_world"], copy=True),
                        colours_rgb=np.array(payload["colours_rgb"], copy=True),
                        geometry_confidence=np.array(
                            payload["geometry_confidence"], copy=True
                        ),
                        mast3r_descriptors=descriptors,
                        descriptor_confidence=descriptor_confidence,
                        source_mask_area=int(entry["mask_area"]),
                        appearance_descriptor=appearance_descriptor,
                        appearance_reliability=appearance_reliability,
                    )
                )
        expected_count = int(frame_manifest["proposal_count"])
        if len(proposals) != expected_count:
            raise ValueError(
                f"frame {frame_id} declares {expected_count} proposals but "
                f"contains {len(proposals)} entries"
            )
        yield SavedProposalFrame(
            frame_id=frame_id,
            timestamp=frame_manifest.get("timestamp"),
            proposals=tuple(proposals),
        )
