"""Short-term proposal links obtained from SAM2 video propagation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from fact3r.association.costs import PairwiseCostMatrix
from fact3r.association.hungarian import solve_hungarian


TRACKLET_FORMAT = "fact3r-sam2-tracklets"
TRACKLET_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrackletLink:
    """One one-to-one propagated-mask link between adjacent keyframes."""

    source_proposal_index: int
    target_proposal_index: int
    source_proposal_id: str
    target_proposal_id: str
    mask_iou: float


@dataclass(frozen=True, slots=True)
class TrackletObservation:
    """Track identity and incoming evidence for one automatic proposal."""

    frame_id: int
    proposal_id: str
    track_id: str
    source_proposal_id: str | None
    link_iou: float | None


@dataclass(frozen=True, slots=True)
class TrackletRun:
    """Validated lookup view of a saved SAM2 tracklet artifact."""

    source_proposals: str
    model: str
    observations_by_frame: Mapping[int, tuple[TrackletObservation, ...]]


def binary_mask_iou_matrix(
    propagated_masks: Sequence[NDArray[np.bool_]],
    target_masks: Sequence[NDArray[np.bool_]],
) -> NDArray[np.float64]:
    """Return target-by-propagated IoU scores without resizing either side."""

    propagated = tuple(np.asarray(mask, dtype=bool) for mask in propagated_masks)
    targets = tuple(np.asarray(mask, dtype=bool) for mask in target_masks)
    shapes = {mask.shape for mask in (*propagated, *targets)}
    if len(shapes) > 1:
        raise ValueError("all propagated and target masks must share one image shape")
    scores = np.zeros((len(targets), len(propagated)), dtype=np.float64)
    for target_index, target in enumerate(targets):
        for source_index, source in enumerate(propagated):
            intersection = int(np.count_nonzero(target & source))
            union = int(np.count_nonzero(target | source))
            scores[target_index, source_index] = (
                0.0 if union == 0 else intersection / union
            )
    return scores


def link_propagated_masks(
    source_proposal_ids: Sequence[str],
    propagated_masks: Sequence[NDArray[np.bool_]],
    target_proposal_ids: Sequence[str],
    target_masks: Sequence[NDArray[np.bool_]],
    *,
    min_mask_iou: float = 0.30,
) -> tuple[TrackletLink, ...]:
    """Jointly match propagated masks to current automatic proposals by IoU."""

    if not 0.0 <= min_mask_iou <= 1.0:
        raise ValueError("min_mask_iou must be in [0, 1]")
    if len(source_proposal_ids) != len(propagated_masks):
        raise ValueError("source proposal IDs and propagated masks must align")
    if len(target_proposal_ids) != len(target_masks):
        raise ValueError("target proposal IDs and masks must align")

    ious = binary_mask_iou_matrix(propagated_masks, target_masks)
    candidate_mask = ious >= min_mask_iou
    costs = np.full(ious.shape, np.inf, dtype=np.float64)
    costs[candidate_mask] = 1.0 - ious[candidate_mask]
    result = solve_hungarian(
        PairwiseCostMatrix(
            proposal_ids=tuple(target_proposal_ids),
            entity_ids=tuple(source_proposal_ids),
            costs=costs,
            candidate_mask=candidate_mask,
            components={"mask_iou": np.where(candidate_mask, 1.0 - ious, np.nan)},
        ),
        max_match_cost=1.0 - min_mask_iou,
    )
    return tuple(
        TrackletLink(
            source_proposal_index=match.entity_index,
            target_proposal_index=match.proposal_index,
            source_proposal_id=match.entity_id,
            target_proposal_id=match.proposal_id,
            mask_iou=float(ious[match.proposal_index, match.entity_index]),
        )
        for match in result.matches
    )


def load_tracklet_run(path: str | Path) -> TrackletRun:
    """Load and validate the compact JSON emitted by the tracklet builder."""

    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != TRACKLET_FORMAT:
        raise ValueError(f"unsupported tracklet artifact in {manifest_path}")
    if payload.get("version") != TRACKLET_VERSION:
        raise ValueError(
            f"unsupported tracklet version {payload.get('version')}"
        )

    by_frame: dict[int, tuple[TrackletObservation, ...]] = {}
    seen_proposals: set[str] = set()
    for frame in payload["frames"]:
        frame_id = int(frame["frame_id"])
        if frame_id in by_frame:
            raise ValueError(f"duplicate tracklet frame {frame_id}")
        observations: list[TrackletObservation] = []
        for entry in frame["observations"]:
            proposal_id = str(entry["proposal_id"])
            if proposal_id in seen_proposals:
                raise ValueError(f"duplicate tracklet proposal {proposal_id!r}")
            seen_proposals.add(proposal_id)
            link_iou = entry.get("link_iou")
            observation = TrackletObservation(
                frame_id=frame_id,
                proposal_id=proposal_id,
                track_id=str(entry["track_id"]),
                source_proposal_id=(
                    None
                    if entry.get("source_proposal_id") is None
                    else str(entry["source_proposal_id"])
                ),
                link_iou=None if link_iou is None else float(link_iou),
            )
            if observation.link_iou is not None and not (
                0.0 <= observation.link_iou <= 1.0
            ):
                raise ValueError("tracklet link_iou must be in [0, 1]")
            if (observation.source_proposal_id is None) != (
                observation.link_iou is None
            ):
                raise ValueError(
                    "source_proposal_id and link_iou must both be set or both be null"
                )
            observations.append(observation)
        by_frame[frame_id] = tuple(observations)

    return TrackletRun(
        source_proposals=str(payload["source_proposals"]),
        model=str(payload["model"]),
        observations_by_frame=by_frame,
    )
