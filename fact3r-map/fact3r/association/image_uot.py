"""Image-space costs for reconstruction-free unbalanced transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fact3r.association.costs import PairwiseCostMatrix


@dataclass(frozen=True, slots=True)
class ImageTrackEvidence:
    entity_id: str
    last_frame_id: int
    last_proposal_id: str
    prototype: NDArray[np.floating]
    last_mask: NDArray[np.bool_]


def mast3r_mask_correspondence_score(
    source_mask: NDArray[np.bool_],
    target_mask: NDArray[np.bool_],
    source_xy: NDArray[np.integer],
    target_xy: NDArray[np.integer],
) -> float:
    """Fraction of mask-supported matches that land inside both masks."""

    if len(source_xy) == 0:
        return 0.0
    source = np.asarray(source_xy, dtype=np.int64)
    target = np.asarray(target_xy, dtype=np.int64)
    source_inside = np.asarray(source_mask, dtype=bool)[source[:, 1], source[:, 0]]
    target_inside = np.asarray(target_mask, dtype=bool)[target[:, 1], target[:, 0]]
    denominator = min(int(source_inside.sum()), int(target_inside.sum()))
    return 0.0 if denominator == 0 else float(np.sum(source_inside & target_inside) / denominator)


def build_image_uot_cost_matrix(
    proposal_ids: Sequence[str],
    proposal_masks: Sequence[NDArray[np.bool_]],
    proposal_embeddings: NDArray[np.floating],
    proposal_source_ids: Sequence[str | None],
    proposal_link_ious: Sequence[float | None],
    tracks: Sequence[ImageTrackEvidence],
    *,
    frame_id: int,
    pair_source_frame_id: int | None,
    source_xy: NDArray[np.integer] | None,
    target_xy: NDArray[np.integer] | None,
    appearance_weight: float = 0.20,
    sam2_weight: float = 0.35,
    mast3r_weight: float = 0.45,
    min_sam2_iou: float = 0.20,
    min_mast3r_support: float = 0.08,
    min_reidentification_similarity: float = 0.80,
    max_track_gap: int = 3,
) -> PairwiseCostMatrix:
    """Fuse appearance, SAM2 propagation and MASt3R pair correspondences."""

    proposal_vectors = np.asarray(proposal_embeddings, dtype=np.float64)
    proposal_vectors /= np.maximum(
        np.linalg.norm(proposal_vectors, axis=1, keepdims=True), 1e-12
    )
    shape = (len(proposal_ids), len(tracks))
    appearance = np.full(shape, np.nan, dtype=np.float64)
    sam2 = np.zeros(shape, dtype=np.float64)
    mast3r = np.zeros(shape, dtype=np.float64)
    costs = np.full(shape, np.inf, dtype=np.float64)
    candidates = np.zeros(shape, dtype=bool)
    sam2_enabled = sam2_weight > 0.0 or np.isfinite(min_sam2_iou)
    mast3r_enabled = mast3r_weight > 0.0 or np.isfinite(min_mast3r_support)
    for proposal_index, proposal_id in enumerate(proposal_ids):
        for track_index, track in enumerate(tracks):
            gap = frame_id - track.last_frame_id
            if gap <= 0 or gap > max_track_gap:
                continue
            prototype = np.asarray(track.prototype, dtype=np.float64)
            prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
            cosine = float(np.clip(proposal_vectors[proposal_index] @ prototype, -1, 1))
            appearance[proposal_index, track_index] = (cosine + 1.0) / 2.0
            source_id = proposal_source_ids[proposal_index]
            if sam2_enabled and source_id == track.last_proposal_id:
                sam2[proposal_index, track_index] = float(
                    proposal_link_ious[proposal_index] or 0.0
                )
            if (
                mast3r_enabled
                and gap == 1
                and pair_source_frame_id == track.last_frame_id
                and source_xy is not None
                and target_xy is not None
            ):
                mast3r[proposal_index, track_index] = mast3r_mask_correspondence_score(
                    track.last_mask,
                    proposal_masks[proposal_index],
                    source_xy,
                    target_xy,
                )
            is_candidate = (
                sam2[proposal_index, track_index] >= min_sam2_iou
                or mast3r[proposal_index, track_index] >= min_mast3r_support
                or appearance[proposal_index, track_index]
                >= min_reidentification_similarity
            )
            if not is_candidate:
                continue
            if gap == 1:
                total_weight = appearance_weight + sam2_weight + mast3r_weight
                cost = (
                    appearance_weight * (1.0 - appearance[proposal_index, track_index])
                    + sam2_weight * (1.0 - sam2[proposal_index, track_index])
                    + mast3r_weight * (1.0 - mast3r[proposal_index, track_index])
                ) / total_weight
            else:
                cost = 1.0 - appearance[proposal_index, track_index]
            candidates[proposal_index, track_index] = True
            costs[proposal_index, track_index] = max(0.0, float(cost))
    return PairwiseCostMatrix(
        proposal_ids=tuple(str(value) for value in proposal_ids),
        entity_ids=tuple(track.entity_id for track in tracks),
        costs=costs,
        candidate_mask=candidates,
        components={
            "appearance_similarity": appearance,
            "sam2_link_iou": sam2,
            "mast3r_mask_support": mast3r,
        },
    )
