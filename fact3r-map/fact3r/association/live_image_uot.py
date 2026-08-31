"""Stateful image-UOT association for causal live-video processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from fact3r.association.image_uot import (
    ImageTrackEvidence,
    build_image_uot_cost_matrix,
)
from fact3r.association.residual_transport import (
    ResidualTransportConfig,
    solve_residual_transport,
)
from fact3r.association.visibility import EntityVisibility
from fact3r.proposals.mask_generator import MaskProposal2D


@dataclass(frozen=True, slots=True)
class LiveTrackletEvidence:
    track_id: str
    source_proposal_id: str | None
    link_iou: float | None


@dataclass(frozen=True, slots=True)
class LiveAssignment:
    entity_id: str
    status: str
    confidence: float


@dataclass(slots=True)
class _TrackState:
    evidence: ImageTrackEvidence
    observation_count: int


class LiveImageUOTMapper:
    """Incrementally associate one current proposal set with active entities."""

    def __init__(
        self,
        *,
        max_track_gap: int = 3,
        max_match_cost: float = 0.70,
    ) -> None:
        if max_track_gap <= 0:
            raise ValueError("max_track_gap must be positive")
        self.max_track_gap = max_track_gap
        self.max_match_cost = max_match_cost
        self._tracks: dict[str, _TrackState] = {}
        self._next_entity = 0
        self._committed_track_entities: dict[str, str] = {}
        self.total_matches = 0
        self.total_births = 0
        self.frames: list[dict[str, object]] = []
        self._transport_config = ResidualTransportConfig(
            min_retained_ratio=0.15,
            min_conditional_probability=0.45,
        )

    @property
    def entity_count(self) -> int:
        return len(self._tracks)

    @property
    def committed_track_entities(self) -> Mapping[str, str]:
        return dict(self._committed_track_entities)

    def update(
        self,
        *,
        frame_id: int,
        proposals: Sequence[MaskProposal2D],
        embeddings: NDArray[np.floating],
        tracklets: Mapping[str, LiveTrackletEvidence],
    ) -> tuple[dict[str, LiveAssignment], dict[str, object]]:
        proposals = tuple(proposals)
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(proposals):
            raise ValueError("one embedding row is required per live proposal")
        proposal_ids = [proposal.proposal_id for proposal in proposals]
        active = [
            state
            for state in self._tracks.values()
            if 0 < frame_id - state.evidence.last_frame_id <= self.max_track_gap
        ]
        source_ids = [
            None
            if tracklets.get(proposal_id) is None
            else tracklets[proposal_id].source_proposal_id
            for proposal_id in proposal_ids
        ]
        link_ious = [
            None
            if tracklets.get(proposal_id) is None
            else tracklets[proposal_id].link_iou
            for proposal_id in proposal_ids
        ]
        matrix = build_image_uot_cost_matrix(
            proposal_ids,
            [proposal.mask for proposal in proposals],
            vectors,
            source_ids,
            link_ious,
            [state.evidence for state in active],
            frame_id=frame_id,
            pair_source_frame_id=None,
            source_xy=None,
            target_xy=None,
            appearance_weight=0.20,
            sam2_weight=0.35,
            mast3r_weight=0.0,
            min_mast3r_support=float("inf"),
            max_track_gap=self.max_track_gap,
        )
        visibility = tuple(
            EntityVisibility(
                entity_id=state.evidence.entity_id,
                score=float(
                    np.exp(-0.5 * (frame_id - state.evidence.last_frame_id - 1))
                ),
                in_frustum_fraction=1.0,
                unoccluded_fraction=1.0,
                sampled_point_count=0,
                projected_point_count=0,
                visible_point_count=0,
                used_intrinsics_fallback=True,
            )
            for state in active
        )
        quality = np.asarray(
            [np.clip(proposal.score, 0.0, 1.0) for proposal in proposals]
        )
        result = solve_residual_transport(
            matrix,
            quality,
            visibility,
            config=self._transport_config,
            max_match_cost=self.max_match_cost,
        )
        best_by_entity: dict[str, tuple[tuple[float, float, float], object]] = {}
        for match in result.matches:
            key = (
                match.conditional_probability,
                match.transport_mass,
                -match.cost,
            )
            current = best_by_entity.get(match.entity_id)
            if current is None or key > current[0]:
                best_by_entity[match.entity_id] = (key, match)
        selected = {
            match.proposal_index: match for _, match in best_by_entity.values()
        }
        duplicates = {
            match.proposal_index: match
            for match in result.matches
            if match.proposal_index not in selected
            and match.entity_id in best_by_entity
        }

        assignments: dict[str, LiveAssignment] = {}
        frame_matches: list[dict[str, object]] = []
        unmatched: list[dict[str, object]] = []
        for proposal_index, proposal in enumerate(proposals):
            proposal_id = proposal.proposal_id
            tracklet = tracklets.get(proposal_id)
            tracklet_payload = (
                None
                if tracklet is None
                else {
                    "track_id": tracklet.track_id,
                    "source_proposal_id": tracklet.source_proposal_id,
                    "link_iou": tracklet.link_iou,
                }
            )
            match = selected.get(proposal_index)
            duplicate = duplicates.get(proposal_index)
            vector = vectors[proposal_index]
            if match is None and duplicate is not None:
                entity_id = duplicate.entity_id
                confidence = float(duplicate.conditional_probability)
                assignments[proposal_id] = LiveAssignment(
                    entity_id, "held_existing", confidence
                )
                unmatched.append(
                    {
                        "proposal_id": proposal_id,
                        "reason": "duplicate_same_entity_transport",
                        "resolved_entity_id": entity_id,
                        "commitment_status": "held_existing",
                        "track_id": None if tracklet is None else tracklet.track_id,
                        "tracklet": tracklet_payload,
                        "cost": duplicate.cost,
                        "conditional_probability": confidence,
                    }
                )
                if tracklet is not None:
                    self._committed_track_entities[tracklet.track_id] = entity_id
                continue
            if match is None:
                entity_id = f"image-entity-{self._next_entity:06d}"
                self._next_entity += 1
                self.total_births += 1
                count = 1
                prototype = vector
                confidence = float(np.clip(proposal.score, 0.0, 1.0))
                assignments[proposal_id] = LiveAssignment(
                    entity_id, "confirmed_birth", confidence
                )
                unmatched.append(
                    {
                        "proposal_id": proposal_id,
                        "reason": "uot_birth_residual",
                        "created_entity_id": entity_id,
                        "commitment_status": "confirmed",
                        "track_id": None if tracklet is None else tracklet.track_id,
                        "tracklet": tracklet_payload,
                    }
                )
            else:
                entity_id = match.entity_id
                self.total_matches += 1
                state = self._tracks[entity_id]
                count = state.observation_count + 1
                prototype = (
                    state.evidence.prototype * state.observation_count + vector
                ) / count
                confidence = float(match.conditional_probability)
                assignments[proposal_id] = LiveAssignment(
                    entity_id, "matched", confidence
                )
                frame_matches.append(
                    {
                        "proposal_id": proposal_id,
                        "entity_id": entity_id,
                        "cost": match.cost,
                        "transport_mass": match.transport_mass,
                        "retained_ratio": match.retained_ratio,
                        "conditional_probability": confidence,
                        "tracklet": tracklet_payload,
                        "appearance_similarity": float(
                            matrix.components["appearance_similarity"][
                                proposal_index, match.entity_index
                            ]
                        ),
                        "sam2_link_iou": float(
                            matrix.components["sam2_link_iou"][
                                proposal_index, match.entity_index
                            ]
                        ),
                        "mast3r_mask_support": 0.0,
                    }
                )
            self._tracks[entity_id] = _TrackState(
                evidence=ImageTrackEvidence(
                    entity_id=entity_id,
                    last_frame_id=frame_id,
                    last_proposal_id=proposal_id,
                    prototype=np.asarray(prototype, dtype=np.float32),
                    last_mask=proposal.mask,
                ),
                observation_count=count,
            )
            if tracklet is not None:
                self._committed_track_entities[tracklet.track_id] = entity_id

        frame_payload = {
            "frame_id": frame_id,
            "matches": frame_matches,
            "unmatched_proposals": unmatched,
            "created_entity_ids": [
                str(item["created_entity_id"])
                for item in unmatched
                if item.get("created_entity_id") is not None
            ],
            "entity_count_after": self.entity_count,
            "uot": {
                "converged": result.converged,
                "iterations": result.iterations,
                "fixed_point_error": result.fixed_point_error,
                "unmatched_reason_counts": result.unmatched_reason_counts,
            },
        }
        self.frames.append(frame_payload)
        return assignments, frame_payload
