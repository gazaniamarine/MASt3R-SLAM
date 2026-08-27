"""Persistent-map loop for visibility-conditioned residual transport."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from fact3r.association.baseline_mapper import (
    HungarianEntityMapper,
    HungarianMapConfig,
)
from fact3r.association.costs import TemporalEntityHint
from fact3r.association.residual_transport import (
    ResidualTransportConfig,
    ResidualTransportResult,
    associate_residual_transport,
)
from fact3r.association.tracklets import TrackletObservation
from fact3r.association.visibility import (
    VisibilityConfig,
    estimate_entity_visibility,
)
from fact3r.proposals.lift_to_3d import LiftedProposal
from fact3r.reconstruction.keyframes import KeyframeRecord


@dataclass(frozen=True, slots=True)
class DelayedCommitmentConfig:
    """Evidence required before an unmatched track becomes a map entity."""

    min_observations: int = 3
    min_mean_birth_residual_ratio: float = 0.55
    min_median_link_iou: float = 0.60
    max_centroid_step_m: float = 0.30
    max_missed_frames: int = 0

    def __post_init__(self) -> None:
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")
        for name in (
            "min_mean_birth_residual_ratio",
            "min_median_link_iou",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            not np.isfinite(self.max_centroid_step_m)
            or self.max_centroid_step_m < 0.0
        ):
            raise ValueError("max_centroid_step_m must be finite and non-negative")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")


class BirthCommitmentStatus(str, Enum):
    """Lifecycle decision applied after one complete-frame UOT solve."""

    IMMEDIATE = "immediate"
    DEFERRED = "deferred"
    CONFIRMED = "confirmed"
    HELD_EXISTING = "held_existing"


@dataclass(frozen=True, slots=True)
class PendingBirthSummary:
    track_id: str
    first_frame_id: int
    last_frame_id: int
    observation_count: int
    mean_birth_residual_ratio: float
    median_link_iou: float | None
    max_centroid_step_m: float
    blocking_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BirthCommitmentDecision:
    proposal_index: int
    proposal_id: str
    track_id: str | None
    status: BirthCommitmentStatus
    resolved_entity_id: str | None
    created_entity_id: str | None
    observation_count: int | None
    mean_birth_residual_ratio: float | None
    median_link_iou: float | None
    max_centroid_step_m: float | None
    blocking_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AppearanceMemoryDecision:
    proposal_id: str
    entity_id: str
    updated: bool
    proposal_reliability: float | None
    conditional_probability: float
    retained_ratio: float
    track_iou: float | None
    blocking_reasons: tuple[str, ...]


@dataclass(slots=True)
class _PendingBirthTrack:
    track_id: str
    first_frame_id: int
    first_timestamp: float | str | None
    last_frame_id: int
    last_seen_step: int
    observation_count: int
    birth_residual_ratio_sum: float
    link_ious: list[float]
    last_centroid_xyz: np.ndarray
    max_centroid_step_m: float
    latest_proposal: LiftedProposal

    @classmethod
    def start(
        cls,
        proposal: LiftedProposal,
        observation: TrackletObservation,
        *,
        birth_residual_ratio: float,
        frame_step: int,
    ) -> _PendingBirthTrack:
        link_ious = (
            [] if observation.link_iou is None else [observation.link_iou]
        )
        return cls(
            track_id=observation.track_id,
            first_frame_id=proposal.frame_id,
            first_timestamp=proposal.timestamp,
            last_frame_id=proposal.frame_id,
            last_seen_step=frame_step,
            observation_count=1,
            birth_residual_ratio_sum=birth_residual_ratio,
            link_ious=link_ious,
            last_centroid_xyz=np.asarray(proposal.centroid_xyz, dtype=np.float64),
            max_centroid_step_m=0.0,
            latest_proposal=proposal,
        )

    def observe(
        self,
        proposal: LiftedProposal,
        observation: TrackletObservation,
        *,
        birth_residual_ratio: float,
        frame_step: int,
    ) -> None:
        if proposal.frame_id == self.last_frame_id:
            raise ValueError("a tracklet may contribute at most one proposal per frame")
        centroid = np.asarray(proposal.centroid_xyz, dtype=np.float64)
        step_distance = float(np.linalg.norm(centroid - self.last_centroid_xyz))
        self.max_centroid_step_m = max(self.max_centroid_step_m, step_distance)
        self.last_centroid_xyz = centroid
        self.last_frame_id = proposal.frame_id
        self.last_seen_step = frame_step
        self.observation_count += 1
        self.birth_residual_ratio_sum += birth_residual_ratio
        if observation.link_iou is not None:
            self.link_ious.append(observation.link_iou)
        self.latest_proposal = proposal

    def summary(
        self, config: DelayedCommitmentConfig
    ) -> PendingBirthSummary:
        mean_birth = self.birth_residual_ratio_sum / self.observation_count
        median_iou = (
            None if not self.link_ious else float(np.median(self.link_ious))
        )
        blocking: list[str] = []
        if self.observation_count < config.min_observations:
            blocking.append("insufficient_observations")
        if mean_birth < config.min_mean_birth_residual_ratio:
            blocking.append("weak_birth_residual")
        if median_iou is None or median_iou < config.min_median_link_iou:
            blocking.append("weak_temporal_link")
        if self.max_centroid_step_m > config.max_centroid_step_m:
            blocking.append("inconsistent_3d_centroid")
        return PendingBirthSummary(
            track_id=self.track_id,
            first_frame_id=self.first_frame_id,
            last_frame_id=self.last_frame_id,
            observation_count=self.observation_count,
            mean_birth_residual_ratio=mean_birth,
            median_link_iou=median_iou,
            max_centroid_step_m=self.max_centroid_step_m,
            blocking_reasons=tuple(blocking),
        )


@dataclass(frozen=True, slots=True)
class ResidualTransportFrameMappingResult:
    frame_id: int
    timestamp: float | str | None
    proposal_count: int
    entity_count_before: int
    assignment: ResidualTransportResult
    created_entity_ids: tuple[str, ...]
    entity_count_after: int
    birth_decisions: tuple[BirthCommitmentDecision, ...]
    expired_pending_track_ids: tuple[str, ...]
    resolved_pending_track_ids: tuple[str, ...]
    pending_track_count_after: int
    appearance_memory_decisions: tuple[AppearanceMemoryDecision, ...]


class VisibilityResidualEntityMapper(HungarianEntityMapper):
    """Condition transport on current visibility, then reuse map updates.

    With no delayed-commitment configuration this reproduces the immediate-birth
    UOT comparison. When configured, rejected proposals accumulate evidence under
    their short-term track ID before any persistent entity is created.
    """

    def __init__(
        self,
        map_config: HungarianMapConfig | None = None,
        transport_config: ResidualTransportConfig | None = None,
        visibility_config: VisibilityConfig | None = None,
        delayed_commitment_config: DelayedCommitmentConfig | None = None,
    ) -> None:
        super().__init__(map_config)
        self.transport_config = (
            ResidualTransportConfig()
            if transport_config is None
            else transport_config
        )
        self.visibility_config = (
            VisibilityConfig() if visibility_config is None else visibility_config
        )
        self.delayed_commitment_config = delayed_commitment_config
        self._pending_births: dict[str, _PendingBirthTrack] = {}
        self._track_entity_ids: dict[str, str] = {}
        self._frame_step = 0

    @property
    def pending_births(self) -> tuple[PendingBirthSummary, ...]:
        if self.delayed_commitment_config is None:
            return ()
        return tuple(
            self._pending_births[track_id].summary(
                self.delayed_commitment_config
            )
            for track_id in sorted(self._pending_births)
        )

    @property
    def committed_track_entities(self) -> Mapping[str, str]:
        return dict(self._track_entity_ids)

    def process_frame(
        self,
        proposals: Sequence[LiftedProposal],
        *,
        keyframe: KeyframeRecord,
        temporal_hints: Mapping[str, TemporalEntityHint] | None = None,
        tracklet_observations: Mapping[str, TrackletObservation] | None = None,
    ) -> ResidualTransportFrameMappingResult:
        proposals = tuple(proposals)
        if any(proposal.frame_id != keyframe.frame_id for proposal in proposals):
            raise ValueError("every proposal must belong to the keyframe")
        observations = (
            {} if tracklet_observations is None else dict(tracklet_observations)
        )
        proposal_ids = {proposal.proposal_id for proposal in proposals}
        unknown_observations = set(observations) - proposal_ids
        if unknown_observations:
            raise ValueError("tracklet observations contain unknown proposals")
        if any(
            observation.proposal_id != proposal_id
            or observation.frame_id != keyframe.frame_id
            for proposal_id, observation in observations.items()
        ):
            raise ValueError(
                "tracklet observation identity must match its proposal and frame"
            )
        if self.delayed_commitment_config is not None:
            missing = proposal_ids - set(observations)
            if missing:
                raise ValueError(
                    "delayed commitment requires one tracklet observation "
                    "for every proposal"
                )
            track_ids = [
                observations[proposal.proposal_id].track_id
                for proposal in proposals
            ]
            if len(track_ids) != len(set(track_ids)):
                raise ValueError("a tracklet ID may occur only once per frame")
        entity_count_before = len(self._entities)
        visibility = estimate_entity_visibility(
            tuple(self._entities), keyframe, self.visibility_config
        )
        assignment = associate_residual_transport(
            proposals,
            tuple(self._entities),
            visibility,
            cost_config=self.config.pairwise_cost,
            transport_config=self.transport_config,
            max_match_cost=self.config.max_match_cost,
            temporal_hints=temporal_hints,
        )

        resolved_pending: list[str] = []
        appearance_decisions: list[AppearanceMemoryDecision] = []
        for match in assignment.matches:
            proposal = proposals[match.proposal_index]
            observation = observations.get(match.proposal_id)
            track_iou = (
                None if observation is None else observation.link_iou
            )
            memory_config = self.config.appearance_memory
            blocking: list[str] = []
            if proposal.appearance_descriptor is None:
                blocking.append("appearance_unavailable")
            if (
                proposal.appearance_reliability is not None
                and proposal.appearance_reliability
                < memory_config.min_update_reliability
            ):
                blocking.append("low_appearance_reliability")
            if (
                match.conditional_probability
                < memory_config.min_conditional_probability
            ):
                blocking.append("low_conditional_probability")
            if match.retained_ratio < memory_config.min_retained_ratio:
                blocking.append("low_retained_ratio")
            if track_iou is not None and track_iou < memory_config.min_track_iou:
                blocking.append("weak_temporal_link")
            appearance_updated = self._update_entity(
                self._entities[match.entity_index],
                proposal,
                keyframe.timestamp,
                update_appearance=not blocking,
            )
            if not blocking and proposal.appearance_descriptor is not None:
                if not appearance_updated:
                    blocking.append("redundant_or_lower_quality_view")
            appearance_decisions.append(
                AppearanceMemoryDecision(
                    proposal_id=match.proposal_id,
                    entity_id=match.entity_id,
                    updated=appearance_updated,
                    proposal_reliability=proposal.appearance_reliability,
                    conditional_probability=match.conditional_probability,
                    retained_ratio=match.retained_ratio,
                    track_iou=track_iou,
                    blocking_reasons=tuple(blocking),
                )
            )
            if observation is not None:
                self._track_entity_ids[observation.track_id] = match.entity_id
                if self._pending_births.pop(observation.track_id, None) is not None:
                    resolved_pending.append(observation.track_id)

        decisions: list[BirthCommitmentDecision] = []
        created_entity_ids: list[str] = []
        if self.delayed_commitment_config is None:
            for unmatched in assignment.unmatched_proposals:
                entity = self._create_entity(
                    proposals[unmatched.proposal_index], keyframe.timestamp
                )
                created_entity_ids.append(entity.id)
                observation = observations.get(unmatched.proposal_id)
                if observation is not None:
                    self._track_entity_ids[observation.track_id] = entity.id
                decisions.append(
                    BirthCommitmentDecision(
                        proposal_index=unmatched.proposal_index,
                        proposal_id=unmatched.proposal_id,
                        track_id=(
                            None if observation is None else observation.track_id
                        ),
                        status=BirthCommitmentStatus.IMMEDIATE,
                        resolved_entity_id=entity.id,
                        created_entity_id=entity.id,
                        observation_count=None,
                        mean_birth_residual_ratio=None,
                        median_link_iou=None,
                        max_centroid_step_m=None,
                    )
                )
        else:
            config = self.delayed_commitment_config
            for unmatched in assignment.unmatched_proposals:
                proposal = proposals[unmatched.proposal_index]
                observation = observations[unmatched.proposal_id]
                known_entity_id = self._track_entity_ids.get(observation.track_id)
                if known_entity_id is not None:
                    decisions.append(
                        BirthCommitmentDecision(
                            proposal_index=unmatched.proposal_index,
                            proposal_id=unmatched.proposal_id,
                            track_id=observation.track_id,
                            status=BirthCommitmentStatus.HELD_EXISTING,
                            resolved_entity_id=known_entity_id,
                            created_entity_id=None,
                            observation_count=None,
                            mean_birth_residual_ratio=None,
                            median_link_iou=observation.link_iou,
                            max_centroid_step_m=None,
                        )
                    )
                    continue

                proposal_mass = float(
                    assignment.proposal_masses[unmatched.proposal_index]
                )
                birth_residual_ratio = (
                    0.0
                    if proposal_mass <= 0.0
                    else float(
                        assignment.proposal_birth_residuals[
                            unmatched.proposal_index
                        ]
                        / proposal_mass
                    )
                )
                pending = self._pending_births.get(observation.track_id)
                if pending is None:
                    pending = _PendingBirthTrack.start(
                        proposal,
                        observation,
                        birth_residual_ratio=birth_residual_ratio,
                        frame_step=self._frame_step,
                    )
                    self._pending_births[observation.track_id] = pending
                else:
                    pending.observe(
                        proposal,
                        observation,
                        birth_residual_ratio=birth_residual_ratio,
                        frame_step=self._frame_step,
                    )
                summary = pending.summary(config)
                entity_id: str | None = None
                status = BirthCommitmentStatus.DEFERRED
                if not summary.blocking_reasons:
                    entity = self._create_entity(
                        pending.latest_proposal, keyframe.timestamp
                    )
                    entity.first_seen_timestamp = pending.first_timestamp
                    entity_id = entity.id
                    created_entity_ids.append(entity.id)
                    status = BirthCommitmentStatus.CONFIRMED
                    self._track_entity_ids[observation.track_id] = entity.id
                    del self._pending_births[observation.track_id]
                decisions.append(
                    BirthCommitmentDecision(
                        proposal_index=unmatched.proposal_index,
                        proposal_id=unmatched.proposal_id,
                        track_id=observation.track_id,
                        status=status,
                        resolved_entity_id=entity_id,
                        created_entity_id=entity_id,
                        observation_count=summary.observation_count,
                        mean_birth_residual_ratio=(
                            summary.mean_birth_residual_ratio
                        ),
                        median_link_iou=summary.median_link_iou,
                        max_centroid_step_m=summary.max_centroid_step_m,
                        blocking_reasons=summary.blocking_reasons,
                    )
                )

        observed_track_ids = {
            observation.track_id for observation in observations.values()
        }
        expired_pending: list[str] = []
        if self.delayed_commitment_config is not None:
            for track_id, pending in tuple(self._pending_births.items()):
                if track_id in observed_track_ids:
                    continue
                missed_frames = self._frame_step - pending.last_seen_step
                if missed_frames > self.delayed_commitment_config.max_missed_frames:
                    expired_pending.append(track_id)
                    del self._pending_births[track_id]

        self._frame_step += 1
        return ResidualTransportFrameMappingResult(
            frame_id=keyframe.frame_id,
            timestamp=keyframe.timestamp,
            proposal_count=len(proposals),
            entity_count_before=entity_count_before,
            assignment=assignment,
            created_entity_ids=tuple(created_entity_ids),
            entity_count_after=len(self._entities),
            birth_decisions=tuple(decisions),
            expired_pending_track_ids=tuple(sorted(expired_pending)),
            resolved_pending_track_ids=tuple(sorted(set(resolved_pending))),
            pending_track_count_after=len(self._pending_births),
            appearance_memory_decisions=tuple(appearance_decisions),
        )
