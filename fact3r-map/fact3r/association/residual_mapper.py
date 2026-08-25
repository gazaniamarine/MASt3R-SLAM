"""Persistent-map loop for visibility-conditioned residual transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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
from fact3r.association.visibility import (
    VisibilityConfig,
    estimate_entity_visibility,
)
from fact3r.proposals.lift_to_3d import LiftedProposal
from fact3r.reconstruction.keyframes import KeyframeRecord


@dataclass(frozen=True, slots=True)
class ResidualTransportFrameMappingResult:
    frame_id: int
    timestamp: float | str | None
    proposal_count: int
    entity_count_before: int
    assignment: ResidualTransportResult
    created_entity_ids: tuple[str, ...]
    entity_count_after: int


class VisibilityResidualEntityMapper(HungarianEntityMapper):
    """Condition transport on current visibility, then reuse map updates.

    This comparison stage still creates an entity immediately from every rejected
    proposal. Delayed birth confirmation is intentionally kept as the next isolated
    change; the saved birth residuals are the evidence it will accumulate.
    """

    def __init__(
        self,
        map_config: HungarianMapConfig | None = None,
        transport_config: ResidualTransportConfig | None = None,
        visibility_config: VisibilityConfig | None = None,
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

    def process_frame(
        self,
        proposals: Sequence[LiftedProposal],
        *,
        keyframe: KeyframeRecord,
        temporal_hints: Mapping[str, TemporalEntityHint] | None = None,
    ) -> ResidualTransportFrameMappingResult:
        proposals = tuple(proposals)
        if any(proposal.frame_id != keyframe.frame_id for proposal in proposals):
            raise ValueError("every proposal must belong to the keyframe")
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

        for match in assignment.matches:
            self._update_entity(
                self._entities[match.entity_index],
                proposals[match.proposal_index],
                keyframe.timestamp,
            )
        created_entity_ids = tuple(
            self._create_entity(proposals[index], keyframe.timestamp).id
            for index in assignment.unmatched_proposal_indices
        )
        return ResidualTransportFrameMappingResult(
            frame_id=keyframe.frame_id,
            timestamp=keyframe.timestamp,
            proposal_count=len(proposals),
            entity_count_before=entity_count_before,
            assignment=assignment,
            created_entity_ids=created_entity_ids,
            entity_count_after=len(self._entities),
        )
