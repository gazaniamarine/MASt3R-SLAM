"""Persistent-map loop for the balanced Sinkhorn comparison baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from fact3r.association.baseline_mapper import (
    HungarianEntityMapper,
    HungarianMapConfig,
)
from fact3r.association.costs import TemporalEntityHint
from fact3r.association.sinkhorn import (
    BalancedSinkhornConfig,
    BalancedSinkhornResult,
    associate_balanced_sinkhorn,
)
from fact3r.proposals.lift_to_3d import LiftedProposal


@dataclass(frozen=True, slots=True)
class SinkhornFrameMappingResult:
    frame_id: int
    timestamp: float | str | None
    proposal_count: int
    entity_count_before: int
    assignment: BalancedSinkhornResult
    created_entity_ids: tuple[str, ...]
    entity_count_after: int


class BalancedSinkhornEntityMapper(HungarianEntityMapper):
    """Apply row-wise transport commitments while reusing entity updates."""

    def __init__(
        self,
        map_config: HungarianMapConfig | None = None,
        sinkhorn_config: BalancedSinkhornConfig | None = None,
    ) -> None:
        super().__init__(map_config)
        self.sinkhorn_config = (
            BalancedSinkhornConfig()
            if sinkhorn_config is None
            else sinkhorn_config
        )

    def process_frame(
        self,
        proposals: Sequence[LiftedProposal],
        *,
        frame_id: int,
        timestamp: float | str | None = None,
        temporal_hints: Mapping[str, TemporalEntityHint] | None = None,
    ) -> SinkhornFrameMappingResult:
        proposals = tuple(proposals)
        if any(proposal.frame_id != frame_id for proposal in proposals):
            raise ValueError("every proposal must belong to the processed frame")
        entity_count_before = len(self._entities)
        assignment = associate_balanced_sinkhorn(
            proposals,
            tuple(self._entities),
            cost_config=self.config.pairwise_cost,
            sinkhorn_config=self.sinkhorn_config,
            max_match_cost=self.config.max_match_cost,
            temporal_hints=temporal_hints,
        )

        for match in assignment.matches:
            self._update_entity(
                self._entities[match.entity_index],
                proposals[match.proposal_index],
                timestamp,
            )
        created_entity_ids = tuple(
            self._create_entity(proposals[index], timestamp).id
            for index in assignment.unmatched_proposal_indices
        )
        return SinkhornFrameMappingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            proposal_count=len(proposals),
            entity_count_before=entity_count_before,
            assignment=assignment,
            created_entity_ids=created_entity_ids,
            entity_count_after=len(self._entities),
        )
