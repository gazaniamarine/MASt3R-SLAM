"""Proposal-to-entity association models."""

from fact3r.association.baseline_mapper import (
    AppearanceMemoryConfig,
    FrameMappingResult,
    HungarianEntityMapper,
    HungarianMapConfig,
)
from fact3r.association.costs import (
    PairwiseCostConfig,
    PairwiseCostMatrix,
    TemporalEntityHint,
    build_pairwise_cost_matrix,
)
from fact3r.association.hungarian import (
    HardMatch,
    HungarianResult,
    UnmatchedProposal,
    UnmatchedReason,
    associate_hungarian,
    solve_hungarian,
)
from fact3r.association.sinkhorn import (
    BalancedSinkhornConfig,
    BalancedSinkhornResult,
    TransportMatch,
    associate_balanced_sinkhorn,
    solve_balanced_sinkhorn,
)
from fact3r.association.sinkhorn_mapper import (
    BalancedSinkhornEntityMapper,
    SinkhornFrameMappingResult,
)
from fact3r.association.residual_mapper import (
    AppearanceMemoryDecision,
    BirthCommitmentDecision,
    BirthCommitmentStatus,
    DelayedCommitmentConfig,
    PendingBirthSummary,
    ResidualTransportFrameMappingResult,
    VisibilityResidualEntityMapper,
)
from fact3r.association.residual_transport import (
    ResidualTransportConfig,
    ResidualTransportMatch,
    ResidualTransportResult,
    ResidualUnmatchedProposal,
    ResidualUnmatchedReason,
    associate_residual_transport,
    proposal_quality_scores,
    solve_residual_transport,
)
from fact3r.association.tracklets import (
    TrackletLink,
    TrackletObservation,
    TrackletRun,
    link_propagated_masks,
    load_tracklet_run,
)
from fact3r.association.visibility import (
    EntityVisibility,
    VisibilityConfig,
    estimate_entity_visibility,
)

__all__ = [
    "AppearanceMemoryConfig",
    "AppearanceMemoryDecision",
    "BalancedSinkhornConfig",
    "BalancedSinkhornEntityMapper",
    "BalancedSinkhornResult",
    "BirthCommitmentDecision",
    "BirthCommitmentStatus",
    "DelayedCommitmentConfig",
    "FrameMappingResult",
    "HardMatch",
    "HungarianEntityMapper",
    "HungarianMapConfig",
    "HungarianResult",
    "PairwiseCostConfig",
    "PairwiseCostMatrix",
    "PendingBirthSummary",
    "EntityVisibility",
    "ResidualTransportConfig",
    "ResidualTransportFrameMappingResult",
    "ResidualTransportMatch",
    "ResidualTransportResult",
    "ResidualUnmatchedProposal",
    "ResidualUnmatchedReason",
    "TemporalEntityHint",
    "TrackletLink",
    "TrackletObservation",
    "TrackletRun",
    "TransportMatch",
    "UnmatchedProposal",
    "UnmatchedReason",
    "VisibilityConfig",
    "VisibilityResidualEntityMapper",
    "associate_residual_transport",
    "associate_balanced_sinkhorn",
    "associate_hungarian",
    "build_pairwise_cost_matrix",
    "link_propagated_masks",
    "SinkhornFrameMappingResult",
    "load_tracklet_run",
    "estimate_entity_visibility",
    "proposal_quality_scores",
    "solve_balanced_sinkhorn",
    "solve_hungarian",
    "solve_residual_transport",
]
