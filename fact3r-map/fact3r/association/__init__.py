"""Proposal-to-entity association models."""

from fact3r.association.baseline_mapper import (
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
from fact3r.association.tracklets import (
    TrackletLink,
    TrackletObservation,
    TrackletRun,
    link_propagated_masks,
    load_tracklet_run,
)

__all__ = [
    "BalancedSinkhornConfig",
    "BalancedSinkhornEntityMapper",
    "BalancedSinkhornResult",
    "FrameMappingResult",
    "HardMatch",
    "HungarianEntityMapper",
    "HungarianMapConfig",
    "HungarianResult",
    "PairwiseCostConfig",
    "PairwiseCostMatrix",
    "TemporalEntityHint",
    "TrackletLink",
    "TrackletObservation",
    "TrackletRun",
    "TransportMatch",
    "UnmatchedProposal",
    "UnmatchedReason",
    "associate_balanced_sinkhorn",
    "associate_hungarian",
    "build_pairwise_cost_matrix",
    "link_propagated_masks",
    "SinkhornFrameMappingResult",
    "load_tracklet_run",
    "solve_balanced_sinkhorn",
    "solve_hungarian",
]
