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
from fact3r.association.tracklets import (
    TrackletLink,
    TrackletObservation,
    TrackletRun,
    link_propagated_masks,
    load_tracklet_run,
)

__all__ = [
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
    "UnmatchedProposal",
    "UnmatchedReason",
    "associate_hungarian",
    "build_pairwise_cost_matrix",
    "link_propagated_masks",
    "load_tracklet_run",
    "solve_hungarian",
]
