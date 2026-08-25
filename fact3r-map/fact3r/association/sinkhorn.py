"""Balanced entropic optimal transport over the shared association costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from fact3r.association.costs import (
    PairwiseCostConfig,
    PairwiseCostMatrix,
    TemporalEntityHint,
    build_pairwise_cost_matrix,
)
from fact3r.association.hungarian import UnmatchedProposal, UnmatchedReason
from fact3r.entities.entity import Entity
from fact3r.proposals.lift_to_3d import LiftedProposal


@dataclass(frozen=True, slots=True)
class BalancedSinkhornConfig:
    """Numerical parameters for the deliberately strict balanced baseline."""

    entropy_temperature: float = 0.05
    max_iterations: int = 300
    marginal_tolerance: float = 1e-6
    noncandidate_cost: float = 2.0

    def __post_init__(self) -> None:
        if self.entropy_temperature <= 0.0 or not np.isfinite(
            self.entropy_temperature
        ):
            raise ValueError("entropy_temperature must be finite and positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.marginal_tolerance <= 0.0 or not np.isfinite(
            self.marginal_tolerance
        ):
            raise ValueError("marginal_tolerance must be finite and positive")
        if self.noncandidate_cost <= 1.0 or not np.isfinite(
            self.noncandidate_cost
        ):
            raise ValueError("noncandidate_cost must be finite and greater than 1")


@dataclass(frozen=True, slots=True)
class TransportMatch:
    proposal_index: int
    entity_index: int
    proposal_id: str
    entity_id: str
    cost: float
    transport_mass: float
    row_probability: float


@dataclass(frozen=True, slots=True)
class BalancedSinkhornResult:
    cost_matrix: PairwiseCostMatrix
    transport_plan: NDArray[np.floating]
    proposal_marginals: NDArray[np.floating]
    entity_marginals: NDArray[np.floating]
    matches: tuple[TransportMatch, ...]
    unmatched_proposal_indices: tuple[int, ...]
    unmatched_entity_indices: tuple[int, ...]
    unmatched_proposals: tuple[UnmatchedProposal, ...]
    converged: bool
    iterations: int
    marginal_error: float
    noncandidate_mass: float

    @property
    def unmatched_proposal_ids(self) -> tuple[str, ...]:
        return tuple(
            self.cost_matrix.proposal_ids[index]
            for index in self.unmatched_proposal_indices
        )

    @property
    def unmatched_entity_ids(self) -> tuple[str, ...]:
        return tuple(
            self.cost_matrix.entity_ids[index]
            for index in self.unmatched_entity_indices
        )

    @property
    def unmatched_reason_counts(self) -> dict[str, int]:
        counts = {reason.value: 0 for reason in UnmatchedReason}
        for proposal in self.unmatched_proposals:
            counts[proposal.reason.value] += 1
        return counts


def _logsumexp(values: NDArray[np.floating], axis: int) -> NDArray[np.float64]:
    maximum = np.max(values, axis=axis, keepdims=True)
    reduced = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(reduced, axis=axis)


def _balanced_plan(
    costs: NDArray[np.floating],
    config: BalancedSinkhornConfig,
) -> tuple[NDArray[np.float64], bool, int, float]:
    row_count, column_count = costs.shape
    row_marginals = np.full(row_count, 1.0 / row_count, dtype=np.float64)
    column_marginals = np.full(
        column_count, 1.0 / column_count, dtype=np.float64
    )
    log_kernel = -np.asarray(costs, dtype=np.float64) / config.entropy_temperature
    log_rows = np.log(row_marginals)
    log_columns = np.log(column_marginals)
    log_u = np.zeros(row_count, dtype=np.float64)
    log_v = np.zeros(column_count, dtype=np.float64)
    converged = False
    marginal_error = np.inf
    plan = np.zeros_like(log_kernel)

    for iteration in range(1, config.max_iterations + 1):
        log_u = log_rows - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_columns - _logsumexp(
            log_kernel + log_u[:, None], axis=0
        )
        plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
        marginal_error = max(
            float(np.max(np.abs(plan.sum(axis=1) - row_marginals))),
            float(np.max(np.abs(plan.sum(axis=0) - column_marginals))),
        )
        if marginal_error <= config.marginal_tolerance:
            converged = True
            break
    return plan, converged, iteration, marginal_error


def solve_balanced_sinkhorn(
    cost_matrix: PairwiseCostMatrix,
    *,
    config: BalancedSinkhornConfig | None = None,
    max_match_cost: float = 0.65,
) -> BalancedSinkhornResult:
    """Solve fixed uniform marginals and immediately commit each viable row.

    Rows and columns with no spatial edge are excluded from the balanced problem
    and remain unmatched/unobserved. Inside the active submatrix, forbidden edges
    receive a high finite numerical cost so fixed marginals remain feasible. Hard
    commitments are still restricted to the original candidate mask.
    """

    config = BalancedSinkhornConfig() if config is None else config
    if max_match_cost < 0.0 or not np.isfinite(max_match_cost):
        raise ValueError("max_match_cost must be finite and non-negative")
    proposal_count, entity_count = cost_matrix.costs.shape
    plan = np.zeros((proposal_count, entity_count), dtype=np.float64)
    proposal_marginals = np.zeros(proposal_count, dtype=np.float64)
    entity_marginals = np.zeros(entity_count, dtype=np.float64)
    active_proposals = np.flatnonzero(np.any(cost_matrix.candidate_mask, axis=1))
    active_entities = np.flatnonzero(np.any(cost_matrix.candidate_mask, axis=0))

    converged = True
    iterations = 0
    marginal_error = 0.0
    noncandidate_mass = 0.0
    if len(active_proposals) and len(active_entities):
        candidate_submatrix = cost_matrix.candidate_mask[
            np.ix_(active_proposals, active_entities)
        ]
        cost_submatrix = np.where(
            candidate_submatrix,
            cost_matrix.costs[np.ix_(active_proposals, active_entities)],
            config.noncandidate_cost,
        )
        subplan, converged, iterations, marginal_error = _balanced_plan(
            cost_submatrix, config
        )
        plan[np.ix_(active_proposals, active_entities)] = subplan
        proposal_marginals[active_proposals] = 1.0 / len(active_proposals)
        entity_marginals[active_entities] = 1.0 / len(active_entities)
        noncandidate_mass = float(np.sum(subplan[~candidate_submatrix]))

    matches: list[TransportMatch] = []
    unmatched: list[UnmatchedProposal] = []
    for proposal_index in range(proposal_count):
        proposal_id = cost_matrix.proposal_ids[proposal_index]
        if entity_count == 0:
            unmatched.append(
                UnmatchedProposal(
                    proposal_index=proposal_index,
                    proposal_id=proposal_id,
                    reason=UnmatchedReason.EMPTY_MAP,
                )
            )
            continue
        candidate_indices = np.flatnonzero(
            cost_matrix.candidate_mask[proposal_index]
        )
        if len(candidate_indices) == 0:
            unmatched.append(
                UnmatchedProposal(
                    proposal_index=proposal_index,
                    proposal_id=proposal_id,
                    reason=UnmatchedReason.NO_SPATIAL_CANDIDATE,
                )
            )
            continue
        candidate_costs = cost_matrix.costs[proposal_index, candidate_indices]
        viable = candidate_indices[candidate_costs <= max_match_cost]
        if len(viable) == 0:
            best_offset = int(np.argmin(candidate_costs))
            best_entity_index = int(candidate_indices[best_offset])
            unmatched.append(
                UnmatchedProposal(
                    proposal_index=proposal_index,
                    proposal_id=proposal_id,
                    reason=UnmatchedReason.COST_ABOVE_THRESHOLD,
                    best_entity_index=best_entity_index,
                    best_entity_id=cost_matrix.entity_ids[best_entity_index],
                    best_cost=float(candidate_costs[best_offset]),
                )
            )
            continue
        viable_mass = plan[proposal_index, viable]
        entity_index = int(viable[int(np.argmax(viable_mass))])
        row_mass = proposal_marginals[proposal_index]
        match_mass = float(plan[proposal_index, entity_index])
        matches.append(
            TransportMatch(
                proposal_index=proposal_index,
                entity_index=entity_index,
                proposal_id=proposal_id,
                entity_id=cost_matrix.entity_ids[entity_index],
                cost=float(cost_matrix.costs[proposal_index, entity_index]),
                transport_mass=match_mass,
                row_probability=(
                    0.0 if row_mass <= 0.0 else match_mass / row_mass
                ),
            )
        )

    unmatched_indices = tuple(item.proposal_index for item in unmatched)
    matched_entities = {match.entity_index for match in matches}
    for array in (plan, proposal_marginals, entity_marginals):
        array.setflags(write=False)
    return BalancedSinkhornResult(
        cost_matrix=cost_matrix,
        transport_plan=plan,
        proposal_marginals=proposal_marginals,
        entity_marginals=entity_marginals,
        matches=tuple(matches),
        unmatched_proposal_indices=unmatched_indices,
        unmatched_entity_indices=tuple(
            index for index in range(entity_count) if index not in matched_entities
        ),
        unmatched_proposals=tuple(unmatched),
        converged=converged,
        iterations=iterations,
        marginal_error=marginal_error,
        noncandidate_mass=noncandidate_mass,
    )


def associate_balanced_sinkhorn(
    proposals: Sequence[LiftedProposal],
    entities: Sequence[Entity],
    *,
    cost_config: PairwiseCostConfig | None = None,
    sinkhorn_config: BalancedSinkhornConfig | None = None,
    max_match_cost: float = 0.65,
    temporal_hints: Mapping[str, TemporalEntityHint] | None = None,
) -> BalancedSinkhornResult:
    """Build the unchanged evidence matrix and solve balanced transport."""

    return solve_balanced_sinkhorn(
        build_pairwise_cost_matrix(
            proposals,
            entities,
            cost_config,
            temporal_hints=temporal_hints,
        ),
        config=sinkhorn_config,
        max_match_cost=max_match_cost,
    )
