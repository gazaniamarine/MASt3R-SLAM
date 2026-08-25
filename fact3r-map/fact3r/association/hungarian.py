"""Exact one-to-one Hungarian baseline over a precomputed cost matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from fact3r.association.costs import (
    PairwiseCostConfig,
    PairwiseCostMatrix,
    TemporalEntityHint,
    build_pairwise_cost_matrix,
)
from fact3r.entities.entity import Entity
from fact3r.proposals.lift_to_3d import LiftedProposal


@dataclass(frozen=True, slots=True)
class HardMatch:
    proposal_index: int
    entity_index: int
    proposal_id: str
    entity_id: str
    cost: float


class UnmatchedReason(str, Enum):
    """Observable reason why a proposal did not receive a hard entity ID."""

    EMPTY_MAP = "empty_map"
    NO_SPATIAL_CANDIDATE = "no_spatial_candidate"
    COST_ABOVE_THRESHOLD = "cost_above_threshold"
    ASSIGNMENT_COMPETITION = "assignment_competition"


@dataclass(frozen=True, slots=True)
class UnmatchedProposal:
    proposal_index: int
    proposal_id: str
    reason: UnmatchedReason
    best_entity_index: int | None = None
    best_entity_id: str | None = None
    best_cost: float | None = None


@dataclass(frozen=True, slots=True)
class HungarianResult:
    cost_matrix: PairwiseCostMatrix
    matches: tuple[HardMatch, ...]
    unmatched_proposal_indices: tuple[int, ...]
    unmatched_entity_indices: tuple[int, ...]
    unmatched_proposals: tuple[UnmatchedProposal, ...]

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


def _describe_unmatched_proposals(
    cost_matrix: PairwiseCostMatrix,
    unmatched_indices: tuple[int, ...],
    max_match_cost: float,
) -> tuple[UnmatchedProposal, ...]:
    entity_count = len(cost_matrix.entity_ids)
    descriptions: list[UnmatchedProposal] = []
    for proposal_index in unmatched_indices:
        if entity_count == 0:
            descriptions.append(
                UnmatchedProposal(
                    proposal_index=proposal_index,
                    proposal_id=cost_matrix.proposal_ids[proposal_index],
                    reason=UnmatchedReason.EMPTY_MAP,
                )
            )
            continue

        candidate_indices = np.flatnonzero(
            cost_matrix.candidate_mask[proposal_index]
        )
        if len(candidate_indices) == 0:
            descriptions.append(
                UnmatchedProposal(
                    proposal_index=proposal_index,
                    proposal_id=cost_matrix.proposal_ids[proposal_index],
                    reason=UnmatchedReason.NO_SPATIAL_CANDIDATE,
                )
            )
            continue

        candidate_costs = cost_matrix.costs[proposal_index, candidate_indices]
        best_offset = int(np.argmin(candidate_costs))
        best_entity_index = int(candidate_indices[best_offset])
        best_cost = float(candidate_costs[best_offset])
        reason = (
            UnmatchedReason.COST_ABOVE_THRESHOLD
            if best_cost > max_match_cost
            else UnmatchedReason.ASSIGNMENT_COMPETITION
        )
        descriptions.append(
            UnmatchedProposal(
                proposal_index=proposal_index,
                proposal_id=cost_matrix.proposal_ids[proposal_index],
                reason=reason,
                best_entity_index=best_entity_index,
                best_entity_id=cost_matrix.entity_ids[best_entity_index],
                best_cost=best_cost,
            )
        )
    return tuple(descriptions)


def _rectangular_linear_sum_assignment(
    costs: NDArray[np.floating],
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """Solve a finite rectangular assignment problem without SciPy.

    This is the shortest-augmenting-path form of the Hungarian algorithm. The
    implementation expects at most as many rows as columns, which the public
    baseline guarantees by adding one private unmatched column per proposal.
    """

    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2:
        raise ValueError("assignment costs must be a matrix")
    row_count, column_count = costs.shape
    if row_count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if row_count > column_count:
        raise ValueError("Hungarian implementation requires rows <= columns")
    if not np.all(np.isfinite(costs)):
        raise ValueError("Hungarian implementation requires finite costs")

    row_potential = np.zeros(row_count + 1, dtype=np.float64)
    column_potential = np.zeros(column_count + 1, dtype=np.float64)
    column_to_row = np.zeros(column_count + 1, dtype=np.int64)
    predecessor = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        column_to_row[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        visited = np.zeros(column_count + 1, dtype=bool)
        current_column = 0
        while True:
            visited[current_column] = True
            current_row = column_to_row[current_column]
            delta = np.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if visited[column]:
                    continue
                reduced_cost = (
                    costs[current_row - 1, column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced_cost < minimum[column]:
                    minimum[column] = reduced_cost
                    predecessor[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(column_count + 1):
                if visited[column]:
                    row_potential[column_to_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if column_to_row[current_column] == 0:
                break

        while True:
            previous_column = predecessor[current_column]
            column_to_row[current_column] = column_to_row[previous_column]
            current_column = previous_column
            if current_column == 0:
                break

    assigned_columns = np.empty(row_count, dtype=np.int64)
    for column in range(1, column_count + 1):
        assigned_row = column_to_row[column]
        if assigned_row != 0:
            assigned_columns[assigned_row - 1] = column - 1
    return np.arange(row_count, dtype=np.int64), assigned_columns


def solve_hungarian(
    cost_matrix: PairwiseCostMatrix,
    *,
    max_match_cost: float = 0.65,
) -> HungarianResult:
    """Return one-to-one hard matches and explicitly unmatched indices.

    Private unmatched columns are an implementation detail of this hard baseline,
    not the shared dustbin rows/columns introduced by the later Sinkhorn model.
    """

    if max_match_cost < 0.0 or not np.isfinite(max_match_cost):
        raise ValueError("max_match_cost must be finite and non-negative")
    proposal_count, entity_count = cost_matrix.costs.shape
    if proposal_count == 0 or entity_count == 0:
        unmatched_indices = tuple(range(proposal_count))
        return HungarianResult(
            cost_matrix=cost_matrix,
            matches=(),
            unmatched_proposal_indices=unmatched_indices,
            unmatched_entity_indices=tuple(range(entity_count)),
            unmatched_proposals=_describe_unmatched_proposals(
                cost_matrix, unmatched_indices, max_match_cost
            ),
        )

    forbidden_cost = max_match_cost + max(1.0, max_match_cost)
    augmented = np.full(
        (proposal_count, entity_count + proposal_count),
        forbidden_cost,
        dtype=np.float64,
    )
    allowed = cost_matrix.candidate_mask & (
        cost_matrix.costs <= max_match_cost
    )
    augmented[:, :entity_count][allowed] = cost_matrix.costs[allowed]
    for proposal_index in range(proposal_count):
        augmented[
            proposal_index, entity_count + proposal_index
        ] = max_match_cost

    rows, columns = _rectangular_linear_sum_assignment(augmented)
    matches: list[HardMatch] = []
    matched_proposals: set[int] = set()
    matched_entities: set[int] = set()
    for proposal_index, entity_index in zip(rows.tolist(), columns.tolist()):
        if entity_index >= entity_count or not allowed[proposal_index, entity_index]:
            continue
        matches.append(
            HardMatch(
                proposal_index=proposal_index,
                entity_index=entity_index,
                proposal_id=cost_matrix.proposal_ids[proposal_index],
                entity_id=cost_matrix.entity_ids[entity_index],
                cost=float(cost_matrix.costs[proposal_index, entity_index]),
            )
        )
        matched_proposals.add(proposal_index)
        matched_entities.add(entity_index)

    unmatched_proposal_indices = tuple(
        index
        for index in range(proposal_count)
        if index not in matched_proposals
    )
    return HungarianResult(
        cost_matrix=cost_matrix,
        matches=tuple(matches),
        unmatched_proposal_indices=unmatched_proposal_indices,
        unmatched_entity_indices=tuple(
            index
            for index in range(entity_count)
            if index not in matched_entities
        ),
        unmatched_proposals=_describe_unmatched_proposals(
            cost_matrix, unmatched_proposal_indices, max_match_cost
        ),
    )


def associate_hungarian(
    proposals: Sequence[LiftedProposal],
    entities: Sequence[Entity],
    *,
    cost_config: PairwiseCostConfig | None = None,
    max_match_cost: float = 0.65,
    temporal_hints: Mapping[str, TemporalEntityHint] | None = None,
) -> HungarianResult:
    """Build the shared costs and run the hard Hungarian baseline."""

    return solve_hungarian(
        build_pairwise_cost_matrix(
            proposals,
            entities,
            cost_config,
            temporal_hints=temporal_hints,
        ),
        max_match_cost=max_match_cost,
    )
