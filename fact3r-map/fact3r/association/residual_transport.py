"""Visibility-conditioned unbalanced transport without shared dustbins."""

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
from fact3r.association.visibility import EntityVisibility
from fact3r.entities.entity import Entity
from fact3r.proposals.lift_to_3d import LiftedProposal


@dataclass(frozen=True, slots=True)
class ResidualTransportConfig:
    """Numerical and evidence parameters for generalized Sinkhorn scaling."""

    entropy_temperature: float = 0.05
    max_iterations: int = 2000
    fixed_point_tolerance: float = 1e-7
    proposal_relaxation_min: float = 0.05
    proposal_relaxation_max: float = 0.50
    entity_relaxation_min: float = 0.02
    entity_relaxation_max: float = 0.50
    mass_floor: float = 1e-3
    geometry_retention_weight: float = 0.70
    min_retained_ratio: float = 0.25
    min_conditional_probability: float = 0.50

    def __post_init__(self) -> None:
        positive = {
            "entropy_temperature": self.entropy_temperature,
            "fixed_point_tolerance": self.fixed_point_tolerance,
            "proposal_relaxation_min": self.proposal_relaxation_min,
            "proposal_relaxation_max": self.proposal_relaxation_max,
            "entity_relaxation_min": self.entity_relaxation_min,
            "entity_relaxation_max": self.entity_relaxation_max,
            "mass_floor": self.mass_floor,
        }
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in positive.values()
        ):
            raise ValueError(
                "transport temperatures, penalties, and floors must be positive"
            )
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.proposal_relaxation_min > self.proposal_relaxation_max:
            raise ValueError("proposal relaxation bounds are reversed")
        if self.entity_relaxation_min > self.entity_relaxation_max:
            raise ValueError("entity relaxation bounds are reversed")
        for name, value in (
            ("geometry_retention_weight", self.geometry_retention_weight),
            ("min_retained_ratio", self.min_retained_ratio),
            ("min_conditional_probability", self.min_conditional_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


class ResidualUnmatchedReason(str, Enum):
    EMPTY_MAP = "empty_map"
    NO_SPATIAL_CANDIDATE = "no_spatial_candidate"
    COST_ABOVE_THRESHOLD = "cost_above_threshold"
    LOW_RETAINED_MASS = "low_retained_mass"
    AMBIGUOUS_TRANSPORT = "ambiguous_transport"


@dataclass(frozen=True, slots=True)
class ResidualTransportMatch:
    proposal_index: int
    entity_index: int
    proposal_id: str
    entity_id: str
    cost: float
    transport_mass: float
    retained_ratio: float
    conditional_probability: float


@dataclass(frozen=True, slots=True)
class ResidualUnmatchedProposal:
    proposal_index: int
    proposal_id: str
    reason: ResidualUnmatchedReason
    best_entity_index: int | None = None
    best_entity_id: str | None = None
    best_cost: float | None = None
    retained_ratio: float | None = None
    conditional_probability: float | None = None


@dataclass(frozen=True, slots=True)
class ResidualTransportResult:
    cost_matrix: PairwiseCostMatrix
    transport_plan: NDArray[np.floating]
    proposal_masses: NDArray[np.floating]
    entity_masses: NDArray[np.floating]
    transported_proposal_masses: NDArray[np.floating]
    transported_entity_masses: NDArray[np.floating]
    proposal_birth_residuals: NDArray[np.floating]
    entity_miss_residuals: NDArray[np.floating]
    proposal_excess_masses: NDArray[np.floating]
    entity_excess_masses: NDArray[np.floating]
    proposal_quality: NDArray[np.floating]
    proposal_relaxation: NDArray[np.floating]
    entity_relaxation: NDArray[np.floating]
    entity_visibility: tuple[EntityVisibility, ...]
    matches: tuple[ResidualTransportMatch, ...]
    unmatched_proposal_indices: tuple[int, ...]
    unmatched_entity_indices: tuple[int, ...]
    unmatched_proposals: tuple[ResidualUnmatchedProposal, ...]
    converged: bool
    iterations: int
    fixed_point_error: float
    forbidden_mass: float

    @property
    def unmatched_entity_ids(self) -> tuple[str, ...]:
        return tuple(
            self.cost_matrix.entity_ids[index]
            for index in self.unmatched_entity_indices
        )

    @property
    def unmatched_reason_counts(self) -> dict[str, int]:
        counts = {reason.value: 0 for reason in ResidualUnmatchedReason}
        for proposal in self.unmatched_proposals:
            counts[proposal.reason.value] += 1
        return counts


def proposal_quality_scores(
    proposals: Sequence[LiftedProposal],
    temporal_hints: Mapping[str, TemporalEntityHint] | None = None,
    *,
    geometry_retention_weight: float = 0.70,
) -> NDArray[np.float64]:
    """Combine mask-to-3D retention with optional SAM2 tracklet support."""

    temporal_hints = {} if temporal_hints is None else temporal_hints
    qualities = np.empty(len(proposals), dtype=np.float64)
    for index, proposal in enumerate(proposals):
        retention = min(
            1.0,
            len(proposal.points_world) / max(1, proposal.source_mask_area),
        )
        hint = temporal_hints.get(proposal.proposal_id)
        temporal_support = 0.5 if hint is None else hint.confidence
        qualities[index] = (
            geometry_retention_weight * retention
            + (1.0 - geometry_retention_weight) * temporal_support
        )
    return qualities


def _masked_logsumexp(
    values: NDArray[np.float64], axis: int
) -> NDArray[np.float64]:
    maximum = np.max(values, axis=axis, keepdims=True)
    if np.any(~np.isfinite(maximum)):
        raise ValueError("every active transport row and column needs support")
    reduced = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(reduced, axis=axis)


def _unbalanced_plan(
    costs: NDArray[np.floating],
    support: NDArray[np.bool_],
    row_masses: NDArray[np.floating],
    column_masses: NDArray[np.floating],
    row_relaxation: NDArray[np.floating],
    column_relaxation: NDArray[np.floating],
    config: ResidualTransportConfig,
) -> tuple[NDArray[np.float64], bool, int, float]:
    log_kernel = np.full(costs.shape, -np.inf, dtype=np.float64)
    log_kernel[support] = (
        -np.asarray(costs, dtype=np.float64)[support]
        / config.entropy_temperature
    )
    log_rows = np.log(np.asarray(row_masses, dtype=np.float64))
    log_columns = np.log(np.asarray(column_masses, dtype=np.float64))
    row_power = np.asarray(row_relaxation, dtype=np.float64) / (
        np.asarray(row_relaxation, dtype=np.float64)
        + config.entropy_temperature
    )
    column_power = np.asarray(column_relaxation, dtype=np.float64) / (
        np.asarray(column_relaxation, dtype=np.float64)
        + config.entropy_temperature
    )
    log_u = np.zeros(len(row_masses), dtype=np.float64)
    log_v = np.zeros(len(column_masses), dtype=np.float64)
    converged = False
    fixed_point_error = np.inf

    for iteration in range(1, config.max_iterations + 1):
        next_log_u = row_power * (
            log_rows
            - _masked_logsumexp(log_kernel + log_v[None, :], axis=1)
        )
        next_log_v = column_power * (
            log_columns
            - _masked_logsumexp(log_kernel + next_log_u[:, None], axis=0)
        )
        fixed_point_error = max(
            float(np.max(np.abs(next_log_u - log_u))),
            float(np.max(np.abs(next_log_v - log_v))),
        )
        log_u, log_v = next_log_u, next_log_v
        if fixed_point_error <= config.fixed_point_tolerance:
            converged = True
            break

    plan = np.zeros(costs.shape, dtype=np.float64)
    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan[support] = np.exp(log_plan[support])
    return plan, converged, iteration, fixed_point_error


def solve_residual_transport(
    cost_matrix: PairwiseCostMatrix,
    proposal_quality: NDArray[np.floating],
    entity_visibility: Sequence[EntityVisibility],
    *,
    config: ResidualTransportConfig | None = None,
    max_match_cost: float = 0.65,
) -> ResidualTransportResult:
    """Solve strict-support UOT and expose unmatched mass as typed residuals."""

    config = ResidualTransportConfig() if config is None else config
    if max_match_cost < 0.0 or not np.isfinite(max_match_cost):
        raise ValueError("max_match_cost must be finite and non-negative")
    proposal_count, entity_count = cost_matrix.costs.shape
    quality = np.array(proposal_quality, dtype=np.float64, copy=True).reshape(-1)
    if quality.shape != (proposal_count,) or np.any(~np.isfinite(quality)):
        raise ValueError("proposal_quality must be one finite value per proposal")
    if np.any((quality < 0.0) | (quality > 1.0)):
        raise ValueError("proposal_quality values must be in [0, 1]")
    visibility = tuple(entity_visibility)
    if len(visibility) != entity_count:
        raise ValueError("entity_visibility must contain one entry per entity")
    if tuple(item.entity_id for item in visibility) != cost_matrix.entity_ids:
        raise ValueError("entity visibility order must match the cost matrix")
    visibility_scores = np.asarray(
        [item.score for item in visibility], dtype=np.float64
    )
    if np.any(~np.isfinite(visibility_scores)) or np.any(
        (visibility_scores < 0.0) | (visibility_scores > 1.0)
    ):
        raise ValueError("visibility scores must be finite and in [0, 1]")

    proposal_masses = np.maximum(quality, config.mass_floor)
    entity_masses = visibility_scores.copy()
    proposal_relaxation = (
        config.proposal_relaxation_min
        + quality
        * (config.proposal_relaxation_max - config.proposal_relaxation_min)
    )
    entity_relaxation = (
        config.entity_relaxation_min
        + visibility_scores
        * (config.entity_relaxation_max - config.entity_relaxation_min)
    )
    plan = np.zeros((proposal_count, entity_count), dtype=np.float64)
    active_proposals = np.flatnonzero(np.any(cost_matrix.candidate_mask, axis=1))
    active_entities = np.flatnonzero(np.any(cost_matrix.candidate_mask, axis=0))
    if len(active_entities):
        entity_masses[active_entities] = np.maximum(
            entity_masses[active_entities], config.mass_floor
        )

    converged = True
    iterations = 0
    fixed_point_error = 0.0
    if len(active_proposals) and len(active_entities):
        support = cost_matrix.candidate_mask[
            np.ix_(active_proposals, active_entities)
        ]
        subplan, converged, iterations, fixed_point_error = _unbalanced_plan(
            cost_matrix.costs[np.ix_(active_proposals, active_entities)],
            support,
            proposal_masses[active_proposals],
            entity_masses[active_entities],
            proposal_relaxation[active_proposals],
            entity_relaxation[active_entities],
            config,
        )
        plan[np.ix_(active_proposals, active_entities)] = subplan

    transported_rows = plan.sum(axis=1)
    transported_columns = plan.sum(axis=0)
    birth_residuals = np.maximum(proposal_masses - transported_rows, 0.0)
    miss_residuals = np.maximum(entity_masses - transported_columns, 0.0)
    proposal_excess = np.maximum(transported_rows - proposal_masses, 0.0)
    entity_excess = np.maximum(transported_columns - entity_masses, 0.0)

    matches: list[ResidualTransportMatch] = []
    unmatched: list[ResidualUnmatchedProposal] = []
    for proposal_index in range(proposal_count):
        proposal_id = cost_matrix.proposal_ids[proposal_index]
        if entity_count == 0:
            unmatched.append(
                ResidualUnmatchedProposal(
                    proposal_index,
                    proposal_id,
                    ResidualUnmatchedReason.EMPTY_MAP,
                    retained_ratio=0.0,
                )
            )
            continue
        candidates = np.flatnonzero(cost_matrix.candidate_mask[proposal_index])
        if len(candidates) == 0:
            unmatched.append(
                ResidualUnmatchedProposal(
                    proposal_index,
                    proposal_id,
                    ResidualUnmatchedReason.NO_SPATIAL_CANDIDATE,
                    retained_ratio=0.0,
                )
            )
            continue
        candidate_costs = cost_matrix.costs[proposal_index, candidates]
        viable = candidates[candidate_costs <= max_match_cost]
        best_offset = int(np.argmin(candidate_costs))
        best_entity_index = int(candidates[best_offset])
        best_fields = {
            "best_entity_index": best_entity_index,
            "best_entity_id": cost_matrix.entity_ids[best_entity_index],
            "best_cost": float(candidate_costs[best_offset]),
        }
        retained_ratio = float(
            transported_rows[proposal_index] / proposal_masses[proposal_index]
        )
        if len(viable) == 0:
            unmatched.append(
                ResidualUnmatchedProposal(
                    proposal_index,
                    proposal_id,
                    ResidualUnmatchedReason.COST_ABOVE_THRESHOLD,
                    retained_ratio=retained_ratio,
                    **best_fields,
                )
            )
            continue
        if retained_ratio < config.min_retained_ratio:
            unmatched.append(
                ResidualUnmatchedProposal(
                    proposal_index,
                    proposal_id,
                    ResidualUnmatchedReason.LOW_RETAINED_MASS,
                    retained_ratio=retained_ratio,
                    **best_fields,
                )
            )
            continue
        viable_mass = plan[proposal_index, viable]
        entity_index = int(viable[int(np.argmax(viable_mass))])
        row_mass = float(transported_rows[proposal_index])
        match_mass = float(plan[proposal_index, entity_index])
        conditional_probability = (
            0.0 if row_mass <= 0.0 else match_mass / row_mass
        )
        if conditional_probability < config.min_conditional_probability:
            unmatched.append(
                ResidualUnmatchedProposal(
                    proposal_index,
                    proposal_id,
                    ResidualUnmatchedReason.AMBIGUOUS_TRANSPORT,
                    retained_ratio=retained_ratio,
                    conditional_probability=conditional_probability,
                    **best_fields,
                )
            )
            continue
        matches.append(
            ResidualTransportMatch(
                proposal_index=proposal_index,
                entity_index=entity_index,
                proposal_id=proposal_id,
                entity_id=cost_matrix.entity_ids[entity_index],
                cost=float(cost_matrix.costs[proposal_index, entity_index]),
                transport_mass=match_mass,
                retained_ratio=retained_ratio,
                conditional_probability=conditional_probability,
            )
        )

    unmatched_indices = tuple(item.proposal_index for item in unmatched)
    matched_entities = {match.entity_index for match in matches}
    arrays = (
        plan,
        proposal_masses,
        entity_masses,
        transported_rows,
        transported_columns,
        birth_residuals,
        miss_residuals,
        proposal_excess,
        entity_excess,
        quality,
        proposal_relaxation,
        entity_relaxation,
    )
    for array in arrays:
        array.setflags(write=False)
    forbidden_mass = float(np.sum(plan[~cost_matrix.candidate_mask]))
    return ResidualTransportResult(
        cost_matrix=cost_matrix,
        transport_plan=plan,
        proposal_masses=proposal_masses,
        entity_masses=entity_masses,
        transported_proposal_masses=transported_rows,
        transported_entity_masses=transported_columns,
        proposal_birth_residuals=birth_residuals,
        entity_miss_residuals=miss_residuals,
        proposal_excess_masses=proposal_excess,
        entity_excess_masses=entity_excess,
        proposal_quality=quality,
        proposal_relaxation=proposal_relaxation,
        entity_relaxation=entity_relaxation,
        entity_visibility=visibility,
        matches=tuple(matches),
        unmatched_proposal_indices=unmatched_indices,
        unmatched_entity_indices=tuple(
            index for index in range(entity_count) if index not in matched_entities
        ),
        unmatched_proposals=tuple(unmatched),
        converged=converged,
        iterations=iterations,
        fixed_point_error=fixed_point_error,
        forbidden_mass=forbidden_mass,
    )


def associate_residual_transport(
    proposals: Sequence[LiftedProposal],
    entities: Sequence[Entity],
    entity_visibility: Sequence[EntityVisibility],
    *,
    cost_config: PairwiseCostConfig | None = None,
    transport_config: ResidualTransportConfig | None = None,
    max_match_cost: float = 0.65,
    temporal_hints: Mapping[str, TemporalEntityHint] | None = None,
) -> ResidualTransportResult:
    """Build shared costs, evidence-conditioned masses, and strict-support UOT."""

    config = (
        ResidualTransportConfig()
        if transport_config is None
        else transport_config
    )
    return solve_residual_transport(
        build_pairwise_cost_matrix(
            proposals,
            entities,
            cost_config,
            temporal_hints=temporal_hints,
        ),
        proposal_quality_scores(
            proposals,
            temporal_hints,
            geometry_retention_weight=config.geometry_retention_weight,
        ),
        entity_visibility,
        config=config,
        max_match_cost=max_match_cost,
    )
