"""Reusable proposal-to-entity costs for all association backends."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from fact3r.entities.entity import Entity
from fact3r.proposals.lift_to_3d import LiftedProposal


@dataclass(frozen=True, slots=True)
class PairwiseCostConfig:
    """Geometry-first configuration for the non-learned association baseline."""

    max_centroid_distance_m: float = 1.0
    bounding_box_padding_m: float = 0.05
    geometry_match_distance_m: float = 0.08
    max_geometry_points: int = 256
    centroid_weight: float = 0.20
    bounding_box_weight: float = 0.15
    geometry_weight: float = 0.45
    colour_weight: float = 0.05
    descriptor_weight: float = 0.15
    appearance_weight: float = 0.25
    appearance_temperature: float = 0.07
    temporal_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.max_centroid_distance_m <= 0.0:
            raise ValueError("max_centroid_distance_m must be positive")
        if self.bounding_box_padding_m < 0.0:
            raise ValueError("bounding_box_padding_m cannot be negative")
        if self.geometry_match_distance_m <= 0.0:
            raise ValueError("geometry_match_distance_m must be positive")
        if self.max_geometry_points <= 0:
            raise ValueError("max_geometry_points must be positive")
        weights = (
            self.centroid_weight,
            self.bounding_box_weight,
            self.geometry_weight,
            self.colour_weight,
            self.descriptor_weight,
            self.appearance_weight,
            self.temporal_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("pairwise cost weights cannot be negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one pairwise cost weight must be positive")
        if self.appearance_temperature <= 0.0:
            raise ValueError("appearance_temperature must be positive")


@dataclass(frozen=True, slots=True)
class TemporalEntityHint:
    """A SAM2 tracklet preference resolved to an existing map entity."""

    entity_id: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("temporal hint entity_id cannot be empty")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("temporal hint confidence must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class PairwiseCostMatrix:
    """One cost matrix shared by Hungarian and future transport solvers.

    Non-candidate pairs have infinite total cost. Component matrices use NaN for
    unavailable cues and for pairs removed by spatial gating.
    """

    proposal_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    costs: NDArray[np.floating]
    candidate_mask: NDArray[np.bool_]
    components: Mapping[str, NDArray[np.floating]]

    def __post_init__(self) -> None:
        shape = (len(self.proposal_ids), len(self.entity_ids))
        costs = np.asarray(self.costs, dtype=np.float64)
        candidate_mask = np.asarray(self.candidate_mask, dtype=bool)
        if costs.shape != shape:
            raise ValueError(f"costs must have shape {shape}")
        if candidate_mask.shape != shape:
            raise ValueError(f"candidate_mask must have shape {shape}")
        if np.any(np.isnan(costs)):
            raise ValueError("total costs cannot contain NaN")
        if np.any(costs[candidate_mask] < 0.0):
            raise ValueError("candidate costs cannot be negative")
        if np.any(~np.isfinite(costs[candidate_mask])):
            raise ValueError("candidate pairs must have finite costs")
        if np.any(np.isfinite(costs[~candidate_mask])):
            raise ValueError("non-candidate pairs must have infinite cost")

        component_arrays: dict[str, NDArray[np.floating]] = {}
        for name, values in self.components.items():
            array = np.asarray(values, dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"component {name!r} must have shape {shape}")
            array = np.ascontiguousarray(array)
            array.setflags(write=False)
            component_arrays[name] = array

        costs = np.ascontiguousarray(costs)
        candidate_mask = np.ascontiguousarray(candidate_mask)
        costs.setflags(write=False)
        candidate_mask.setflags(write=False)
        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "candidate_mask", candidate_mask)
        object.__setattr__(
            self, "components", MappingProxyType(component_arrays)
        )


def _expanded_boxes_overlap(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    padding: float,
) -> bool:
    return bool(
        np.all(left[0] - padding <= right[1] + padding)
        and np.all(right[0] - padding <= left[1] + padding)
    )


def _padded_box_iou(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    padding: float,
) -> float:
    left_min, left_max = left[0] - padding, left[1] + padding
    right_min, right_max = right[0] - padding, right[1] + padding
    intersection_extent = np.maximum(
        0.0, np.minimum(left_max, right_max) - np.maximum(left_min, right_min)
    )
    intersection = float(np.prod(intersection_extent))
    left_volume = float(np.prod(np.maximum(0.0, left_max - left_min)))
    right_volume = float(np.prod(np.maximum(0.0, right_max - right_min)))
    union = left_volume + right_volume - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _sample_points(
    points: NDArray[np.floating], max_points: int
) -> NDArray[np.floating]:
    points = np.asarray(points, dtype=np.float32)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def _symmetric_geometry_consistency(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    distance_threshold: float,
    max_points: int,
) -> float:
    """Return the symmetric fraction of points with a nearby counterpart."""

    left = _sample_points(left, max_points)
    right = _sample_points(right, max_points)
    if len(left) == 0 or len(right) == 0:
        return 0.0
    threshold_squared = distance_threshold * distance_threshold
    left_min = np.full(len(left), np.inf, dtype=np.float64)
    right_min = np.full(len(right), np.inf, dtype=np.float64)
    block_size = 64
    for start in range(0, len(left), block_size):
        stop = min(start + block_size, len(left))
        difference = left[start:stop, None, :] - right[None, :, :]
        distance_squared = np.einsum(
            "ijk,ijk->ij", difference, difference, optimize=True
        )
        left_min[start:stop] = distance_squared.min(axis=1)
        right_min = np.minimum(right_min, distance_squared.min(axis=0))
    return 0.5 * (
        float(np.mean(left_min <= threshold_squared))
        + float(np.mean(right_min <= threshold_squared))
    )


def _normalised_rgb(values: NDArray[np.generic]) -> NDArray[np.floating] | None:
    try:
        rgb = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if rgb.shape != (3,) or not np.all(np.isfinite(rgb)):
        return None
    if np.max(np.abs(rgb)) > 1.5:
        rgb = rgb / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _entity_colour(entity: Entity) -> NDArray[np.floating] | None:
    for key in ("median_rgb", "mean_rgb"):
        if key in entity.colour_statistics:
            return _normalised_rgb(entity.colour_statistics[key])
    return None


def _pooled_descriptor(
    descriptors: NDArray[np.floating] | None,
    confidence: NDArray[np.floating] | None,
) -> NDArray[np.floating] | None:
    if descriptors is None:
        return None
    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.ndim != 2 or len(descriptors) == 0:
        return None
    finite_rows = np.all(np.isfinite(descriptors), axis=1)
    if not np.any(finite_rows):
        return None
    descriptors = descriptors[finite_rows]
    if confidence is None:
        pooled = descriptors.mean(axis=0)
    else:
        weights = np.asarray(confidence, dtype=np.float64).reshape(-1)[finite_rows]
        weights = np.where(np.isfinite(weights), np.maximum(weights, 0.0), 0.0)
        pooled = (
            descriptors.mean(axis=0)
            if float(weights.sum()) <= 0.0
            else np.average(descriptors, axis=0, weights=weights)
        )
    norm = float(np.linalg.norm(pooled))
    return None if norm <= 1e-12 else pooled / norm


def _descriptor_cost(
    left: NDArray[np.floating] | None,
    right: NDArray[np.floating] | None,
) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return 0.5 * (1.0 - cosine)


def _appearance_cost(
    proposal_descriptor: NDArray[np.floating] | None,
    entity_bank: NDArray[np.floating] | None,
    entity_reliability: NDArray[np.floating] | None,
    temperature: float,
) -> tuple[float, float] | None:
    """Return soft best-view cosine cost and entity-memory reliability."""

    if proposal_descriptor is None or entity_bank is None:
        return None
    query = np.asarray(proposal_descriptor, dtype=np.float64).reshape(-1)
    bank = np.asarray(entity_bank, dtype=np.float64)
    if bank.ndim != 2 or len(bank) == 0 or bank.shape[1:] != query.shape:
        return None
    query_norm = float(np.linalg.norm(query))
    bank_norms = np.linalg.norm(bank, axis=1)
    valid = (
        np.all(np.isfinite(bank), axis=1)
        & np.isfinite(bank_norms)
        & (bank_norms > 1e-12)
    )
    if not np.isfinite(query_norm) or query_norm <= 1e-12 or not np.any(valid):
        return None
    query = query / query_norm
    bank = bank[valid] / bank_norms[valid, None]
    similarities = np.clip(bank @ query, -1.0, 1.0)
    if entity_reliability is None:
        reliability = np.ones(len(bank), dtype=np.float64)
    else:
        reliability = np.asarray(entity_reliability, dtype=np.float64).reshape(-1)
        reliability = reliability[valid]
        reliability = np.where(
            np.isfinite(reliability), np.clip(reliability, 0.0, 1.0), 0.0
        )
        if float(reliability.sum()) <= 1e-12:
            reliability = np.ones(len(bank), dtype=np.float64)
    alpha = reliability / reliability.sum()
    logits = np.log(np.maximum(alpha, 1e-12)) + similarities / temperature
    maximum = float(np.max(logits))
    soft_similarity = temperature * (
        maximum + float(np.log(np.exp(logits - maximum).sum()))
    )
    cost = 0.5 * (1.0 - float(np.clip(soft_similarity, -1.0, 1.0)))
    return cost, float(np.mean(np.clip(reliability, 0.0, 1.0)))


def build_pairwise_cost_matrix(
    proposals: Sequence[LiftedProposal],
    entities: Sequence[Entity],
    config: PairwiseCostConfig | None = None,
    *,
    temporal_hints: Mapping[str, TemporalEntityHint] | None = None,
) -> PairwiseCostMatrix:
    """Gate candidate pairs and calculate a normalized geometry-first cost.

    Available cue weights are renormalized per pair. Consequently, an absent
    descriptor or entity colour statistic does not itself penalize the pair.
    """

    config = PairwiseCostConfig() if config is None else config
    shape = (len(proposals), len(entities))
    costs = np.full(shape, np.inf, dtype=np.float64)
    candidate_mask = np.zeros(shape, dtype=bool)
    components = {
        "centroid": np.full(shape, np.nan, dtype=np.float64),
        "bounding_box": np.full(shape, np.nan, dtype=np.float64),
        "geometry": np.full(shape, np.nan, dtype=np.float64),
        "colour": np.full(shape, np.nan, dtype=np.float64),
        "descriptor": np.full(shape, np.nan, dtype=np.float64),
        "appearance": np.full(shape, np.nan, dtype=np.float64),
        "appearance_reliability": np.full(
            shape, np.nan, dtype=np.float64
        ),
        "temporal": np.full(shape, np.nan, dtype=np.float64),
    }

    proposal_colours = [
        _normalised_rgb(np.median(proposal.colours_rgb, axis=0))
        for proposal in proposals
    ]
    entity_colours = [_entity_colour(entity) for entity in entities]
    proposal_descriptors = [
        _pooled_descriptor(
            proposal.mast3r_descriptors, proposal.descriptor_confidence
        )
        for proposal in proposals
    ]
    entity_descriptors = [
        _pooled_descriptor(
            entity.mast3r_descriptor_bank, entity.descriptor_confidence
        )
        for entity in entities
    ]
    temporal_hints = {} if temporal_hints is None else temporal_hints
    entity_id_set = {entity.id for entity in entities}

    for proposal_index, proposal in enumerate(proposals):
        for entity_index, entity in enumerate(entities):
            centroid_distance = float(
                np.linalg.norm(proposal.centroid_xyz - entity.centroid_xyz)
            )
            boxes_overlap = _expanded_boxes_overlap(
                proposal.bounding_box_xyz,
                entity.bounding_box_xyz,
                config.bounding_box_padding_m,
            )
            if (
                centroid_distance > config.max_centroid_distance_m
                and not boxes_overlap
            ):
                continue
            candidate_mask[proposal_index, entity_index] = True

            cue_values: list[tuple[float, float]] = []
            centroid_cost = min(
                centroid_distance / config.max_centroid_distance_m, 1.0
            )
            components["centroid"][proposal_index, entity_index] = centroid_cost
            cue_values.append((config.centroid_weight, centroid_cost))

            bounding_box_cost = 1.0 - _padded_box_iou(
                proposal.bounding_box_xyz,
                entity.bounding_box_xyz,
                config.bounding_box_padding_m,
            )
            components["bounding_box"][proposal_index, entity_index] = (
                bounding_box_cost
            )
            cue_values.append((config.bounding_box_weight, bounding_box_cost))

            geometry_cost = 1.0 - _symmetric_geometry_consistency(
                proposal.points_world,
                entity.surfel_or_voxel_geometry,
                config.geometry_match_distance_m,
                config.max_geometry_points,
            )
            components["geometry"][proposal_index, entity_index] = geometry_cost
            cue_values.append((config.geometry_weight, geometry_cost))

            left_colour = proposal_colours[proposal_index]
            right_colour = entity_colours[entity_index]
            if left_colour is not None and right_colour is not None:
                colour_cost = float(
                    np.linalg.norm(left_colour - right_colour) / np.sqrt(3.0)
                )
                components["colour"][proposal_index, entity_index] = colour_cost
                cue_values.append((config.colour_weight, colour_cost))

            descriptor_cost = _descriptor_cost(
                proposal_descriptors[proposal_index],
                entity_descriptors[entity_index],
            )
            if descriptor_cost is not None:
                components["descriptor"][proposal_index, entity_index] = (
                    descriptor_cost
                )
                cue_values.append((config.descriptor_weight, descriptor_cost))

            appearance = _appearance_cost(
                proposal.appearance_descriptor,
                entity.appearance_descriptor_bank,
                entity.appearance_reliability,
                config.appearance_temperature,
            )
            if appearance is not None:
                appearance_cost, entity_reliability = appearance
                proposal_reliability = (
                    1.0
                    if proposal.appearance_reliability is None
                    else float(proposal.appearance_reliability)
                )
                pair_reliability = float(
                    np.sqrt(
                        np.clip(proposal_reliability, 0.0, 1.0)
                        * np.clip(entity_reliability, 0.0, 1.0)
                    )
                )
                components["appearance"][proposal_index, entity_index] = (
                    appearance_cost
                )
                components["appearance_reliability"][
                    proposal_index, entity_index
                ] = pair_reliability
                cue_values.append(
                    (
                        config.appearance_weight * pair_reliability,
                        appearance_cost,
                    )
                )

            temporal_hint = temporal_hints.get(proposal.proposal_id)
            if (
                temporal_hint is not None
                and temporal_hint.entity_id in entity_id_set
                and temporal_hint.confidence > 0.0
            ):
                temporal_cost = float(entity.id != temporal_hint.entity_id)
                components["temporal"][proposal_index, entity_index] = (
                    temporal_cost
                )
                cue_values.append(
                    (
                        config.temporal_weight * temporal_hint.confidence,
                        temporal_cost,
                    )
                )

            active_weight = sum(
                weight for weight, _ in cue_values if weight > 0.0
            )
            if active_weight <= 0.0:
                candidate_mask[proposal_index, entity_index] = False
                continue
            costs[proposal_index, entity_index] = sum(
                weight * value
                for weight, value in cue_values
                if weight > 0.0
            ) / active_weight

    return PairwiseCostMatrix(
        proposal_ids=tuple(proposal.proposal_id for proposal in proposals),
        entity_ids=tuple(entity.id for entity in entities),
        costs=costs,
        candidate_mask=candidate_mask,
        components=components,
    )
