"""Minimal persistent-map loop for the frame-level Hungarian baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fact3r.association.costs import PairwiseCostConfig
from fact3r.association.hungarian import HungarianResult, associate_hungarian
from fact3r.entities.entity import Entity, EntityStatus
from fact3r.proposals.lift_to_3d import LiftedProposal


@dataclass(frozen=True, slots=True)
class HungarianMapConfig:
    """State-update parameters intentionally limited to the hard baseline."""

    pairwise_cost: PairwiseCostConfig = field(default_factory=PairwiseCostConfig)
    max_match_cost: float = 0.65
    entity_voxel_size_m: float = 0.04
    max_entity_points: int = 4096
    max_descriptor_samples: int = 512

    def __post_init__(self) -> None:
        if self.max_match_cost < 0.0 or not np.isfinite(self.max_match_cost):
            raise ValueError("max_match_cost must be finite and non-negative")
        if self.entity_voxel_size_m <= 0.0:
            raise ValueError("entity_voxel_size_m must be positive")
        if self.max_entity_points <= 0:
            raise ValueError("max_entity_points must be positive")
        if self.max_descriptor_samples <= 0:
            raise ValueError("max_descriptor_samples must be positive")


@dataclass(frozen=True, slots=True)
class FrameMappingResult:
    """Result of one joint assignment over every proposal in a keyframe."""

    frame_id: int
    timestamp: float | str | None
    proposal_count: int
    entity_count_before: int
    assignment: HungarianResult
    created_entity_ids: tuple[str, ...]
    entity_count_after: int


def _bounded_rows(
    values: NDArray[np.floating], maximum: int
) -> NDArray[np.floating]:
    if len(values) <= maximum:
        return np.ascontiguousarray(values)
    indices = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return np.ascontiguousarray(values[indices])


def _voxel_reduce(
    points: NDArray[np.floating], voxel_size: float, maximum: int
) -> NDArray[np.floating]:
    points = np.asarray(points, dtype=np.float32)
    finite = points[np.all(np.isfinite(points), axis=1)]
    if len(finite) == 0:
        raise ValueError("entity geometry update contains no finite points")
    cells = np.floor(finite / voxel_size).astype(np.int64)
    _, first_indices = np.unique(cells, axis=0, return_index=True)
    reduced = finite[np.sort(first_indices)]
    return _bounded_rows(reduced, maximum)


def _proposal_colour(proposal: LiftedProposal) -> NDArray[np.floating]:
    colour = np.median(proposal.colours_rgb, axis=0).astype(np.float64)
    if float(np.max(np.abs(colour))) <= 1.5:
        colour *= 255.0
    return np.clip(colour, 0.0, 255.0)


def _proposal_descriptor_samples(
    proposal: LiftedProposal, maximum: int
) -> tuple[NDArray[np.floating] | None, NDArray[np.floating] | None]:
    if proposal.mast3r_descriptors is None:
        return None, None
    descriptors = np.asarray(proposal.mast3r_descriptors, dtype=np.float32)
    confidence = (
        np.ones(len(descriptors), dtype=np.float32)
        if proposal.descriptor_confidence is None
        else np.asarray(proposal.descriptor_confidence, dtype=np.float32)
    )
    if len(descriptors) > maximum:
        indices = np.linspace(0, len(descriptors) - 1, maximum, dtype=np.int64)
        descriptors = descriptors[indices]
        confidence = confidence[indices]
    return np.ascontiguousarray(descriptors), np.ascontiguousarray(confidence)


class HungarianEntityMapper:
    """Run one hard assignment per complete frame and retain persistent IDs.

    Unmatched proposals immediately create provisional entities. Unmatched map
    entities are retained. No confirmation transition, delayed belief, confidence
    gate, split/merge operation, or semantic update is performed here; those are
    deliberately reserved for later comparison stages.
    """

    def __init__(self, config: HungarianMapConfig | None = None) -> None:
        self.config = HungarianMapConfig() if config is None else config
        self._entities: list[Entity] = []
        self._next_entity_index = 0

    @property
    def entities(self) -> tuple[Entity, ...]:
        return tuple(self._entities)

    def process_frame(
        self,
        proposals: Sequence[LiftedProposal],
        *,
        frame_id: int,
        timestamp: float | str | None = None,
    ) -> FrameMappingResult:
        """Associate all proposals jointly, then apply baseline map updates."""

        proposals = tuple(proposals)
        if any(proposal.frame_id != frame_id for proposal in proposals):
            raise ValueError("every proposal must belong to the processed frame")
        entity_count_before = len(self._entities)
        assignment = associate_hungarian(
            proposals,
            tuple(self._entities),
            cost_config=self.config.pairwise_cost,
            max_match_cost=self.config.max_match_cost,
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
        return FrameMappingResult(
            frame_id=frame_id,
            timestamp=timestamp,
            proposal_count=len(proposals),
            entity_count_before=entity_count_before,
            assignment=assignment,
            created_entity_ids=created_entity_ids,
            entity_count_after=len(self._entities),
        )

    def _create_entity(
        self, proposal: LiftedProposal, timestamp: float | str | None
    ) -> Entity:
        geometry = _voxel_reduce(
            proposal.points_world,
            self.config.entity_voxel_size_m,
            self.config.max_entity_points,
        )
        descriptors, confidence = _proposal_descriptor_samples(
            proposal, self.config.max_descriptor_samples
        )
        entity = Entity(
            id=f"entity-{self._next_entity_index:06d}",
            status=EntityStatus.PROVISIONAL,
            centroid_xyz=proposal.centroid_xyz,
            bounding_box_xyz=proposal.bounding_box_xyz,
            surfel_or_voxel_geometry=geometry,
            colour_statistics={"mean_rgb": _proposal_colour(proposal)},
            mast3r_descriptor_bank=descriptors,
            descriptor_confidence=confidence,
            observation_count=1,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            persistence_probability=0.0,
        )
        self._next_entity_index += 1
        self._entities.append(entity)
        return entity

    def _update_entity(
        self,
        entity: Entity,
        proposal: LiftedProposal,
        timestamp: float | str | None,
    ) -> None:
        combined_points = np.concatenate(
            (entity.surfel_or_voxel_geometry, proposal.points_world), axis=0
        )
        entity.surfel_or_voxel_geometry = _voxel_reduce(
            combined_points,
            self.config.entity_voxel_size_m,
            self.config.max_entity_points,
        )
        entity.centroid_xyz = (
            (
                entity.centroid_xyz * entity.observation_count
                + proposal.centroid_xyz
            )
            / (entity.observation_count + 1)
        ).astype(np.float32)
        entity.bounding_box_xyz = np.stack(
            (
                np.minimum(entity.bounding_box_xyz[0], proposal.bounding_box_xyz[0]),
                np.maximum(entity.bounding_box_xyz[1], proposal.bounding_box_xyz[1]),
            ),
            axis=0,
        ).astype(np.float32)

        previous_colour = np.asarray(
            entity.colour_statistics.get(
                "mean_rgb", entity.colour_statistics.get("median_rgb", [0, 0, 0])
            ),
            dtype=np.float64,
        )
        entity.colour_statistics = {
            "mean_rgb": (
                previous_colour * entity.observation_count
                + _proposal_colour(proposal)
            )
            / (entity.observation_count + 1)
        }

        descriptors, confidence = _proposal_descriptor_samples(
            proposal, self.config.max_descriptor_samples
        )
        if descriptors is not None:
            if entity.mast3r_descriptor_bank is None:
                entity.mast3r_descriptor_bank = descriptors
                entity.descriptor_confidence = confidence
            elif entity.mast3r_descriptor_bank.shape[1] == descriptors.shape[1]:
                entity.mast3r_descriptor_bank = _bounded_rows(
                    np.concatenate((entity.mast3r_descriptor_bank, descriptors)),
                    self.config.max_descriptor_samples,
                ).astype(np.float32)
                entity.descriptor_confidence = _bounded_rows(
                    np.concatenate((entity.descriptor_confidence, confidence)),
                    self.config.max_descriptor_samples,
                ).astype(np.float32)

        entity.observation_count += 1
        entity.last_seen_timestamp = timestamp
