"""Persistent 3D entity contract; association behavior comes in Milestone 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class EntityStatus(str, Enum):
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
    INACTIVE = "inactive"
    DYNAMIC = "dynamic"


@dataclass(slots=True)
class Entity:
    """State owned by one persistent object or part entity."""

    id: str
    status: EntityStatus
    centroid_xyz: NDArray[np.floating]
    bounding_box_xyz: NDArray[np.floating]
    surfel_or_voxel_geometry: NDArray[np.floating]
    parent_region_id: str | None = None
    parent_entity_id: str | None = None
    normal_statistics: dict[str, Any] = field(default_factory=dict)
    colour_statistics: dict[str, Any] = field(default_factory=dict)
    mast3r_descriptor_bank: NDArray[np.floating] | None = None
    descriptor_confidence: NDArray[np.floating] | None = None
    appearance_descriptor_bank: NDArray[np.floating] | None = None
    appearance_reliability: NDArray[np.floating] | None = None
    observation_count: int = 0
    observed_view_directions: NDArray[np.floating] | None = None
    best_observation_pose: NDArray[np.floating] | None = None
    first_seen_timestamp: float | str | None = None
    last_seen_timestamp: float | str | None = None
    persistence_probability: float = 0.0
    semantic_fact_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.status = EntityStatus(self.status)
        self.centroid_xyz = np.asarray(self.centroid_xyz, dtype=np.float32)
        self.bounding_box_xyz = np.asarray(self.bounding_box_xyz, dtype=np.float32)
        self.surfel_or_voxel_geometry = np.asarray(
            self.surfel_or_voxel_geometry, dtype=np.float32
        )
        if self.centroid_xyz.shape != (3,):
            raise ValueError("centroid_xyz must have shape (3,)")
        if self.bounding_box_xyz.shape != (2, 3):
            raise ValueError("bounding_box_xyz must have shape (2, 3)")
        if np.any(self.bounding_box_xyz[0] > self.bounding_box_xyz[1]):
            raise ValueError("bounding_box_xyz min corner must not exceed max corner")
        if (
            self.surfel_or_voxel_geometry.ndim != 2
            or self.surfel_or_voxel_geometry.shape[1] != 3
        ):
            raise ValueError("surfel_or_voxel_geometry must have shape (N, 3)")
        if self.observation_count < 0:
            raise ValueError("observation_count cannot be negative")
        if not 0.0 <= self.persistence_probability <= 1.0:
            raise ValueError("persistence_probability must be in [0, 1]")

        if self.mast3r_descriptor_bank is not None:
            self.mast3r_descriptor_bank = np.asarray(
                self.mast3r_descriptor_bank, dtype=np.float32
            )
            if self.mast3r_descriptor_bank.ndim != 2:
                raise ValueError("mast3r_descriptor_bank must have shape (N, D)")

        if self.descriptor_confidence is not None:
            self.descriptor_confidence = np.asarray(
                self.descriptor_confidence, dtype=np.float32
            ).reshape(-1)
            if self.mast3r_descriptor_bank is None:
                raise ValueError(
                    "descriptor_confidence requires mast3r_descriptor_bank"
                )
            if len(self.descriptor_confidence) != len(self.mast3r_descriptor_bank):
                raise ValueError(
                    "descriptor_confidence and descriptor bank must have equal length"
                )

        if self.appearance_descriptor_bank is not None:
            self.appearance_descriptor_bank = np.asarray(
                self.appearance_descriptor_bank, dtype=np.float32
            )
            if (
                self.appearance_descriptor_bank.ndim != 2
                or self.appearance_descriptor_bank.shape[1] == 0
                or not np.all(np.isfinite(self.appearance_descriptor_bank))
            ):
                raise ValueError(
                    "appearance_descriptor_bank must have finite shape (N, D)"
                )
            norms = np.linalg.norm(
                self.appearance_descriptor_bank, axis=1, keepdims=True
            )
            if np.any(norms <= 1e-12):
                raise ValueError(
                    "appearance_descriptor_bank cannot contain zero vectors"
                )
            self.appearance_descriptor_bank = np.ascontiguousarray(
                self.appearance_descriptor_bank / norms, dtype=np.float32
            )

        if self.appearance_reliability is not None:
            self.appearance_reliability = np.asarray(
                self.appearance_reliability, dtype=np.float32
            ).reshape(-1)
            if self.appearance_descriptor_bank is None:
                raise ValueError(
                    "appearance_reliability requires appearance_descriptor_bank"
                )
            if len(self.appearance_reliability) != len(
                self.appearance_descriptor_bank
            ):
                raise ValueError(
                    "appearance reliability and bank must have equal length"
                )
            if not np.all(np.isfinite(self.appearance_reliability)) or np.any(
                (self.appearance_reliability < 0.0)
                | (self.appearance_reliability > 1.0)
            ):
                raise ValueError(
                    "appearance_reliability must be finite and in [0, 1]"
                )

        if self.observed_view_directions is not None:
            self.observed_view_directions = np.asarray(
                self.observed_view_directions, dtype=np.float32
            )
            if (
                self.observed_view_directions.ndim != 2
                or self.observed_view_directions.shape[1] != 3
            ):
                raise ValueError("observed_view_directions must have shape (N, 3)")

        if self.best_observation_pose is not None:
            self.best_observation_pose = np.asarray(
                self.best_observation_pose, dtype=np.float32
            )
            if self.best_observation_pose.shape != (4, 4):
                raise ValueError("best_observation_pose must have shape (4, 4)")
