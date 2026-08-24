"""Geometry-aware cleanup for overlapping class-agnostic masks."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from fact3r.proposals.mask_generator import MaskProposal2D
from fact3r.reconstruction.keyframes import KeyframeRecord


@dataclass(frozen=True, slots=True)
class MaskFilterConfig:
    min_score: float = 0.88
    min_area_pixels: int = 100
    min_area_fraction: float = 0.001
    max_area_fraction: float = 0.8
    erosion_pixels: int = 1
    min_component_pixels: int = 50
    duplicate_iou_threshold: float = 0.9
    min_geometry_confidence: float = 0.0
    min_descriptor_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.min_area_pixels < 1 or self.min_component_pixels < 1:
            raise ValueError("pixel thresholds must be positive")
        if self.erosion_pixels < 0:
            raise ValueError("erosion_pixels cannot be negative")
        for name in (
            "min_area_fraction",
            "max_area_fraction",
            "duplicate_iou_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_area_fraction > self.max_area_fraction:
            raise ValueError("min_area_fraction cannot exceed max_area_fraction")


def erode_mask(mask: NDArray[np.bool_], iterations: int) -> NDArray[np.bool_]:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_and.reduce(
            [
                padded[row : row + result.shape[0], column : column + result.shape[1]]
                for row in range(3)
                for column in range(3)
            ]
        )
    return result


def remove_small_components(
    mask: NDArray[np.bool_], min_component_pixels: int
) -> NDArray[np.bool_]:
    """Keep 4-connected components large enough to be plausible 3D support."""

    height, width = mask.shape
    remaining = set(map(tuple, np.argwhere(mask)))
    cleaned = np.zeros_like(mask, dtype=bool)
    while remaining:
        seed = remaining.pop()
        component = [seed]
        frontier = [seed]
        while frontier:
            row, column = frontier.pop()
            for neighbour in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= neighbour[0] < height
                    and 0 <= neighbour[1] < width
                    and neighbour in remaining
                ):
                    remaining.remove(neighbour)
                    component.append(neighbour)
                    frontier.append(neighbour)
        if len(component) >= min_component_pixels:
            rows, columns = zip(*component)
            cleaned[rows, columns] = True
    return cleaned


def mask_iou(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> float:
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return 0.0 if union == 0 else float(intersection / union)


def filter_mask_proposals(
    proposals: list[MaskProposal2D],
    keyframe: KeyframeRecord,
    config: MaskFilterConfig,
) -> list[MaskProposal2D]:
    """Apply image, MASt3R-confidence, component, and duplicate filtering."""

    image_area = keyframe.image_shape[0] * keyframe.image_shape[1]
    min_area = max(
        config.min_area_pixels, int(np.ceil(config.min_area_fraction * image_area))
    )
    max_area = int(np.floor(config.max_area_fraction * image_area))
    if config.min_descriptor_confidence is not None:
        if keyframe.descriptor_confidence is None:
            raise ValueError(
                "descriptor filtering requested, but exported keyframe has no Q map"
            )

    cleaned: list[MaskProposal2D] = []
    for proposal in proposals:
        if proposal.frame_id != keyframe.frame_id:
            raise ValueError("proposal frame_id does not match keyframe")
        if proposal.mask.shape != keyframe.image_shape:
            raise ValueError("proposal mask shape does not match keyframe")
        if proposal.score < config.min_score:
            continue
        mask = proposal.mask.copy()
        mask &= keyframe.geometry_confidence > config.min_geometry_confidence
        if config.min_descriptor_confidence is not None:
            mask &= (
                keyframe.descriptor_confidence
                > config.min_descriptor_confidence
            )
        mask = erode_mask(mask, config.erosion_pixels)
        mask = remove_small_components(mask, config.min_component_pixels)
        area = int(mask.sum())
        if area < min_area or area > max_area:
            continue
        cleaned.append(replace(proposal, mask=mask))

    kept: list[MaskProposal2D] = []
    for proposal in sorted(cleaned, key=lambda item: item.score, reverse=True):
        if all(
            mask_iou(proposal.mask, existing.mask)
            < config.duplicate_iou_threshold
            for existing in kept
        ):
            kept.append(proposal)
    return kept

