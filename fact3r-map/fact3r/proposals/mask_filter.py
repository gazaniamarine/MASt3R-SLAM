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
    min_lifted_points: int = 16
    full_anchor_coverage: float = 0.50
    reject_border_slivers: bool = True
    border_sliver_max_thickness_fraction: float = 0.05
    border_sliver_min_aspect_ratio: float = 12.0

    def __post_init__(self) -> None:
        if (
            self.min_area_pixels < 1
            or self.min_component_pixels < 1
            or self.min_lifted_points < 1
        ):
            raise ValueError("pixel thresholds must be positive")
        if self.erosion_pixels < 0:
            raise ValueError("erosion_pixels cannot be negative")
        for name in (
            "min_area_fraction",
            "max_area_fraction",
            "duplicate_iou_threshold",
            "full_anchor_coverage",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_area_fraction > self.max_area_fraction:
            raise ValueError("min_area_fraction cannot exceed max_area_fraction")
        if not 0.0 < self.border_sliver_max_thickness_fraction <= 1.0:
            raise ValueError("border sliver thickness fraction must be in (0, 1]")
        if self.border_sliver_min_aspect_ratio < 1.0:
            raise ValueError("border sliver aspect ratio must be at least 1")


def is_pathological_border_sliver(
    mask: NDArray[np.bool_], config: MaskFilterConfig
) -> bool:
    """Reject narrow SAM regions that merely follow an image boundary."""

    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return True
    rows, columns = np.nonzero(selected)
    height, width = selected.shape
    box_height = int(rows.max() - rows.min() + 1)
    box_width = int(columns.max() - columns.min() + 1)
    aspect = max(box_height / max(box_width, 1), box_width / max(box_height, 1))
    edge_tolerance = max(2, int(round(0.005 * max(height, width))))
    touches_edge = (
        rows.min() <= edge_tolerance
        or rows.max() >= height - 1 - edge_tolerance
        or columns.min() <= edge_tolerance
        or columns.max() >= width - 1 - edge_tolerance
    )
    thin = (
        box_width / width <= config.border_sliver_max_thickness_fraction
        or box_height / height <= config.border_sliver_max_thickness_fraction
    )
    return bool(
        config.reject_border_slivers
        and touches_edge
        and thin
        and aspect >= config.border_sliver_min_aspect_ratio
    )


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

    selected = np.asarray(mask, dtype=bool)
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
            selected.astype(np.uint8, copy=False), connectivity=4
        )
        if component_count <= 1:
            return np.zeros_like(selected, dtype=bool)
        keep = np.zeros(component_count, dtype=bool)
        keep[1:] = statistics[1:, cv2.CC_STAT_AREA] >= min_component_pixels
        return np.ascontiguousarray(keep[labels])

    height, width = mask.shape
    remaining = set(map(tuple, np.argwhere(selected)))
    cleaned = np.zeros_like(selected, dtype=bool)
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

    if config.min_descriptor_confidence is not None:
        if keyframe.descriptor_confidence is None:
            raise ValueError(
                "descriptor filtering requested, but exported keyframe has no Q map"
            )
    allowed_pixels = (
        keyframe.geometry_confidence > config.min_geometry_confidence
    )
    if config.min_descriptor_confidence is not None:
        allowed_pixels &= (
            keyframe.descriptor_confidence > config.min_descriptor_confidence
        )
    return _filter_with_allowed_pixels(
        proposals,
        image_shape=keyframe.image_shape,
        expected_frame_id=keyframe.frame_id,
        config=config,
        allowed_pixels=allowed_pixels,
    )


def filter_image_mask_proposals(
    proposals: list[MaskProposal2D],
    image_shape: tuple[int, int],
    config: MaskFilterConfig,
    *,
    frame_id: int | None = None,
) -> list[MaskProposal2D]:
    """Apply the shared mask cleanup without requiring reconstructed geometry.

    This is intended for image-only temporal diagnostics. The resulting masks
    cannot be lifted into the persistent 3D map until aligned geometry is supplied.
    """

    if config.min_descriptor_confidence is not None:
        raise ValueError(
            "descriptor confidence filtering is unavailable without a keyframe"
        )
    expected_frame_id = (
        frame_id
        if frame_id is not None
        else (None if not proposals else proposals[0].frame_id)
    )
    return _filter_with_allowed_pixels(
        proposals,
        image_shape=image_shape,
        expected_frame_id=expected_frame_id,
        config=config,
        allowed_pixels=None,
    )


def _filter_with_allowed_pixels(
    proposals: list[MaskProposal2D],
    *,
    image_shape: tuple[int, int],
    expected_frame_id: int | None,
    config: MaskFilterConfig,
    allowed_pixels: NDArray[np.bool_] | None,
) -> list[MaskProposal2D]:
    image_area = image_shape[0] * image_shape[1]
    min_area = max(
        config.min_area_pixels, int(np.ceil(config.min_area_fraction * image_area))
    )
    max_area = int(np.floor(config.max_area_fraction * image_area))

    cleaned: list[MaskProposal2D] = []
    for proposal in proposals:
        if (
            expected_frame_id is not None
            and proposal.frame_id != expected_frame_id
        ):
            raise ValueError("proposal frame_id does not match the expected frame")
        if proposal.mask.shape != image_shape:
            raise ValueError("proposal mask shape does not match the expected image")
        if proposal.score < config.min_score:
            continue
        mask = proposal.mask.copy()
        if allowed_pixels is not None:
            mask &= allowed_pixels
        mask = erode_mask(mask, config.erosion_pixels)
        mask = remove_small_components(mask, config.min_component_pixels)
        if is_pathological_border_sliver(mask, config):
            continue
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
