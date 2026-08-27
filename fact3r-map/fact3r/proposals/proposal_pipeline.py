"""End-to-end per-keyframe path from RGB to filtered, lifted proposals."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from fact3r.proposals.lift_to_3d import LiftedProposal, lift_mask_to_3d
from fact3r.proposals.mask_filter import (
    MaskFilterConfig,
    filter_image_mask_proposals,
    remove_small_components,
)
from fact3r.proposals.mask_generator import MaskGenerator, MaskProposal2D
from fact3r.reconstruction.keyframes import KeyframeRecord


class GeometryStatus(str, Enum):
    """How much of a retained 2D observation is anchored in the pointmap."""

    ANCHORED_3D = "anchored_3d"
    PARTIAL_3D = "partial_3d"
    UNANCHORED_2D = "unanchored_2d"


@dataclass(frozen=True, slots=True)
class GeneratedProposal:
    mask_2d: MaskProposal2D
    lifted_3d: LiftedProposal | None
    geometry_status: GeometryStatus = GeometryStatus.ANCHORED_3D
    geometry_coverage: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.geometry_coverage <= 1.0:
            raise ValueError("geometry_coverage must be in [0, 1]")
        if (self.lifted_3d is None) != (
            self.geometry_status == GeometryStatus.UNANCHORED_2D
        ):
            raise ValueError("geometry status and lifted proposal disagree")


def generate_lifted_proposals(
    keyframe: KeyframeRecord,
    generator: MaskGenerator,
    filter_config: MaskFilterConfig,
) -> list[GeneratedProposal]:
    raw = list(generator.generate(keyframe.rgb, frame_id=keyframe.frame_id))
    image_config = replace(filter_config, min_descriptor_confidence=None)
    filtered = filter_image_mask_proposals(
        raw,
        keyframe.image_shape,
        image_config,
        frame_id=keyframe.frame_id,
    )
    points_world = keyframe.points_world()
    geometry_allowed = (
        keyframe.geometry_confidence > filter_config.min_geometry_confidence
    ) & np.all(np.isfinite(points_world), axis=-1)
    if filter_config.min_descriptor_confidence is not None:
        if keyframe.descriptor_confidence is None:
            raise ValueError(
                "descriptor filtering requested, but exported keyframe has no Q map"
            )
        geometry_allowed &= (
            keyframe.descriptor_confidence
            > filter_config.min_descriptor_confidence
        )

    generated: list[GeneratedProposal] = []
    for proposal in filtered:
        supported = remove_small_components(
            proposal.mask & geometry_allowed,
            min(
                filter_config.min_component_pixels,
                filter_config.min_lifted_points,
            ),
        )
        supported_count = int(np.count_nonzero(supported))
        coverage = supported_count / max(proposal.area, 1)
        if supported_count < filter_config.min_lifted_points:
            generated.append(
                GeneratedProposal(
                    mask_2d=proposal,
                    lifted_3d=None,
                    geometry_status=GeometryStatus.UNANCHORED_2D,
                    geometry_coverage=coverage,
                )
            )
            continue
        status = (
            GeometryStatus.ANCHORED_3D
            if coverage >= filter_config.full_anchor_coverage
            else GeometryStatus.PARTIAL_3D
        )
        generated.append(
            GeneratedProposal(
                mask_2d=proposal,
                lifted_3d=lift_mask_to_3d(
                    keyframe,
                    supported,
                    proposal_id=proposal.proposal_id,
                    source_mask_area=proposal.area,
                ),
                geometry_status=status,
                geometry_coverage=coverage,
            )
        )
    return generated
