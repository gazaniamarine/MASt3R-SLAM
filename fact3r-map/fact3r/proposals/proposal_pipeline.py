"""End-to-end per-keyframe path from RGB to filtered, lifted proposals."""

from __future__ import annotations

from dataclasses import dataclass

from fact3r.proposals.lift_to_3d import LiftedProposal, lift_mask_to_3d
from fact3r.proposals.mask_filter import MaskFilterConfig, filter_mask_proposals
from fact3r.proposals.mask_generator import MaskGenerator, MaskProposal2D
from fact3r.reconstruction.keyframes import KeyframeRecord


@dataclass(frozen=True, slots=True)
class GeneratedProposal:
    mask_2d: MaskProposal2D
    lifted_3d: LiftedProposal


def generate_lifted_proposals(
    keyframe: KeyframeRecord,
    generator: MaskGenerator,
    filter_config: MaskFilterConfig,
) -> list[GeneratedProposal]:
    raw = list(generator.generate(keyframe.rgb, frame_id=keyframe.frame_id))
    filtered = filter_mask_proposals(raw, keyframe, filter_config)
    return [
        GeneratedProposal(
            mask_2d=proposal,
            lifted_3d=lift_mask_to_3d(
                keyframe,
                proposal.mask,
                proposal_id=proposal.proposal_id,
            ),
        )
        for proposal in filtered
    ]

