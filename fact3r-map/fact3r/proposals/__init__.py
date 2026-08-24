"""Class-agnostic proposal generation, filtering, and lifting interfaces."""

from fact3r.proposals.lift_to_3d import LiftedProposal, lift_mask_to_3d
from fact3r.proposals.mask_filter import MaskFilterConfig, filter_mask_proposals
from fact3r.proposals.mask_generator import MaskGenerator, MaskProposal2D
from fact3r.proposals.proposal_pipeline import GeneratedProposal, generate_lifted_proposals
from fact3r.proposals.sam2_generator import SAM2AutomaticMaskGenerator

__all__ = [
    "GeneratedProposal",
    "LiftedProposal",
    "MaskFilterConfig",
    "MaskGenerator",
    "MaskProposal2D",
    "SAM2AutomaticMaskGenerator",
    "filter_mask_proposals",
    "generate_lifted_proposals",
    "lift_mask_to_3d",
]
