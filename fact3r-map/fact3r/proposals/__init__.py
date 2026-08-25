"""Class-agnostic proposal generation, filtering, and lifting interfaces."""

from fact3r.proposals.lift_to_3d import LiftedProposal, lift_mask_to_3d
from fact3r.proposals.mask_filter import MaskFilterConfig, filter_mask_proposals
from fact3r.proposals.mask_generator import MaskGenerator, MaskProposal2D
from fact3r.proposals.proposal_pipeline import GeneratedProposal, generate_lifted_proposals
from fact3r.proposals.sam2_generator import SAM2AutomaticMaskGenerator
from fact3r.proposals.sam2_official_generator import SAM2OfficialMaskGenerator
from fact3r.proposals.storage import (
    SavedProposalFrame,
    iter_saved_proposal_frames,
    load_proposal_run_manifest,
)

__all__ = [
    "GeneratedProposal",
    "LiftedProposal",
    "MaskFilterConfig",
    "MaskGenerator",
    "MaskProposal2D",
    "SAM2AutomaticMaskGenerator",
    "SAM2OfficialMaskGenerator",
    "SavedProposalFrame",
    "filter_mask_proposals",
    "generate_lifted_proposals",
    "iter_saved_proposal_frames",
    "lift_mask_to_3d",
    "load_proposal_run_manifest",
]
