"""Attach pre-UOT SigLIP observations as reliability-calibrated appearance cues."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from fact3r.association.tracklets import TrackletObservation
from fact3r.proposals.lift_to_3d import LiftedProposal
from fact3r.proposals.storage import SavedProposalFrame
from fact3r.semantics.observation_index import load_observation_index


@dataclass(frozen=True, slots=True)
class AppearanceReliabilityConfig:
    """Existing confidence signals combined without calibration labels."""

    reference_mask_area: float = 4096.0
    missing_track_quality: float = 0.75

    def __post_init__(self) -> None:
        if self.reference_mask_area <= 0.0:
            raise ValueError("reference_mask_area must be positive")
        if not 0.0 <= self.missing_track_quality <= 1.0:
            raise ValueError("missing_track_quality must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AppearanceEvidence:
    proposal_id: str
    sam_quality: float
    lifted_retention: float
    track_quality: float
    resolution_quality: float
    reliability: float


def proposal_appearance_reliability(
    proposal: LiftedProposal,
    observation: Mapping[str, object],
    tracklet: TrackletObservation | None,
    config: AppearanceReliabilityConfig | None = None,
) -> AppearanceEvidence:
    """Combine SAM, lift, tracking and resolution evidence geometrically."""

    config = AppearanceReliabilityConfig() if config is None else config
    sam_quality = float(
        np.clip(float(observation.get("proposal_score", 1.0)), 0.0, 1.0)
    )
    lifted_retention = float(
        np.clip(
            len(proposal.points_world) / max(proposal.source_mask_area, 1),
            0.0,
            1.0,
        )
    )
    track_quality = (
        config.missing_track_quality
        if tracklet is None or tracklet.link_iou is None
        else float(np.clip(tracklet.link_iou, 0.0, 1.0))
    )
    mask_area = max(
        0.0,
        float(observation.get("mask_area", proposal.source_mask_area)),
    )
    resolution_quality = float(
        min(1.0, np.sqrt(mask_area / config.reference_mask_area))
    )
    reliability = float(
        np.power(
            sam_quality
            * lifted_retention
            * track_quality
            * resolution_quality,
            0.25,
        )
    )
    return AppearanceEvidence(
        proposal_id=proposal.proposal_id,
        sam_quality=sam_quality,
        lifted_retention=lifted_retention,
        track_quality=track_quality,
        resolution_quality=resolution_quality,
        reliability=reliability,
    )


class SiglipAppearanceIndex:
    """Lookup over the existing SigLIP observation artifact."""

    def __init__(
        self,
        manifest_path: Path,
        manifest: dict[str, object],
        embeddings: NDArray[np.float32],
        rows: Mapping[tuple[int, str], int],
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.embeddings = embeddings
        self._rows = dict(rows)

    def enrich_frame(
        self,
        frame: SavedProposalFrame,
        tracklets: Mapping[str, TrackletObservation] | None = None,
        config: AppearanceReliabilityConfig | None = None,
    ) -> tuple[SavedProposalFrame, dict[str, AppearanceEvidence]]:
        tracklets = {} if tracklets is None else tracklets
        evidence: dict[str, AppearanceEvidence] = {}
        enriched: list[LiftedProposal] = []
        observations = self.manifest["observations"]
        for proposal in frame.proposals:
            key = (frame.frame_id, proposal.proposal_id)
            if key not in self._rows:
                raise KeyError(
                    f"SigLIP index has no row for frame {frame.frame_id}, "
                    f"proposal {proposal.proposal_id!r}"
                )
            row = self._rows[key]
            item = proposal_appearance_reliability(
                proposal,
                observations[row],
                tracklets.get(proposal.proposal_id),
                config,
            )
            evidence[proposal.proposal_id] = item
            enriched.append(
                replace(
                    proposal,
                    appearance_descriptor=self.embeddings[row],
                    appearance_reliability=item.reliability,
                )
            )
        return (
            SavedProposalFrame(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                proposals=tuple(enriched),
            ),
            evidence,
        )


def load_siglip_appearance_index(
    index: str | Path,
    *,
    require_pre_uot: bool = True,
) -> SiglipAppearanceIndex:
    manifest_path, manifest, embeddings = load_observation_index(index)
    if require_pre_uot and manifest.get("source_mapping") is not None:
        raise ValueError(
            "UOT appearance memory requires a pre-UOT SigLIP index built "
            "without --mapping"
        )
    rows: dict[tuple[int, str], int] = {}
    for fallback_row, observation in enumerate(manifest["observations"]):
        row = int(observation.get("index", fallback_row))
        key = (int(observation["frame_id"]), str(observation["proposal_id"]))
        if key in rows:
            raise ValueError(f"duplicate SigLIP observation identity {key}")
        if not 0 <= row < len(embeddings):
            raise ValueError(f"SigLIP observation row {row} is out of range")
        rows[key] = row
    return SiglipAppearanceIndex(manifest_path, manifest, embeddings, rows)
