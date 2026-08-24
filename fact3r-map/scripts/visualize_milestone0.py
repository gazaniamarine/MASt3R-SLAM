#!/usr/bin/env python3
"""Generate the Milestone 0 world-alignment regression visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.proposals.lift_to_3d import lift_mask_to_3d  # noqa: E402
from fact3r.regression import load_regression_sequence  # noqa: E402
from fact3r.visualization.alignment import write_alignment_ply  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=PROJECT_ROOT / "tests/fixtures/milestone0_sequence.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/milestone0_alignment.ply",
    )
    args = parser.parse_args()

    sequence = load_regression_sequence(args.fixture)
    keyframes = {keyframe.frame_id: keyframe for keyframe in sequence.keyframes}
    proposals = tuple(
        lift_mask_to_3d(
            keyframes[mask.frame_id],
            mask.mask,
            proposal_id=mask.proposal_id,
            min_geometry_confidence=1.0,
            min_descriptor_confidence=1.0,
        )
        for mask in sequence.masks
    )
    output = write_alignment_ply(args.output, sequence.keyframes, proposals)
    print(
        f"Wrote {output} with {len(sequence.keyframes)} keyframes and "
        f"{len(proposals)} lifted proposals"
    )
    for proposal in proposals:
        centroid = ", ".join(f"{value:.3f}" for value in proposal.centroid_xyz)
        print(
            f"{proposal.proposal_id}: {len(proposal.points_world)} points, "
            f"world centroid=({centroid})"
        )


if __name__ == "__main__":
    main()

