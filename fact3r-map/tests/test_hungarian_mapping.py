from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fact3r.association import (
    HungarianEntityMapper,
    HungarianMapConfig,
    PairwiseCostConfig,
    TemporalEntityHint,
    UnmatchedReason,
)
from fact3r.proposals.lift_to_3d import LiftedProposal
from fact3r.proposals.storage import iter_saved_proposal_frames


def _proposal(
    proposal_id: str,
    frame_id: int,
    centre_x: float,
) -> LiftedProposal:
    points = np.asarray(
        [
            [centre_x - 0.01, -0.01, 1.0],
            [centre_x + 0.01, -0.01, 1.0],
            [centre_x - 0.01, 0.01, 1.0],
            [centre_x + 0.01, 0.01, 1.0],
        ],
        dtype=np.float32,
    )
    return LiftedProposal(
        proposal_id=proposal_id,
        frame_id=frame_id,
        timestamp=float(frame_id),
        pixel_rc=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
        points_world=points,
        colours_rgb=np.full((4, 3), 128, dtype=np.uint8),
        geometry_confidence=np.ones(4, dtype=np.float32),
        mast3r_descriptors=None,
        descriptor_confidence=None,
        source_mask_area=4,
    )


def _write_official_proposal_run(
    directory: Path, frames: list[list[LiftedProposal]]
) -> None:
    run_entries = []
    for proposals in frames:
        frame_id = proposals[0].frame_id
        frame_directory = directory / f"frame_{frame_id:06d}"
        frame_directory.mkdir(parents=True)
        proposal_entries = []
        for index, proposal in enumerate(proposals):
            filename = f"proposal_{index:04d}.npz"
            mask = np.ones((2, 2), dtype=bool)
            np.savez_compressed(
                frame_directory / filename,
                mask=mask,
                pixel_rc=proposal.pixel_rc,
                points_world=proposal.points_world,
                colours_rgb=proposal.colours_rgb,
                geometry_confidence=proposal.geometry_confidence,
            )
            proposal_entries.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "file": filename,
                    "source": "sam2-official:mock",
                    "score": 0.95,
                    "mask_area": 4,
                    "lifted_point_count": 4,
                    "centroid_xyz": proposal.centroid_xyz.tolist(),
                    "bounding_box_xyz": proposal.bounding_box_xyz.tolist(),
                    "bounding_box_xyxy": [0, 0, 2, 2],
                }
            )
        frame_manifest = {
            "frame_id": frame_id,
            "timestamp": float(frame_id),
            "image_shape": [2, 2],
            "proposal_count": len(proposals),
            "visualization": "alignment.ply",
            "proposals": proposal_entries,
        }
        manifest_relative = f"frame_{frame_id:06d}/manifest.json"
        (directory / manifest_relative).write_text(
            json.dumps(frame_manifest), encoding="utf-8"
        )
        run_entries.append(
            {
                "frame_id": frame_id,
                "proposal_count": len(proposals),
                "manifest": manifest_relative,
            }
        )
    run_manifest = {
        "format": "fact3r-sam2-proposals",
        "version": 1,
        "backend": "official",
        "model": "mock",
        "frame_count": len(frames),
        "frames": run_entries,
    }
    (directory / "manifest.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )


class HungarianMappingIntegrationTests(unittest.TestCase):
    def test_official_sam2_artifacts_load_as_complete_frame_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _write_official_proposal_run(
                directory,
                [[_proposal("left", 10, 0.0), _proposal("right", 10, 0.5)]],
            )
            frames = list(iter_saved_proposal_frames(directory))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame_id, 10)
        self.assertEqual(
            tuple(proposal.proposal_id for proposal in frames[0].proposals),
            ("left", "right"),
        )

    def test_mapper_runs_one_joint_assignment_and_keeps_ids(self) -> None:
        mapper = HungarianEntityMapper(
            HungarianMapConfig(
                pairwise_cost=PairwiseCostConfig(
                    max_centroid_distance_m=0.25,
                    geometry_match_distance_m=0.05,
                ),
                max_match_cost=0.65,
            )
        )
        first = mapper.process_frame(
            [_proposal("left-0", 0, 0.0), _proposal("right-0", 0, 0.5)],
            frame_id=0,
            timestamp=0.0,
        )
        self.assertEqual(first.assignment.cost_matrix.costs.shape, (2, 0))
        self.assertEqual(
            first.created_entity_ids, ("entity-000000", "entity-000001")
        )

        second = mapper.process_frame(
            [_proposal("right-1", 1, 0.51), _proposal("left-1", 1, 0.01)],
            frame_id=1,
            timestamp=1.0,
        )
        self.assertEqual(second.assignment.cost_matrix.costs.shape, (2, 2))
        self.assertEqual(
            {(match.proposal_id, match.entity_id) for match in second.assignment.matches},
            {
                ("right-1", "entity-000001"),
                ("left-1", "entity-000000"),
            },
        )
        self.assertEqual(second.created_entity_ids, ())
        self.assertEqual(len(mapper.entities), 2)
        self.assertEqual(
            tuple(entity.observation_count for entity in mapper.entities), (2, 2)
        )

    def test_unmatched_mask_creates_entity_and_old_entities_are_retained(self) -> None:
        mapper = HungarianEntityMapper()
        mapper.process_frame([_proposal("first", 0, 0.0)], frame_id=0)
        result = mapper.process_frame([_proposal("far", 1, 5.0)], frame_id=1)
        self.assertEqual(result.assignment.matches, ())
        self.assertEqual(result.created_entity_ids, ("entity-000001",))
        self.assertEqual(result.assignment.unmatched_entity_ids, ("entity-000000",))
        self.assertEqual(
            result.assignment.unmatched_proposals[0].reason,
            UnmatchedReason.NO_SPATIAL_CANDIDATE,
        )
        self.assertEqual(len(mapper.entities), 2)

    def test_mapper_uses_tracklet_hint_as_a_soft_identity_cue(self) -> None:
        mapper = HungarianEntityMapper()
        mapper.process_frame(
            [_proposal("first-a", 0, 0.0), _proposal("first-b", 0, 0.0)],
            frame_id=0,
        )
        result = mapper.process_frame(
            [_proposal("continued", 1, 0.0)],
            frame_id=1,
            temporal_hints={
                "continued": TemporalEntityHint(
                    entity_id="entity-000001", confidence=1.0
                )
            },
        )
        self.assertEqual(len(result.assignment.matches), 1)
        self.assertEqual(
            result.assignment.matches[0].entity_id, "entity-000001"
        )


if __name__ == "__main__":
    unittest.main()
