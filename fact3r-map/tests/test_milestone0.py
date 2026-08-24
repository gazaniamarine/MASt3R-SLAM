from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from fact3r.entities.entity import Entity, EntityStatus
from fact3r.proposals.lift_to_3d import lift_mask_to_3d
from fact3r.reconstruction.keyframes import KeyframeRecord
from fact3r.reconstruction.pointmap_adapter import keyframe_record_from_mast3r
from fact3r.regression import load_regression_sequence
from fact3r.semantics.fact_graph import SemanticFact, SupportType
from fact3r.visualization.alignment import write_alignment_ply


FIXTURE = Path(__file__).parent / "fixtures/milestone0_sequence.json"


class _FakePose:
    def matrix(self) -> np.ndarray:
        return np.eye(4, dtype=np.float32)[None]


class _FakeFrame:
    frame_id = 7
    uimg = np.zeros((2, 3, 3), dtype=np.float32)
    X_canon = np.arange(18, dtype=np.float32).reshape(6, 3)
    C = np.full((6, 1), 4.0, dtype=np.float32)
    T_WC = _FakePose()
    K = None
    feat = np.full((1, 1, 1024), -999.0, dtype=np.float32)

    def get_average_conf(self) -> np.ndarray:
        return self.C / 2.0


class Milestone0Tests(unittest.TestCase):
    def test_keyframe_contract_transforms_sim3_style_matrix(self) -> None:
        record = KeyframeRecord(
            frame_id=1,
            timestamp=0.0,
            rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            pointmap_camera=np.asarray([[[1.0, 2.0, 3.0]]]),
            geometry_confidence=np.ones((1, 1)),
            pose_world_from_camera=np.asarray(
                [
                    [2.0, 0.0, 0.0, 10.0],
                    [0.0, 2.0, 0.0, 20.0],
                    [0.0, 0.0, 2.0, 30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        np.testing.assert_allclose(record.points_world()[0, 0], [12.0, 24.0, 36.0])

    def test_adapter_reshapes_frame_fields_and_accepts_dense_d_q(self) -> None:
        descriptors = np.arange(24, dtype=np.float32).reshape(6, 4)
        descriptor_confidence = np.full((6, 1), 3.0, dtype=np.float32)
        record = keyframe_record_from_mast3r(
            _FakeFrame(),
            timestamp="t7",
            descriptors=descriptors,
            descriptor_confidence=descriptor_confidence,
        )
        self.assertEqual(record.image_shape, (2, 3))
        self.assertEqual(record.pointmap_camera.shape, (2, 3, 3))
        self.assertEqual(record.mast3r_descriptors.shape, (2, 3, 4))
        np.testing.assert_allclose(record.geometry_confidence, 2.0)
        self.assertEqual(record.timestamp, "t7")

    def test_two_masks_lift_to_the_same_world_patch(self) -> None:
        sequence = load_regression_sequence(FIXTURE)
        keyframes = {keyframe.frame_id: keyframe for keyframe in sequence.keyframes}
        proposals = [
            lift_mask_to_3d(
                keyframes[mask.frame_id],
                mask.mask,
                proposal_id=mask.proposal_id,
                min_geometry_confidence=1.0,
                min_descriptor_confidence=1.0,
            )
            for mask in sequence.masks
        ]
        self.assertEqual(len(proposals[0].points_world), 4)
        np.testing.assert_allclose(
            proposals[0].points_world, proposals[1].points_world, atol=1e-7
        )
        np.testing.assert_allclose(proposals[0].centroid_xyz, [0.15, 0.05, 1.0])

    def test_alignment_visualizer_writes_both_world_pointmaps(self) -> None:
        sequence = load_regression_sequence(FIXTURE)
        keyframes = {keyframe.frame_id: keyframe for keyframe in sequence.keyframes}
        proposals = [
            lift_mask_to_3d(
                keyframes[mask.frame_id], mask.mask, proposal_id=mask.proposal_id
            )
            for mask in sequence.masks
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = write_alignment_ply(
                Path(directory) / "alignment.ply", sequence.keyframes, proposals
            )
            contents = output.read_text(encoding="utf-8")
        self.assertIn("element vertex 12", contents)
        self.assertIn("255 0 255", contents)
        self.assertIn("0 255 255", contents)

    def test_entity_and_fact_contracts_validate_state(self) -> None:
        entity = Entity(
            id="entity-1",
            status=EntityStatus.PROVISIONAL,
            centroid_xyz=np.asarray([0.15, 0.05, 1.0]),
            bounding_box_xyz=np.asarray([[0.1, 0.0, 1.0], [0.2, 0.1, 1.0]]),
            surfel_or_voxel_geometry=np.asarray([[0.1, 0.0, 1.0]]),
            persistence_probability=0.5,
        )
        fact = SemanticFact(
            id="fact-1",
            entity_id=entity.id,
            subject=entity.id,
            predicate="colour",
            value="red",
            support_type=SupportType.ENTIRE_ENTITY,
            support_reference=entity.id,
            posterior_probability=0.8,
        )
        self.assertEqual(fact.entity_id, entity.id)


if __name__ == "__main__":
    unittest.main()

