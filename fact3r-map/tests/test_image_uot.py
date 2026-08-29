from __future__ import annotations

import unittest

import numpy as np

from fact3r.association.image_uot import (
    ImageTrackEvidence,
    build_image_uot_cost_matrix,
    mast3r_mask_correspondence_score,
)


class ImageUOTTests(unittest.TestCase):
    def test_mast3r_correspondences_measure_mask_consistency(self) -> None:
        source = np.zeros((10, 10), dtype=bool)
        target = np.zeros((10, 10), dtype=bool)
        source[2:7, 2:7] = True
        target[2:7, 3:8] = True
        source_xy = np.asarray([[2, 2], [3, 3], [4, 4], [8, 8]])
        target_xy = np.asarray([[3, 2], [4, 3], [5, 4], [1, 1]])
        self.assertAlmostEqual(
            mast3r_mask_correspondence_score(
                source, target, source_xy, target_xy
            ),
            1.0,
        )

    def test_cost_accepts_pairwise_feature_supported_track(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:7, 2:7] = True
        coordinates = np.asarray([[2, 2], [3, 3], [4, 4]])
        track = ImageTrackEvidence(
            entity_id="entity-0",
            last_frame_id=1,
            last_proposal_id="old",
            prototype=np.asarray([1.0, 0.0]),
            last_mask=mask,
        )
        matrix = build_image_uot_cost_matrix(
            ["new"],
            [mask],
            np.asarray([[0.0, 1.0]]),
            [None],
            [None],
            [track],
            frame_id=2,
            pair_source_frame_id=1,
            source_xy=coordinates,
            target_xy=coordinates,
        )
        self.assertTrue(matrix.candidate_mask[0, 0])
        self.assertEqual(matrix.components["mast3r_mask_support"][0, 0], 1.0)

    def test_cost_rejects_unrelated_observation(self) -> None:
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:7, 2:7] = True
        track = ImageTrackEvidence(
            entity_id="entity-0",
            last_frame_id=1,
            last_proposal_id="old",
            prototype=np.asarray([1.0, 0.0]),
            last_mask=mask,
        )
        matrix = build_image_uot_cost_matrix(
            ["new"],
            [mask],
            np.asarray([[-1.0, 0.0]]),
            [None],
            [None],
            [track],
            frame_id=2,
            pair_source_frame_id=1,
            source_xy=np.empty((0, 2), dtype=np.int32),
            target_xy=np.empty((0, 2), dtype=np.int32),
        )
        self.assertFalse(matrix.candidate_mask[0, 0])
        self.assertTrue(np.isinf(matrix.costs[0, 0]))

    def test_disabled_temporal_cues_do_not_contribute(self) -> None:
        mask = np.ones((5, 5), dtype=bool)
        coordinates = np.asarray([[1, 1], [2, 2]], dtype=np.int32)
        track = ImageTrackEvidence(
            entity_id="entity-0",
            last_frame_id=1,
            last_proposal_id="old",
            prototype=np.asarray([1.0, 0.0]),
            last_mask=mask,
        )
        matrix = build_image_uot_cost_matrix(
            ["new"],
            [mask],
            np.asarray([[0.0, 1.0]]),
            ["old"],
            [1.0],
            [track],
            frame_id=2,
            pair_source_frame_id=1,
            source_xy=coordinates,
            target_xy=coordinates,
            sam2_weight=0.0,
            mast3r_weight=0.0,
            min_sam2_iou=float("inf"),
            min_mast3r_support=float("inf"),
        )
        self.assertFalse(matrix.candidate_mask[0, 0])
        self.assertEqual(matrix.components["sam2_link_iou"][0, 0], 0.0)
        self.assertEqual(matrix.components["mast3r_mask_support"][0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
