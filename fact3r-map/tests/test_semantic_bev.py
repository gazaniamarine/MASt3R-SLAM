from __future__ import annotations

import unittest

import numpy as np

from fact3r.semantics.semantic_bev import (
    aggregate_group_embeddings,
    backproject_depth,
    build_semantic_grid,
    camera_points_to_rover_map,
    camera_to_body,
)


class SemanticBEVTests(unittest.TestCase):
    def test_backprojection_preserves_pixel_geometry(self) -> None:
        depth = np.full((2, 2), 2.0, dtype=np.float32)
        points, rows, columns = backproject_depth(
            depth, fx=2.0, fy=2.0, cx=1.0, cy=1.0
        )
        np.testing.assert_array_equal(rows, [0, 0, 1, 1])
        np.testing.assert_array_equal(columns, [0, 1, 0, 1])
        np.testing.assert_allclose(points[0], [-1.0, -1.0, 2.0])
        np.testing.assert_allclose(points[-1], [0.0, 0.0, 2.0])

    def test_camera_to_rover_transform_matches_depth_bev_convention(self) -> None:
        rotation, translation = camera_to_body(0.0, 0.5)
        result = camera_points_to_rover_map(
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            rover_x=2.0,
            rover_y=3.0,
            rover_yaw=0.0,
            rotation_body_from_camera=rotation,
            translation_body_from_camera=translation,
        )
        np.testing.assert_allclose(result[0], [3.0, -0.5, 3.0])

    def test_semantic_grid_uses_strongest_entity_vote(self) -> None:
        points = np.asarray(
            [[0.2, 0.0, 0.2], [0.3, 0.0, 0.3], [0.4, 0.0, 0.4]],
            dtype=np.float32,
        )
        grid = build_semantic_grid(
            points,
            np.asarray([0, 1, 1], dtype=np.int32),
            np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(2, 2),
            lower_xy=np.asarray([0.0, 0.0]),
            resolution=1.0,
            floor_origin=np.zeros(3),
            floor_u=np.asarray([1.0, 0.0, 0.0]),
            floor_v=np.asarray([0.0, 0.0, 1.0]),
        )
        self.assertEqual(grid.entity_ids[0, 0], 1)
        self.assertAlmostEqual(float(grid.support[0, 0]), 2.0)
        self.assertAlmostEqual(float(grid.confidence[0, 0]), 2.0 / 3.0)

    def test_group_prototype_is_quality_weighted_and_normalized(self) -> None:
        embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        observations = [
            {
                "group_id": "entity-a",
                "proposal_score": 1.0,
                "association_confidence": 1.0,
            },
            {
                "group_id": "entity-a",
                "proposal_score": 0.5,
                "association_confidence": 1.0,
            },
        ]
        prototypes = aggregate_group_embeddings(
            embeddings, observations, ["entity-a"]
        )
        self.assertAlmostEqual(float(np.linalg.norm(prototypes[0])), 1.0, places=6)
        self.assertGreater(prototypes[0, 0], prototypes[0, 1])


if __name__ == "__main__":
    unittest.main()
