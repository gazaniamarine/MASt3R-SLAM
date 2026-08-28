from __future__ import annotations

import unittest

import numpy as np

from fact3r.semantics.scene_memory import group_scene_areas


class SceneMemoryTests(unittest.TestCase):
    def test_groups_time_windows_and_visual_changes(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        prototypes, areas = group_scene_areas(
            embeddings,
            [0.0, 1.0, 2.0, 3.0, 8.0],
            area_seconds=5.0,
            visual_split_similarity=0.8,
        )
        self.assertEqual([area["frame_indices"] for area in areas], [[0, 1], [2, 3], [4]])
        self.assertEqual(prototypes.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
