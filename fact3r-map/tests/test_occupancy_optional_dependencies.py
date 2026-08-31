from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import occupancy_grid  # noqa: E402


class OccupancyOptionalDependencyTests(unittest.TestCase):
    def test_numpy_binary_dilation_fallback(self) -> None:
        accelerated = occupancy_grid._scipy_binary_dilation
        try:
            occupancy_grid._scipy_binary_dilation = None
            mask = np.zeros((5, 5), dtype=bool)
            mask[2, 2] = True
            result = occupancy_grid._binary_dilation(
                mask, np.ones((3, 3), dtype=bool)
            )
        finally:
            occupancy_grid._scipy_binary_dilation = accelerated
        self.assertEqual(int(result.sum()), 9)
        self.assertTrue(result[1:4, 1:4].all())

    def test_numpy_path_radius_fallback(self) -> None:
        accelerated = occupancy_grid.cKDTree
        try:
            occupancy_grid.cKDTree = None
            result = occupancy_grid._points_near_path(
                np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
                np.asarray([[0.5, 0.0, 0.0]]),
                1.0,
            )
        finally:
            occupancy_grid.cKDTree = accelerated
        np.testing.assert_array_equal(result, [True, False])


if __name__ == "__main__":
    unittest.main()
