from __future__ import annotations

import unittest

import numpy as np

from fact3r.semantics.semantic_bev import build_semantic_grid
from fact3r.semantics.semantic_goal import (
    cell_centre_xy,
    group_cell_counts,
    nearest_cell_in,
    weighted_centroid_cell,
    world_xy_to_cell,
)


class CellWorldRoundTripTests(unittest.TestCase):
    """The (y, x) / (x, y) boundary, which fails silently when it is wrong."""

    # Deliberately asymmetric: a square grid with a symmetric origin would pass
    # every one of these tests with the axes transposed.
    ORIGIN = (-4.0, 7.5)
    RESOLUTION = 0.05

    def test_cell_to_world_uses_origin_x_for_columns(self) -> None:
        x, y = cell_centre_xy(3, 11, self.ORIGIN, self.RESOLUTION)
        # col drives x off origin[0]; row drives y off origin[1].
        self.assertAlmostEqual(float(x), -4.0 + (11 + 0.5) * 0.05)
        self.assertAlmostEqual(float(y), 7.5 + (3 + 0.5) * 0.05)

    def test_round_trip_recovers_every_cell(self) -> None:
        rows, cols = np.mgrid[0:9, 0:13]
        rows, cols = rows.ravel(), cols.ravel()
        x, y = cell_centre_xy(rows, cols, self.ORIGIN, self.RESOLUTION)
        back_rows, back_cols = world_xy_to_cell(x, y, self.ORIGIN, self.RESOLUTION)
        np.testing.assert_array_equal(back_rows, rows)
        np.testing.assert_array_equal(back_cols, cols)

    def test_round_trip_fails_when_the_axes_are_swapped(self) -> None:
        """A transposed round trip must not quietly agree with a correct one."""
        x, y = cell_centre_xy(3, 11, self.ORIGIN, self.RESOLUTION)
        swapped_row, swapped_col = world_xy_to_cell(
            y, x, self.ORIGIN, self.RESOLUTION
        )
        self.assertNotEqual((int(swapped_row), int(swapped_col)), (3, 11))

    def test_planner_order_is_y_then_x(self) -> None:
        """HM3DMap.cell_to_world returns np.stack([y, x]); mirror it here."""
        x, y = cell_centre_xy(3, 11, self.ORIGIN, self.RESOLUTION)
        goal_yx = [float(y), float(x)]
        self.assertGreater(goal_yx[0], goal_yx[1])  # y = 7.68, x = -3.42
        row, col = world_xy_to_cell(goal_yx[1], goal_yx[0], self.ORIGIN, self.RESOLUTION)
        self.assertEqual((int(row), int(col)), (3, 11))

    def test_matches_the_grid_the_semantic_map_is_built_on(self) -> None:
        """The inverse of `build_semantic_grid`'s own indexing, not a parallel one."""
        resolution, lower = 0.5, np.asarray([-1.0, 2.0])
        # Floor frame chosen so plane_x is world x and plane_y is world z, which
        # is what camera_points_to_rover_map produces.
        points = np.asarray([[0.30, 0.0, 3.70]], dtype=np.float32)
        grid = build_semantic_grid(
            points,
            np.asarray([0], dtype=np.int32),
            np.asarray([1.0], dtype=np.float32),
            shape=(6, 6),
            lower_xy=lower,
            resolution=resolution,
            floor_origin=np.zeros(3),
            floor_u=np.asarray([1.0, 0.0, 0.0]),
            floor_v=np.asarray([0.0, 0.0, 1.0]),
        )
        rows, cols = np.nonzero(grid.entity_ids == 0)
        self.assertEqual(len(rows), 1)
        # The cell the fuse stage put the vote in must be the cell this module
        # reports the position of.
        row, col = world_xy_to_cell(0.30, 3.70, lower, resolution)
        self.assertEqual((int(row), int(col)), (int(rows[0]), int(cols[0])))
        x, y = cell_centre_xy(rows[0], cols[0], lower, resolution)
        self.assertAlmostEqual(float(x), 0.25)
        self.assertAlmostEqual(float(y), 3.75)


class OnMapFilterTests(unittest.TestCase):
    def test_counts_only_cells_the_entity_actually_won(self) -> None:
        semantic_ids = np.asarray([[0, 0, -1], [2, -1, -1]], dtype=np.int32)
        groups = [
            {"semantic_id": 0, "group_id": "a"},
            {"semantic_id": 1, "group_id": "b"},
            {"semantic_id": 2, "group_id": "c"},
        ]
        counts = group_cell_counts(semantic_ids, groups)
        self.assertEqual(counts, {"a": 2, "b": 0, "c": 1})

    def test_entities_beyond_the_grid_ids_count_zero(self) -> None:
        """A manifest lists every group; the grid only ever holds a few."""
        semantic_ids = np.full((2, 2), -1, dtype=np.int32)
        groups = [{"semantic_id": i, "group_id": f"g{i}"} for i in range(4)]
        self.assertEqual(set(group_cell_counts(semantic_ids, groups).values()), {0})

    def test_filter_drops_the_positionless_entities(self) -> None:
        semantic_ids = np.asarray([[5, -1]], dtype=np.int32)
        groups = [
            {"semantic_id": 5, "group_id": "mapped"},
            {"semantic_id": 6, "group_id": "unmapped"},
        ]
        counts = group_cell_counts(semantic_ids, groups)
        on_map = {g for g, n in counts.items() if n > 0}
        self.assertEqual(on_map, {"mapped"})


class WeightedCentroidTests(unittest.TestCase):
    def test_confidence_pulls_the_centroid_toward_the_confident_cells(self) -> None:
        rows = np.asarray([0, 0, 10])
        cols = np.asarray([0, 0, 10])
        row, col = weighted_centroid_cell(rows, cols, np.asarray([1.0, 1.0, 8.0]))
        self.assertAlmostEqual(row, 8.0)
        self.assertAlmostEqual(col, 8.0)

    def test_zero_weights_fall_back_to_the_plain_centroid(self) -> None:
        rows, cols = np.asarray([0, 4]), np.asarray([2, 6])
        row, col = weighted_centroid_cell(rows, cols, np.zeros(2))
        self.assertAlmostEqual(row, 2.0)
        self.assertAlmostEqual(col, 4.0)

    def test_an_entity_with_no_cells_is_an_error_not_a_nan(self) -> None:
        with self.assertRaises(ValueError):
            weighted_centroid_cell(
                np.asarray([], dtype=int), np.asarray([], dtype=int), np.asarray([])
            )


class GoalProjectionTests(unittest.TestCase):
    def test_centroid_inside_an_obstacle_projects_out_of_it(self) -> None:
        """The U-shaped case: the entity's own centroid is in the wall.

        Cells marked X are the entity; the centroid of the U lands in the gap
        between its arms, which is occupied, so no goal may be placed there.

            row 4  . . . . .
            row 3  X . . . X
            row 2  X . # . X      # = the centroid, inside the obstacle
            row 1  X X X X X
            row 0  . . . . .
        """
        navigable = np.zeros((5, 5), dtype=bool)
        navigable[4, :] = True  # the only free row, above the U
        entity_rows = np.asarray([3, 2, 1, 1, 1, 1, 1, 2, 3])
        entity_cols = np.asarray([0, 0, 0, 1, 2, 3, 4, 4, 4])
        row, col = weighted_centroid_cell(
            entity_rows, entity_cols, np.ones(len(entity_rows))
        )
        self.assertFalse(navigable[int(row), int(col)])
        goal_row, goal_col, moved = nearest_cell_in(navigable, row, col)
        self.assertTrue(navigable[goal_row, goal_col])
        self.assertEqual(goal_row, 4)
        self.assertGreater(moved, 0.0)

    def test_a_navigable_centroid_is_left_where_it_is(self) -> None:
        navigable = np.ones((4, 4), dtype=bool)
        row, col, moved = nearest_cell_in(navigable, 2.0, 3.0)
        self.assertEqual((row, col), (2, 3))
        self.assertAlmostEqual(moved, 0.0)

    def test_projection_stays_inside_the_mask_it_was_given(self) -> None:
        """Restricting to one component must not be undone by a nearer cell."""
        component = np.zeros((3, 9), dtype=bool)
        component[1, 8] = True   # the reachable component, far away
        other = np.zeros((3, 9), dtype=bool)
        other[1, 1] = True       # nearer, but across a wall
        row, col, _ = nearest_cell_in(component, 1.0, 0.0)
        self.assertEqual((row, col), (1, 8))
        self.assertFalse(other[row, col])

    def test_clearance_breaks_ties_among_equally_near_cells(self) -> None:
        """Projection lands on a boundary, which is where clearance is worst."""
        navigable = np.zeros((3, 5), dtype=bool)
        navigable[1, 1] = navigable[1, 2] = True
        clearance = np.zeros((3, 5))
        clearance[1, 1] = 0.01   # nearest, but a pinch point
        clearance[1, 2] = 0.30   # one cell further, with room to stand
        plain = nearest_cell_in(navigable, 1.0, 0.0)
        self.assertEqual(plain[:2], (1, 1))
        preferred = nearest_cell_in(
            navigable, 1.0, 0.0, prefer=clearance, tolerance_cells=4.0
        )
        self.assertEqual(preferred[:2], (1, 2))

    def test_no_navigable_cell_raises_rather_than_returning_a_point(self) -> None:
        with self.assertRaises(ValueError):
            nearest_cell_in(np.zeros((4, 4), dtype=bool), 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
