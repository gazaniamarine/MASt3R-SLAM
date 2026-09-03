"""Renderer logic that must hold before MP3D scenes are available.

scripts/render_vlnce_tour.py runs under habitat-vla, but its tour assembly is
pure geometry, so it is exercised here with injected navmesh functions.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_vlnce_tour import (  # noqa: E402
    leg_frame_ranges,
    scene_glb,
    tour_polyline,
    validate_links,
)


def _leg(index, start, goal, landmark="rug", reference_path=None):
    return {
        "index": index,
        "episode_id": 100 + index,
        "trajectory_id": 200 + index,
        "instruction": f"Walk on and stop near the {landmark}.",
        "landmark": landmark,
        "start_position": list(start),
        "goal_position": list(goal),
        "geodesic_distance": float(np.linalg.norm(np.array(goal) - np.array(start))),
        "reference_path": reference_path or [list(start), list(goal)],
    }


def _tour(legs):
    return {
        "scene": "zsNo4HB9uLZ",
        "legs": legs,
        "return_leg_index": 0,
        "return_query": legs[0]["landmark"],
        "return_position": legs[0]["goal_position"],
    }


def _euclidean(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _straight(a, b):
    return [np.asarray(a, dtype=float), np.asarray(b, dtype=float)]


class ValidateLinksTest(unittest.TestCase):
    def test_accepts_a_connected_chain(self):
        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0]), _leg(1, [5, 0, 0], [10, 0, 0])])
        self.assertIsNone(validate_links(_euclidean, tour, 1.0))

    def test_rejects_a_link_beyond_tolerance(self):
        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0]), _leg(1, [9, 0, 0], [12, 0, 0])])
        problem = validate_links(_euclidean, tour, 1.0)
        self.assertIsNotNone(problem)
        self.assertIn("geodesic", problem)

    def test_rejects_an_unreachable_link(self):
        # A wall between the two legs: the navmesh reports no path at all.
        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0]), _leg(1, [5, 0, 0], [10, 0, 0])])
        problem = validate_links(lambda a, b: None, tour, 99.0)
        self.assertIn("unreachable", problem)

    def test_euclidean_link_that_is_geodesically_far_is_rejected(self):
        # The case the whole re-check exists for: 1 m apart, 20 m around a wall.
        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0]), _leg(1, [5.5, 0, 0], [10, 0, 0])])
        self.assertIsNone(validate_links(_euclidean, tour, 1.0))
        self.assertIsNotNone(validate_links(lambda a, b: 20.0, tour, 1.0))

    def test_single_leg_tour_has_no_links(self):
        self.assertIsNone(validate_links(_euclidean, _tour([_leg(0, [0, 0, 0], [1, 0, 0])]), 0.0))


class TourPolylineTest(unittest.TestCase):
    def test_concatenates_legs(self):
        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0]), _leg(1, [5, 0, 0], [10, 0, 0])])
        polyline, straight = tour_polyline(_straight, tour)
        self.assertEqual(straight, 0)
        np.testing.assert_allclose(polyline[0], [0, 0, 0])
        np.testing.assert_allclose(polyline[-1], [10, 0, 0])

    def test_drops_the_duplicated_link_waypoint(self):
        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0]), _leg(1, [5, 0, 0], [10, 0, 0])])
        polyline, _ = tour_polyline(_straight, tour)
        self.assertEqual(len(polyline), 3)

    def test_expands_waypoints_along_the_navmesh(self):
        # The navmesh detours around a corner between the two waypoints.
        def detour(a, b):
            return [np.asarray(a, dtype=float), np.array([2.0, 0.0, 3.0]),
                    np.asarray(b, dtype=float)]

        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0])])
        polyline, straight = tour_polyline(detour, tour)
        self.assertEqual(len(polyline), 3)
        self.assertEqual(straight, 0)
        np.testing.assert_allclose(polyline[1], [2.0, 0.0, 3.0])

    def test_counts_straight_line_fallbacks(self):
        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0])])
        polyline, straight = tour_polyline(lambda a, b: None, tour)
        self.assertEqual(straight, 1)
        self.assertEqual(len(polyline), 2)

    def test_multi_waypoint_reference_path_is_followed(self):
        leg = _leg(0, [0, 0, 0], [3, 0, 0],
                   reference_path=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        polyline, _ = tour_polyline(_straight, _tour([leg]))
        self.assertEqual(len(polyline), 4)


class LegFrameRangesTest(unittest.TestCase):
    def _frames(self, positions):
        return [(np.asarray(p, dtype=float), 0.0) for p in positions]

    def test_boundaries_land_on_the_closest_frame(self):
        frames = self._frames([[i, 0, 0] for i in range(11)])
        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0]), _leg(1, [5, 0, 0], [10, 0, 0])])
        ranges = leg_frame_ranges(frames, tour)
        self.assertEqual(ranges[0]["first_frame"], 0)
        self.assertEqual(ranges[0]["last_frame"], 5)
        self.assertEqual(ranges[1]["first_frame"], 5)
        self.assertEqual(ranges[1]["last_frame"], 10)

    def test_ranges_are_contiguous(self):
        frames = self._frames([[i * 0.5, 0, 0] for i in range(21)])
        tour = _tour([_leg(0, [0, 0, 0], [4, 0, 0]), _leg(1, [4, 0, 0], [10, 0, 0])])
        ranges = leg_frame_ranges(frames, tour)
        self.assertEqual(ranges[0]["last_frame"], ranges[1]["first_frame"])

    def test_doubling_back_does_not_invert_a_range(self):
        # Chained tours revisit earlier rooms: leg 1 ends near leg 0's start.
        # A global nearest-frame search would put its boundary before its own
        # first frame and make the range unsliceable.
        frames = self._frames([[i, 0, 0] for i in range(11)])
        tour = _tour([_leg(0, [0, 0, 0], [10, 0, 0]),
                      _leg(1, [10, 0, 0], [2, 0, 0], landmark="bed")])
        ranges = leg_frame_ranges(frames, tour)
        for entry in ranges:
            self.assertLessEqual(entry["first_frame"], entry["last_frame"])
        self.assertEqual(ranges[1]["first_frame"], 10)

    def test_boundaries_are_non_decreasing(self):
        frames = self._frames([[i, 0, 0] for i in range(21)])
        tour = _tour([_leg(0, [0, 0, 0], [15, 0, 0]),
                      _leg(1, [15, 0, 0], [5, 0, 0]),
                      _leg(2, [5, 0, 0], [8, 0, 0])])
        boundaries = [entry["last_frame"] for entry in leg_frame_ranges(frames, tour)]
        self.assertEqual(boundaries, sorted(boundaries))

    def test_carries_the_landmark_through(self):
        frames = self._frames([[i, 0, 0] for i in range(6)])
        tour = _tour([_leg(0, [0, 0, 0], [5, 0, 0], landmark="red chair")])
        self.assertEqual(leg_frame_ranges(frames, tour)[0]["landmark"], "red chair")


class SceneLookupTest(unittest.TestCase):
    def test_prefers_the_canonical_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            house = Path(tmp) / "2azQ1b91cZZ"
            house.mkdir()
            expected = house / "2azQ1b91cZZ.glb"
            expected.touch()
            self.assertEqual(scene_glb(tmp, "2azQ1b91cZZ"), str(expected))

    def test_finds_a_nested_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "v1" / "scans" / "2azQ1b91cZZ"
            nested.mkdir(parents=True)
            expected = nested / "2azQ1b91cZZ.glb"
            expected.touch()
            self.assertEqual(scene_glb(tmp, "2azQ1b91cZZ"), str(expected))

    def test_returns_the_canonical_path_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = scene_glb(tmp, "nosuchhouse")
            self.assertFalse(os.path.isfile(path))
            self.assertIn("nosuchhouse", path)


if __name__ == "__main__":
    unittest.main()


class ChooseReturnLegTest(unittest.TestCase):
    """The return target is chosen by distance, not by position in the tour."""

    def setUp(self):
        import render_vlnce_tour

        self.module = render_vlnce_tour
        self._snap = render_vlnce_tour.snap
        self._geodesic = render_vlnce_tour.geodesic_distance
        render_vlnce_tour.snap = lambda pf, point: np.asarray(point, dtype=float)
        render_vlnce_tour.geodesic_distance = lambda pf, a, b: float(
            np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
        )

    def tearDown(self):
        self.module.snap = self._snap
        self.module.geodesic_distance = self._geodesic

    def _tour3(self):
        # Leg 0's goal is near the end; leg 1's is far. Leg 2 is the tour end.
        return _tour([
            _leg(0, [0, 0, 0], [1, 0, 0], landmark="rug"),
            _leg(1, [1, 0, 0], [30, 0, 0], landmark="bed"),
            _leg(2, [30, 0, 0], [2, 0, 0], landmark="sink"),
        ])

    def test_picks_the_furthest_landmark_leg(self):
        chosen = self.module.choose_return_leg(None, self._tour3(), [0, 0, 0], 6.0)
        index, query, position, distance = chosen
        self.assertEqual(index, 1)
        self.assertEqual(query, "bed")
        self.assertAlmostEqual(distance, 30.0)

    def test_never_selects_the_final_leg(self):
        # The tour already ends there, so returning would be a no-op.
        chosen = self.module.choose_return_leg(None, self._tour3(), [30, 0, 0], 0.0)
        self.assertNotEqual(chosen[0], 2)

    def test_rejects_when_every_target_is_too_close(self):
        tour = _tour([
            _leg(0, [0, 0, 0], [1, 0, 0], landmark="rug"),
            _leg(1, [1, 0, 0], [2, 0, 0], landmark="bed"),
        ])
        self.assertIsNone(
            self.module.choose_return_leg(None, tour, [0, 0, 0], 6.0)
        )

    def test_skips_legs_without_a_landmark(self):
        tour = _tour([
            _leg(0, [0, 0, 0], [50, 0, 0]),
            _leg(1, [50, 0, 0], [10, 0, 0], landmark="bed"),
            _leg(2, [10, 0, 0], [0, 0, 0], landmark="sink"),
        ])
        tour["legs"][0]["landmark"] = None
        chosen = self.module.choose_return_leg(None, tour, [0, 0, 0], 6.0)
        self.assertEqual(chosen[1], "bed")

    def test_unreachable_leg_is_ignored(self):
        self.module.geodesic_distance = lambda pf, a, b: None
        self.assertIsNone(
            self.module.choose_return_leg(None, self._tour3(), [0, 0, 0], 0.0)
        )


class NearFloorDistanceTest(unittest.TestCase):
    """Camera pitch decides whether ray carving sees any nearby floor."""

    def setUp(self):
        import render_vlnce_tour

        self.near = render_vlnce_tour._near_floor_distance

    def test_level_camera_cannot_see_the_near_floor(self):
        # 1.5 m up, 90 deg HFOV cropped to 4:3 -> nothing closer than 2 m.
        self.assertAlmostEqual(self.near(1.5, 90.0, 0.0), 2.0, places=6)

    def test_pitching_down_brings_the_floor_closer(self):
        level = self.near(1.5, 90.0, 0.0)
        tilted = self.near(1.5, 90.0, 20.0)
        self.assertLess(tilted, level)
        self.assertAlmostEqual(tilted, 0.98, places=2)

    def test_monotonic_in_pitch(self):
        distances = [self.near(1.5, 90.0, p) for p in (0, 10, 20, 30, 40)]
        self.assertEqual(distances, sorted(distances, reverse=True))

    def test_sign_of_pitch_is_ignored(self):
        self.assertAlmostEqual(self.near(1.5, 90.0, 20.0), self.near(1.5, 90.0, -20.0))

    def test_scales_with_camera_height(self):
        self.assertAlmostEqual(self.near(3.0, 90.0, 0.0), 2.0 * self.near(1.5, 90.0, 0.0))

    def test_straight_down_sees_the_floor_underfoot(self):
        self.assertEqual(self.near(1.5, 90.0, 90.0), 0.0)
