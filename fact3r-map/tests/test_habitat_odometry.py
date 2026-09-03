from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fact3r.experiments.habitat_odometry import (
    OdometryRow,
    habitat_to_planar,
    poses_to_odometry,
    read_tum_poses,
    rotate_by_quaternion,
    write_odometry_csv,
    yaw_from_quaternion,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_hm3d_traj import yaw_from_direction, yaw_to_quat  # noqa: E402


class QuaternionTest(unittest.TestCase):
    def test_identity_leaves_the_vector_alone(self):
        np.testing.assert_allclose(
            rotate_by_quaternion((0, 0, -1), (0, 0, 0, 1)), [0, 0, -1], atol=1e-12
        )

    def test_quarter_turn_about_y(self):
        rotated = rotate_by_quaternion((0, 0, -1), yaw_to_quat(math.pi / 2))
        np.testing.assert_allclose(rotated, [-1, 0, 0], atol=1e-9)

    def test_rejects_a_degenerate_quaternion(self):
        with self.assertRaises(ValueError):
            rotate_by_quaternion((0, 0, -1), (0, 0, 0, 0))

    def test_yaw_round_trips_through_the_renderer_convention(self):
        # Ties this module to scripts/render_hm3d_traj.py, which writes the poses.
        for degrees in range(-180, 180, 7):
            yaw = math.radians(degrees)
            recovered = yaw_from_quaternion(yaw_to_quat(yaw))
            self.assertAlmostEqual(math.sin(recovered), math.sin(yaw), places=9)
            self.assertAlmostEqual(math.cos(recovered), math.cos(yaw), places=9)

    def test_agrees_with_the_renderer_direction_helper(self):
        direction = np.array([0.3, 0.0, -0.7])
        yaw = yaw_from_direction(direction)
        self.assertAlmostEqual(yaw_from_quaternion(yaw_to_quat(yaw)), yaw, places=9)


class PlanarFrameTest(unittest.TestCase):
    def test_axes_map_as_documented(self):
        self.assertEqual(habitat_to_planar((2.0, 1.5, -3.0), 0.0), (3.0, -2.0, 0.0))

    def test_forward_at_zero_yaw_increases_x(self):
        # Habitat forward at yaw 0 is -z, which must be +x in the planar frame.
        before = habitat_to_planar((0.0, 1.5, 0.0), 0.0)
        after = habitat_to_planar((0.0, 1.5, -1.0), 0.0)
        self.assertGreater(after[0], before[0])
        self.assertAlmostEqual(after[1], before[1])

    def test_positive_yaw_turns_toward_positive_y(self):
        # A counter-clockwise planar turn must move the forward vector to +y.
        yaw = math.radians(30.0)
        forward = rotate_by_quaternion((0, 0, -1), yaw_to_quat(yaw))
        x, y, _ = habitat_to_planar(forward, yaw)
        self.assertGreater(y, 0.0)
        self.assertAlmostEqual(math.atan2(y, x), yaw, places=9)


class OdometryTest(unittest.TestCase):
    def _straight_poses(self, count=5, step=0.5, fps=30.0):
        # Walking along -z at yaw 0: planar +x.
        return [
            (i / fps, np.array([0.0, 1.5, -i * step]), yaw_to_quat(0.0))
            for i in range(count)
        ]

    def test_straight_walk_advances_x_only(self):
        rows = poses_to_odometry(self._straight_poses())
        self.assertAlmostEqual(rows[0].x, 0.0)
        self.assertAlmostEqual(rows[-1].x, 2.0)
        for row in rows:
            self.assertAlmostEqual(row.y, 0.0)
            self.assertAlmostEqual(row.theta, 0.0)

    def test_velocity_matches_the_step_rate(self):
        rows = poses_to_odometry(self._straight_poses(step=0.5, fps=30.0))
        self.assertAlmostEqual(rows[-1].v, 0.5 * 30.0, places=6)

    def test_first_velocity_borrows_the_second(self):
        rows = poses_to_odometry(self._straight_poses())
        self.assertAlmostEqual(rows[0].v, rows[1].v)

    def test_stationary_turn_has_zero_velocity(self):
        poses = [
            (i / 30.0, np.array([0.0, 1.5, 0.0]), yaw_to_quat(math.radians(i * 2.0)))
            for i in range(5)
        ]
        rows = poses_to_odometry(poses)
        for row in rows[1:]:
            self.assertAlmostEqual(row.v, 0.0)
        self.assertGreater(rows[-1].theta, rows[1].theta)

    def test_rejects_a_single_pose(self):
        with self.assertRaises(ValueError):
            poses_to_odometry(self._straight_poses(count=1))

    def test_rejects_non_increasing_timestamps(self):
        poses = self._straight_poses(count=3)
        poses[2] = (poses[1][0], poses[2][1], poses[2][2])
        with self.assertRaises(ValueError) as caught:
            poses_to_odometry(poses)
        self.assertIn("strictly increase", str(caught.exception))


class RoundTripTest(unittest.TestCase):
    def test_written_csv_is_readable_by_the_bev_builder(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from build_depth_semantic_bev import _load_odometry  # noqa: E402

        poses = [
            (i / 30.0, np.array([0.0, 1.5, -i * 0.04]), yaw_to_quat(math.radians(i)))
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_odometry_csv(poses_to_odometry(poses), Path(tmp) / "odom_t.csv")
            time, x, y, theta, velocity = _load_odometry(path)
        self.assertEqual(len(time), 20)
        self.assertTrue(np.all(np.diff(time) > 0))
        self.assertAlmostEqual(float(x[-1]), 19 * 0.04, places=5)

    def test_tum_reader_skips_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "groundtruth.txt"
            path.write_text(
                "# a comment\n"
                "0.000000 0.0 1.5 0.0 0.0 0.0 0.0 1.0\n"
                "\n"
                "0.033333 0.0 1.5 -0.04 0.0 0.0 0.0 1.0\n"
            )
            poses = read_tum_poses(path)
        self.assertEqual(len(poses), 2)

    def test_empty_pose_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "groundtruth.txt"
            path.write_text("# only a comment\n")
            with self.assertRaises(ValueError):
                read_tum_poses(path)


if __name__ == "__main__":
    unittest.main()
