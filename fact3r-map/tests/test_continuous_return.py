"""Unit tests for Continuous Action Navigation Controller."""

import math
import unittest
import numpy as np

from fact3r.experiments.continuous_return import (
    ContinuousAction,
    ContinuousWaypointFollower,
    continuous_kinematic_step,
    run_continuous_rollout,
    wrap_angle,
)


class TestContinuousReturn(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(0.0), 0.0)
        self.assertAlmostEqual(wrap_angle(math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(3 * math.pi), -math.pi)

    def test_continuous_kinematic_step(self):
        pos = np.array([0.0, 0.0])
        yaw = 0.0
        act = ContinuousAction(v=1.0, w=0.0)
        new_pos, new_yaw = continuous_kinematic_step(pos, yaw, act, dt=1.0)
        self.assertAlmostEqual(new_pos[0], 1.0)
        self.assertAlmostEqual(new_pos[1], 0.0)
        self.assertAlmostEqual(new_yaw, 0.0)

    def test_continuous_follower_straight_line(self):
        waypoints = np.array([[0.0, 0.0], [0.0, 5.0]])  # (y, x) -> route is (0,0) to (5,0)
        follower = ContinuousWaypointFollower(waypoints=waypoints, goal_radius=0.3)
        rollout = run_continuous_rollout(follower, [0.0, 0.0], 0.0, dt=0.05, max_time=30.0)
        self.assertTrue(rollout.stopped)
        self.assertLess(rollout.steps[-1].distance_to_target, 0.3)

    def test_continuous_follower_curved_path(self):
        waypoints = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
        follower = ContinuousWaypointFollower(waypoints=waypoints, goal_radius=0.4)
        rollout = run_continuous_rollout(follower, [0.0, 0.0], 0.0, dt=0.05, max_time=30.0)
        self.assertTrue(rollout.stopped)
        self.assertLess(rollout.steps[-1].distance_to_target, 0.4)


if __name__ == "__main__":
    unittest.main()
