from __future__ import annotations

import math
import unittest

import numpy as np

from fact3r.experiments.habitat_odometry import habitat_to_planar
from fact3r.experiments.vlnce_return import (
    Action,
    TURN_ANGLE_RAD,
    WaypointFollower,
    habitat_to_planar_xy,
    kinematic_step,
    path_length_2d,
    planar_to_habitat,
    run_rollout,
    score_rollout,
    wrap_angle,
)


def _drive(follower, position=(0.0, 0.0), yaw=0.0, max_steps=400):
    """Run the follower under the kinematic model, tracking the pose."""

    state = {"p": np.asarray(position, dtype=np.float64), "yaw": float(yaw)}

    def step(action):
        state["p"], state["yaw"] = kinematic_step(state["p"], state["yaw"], action)
        return state["p"], state["yaw"]

    return run_rollout(
        follower, state["p"], state["yaw"], step, max_steps=max_steps,
        distance_fn=lambda p: float(np.linalg.norm(p - follower.goal_xy)),
    )


class FrameTest(unittest.TestCase):
    def test_planar_and_habitat_round_trip(self):
        habitat = planar_to_habitat(3.0, -2.0, floor_y=0.17)
        self.assertEqual(habitat_to_planar_xy(habitat), (3.0, -2.0))

    def test_agrees_with_the_odometry_converter(self):
        # The two modules must not drift apart: same mapping, opposite directions.
        position = (1.5, 0.2, -4.0)
        x, y, _ = habitat_to_planar(position, 0.0)
        np.testing.assert_allclose(
            planar_to_habitat(x, y, floor_y=0.2), position
        )

    def test_wrap_angle_folds_into_one_turn(self):
        # +-pi name the same heading; this formula lands on -pi, matching the
        # `wrap` already used by scripts/render_hm3d_traj.py.
        for angle in (3.0 * math.pi, -3.0 * math.pi, 5.0, -5.0, 0.3):
            wrapped = wrap_angle(angle)
            self.assertGreaterEqual(wrapped, -math.pi - 1e-12)
            self.assertLessEqual(wrapped, math.pi + 1e-12)
            self.assertAlmostEqual(math.cos(wrapped), math.cos(angle))
            self.assertAlmostEqual(math.sin(wrapped), math.sin(angle))


class WaypointFollowerTest(unittest.TestCase):
    def test_rejects_empty_waypoints(self):
        with self.assertRaises(ValueError):
            WaypointFollower(np.zeros((0, 2)))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            WaypointFollower(np.zeros((4, 3)))

    def test_waypoints_are_read_as_y_x(self):
        # (y, x) = (0, 5) is 5 m along +x, i.e. straight ahead at yaw 0.
        follower = WaypointFollower(np.array([[0.0, 5.0]]))
        np.testing.assert_allclose(follower.goal_xy, [5.0, 0.0])

    def test_drives_straight_to_a_goal_ahead(self):
        follower = WaypointFollower(np.array([[0.0, 3.0]]))
        rollout = _drive(follower)
        self.assertTrue(rollout.stopped)
        self.assertLess(follower.distance_to_goal(rollout.positions[-1]), 0.5)
        self.assertNotIn(Action.TURN_LEFT, [s.action for s in rollout.steps])

    def test_turns_toward_a_goal_to_the_side(self):
        # (y, x) = (3, 0) is 3 m along +y: a 90 degree left turn.
        follower = WaypointFollower(np.array([[3.0, 0.0]]))
        rollout = _drive(follower)
        self.assertTrue(rollout.stopped)
        actions = [step.action for step in rollout.steps]
        self.assertIn(Action.TURN_LEFT, actions)
        self.assertNotIn(Action.TURN_RIGHT, actions)

    def test_turns_right_for_a_goal_clockwise(self):
        follower = WaypointFollower(np.array([[-3.0, 0.0]]))
        rollout = _drive(follower)
        actions = [step.action for step in rollout.steps]
        self.assertIn(Action.TURN_RIGHT, actions)
        self.assertNotIn(Action.TURN_LEFT, actions)

    def test_stops_immediately_when_already_at_the_goal(self):
        follower = WaypointFollower(np.array([[0.0, 0.1]]))
        rollout = _drive(follower)
        self.assertTrue(rollout.stopped)
        self.assertEqual(len(rollout.steps), 1)

    def test_follows_a_multi_waypoint_route(self):
        route = np.array([[0.0, 2.0], [2.0, 2.0], [2.0, 4.0]])
        follower = WaypointFollower(route)
        rollout = _drive(follower)
        self.assertTrue(rollout.stopped)
        final = rollout.positions[-1]
        self.assertLess(follower.distance_to_goal(final), 0.5)

    def test_retires_waypoints_in_order(self):
        route = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        follower = WaypointFollower(route)
        _drive(follower)
        self.assertEqual(follower.index, len(route) - 1)

    def test_does_not_oscillate_on_small_heading_error(self):
        # A heading error under half a turn increment must not trigger a turn.
        follower = WaypointFollower(np.array([[0.0, 5.0]]))
        action = follower.act((0.0, 0.0), math.radians(5.0))
        self.assertEqual(action, Action.MOVE_FORWARD)

    def test_turns_when_error_exceeds_half_an_increment(self):
        follower = WaypointFollower(np.array([[0.0, 5.0]]))
        self.assertEqual(follower.act((0.0, 0.0), math.radians(20.0)), Action.TURN_RIGHT)

    def test_goal_behind_requires_a_half_turn(self):
        follower = WaypointFollower(np.array([[0.0, -3.0]]))
        rollout = _drive(follower)
        self.assertTrue(rollout.stopped)
        turns = sum(
            1 for step in rollout.steps
            if step.action in (Action.TURN_LEFT, Action.TURN_RIGHT)
        )
        # 180 degrees at 15 degrees per action.
        self.assertGreaterEqual(turns, 11)

    def test_stays_stopped_once_stopped(self):
        follower = WaypointFollower(np.array([[0.0, 0.1]]))
        follower.act((0.0, 0.0), 0.0)
        self.assertTrue(follower.stopped)
        self.assertEqual(follower.act((0.0, 0.0), 0.0), Action.STOP)


class RolloutTest(unittest.TestCase):
    def test_budget_exhaustion_is_recorded(self):
        follower = WaypointFollower(np.array([[0.0, 50.0]]))
        rollout = _drive(follower, max_steps=5)
        self.assertFalse(rollout.stopped)
        self.assertTrue(rollout.exhausted)
        self.assertEqual(len(rollout.steps), 6)  # start + 5 actions

    def test_rejects_a_zero_budget(self):
        follower = WaypointFollower(np.array([[0.0, 1.0]]))
        with self.assertRaises(ValueError):
            run_rollout(follower, (0, 0), 0.0, lambda a: ((0, 0), 0.0), max_steps=0)

    def test_first_step_records_the_start_pose(self):
        follower = WaypointFollower(np.array([[0.0, 3.0]]))
        rollout = _drive(follower, position=(1.0, 2.0))
        self.assertEqual(rollout.steps[0].action, "start")
        np.testing.assert_allclose(rollout.steps[0].position, [1.0, 2.0])

    def test_distance_is_logged_each_step(self):
        follower = WaypointFollower(np.array([[0.0, 3.0]]))
        rollout = _drive(follower)
        distances = [step.distance_to_target for step in rollout.steps]
        self.assertGreater(distances[0], distances[-1])

    def test_blocked_agent_still_terminates(self):
        # habitat refuses moves into geometry; the pose simply does not change.
        follower = WaypointFollower(np.array([[0.0, 10.0]]))
        rollout = run_rollout(
            follower, (0.0, 0.0), 0.0, lambda action: (np.zeros(2), 0.0),
            max_steps=20,
        )
        self.assertTrue(rollout.exhausted)
        self.assertFalse(rollout.stopped)

    def test_rollout_serialises(self):
        follower = WaypointFollower(np.array([[0.0, 1.0]]))
        payload = _drive(follower).to_json()
        self.assertIn("steps", payload)
        self.assertEqual(payload["steps"][0]["action"], "start")


class ScoreRolloutTest(unittest.TestCase):
    def _rollout(self, max_steps=400, goal=(0.0, 3.0)):
        return _drive(WaypointFollower(np.array([goal])), max_steps=max_steps)

    def test_successful_return_scores(self):
        metrics = score_rollout(self._rollout(), optimal_length=3.0)
        self.assertTrue(metrics["success"])
        self.assertTrue(metrics["oracle_success"])
        self.assertLess(metrics["navigation_error"], 3.0)
        self.assertGreater(metrics["spl"], 0.0)

    def test_failure_beyond_the_threshold(self):
        rollout = self._rollout(max_steps=2, goal=(0.0, 30.0))
        metrics = score_rollout(rollout, optimal_length=30.0)
        self.assertFalse(metrics["success"])
        self.assertEqual(metrics["spl"], 0.0)
        self.assertTrue(metrics["budget_exhausted"])

    def test_oracle_success_without_success(self):
        # Passes within 3 m of the goal, then is cut off far past it.
        rollout = _drive(WaypointFollower(np.array([[0.0, 40.0]])), max_steps=60)
        metrics = score_rollout(rollout, optimal_length=40.0)
        self.assertFalse(metrics["success"])
        self.assertFalse(metrics["oracle_success"])

    def test_spl_is_capped_at_one(self):
        metrics = score_rollout(self._rollout(), optimal_length=100.0)
        self.assertLessEqual(metrics["spl"], 1.0)

    def test_rejects_a_rollout_without_distances(self):
        follower = WaypointFollower(np.array([[0.0, 1.0]]))
        rollout = run_rollout(
            follower, (0.0, 0.0), 0.0,
            lambda a: kinematic_step((0.0, 0.0), 0.0, a), max_steps=3,
        )
        with self.assertRaises(ValueError):
            score_rollout(rollout, optimal_length=1.0)


class KinematicStepTest(unittest.TestCase):
    def test_forward_moves_along_the_heading(self):
        position, yaw = kinematic_step((0.0, 0.0), math.pi / 2.0, Action.MOVE_FORWARD)
        np.testing.assert_allclose(position, [0.0, 0.25], atol=1e-12)
        self.assertAlmostEqual(yaw, math.pi / 2.0)

    def test_left_turn_increases_yaw(self):
        _, yaw = kinematic_step((0.0, 0.0), 0.0, Action.TURN_LEFT)
        self.assertAlmostEqual(yaw, math.radians(15.0))

    def test_right_turn_decreases_yaw(self):
        _, yaw = kinematic_step((0.0, 0.0), 0.0, Action.TURN_RIGHT)
        self.assertAlmostEqual(yaw, math.radians(-15.0))

    def test_unknown_action_is_rejected(self):
        with self.assertRaises(ValueError):
            kinematic_step((0.0, 0.0), 0.0, "fly")


class PathLengthTest(unittest.TestCase):
    def test_single_point_has_no_length(self):
        self.assertEqual(path_length_2d([[0.0, 0.0]]), 0.0)

    def test_sums_segments(self):
        self.assertAlmostEqual(path_length_2d([[0, 0], [3, 0], [3, 4]]), 7.0)


if __name__ == "__main__":
    unittest.main()


class ExecutorTest(unittest.TestCase):
    """The executor's non-habitat logic, which runs without a scene."""

    def _module(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "scripts" / "execute_vlnce_return.py"
        spec = importlib.util.spec_from_file_location("execute_vlnce_return", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_imports_without_habitat(self):
        module = self._module()
        self.assertIsNone(module.habitat_sim)

    def test_reads_the_final_pose_from_a_real_tour(self):
        import tempfile
        from pathlib import Path

        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "groundtruth.txt"
            path.write_text(
                "# habitat camera poses\n"
                "0.000000 0.0 1.5 0.0 0.0 0.0 0.0 1.0\n"
                "0.033333 1.0 1.5 -2.0 0.0 0.7071068 0.0 0.7071068\n"
            )
            position, yaw = module.final_pose_from_groundtruth(path)
        np.testing.assert_allclose(position, [1.0, 1.5, -2.0])
        self.assertAlmostEqual(yaw, math.pi / 2.0, places=6)

    def test_rejects_a_pose_file_with_no_poses(self):
        import tempfile
        from pathlib import Path

        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "groundtruth.txt"
            path.write_text("# only a comment\n")
            with self.assertRaises(ValueError):
                module.final_pose_from_groundtruth(path)

    def test_shares_one_yaw_implementation(self):
        module = self._module()
        from fact3r.experiments.habitat_odometry import yaw_from_quaternion

        self.assertAlmostEqual(
            module.habitat_odometry.yaw_from_quaternion([0, 0.3826834, 0, 0.9238795]),
            yaw_from_quaternion([0, 0.3826834, 0, 0.9238795]),
        )

    def test_plan_route_unpacks_the_centerline_tuple(self):
        """centerline returns (line, raw); treating it as an array breaks numpy."""
        import sys
        import types

        module = self._module()
        route = np.array([[0.0, 0.0], [1.0, 1.0]])

        fake_map = types.ModuleType("diffuser.hm3d.map")
        fake_map.HM3DMap = type("HM3DMap", (), {"load": staticmethod(lambda *a, **k: "map")})
        fake_planner = types.ModuleType("diffuser.hm3d.planner")
        fake_planner.centerline = lambda *a, **k: (route, "raw_cells")
        package = types.ModuleType("diffuser")
        hm3d = types.ModuleType("diffuser.hm3d")
        saved = {name: sys.modules.get(name) for name in
                 ("diffuser", "diffuser.hm3d", "diffuser.hm3d.map", "diffuser.hm3d.planner")}
        sys.modules.update({
            "diffuser": package, "diffuser.hm3d": hm3d,
            "diffuser.hm3d.map": fake_map, "diffuser.hm3d.planner": fake_planner,
        })
        try:
            args = types.SimpleNamespace(
                planner_root=".", robot_radius=0.2, unknown_slack=0.2,
                exclude_exterior=True, horizon=8,
            )
            result, problem = module.plan_route("grid.npy", (0, 0), (1, 1), args)
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        self.assertIsNone(problem)
        np.testing.assert_allclose(result, route)

    def test_plan_route_reports_a_disconnected_goal(self):
        """A* raises when start and goal are in different components."""
        import sys
        import types

        module = self._module()
        fake_map = types.ModuleType("diffuser.hm3d.map")
        fake_map.HM3DMap = type("HM3DMap", (), {"load": staticmethod(lambda *a, **k: "map")})
        fake_planner = types.ModuleType("diffuser.hm3d.planner")

        def raising(*args, **kwargs):
            raise ValueError("no path between start and goal")

        fake_planner.centerline = raising
        saved = {name: sys.modules.get(name) for name in
                 ("diffuser", "diffuser.hm3d", "diffuser.hm3d.map", "diffuser.hm3d.planner")}
        sys.modules.update({
            "diffuser": types.ModuleType("diffuser"),
            "diffuser.hm3d": types.ModuleType("diffuser.hm3d"),
            "diffuser.hm3d.map": fake_map, "diffuser.hm3d.planner": fake_planner,
        })
        try:
            args = types.SimpleNamespace(
                planner_root=".", robot_radius=0.2, unknown_slack=0.2,
                exclude_exterior=True, horizon=8,
            )
            result, problem = module.plan_route("grid.npy", (0, 0), (1, 1), args)
        finally:
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        self.assertIsNone(result)
        self.assertIn("no path", problem)


class MapFrameTest(unittest.TestCase):
    """The occupancy grid frame is the odometry frame turned +90 degrees."""

    def test_map_and_habitat_round_trip(self):
        from fact3r.experiments.vlnce_return import habitat_to_map_xy, map_to_habitat

        x, y = habitat_to_map_xy((-4.36, 0.07, -0.11))
        np.testing.assert_allclose(map_to_habitat(x, y, 0.07), [-4.36, 0.07, -0.11])

    def test_matches_the_measured_convention(self):
        # Fitted over a real 160-pose track: map (y, x) = (-z_hab, x_hab).
        from fact3r.experiments.vlnce_return import habitat_to_map_xy

        x, y = habitat_to_map_xy((-6.425382, 1.572447, -1.379692))
        self.assertAlmostEqual(x, -6.425382)
        self.assertAlmostEqual(y, 1.379692)

    def test_map_frame_is_planar_rotated_a_quarter_turn(self):
        from fact3r.experiments.vlnce_return import (
            habitat_to_map_xy,
            habitat_to_planar_xy,
        )

        position = (-6.43, 1.57, -1.38)
        px, py = habitat_to_planar_xy(position)
        mx, my = habitat_to_map_xy(position)
        # (mx, my) == R(+90) @ (px, py)
        self.assertAlmostEqual(mx, -py)
        self.assertAlmostEqual(my, px)

    def test_yaw_offset_matches_the_frame_rotation(self):
        from fact3r.experiments.vlnce_return import habitat_yaw_to_map

        self.assertAlmostEqual(habitat_yaw_to_map(0.0), math.pi / 2.0)
        self.assertAlmostEqual(habitat_yaw_to_map(-math.pi / 2.0), 0.0)

    def test_forward_direction_is_consistent_between_frames(self):
        # Moving forward in habitat must advance along the map heading too.
        from fact3r.experiments.vlnce_return import (
            FORWARD_STEP_M,
            habitat_to_map_xy,
            habitat_yaw_to_map,
        )

        yaw = 0.6
        start = np.array([1.0, 1.5, -2.0])
        step = FORWARD_STEP_M * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
        before = np.array(habitat_to_map_xy(start))
        after = np.array(habitat_to_map_xy(start + step))
        heading = habitat_yaw_to_map(yaw)
        moved = after - before
        self.assertAlmostEqual(math.atan2(moved[1], moved[0]), heading, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(moved)), FORWARD_STEP_M, places=9)


class GoalHeadingTest(unittest.TestCase):
    """Returning to a place is not returning to a view."""

    def test_without_a_goal_yaw_it_stops_facing_anywhere(self):
        follower = WaypointFollower(np.array([[0.0, 3.0]]))
        rollout = _drive(follower)
        self.assertTrue(rollout.stopped)
        # Drove along +x, so it ends facing +x regardless of any desired view.
        self.assertAlmostEqual(rollout.steps[-1].yaw, 0.0, places=6)

    def test_goal_yaw_is_reached_on_arrival(self):
        target = math.radians(90.0)
        follower = WaypointFollower(np.array([[0.0, 3.0]]), goal_yaw=target)
        rollout = _drive(follower)
        self.assertTrue(rollout.stopped)
        error = wrap_angle(rollout.steps[-1].yaw - target)
        self.assertLessEqual(abs(error), TURN_ANGLE_RAD / 2.0 + 1e-9)

    def test_it_turns_on_the_spot_after_arriving(self):
        target = math.radians(180.0)
        follower = WaypointFollower(np.array([[0.0, 2.0]]), goal_yaw=target)
        rollout = _drive(follower)
        arrival = next(i for i, s in enumerate(rollout.steps)
                       if follower.distance_to_goal(s.position) < follower.goal_radius)
        after = rollout.steps[arrival + 1:]
        self.assertTrue(after, "expected alignment turns after arrival")
        self.assertTrue(all(s.action in (Action.TURN_LEFT, Action.TURN_RIGHT)
                            for s in after))
        # Turning on the spot must not move the agent.
        np.testing.assert_allclose(after[-1].position, rollout.steps[arrival].position)

    def test_already_aligned_needs_no_extra_turn(self):
        follower = WaypointFollower(np.array([[0.0, 3.0]]), goal_yaw=0.0)
        aligned = _drive(follower)
        plain = _drive(WaypointFollower(np.array([[0.0, 3.0]])))
        self.assertEqual(len(aligned.steps), len(plain.steps))

    def test_turns_the_short_way_round(self):
        follower = WaypointFollower(np.array([[0.0, 1.0]]), goal_yaw=math.radians(-30.0))
        rollout = _drive(follower)
        turns = [s.action for s in rollout.steps
                 if s.action in (Action.TURN_LEFT, Action.TURN_RIGHT)]
        self.assertTrue(turns)
        self.assertNotIn(Action.TURN_LEFT, turns)


class LookAtAndPoseScoreTest(unittest.TestCase):
    def test_look_at_points_from_goal_to_entity(self):
        from fact3r.experiments.vlnce_return import (habitat_yaw_to_map,
                                                     look_at_yaw)
        # Entity is at +y in the map frame from the goal cell.
        yaw = look_at_yaw((0.0, 0.0), (3.0, 0.0))
        self.assertAlmostEqual(habitat_yaw_to_map(yaw), math.pi / 2.0, places=9)

    def test_look_at_is_none_when_coincident(self):
        from fact3r.experiments.vlnce_return import look_at_yaw
        self.assertIsNone(look_at_yaw((1.0, 2.0), (1.0, 2.0)))

    def test_map_yaw_round_trips(self):
        from fact3r.experiments.vlnce_return import (habitat_yaw_to_map,
                                                     map_yaw_to_habitat)
        for yaw in (0.0, 1.3, -2.7, 3.0):
            self.assertAlmostEqual(map_yaw_to_habitat(habitat_yaw_to_map(yaw)), yaw)

    def test_pose_success_requires_both_position_and_heading(self):
        follower = WaypointFollower(np.array([[0.0, 3.0]]))
        rollout = _drive(follower)                      # ends facing +x (yaw 0)
        aligned = score_rollout(rollout, 3.0, target_yaw=0.0)
        turned = score_rollout(rollout, 3.0, target_yaw=math.pi)
        self.assertTrue(aligned["pose_success"])
        self.assertTrue(turned["success"])              # position-only passes
        self.assertFalse(turned["pose_success"])        # the view does not
        self.assertAlmostEqual(abs(turned["heading_error_deg"]), 180.0, places=4)

    def test_heading_is_absent_without_a_target(self):
        rollout = _drive(WaypointFollower(np.array([[0.0, 3.0]])))
        self.assertNotIn("heading_error_deg", score_rollout(rollout, 3.0))
