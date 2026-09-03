"""Execute and score the return leg in the VLN-CE action space.

Everything upstream of this already exists: `resolve_semantic_goal.py` and
`project_semantic_goal.py` turn a query into a standable `goal_yx`, and the
vendored planner's `centerline`/`astar` turn that into a route. What was
missing is the part that actually drives it -- a discrete controller in the
VLN-CE action space (0.25 m forward, 15 degree turns, STOP) -- and the
bookkeeping that scores where it ended up.

The controller is deliberately pure: it sees a pose and returns an action, so
the same policy runs under a kinematic model in the tests and under habitat in
`scripts/execute_vlnce_return.py`.

**The plan must come from the agent's own map.** The simulator pathfinder
appears in the executor only to measure distance for scoring. Planning with it
would turn this into oracle navigation and the numbers would mean nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# The RxR-Habitat challenge action space, which R2R-CE baselines also use.
FORWARD_STEP_M = 0.25
TURN_ANGLE_RAD = math.radians(15.0)


class Action:
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"


# Habitat's own action names, so the executor can pass these straight through.
HABITAT_ACTIONS = {
    Action.MOVE_FORWARD: "move_forward",
    Action.TURN_LEFT: "turn_left",
    Action.TURN_RIGHT: "turn_right",
}


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def planar_to_habitat(x: float, y: float, floor_y: float) -> np.ndarray:
    """Planar (x, y) -> habitat (x, y, z), inverting `habitat_to_planar`.

    That mapping is x_planar = -z_habitat and y_planar = -x_habitat, so the
    inverse is exact and needs no fitted transform -- unlike the reconstruction
    grids, whose plane basis has to be carried around as a Sim(3).
    """

    return np.array([-float(y), float(floor_y), -float(x)], dtype=np.float64)


def habitat_to_planar_xy(position: Sequence[float]) -> Tuple[float, float]:
    x, _, z = (float(value) for value in position)
    return -z, -x


# The occupancy grid does NOT share the odometry frame. build_depth_semantic_bev
# hands the cloud to the gridder as [world_x, height, world_y] and the gridder
# lays those out rotated a quarter turn, so the map frame is the odometry frame
# rotated +90 degrees. Measured over a 160-pose track the fit is exactly
# 90.000 deg with zero translation and zero residual, so it is a convention
# rather than something to re-fit per run. Confusing the two plans confidently
# to the wrong room instead of raising.
MAP_FRAME_YAW_OFFSET = math.pi / 2.0


def habitat_to_map_xy(position: Sequence[float]) -> Tuple[float, float]:
    """Habitat position -> (x, y) metres in the occupancy grid's frame."""

    x, _, z = (float(value) for value in position)
    return x, -z


def map_to_habitat(x: float, y: float, floor_y: float) -> np.ndarray:
    """Inverse of `habitat_to_map_xy`, at a given floor height."""

    return np.array([float(x), float(floor_y), -float(y)], dtype=np.float64)


def habitat_yaw_to_map(yaw: float) -> float:
    """Heading in the grid frame, which is the odometry frame turned +90 deg."""

    return wrap_angle(float(yaw) + MAP_FRAME_YAW_OFFSET)


def map_yaw_to_habitat(yaw: float) -> float:
    """Inverse of `habitat_yaw_to_map`."""

    return wrap_angle(float(yaw) - MAP_FRAME_YAW_OFFSET)


def look_at_yaw(from_yx: Sequence[float], to_yx: Sequence[float]) -> Optional[float]:
    """Habitat heading that points from one map-frame (y, x) at another.

    Used to make the agent arrive facing the entity it was sent to, so the
    camera can actually re-observe it. Returns None when the two coincide and
    no heading is defined.
    """

    a = np.asarray(from_yx, dtype=np.float64)
    b = np.asarray(to_yx, dtype=np.float64)
    delta = b - a
    if float(np.linalg.norm(delta)) < 1e-6:
        return None
    return map_yaw_to_habitat(math.atan2(delta[0], delta[1]))


@dataclass
class WaypointFollower:
    """Drive a polyline in the VLN-CE action space.

    `waypoints` are planner-frame (y, x) metres, matching `goal_yx` and
    `centerline` output; they are converted to (x, y) internally so the heading
    maths reads normally.
    """

    waypoints: np.ndarray
    goal_radius: float = 0.5
    waypoint_radius: float = 0.3
    forward_step: float = FORWARD_STEP_M
    turn_angle: float = TURN_ANGLE_RAD
    # Coming back to a place is not the same as coming back to a view. With a
    # goal heading the agent turns on the spot once it arrives, so the camera
    # re-observes what it saw; without one it stops facing wherever it drove in
    # from, which position-only metrics score as a success anyway.
    goal_yaw: Optional[float] = None
    index: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        route = np.asarray(self.waypoints, dtype=np.float64)
        if route.ndim != 2 or route.shape[1] != 2:
            raise ValueError("waypoints must be an (N, 2) array of (y, x) metres")
        if len(route) == 0:
            raise ValueError("waypoints must not be empty")
        # (y, x) in, (x, y) held.
        self._route = route[:, ::-1].copy()

    @property
    def goal_xy(self) -> np.ndarray:
        return self._route[-1]

    def distance_to_goal(self, position: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(position, dtype=np.float64) - self.goal_xy))

    def act(self, position: Sequence[float], yaw: float) -> str:
        """One action for the current pose."""

        if self.stopped:
            return Action.STOP
        here = np.asarray(position, dtype=np.float64)

        # Retire waypoints already reached. The last one is retired only by the
        # goal test below, so the follower never runs out of a target to steer at.
        while (
            self.index < len(self._route) - 1
            and float(np.linalg.norm(self._route[self.index] - here)) < self.waypoint_radius
        ):
            self.index += 1

        if self.distance_to_goal(here) < self.goal_radius:
            if self.goal_yaw is not None:
                heading_error = wrap_angle(float(self.goal_yaw) - float(yaw))
                if abs(heading_error) > self.turn_angle / 2.0:
                    return Action.TURN_LEFT if heading_error > 0.0 else Action.TURN_RIGHT
            self.stopped = True
            return Action.STOP

        target = self._route[self.index]
        delta = target - here
        if float(np.linalg.norm(delta)) < 1e-9:
            # Standing exactly on a non-final waypoint: step on rather than
            # taking atan2 of a zero vector.
            self.index = min(self.index + 1, len(self._route) - 1)
            target = self._route[self.index]
            delta = target - here

        error = wrap_angle(math.atan2(delta[1], delta[0]) - float(yaw))
        # Turning costs a step, so only turn when the heading is off by more
        # than half a turn increment; otherwise the agent oscillates in place.
        if abs(error) > self.turn_angle / 2.0:
            return Action.TURN_LEFT if error > 0.0 else Action.TURN_RIGHT
        return Action.MOVE_FORWARD


@dataclass
class RolloutStep:
    index: int
    action: str
    position: np.ndarray
    yaw: float
    distance_to_target: float


@dataclass
class Rollout:
    steps: List[RolloutStep] = field(default_factory=list)
    stopped: bool = False
    exhausted: bool = False

    @property
    def positions(self) -> np.ndarray:
        return np.stack([step.position for step in self.steps])

    def to_json(self) -> dict:
        return {
            "stopped": self.stopped,
            "budget_exhausted": self.exhausted,
            "steps": [
                {
                    "index": step.index,
                    "action": step.action,
                    "position": step.position.tolist(),
                    "yaw": step.yaw,
                    "distance_to_target": step.distance_to_target,
                }
                for step in self.steps
            ],
        }


def run_rollout(
    follower: WaypointFollower,
    initial_position: Sequence[float],
    initial_yaw: float,
    step_fn: Callable[[str], Tuple[np.ndarray, float]],
    *,
    max_steps: int = 500,
    distance_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Rollout:
    """Drive `follower` until it stops or the step budget runs out.

    `step_fn` applies one action and returns the pose that resulted, so habitat
    (which refuses moves into geometry) and the kinematic test model plug in
    identically. `distance_fn` records distance to the true target each step;
    under habitat it should be the geodesic, which is what the metrics want.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    measure = distance_fn or (lambda position: float("nan"))

    position = np.asarray(initial_position, dtype=np.float64)
    yaw = float(initial_yaw)
    rollout = Rollout()
    rollout.steps.append(
        RolloutStep(0, "start", position.copy(), yaw, measure(position))
    )

    for index in range(1, max_steps + 1):
        action = follower.act(position, yaw)
        if action == Action.STOP:
            rollout.stopped = True
            break
        position, yaw = step_fn(action)
        position = np.asarray(position, dtype=np.float64)
        yaw = float(yaw)
        rollout.steps.append(
            RolloutStep(index, action, position.copy(), yaw, measure(position))
        )
    else:
        # Falling out of the loop means the agent never chose to stop.
        rollout.exhausted = True
    return rollout


def kinematic_step(
    position: Sequence[float],
    yaw: float,
    action: str,
    *,
    forward_step: float = FORWARD_STEP_M,
    turn_angle: float = TURN_ANGLE_RAD,
) -> Tuple[np.ndarray, float]:
    """Habitat's action model without a scene: no walls, no sliding."""

    here = np.asarray(position, dtype=np.float64)
    if action == Action.MOVE_FORWARD:
        return here + forward_step * np.array([math.cos(yaw), math.sin(yaw)]), yaw
    if action == Action.TURN_LEFT:
        return here, wrap_angle(yaw + turn_angle)
    if action == Action.TURN_RIGHT:
        return here, wrap_angle(yaw - turn_angle)
    if action == Action.STOP:
        return here, yaw
    raise ValueError("unknown action: %r" % action)


def score_rollout(
    rollout: Rollout,
    optimal_length: float,
    *,
    success_distance: float = 3.0,
    target_yaw: Optional[float] = None,
    heading_tolerance: float = TURN_ANGLE_RAD,
) -> dict:
    """VLN-CE metrics from a finished rollout.

    Distances come from whatever `run_rollout` logged, which under habitat is
    the navmesh geodesic to the true target -- measured with the simulator, not
    with the agent's map, so a distorted map cannot flatter the score.
    """

    if not rollout.steps:
        raise ValueError("rollout has no steps")
    distances = [step.distance_to_target for step in rollout.steps]
    if any(not np.isfinite(value) for value in distances):
        raise ValueError("rollout has no usable distance log")

    final_error = float(distances[-1])
    travelled = path_length_2d([step.position for step in rollout.steps])
    optimal = float(max(optimal_length, 0.0))
    success = final_error < success_distance
    spl = optimal / max(optimal, travelled) if success and travelled > 0.0 else 0.0
    metrics = {
        "navigation_error": final_error,
        "success": bool(success),
        "oracle_success": bool(min(distances) < success_distance),
        "path_length": travelled,
        "optimal_length": optimal,
        "spl": float(spl),
        "steps": len(rollout.steps) - 1,
        "stopped": rollout.stopped,
        "budget_exhausted": rollout.exhausted,
    }
    if target_yaw is not None:
        # Coming back to a place is not coming back to a view, and a
        # position-only criterion scores both the same. Report them apart.
        error = wrap_angle(float(rollout.steps[-1].yaw) - float(target_yaw))
        metrics["heading_error_deg"] = math.degrees(error)
        metrics["pose_success"] = bool(success and abs(error) <= heading_tolerance)
    return metrics


def path_length_2d(positions: Sequence[Sequence[float]]) -> float:
    points = np.asarray(positions, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
