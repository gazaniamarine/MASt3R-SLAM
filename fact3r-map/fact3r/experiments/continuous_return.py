"""Continuous Action Controller and Navigation Rollout Module.

Provides continuous velocity control (v, w) over 2D/3D waypoints for continuous
outbound traversal to Object A and return traversal to Object B.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ContinuousAction:
    v: float  # linear velocity m/s
    w: float  # angular velocity rad/s


@dataclass
class ContinuousWaypointFollower:
    """Follows a polyline route using continuous velocity control (v, w)."""

    waypoints: np.ndarray  # (N, 2) array of (y, x) metres
    goal_radius: float = 0.3
    waypoint_radius: float = 0.4
    v_max: float = 0.5  # m/s
    w_max: float = 1.2  # rad/s
    kp_linear: float = 1.0
    kp_angular: float = 2.5
    goal_yaw: Optional[float] = None
    dt: float = 0.05  # 20 Hz control loop
    index: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        route = np.asarray(self.waypoints, dtype=np.float64)
        if route.ndim != 2 or route.shape[1] != 2:
            raise ValueError("waypoints must be an (N, 2) array of (y, x) metres")
        if len(route) == 0:
            raise ValueError("waypoints must not be empty")
        self._route = route[:, ::-1].copy()  # (x, y) held internally

    @property
    def goal_xy(self) -> np.ndarray:
        return self._route[-1]

    def distance_to_goal(self, position: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(position, dtype=np.float64) - self.goal_xy))

    def act(self, position: Sequence[float], yaw: float) -> ContinuousAction:
        """Compute continuous linear velocity (v) and angular velocity (w)."""
        if self.stopped:
            return ContinuousAction(0.0, 0.0)

        here = np.asarray(position, dtype=np.float64)

        # Retire waypoints reached
        while (
            self.index < len(self._route) - 1
            and float(np.linalg.norm(self._route[self.index] - here)) < self.waypoint_radius
        ):
            self.index += 1

        dist_goal = self.distance_to_goal(here)
        if dist_goal < self.goal_radius:
            if self.goal_yaw is not None:
                heading_err = wrap_angle(float(self.goal_yaw) - float(yaw))
                if abs(heading_err) > 0.05:
                    w = np.clip(self.kp_angular * heading_err, -self.w_max, self.w_max)
                    return ContinuousAction(0.0, float(w))
            self.stopped = True
            return ContinuousAction(0.0, 0.0)

        target = self._route[self.index]
        delta = target - here
        dist_target = float(np.linalg.norm(delta))

        if dist_target < 1e-6:
            self.index = min(self.index + 1, len(self._route) - 1)
            target = self._route[self.index]
            delta = target - here
            dist_target = float(np.linalg.norm(delta))

        target_yaw = math.atan2(delta[1], delta[0])
        heading_error = wrap_angle(target_yaw - float(yaw))

        # Continuous velocity calculation
        # Slow down linear speed when heading error is large
        speed_factor = max(0.0, math.cos(heading_error))
        v = min(self.v_max, self.kp_linear * dist_target) * speed_factor
        w = np.clip(self.kp_angular * heading_error, -self.w_max, self.w_max)

        return ContinuousAction(float(v), float(w))


@dataclass
class ContinuousStep:
    time: float
    v: float
    w: float
    position: np.ndarray
    yaw: float
    distance_to_target: float


@dataclass
class ContinuousRollout:
    steps: List[ContinuousStep] = field(default_factory=list)
    stopped: bool = False
    exhausted: bool = False

    @property
    def positions(self) -> np.ndarray:
        return np.stack([step.position for step in self.steps])

    @property
    def total_time(self) -> float:
        return self.steps[-1].time if self.steps else 0.0

    def to_json(self) -> dict:
        return {
            "stopped": self.stopped,
            "budget_exhausted": self.exhausted,
            "total_time_seconds": self.total_time,
            "steps": [
                {
                    "time": step.time,
                    "v": step.v,
                    "w": step.w,
                    "position": step.position.tolist(),
                    "yaw": step.yaw,
                    "distance_to_target": step.distance_to_target,
                }
                for step in self.steps
            ],
        }


def continuous_kinematic_step(
    position: Sequence[float],
    yaw: float,
    action: ContinuousAction,
    dt: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """Integrate continuous velocity (v, w) over timestep dt."""
    here = np.asarray(position, dtype=np.float64)
    new_yaw = wrap_angle(float(yaw) + action.w * dt)
    new_pos = here + action.v * dt * np.array([math.cos(new_yaw), math.sin(new_yaw)])
    return new_pos, new_yaw


def run_continuous_rollout(
    follower: ContinuousWaypointFollower,
    initial_position: Sequence[float],
    initial_yaw: float,
    *,
    dt: float = 0.05,
    max_time: float = 120.0,
    distance_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> ContinuousRollout:
    """Execute continuous action rollout using differential drive kinematics."""
    measure = distance_fn or (lambda pos: float(np.linalg.norm(pos - follower.goal_xy)))
    position = np.asarray(initial_position, dtype=np.float64)
    yaw = float(initial_yaw)
    t = 0.0

    rollout = ContinuousRollout()
    rollout.steps.append(
        ContinuousStep(t, 0.0, 0.0, position.copy(), yaw, measure(position))
    )

    max_steps = int(max_time / dt)
    for _ in range(max_steps):
        action = follower.act(position, yaw)
        if action.v == 0.0 and action.w == 0.0 and follower.stopped:
            rollout.stopped = True
            break
        position, yaw = continuous_kinematic_step(position, yaw, action, dt=dt)
        t += dt
        rollout.steps.append(
            ContinuousStep(t, action.v, action.w, position.copy(), yaw, measure(position))
        )
    else:
        rollout.exhausted = True

    return rollout
