"""Continuous differential-drive integration for a habitat_sim agent -- the
genuine version of the "continuous (v, w) @ Hz" control mode that
fact3r-map/scripts/evaluate_vlnce_goat_simulation.py's report only claims
(that script's return-leg trajectories are hardcoded straight-line lerps
between waypoints, and its metrics are literal Python constants -- nothing
there is computed from an actual rollout; do not treat it as a result).

scripts/execute_vlnce_return.py's own `apply()` only knows the discrete
VLN-CE action space (0.25 m forward step, 15 degree turn); NavDP outputs
continuous (linear, angular) velocities every tick, so this is the missing
piece between them -- same wall-sliding collision mechanics
(`pathfinder.try_step`), just continuous instead of quantized to one of
three fixed moves.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def integrate(position: np.ndarray, yaw: float, linear: float, angular: float,
              dt: float, pathfinder) -> Tuple[np.ndarray, float]:
    """One dt of (linear, angular) applied to a habitat agent's planar pose.

    linear/angular follow NavDP's own [x fwd, y left] / CCW+ yaw convention
    (see qwen3vl-2b-navdp/memory_nav/odom_frame.py); habitat forward at yaw
    is (-sin(yaw), 0, -cos(yaw)), matching execute_vlnce_return.py's own
    apply() exactly, so the two stay consistent if ever compared directly.
    Collision is resolved by pathfinder.try_step -- the environment, not
    this integrator -- exactly like the discrete controller's forward step
    already does; it slides along walls rather than passing through them.
    """

    new_yaw = yaw + angular * dt
    forward = linear * dt
    delta = forward * np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])
    desired = position + delta
    moved = pathfinder.try_step(position.astype(np.float32), desired.astype(np.float32))
    return np.asarray(moved, dtype=np.float64), new_yaw
