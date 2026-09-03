"""Convert habitat camera poses into the planar odometry the BEV builder reads.

`scripts/render_vlnce_tour.py` writes 7-DoF TUM poses in habitat's world frame
(y-up, camera looking along local -z). `build_depth_semantic_bev.py` wants
`odom_*.csv` with `t,x,y,theta,v` in the usual planar robotics convention
(x forward at theta = 0, theta counter-clockwise).

The mapping between them is fixed by habitat's axes:

    x_planar = -z_habitat
    y_planar = -x_habitat
    theta    =  yaw about +y

At yaw = 0 the camera looks along -z, which is +x_planar; increasing yaw tilts
the view toward -x_habitat, which is +y_planar. So the frames agree in
handedness and no sign flip is needed anywhere else.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class OdometryRow:
    t: float
    x: float
    y: float
    theta: float
    v: float


def rotate_by_quaternion(vector: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    """Rotate `vector` by a quaternion given as (x, y, z, w)."""

    qx, qy, qz, qw = (float(value) for value in quaternion)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("quaternion has zero norm")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    u = np.array([qx, qy, qz], dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    return (
        2.0 * float(np.dot(u, v)) * u
        + (qw * qw - float(np.dot(u, u))) * v
        + 2.0 * qw * np.cross(u, v)
    )


def yaw_from_quaternion(quaternion: Sequence[float]) -> float:
    """Habitat yaw about +y, recovered from the camera's forward axis.

    Taking the rotated forward vector rather than reading `qy` directly keeps
    this correct if a pose ever carries pitch or roll.
    """

    forward = rotate_by_quaternion((0.0, 0.0, -1.0), quaternion)
    return math.atan2(-forward[0], -forward[2])


def habitat_to_planar(position: Sequence[float], yaw: float) -> Tuple[float, float, float]:
    x, _, z = (float(value) for value in position)
    return -z, -x, yaw


def read_tum_poses(path) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Read `timestamp tx ty tz qx qy qz qw` lines, skipping comments."""

    poses = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        values = [float(part) for part in parts]
        poses.append(
            (
                values[0],
                np.asarray(values[1:4], dtype=np.float64),
                np.asarray(values[4:8], dtype=np.float64),
            )
        )
    if not poses:
        raise ValueError(f"no poses found in {path}")
    return poses


def poses_to_odometry(
    poses: Sequence[Tuple[float, Sequence[float], Sequence[float]]],
    *,
    camera_height: float | None = None,
) -> List[OdometryRow]:
    """Turn TUM habitat poses into planar odometry rows.

    `camera_height` is accepted for symmetry with the renderer but is not used:
    the planar frame drops the vertical axis entirely.
    """

    if len(poses) < 2:
        raise ValueError("odometry needs at least two poses")

    rows: List[OdometryRow] = []
    previous = None
    for timestamp, position, quaternion in poses:
        x, y, theta = habitat_to_planar(position, yaw_from_quaternion(quaternion))
        if previous is None:
            velocity = 0.0
        else:
            dt = timestamp - previous[0]
            if dt <= 0.0:
                # The consumer rejects non-increasing stamps outright, so fail
                # here with a message that names the cause.
                raise ValueError(
                    "pose timestamps must strictly increase; got %r after %r"
                    % (timestamp, previous[0])
                )
            velocity = math.hypot(x - previous[1], y - previous[2]) / dt
        rows.append(OdometryRow(t=timestamp, x=x, y=y, theta=theta, v=velocity))
        previous = (timestamp, x, y)

    # The first sample has no predecessor; reuse the second so a stationary
    # start is not mistaken for a velocity discontinuity.
    if len(rows) > 1:
        rows[0] = OdometryRow(rows[0].t, rows[0].x, rows[0].y, rows[0].theta, rows[1].v)
    return rows


def write_odometry_csv(rows: Sequence[OdometryRow], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "theta", "v"])
        for row in rows:
            writer.writerow(
                [
                    "%.6f" % row.t,
                    "%.6f" % row.x,
                    "%.6f" % row.y,
                    "%.6f" % row.theta,
                    "%.6f" % row.v,
                ]
            )
    return path
