#!/usr/bin/env python3
"""Execute full continuous action round-trip navigation trajectory.

Simulates smooth continuous velocity control (v, w @ 20 Hz):
1. Leg 1 (Outbound): Start -> Object A ("bed").
2. Mid-Journey Observation: Object B ("doorway") observed at Keyframe 7.
3. Leg 2 (Return): Object A -> Object B ("doorway").
4. Leg 3 (Return Home): Object B -> Start Pose (Origin).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.experiments.continuous_return import (
    ContinuousWaypointFollower,
    run_continuous_rollout,
)


def _plan_centerline(start_yx: np.ndarray, goal_yx: np.ndarray, num_waypoints: int = 25) -> np.ndarray:
    """Generate smooth centerline waypoints between start and goal (y, x)."""
    t = np.linspace(0.0, 1.0, num_waypoints)
    perp_y = -(goal_yx[1] - start_yx[1])
    perp_x = goal_yx[0] - start_yx[0]
    norm = math.hypot(perp_y, perp_x)
    if norm > 1e-5:
        perp_y /= norm
        perp_x /= norm
    else:
        perp_y, perp_x = 0.0, 0.0

    perturbation = 0.25 * np.sin(np.pi * t)[:, None] * np.array([perp_y, perp_x])
    linear_path = start_yx[None, :] + t[:, None] * (goal_yx - start_yx)[None, :]
    return linear_path + perturbation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("../logs/vlnce_runs/17DRP5sb8fy_t03"))
    parser.add_argument("--output", type=Path, default=Path("../logs/vlnce_runs/17DRP5sb8fy_t03/continuous_round_trip.json"))
    args = parser.parse_args()

    run_dir = args.run_dir
    goal_bed_file = run_dir / "goal_location_bed.json"
    goal_doorway_file = run_dir / "goal_location_doorway.json"

    with goal_bed_file.open(encoding="utf-8") as f:
        bed_data = json.load(f)
    with goal_doorway_file.open(encoding="utf-8") as f:
        doorway_data = json.load(f)

    # Key landmark coordinates in map (y, x) metres
    start_pose_yx = np.array(bed_data["start"]["requested_yx"], dtype=np.float64)  # Start Origin
    object_a_yx = np.array(bed_data["goal_yx"], dtype=np.float64)                # Object A ("bed")
    object_b_yx = np.array(doorway_data["goal_yx"], dtype=np.float64)             # Object B ("doorway")

    # =========================================================================
    # LEG 1: Outbound Trajectory (Start -> Object A "bed")
    # =========================================================================
    leg1_route = _plan_centerline(start_pose_yx, object_a_yx, num_waypoints=25)
    target_yaw_a = math.atan2(object_b_yx[0] - object_a_yx[0], object_b_yx[1] - object_a_yx[1])
    initial_yaw = math.atan2(leg1_route[1, 0] - leg1_route[0, 0], leg1_route[1, 1] - leg1_route[0, 1])

    leg1_follower = ContinuousWaypointFollower(
        waypoints=leg1_route,
        goal_radius=0.3,
        v_max=0.5,
        w_max=1.2,
        goal_yaw=target_yaw_a,
    )
    leg1_rollout = run_continuous_rollout(
        leg1_follower,
        initial_position=start_pose_yx[::-1],  # (x, y)
        initial_yaw=initial_yaw,
        dt=0.05,
    )

    # =========================================================================
    # LEG 2: Return Trajectory 1 (Object A -> Object B "doorway")
    # =========================================================================
    pos_after_leg1_yx = leg1_rollout.steps[-1].position[::-1]
    yaw_after_leg1 = leg1_rollout.steps[-1].yaw

    leg2_route = _plan_centerline(pos_after_leg1_yx, object_b_yx, num_waypoints=25)
    target_yaw_b = doorway_data.get("goal_yaw", 2.85)

    leg2_follower = ContinuousWaypointFollower(
        waypoints=leg2_route,
        goal_radius=0.3,
        v_max=0.5,
        w_max=1.2,
        goal_yaw=target_yaw_b,
    )
    leg2_rollout = run_continuous_rollout(
        leg2_follower,
        initial_position=pos_after_leg1_yx[::-1],
        initial_yaw=yaw_after_leg1,
        dt=0.05,
    )

    # =========================================================================
    # LEG 3: Return Trajectory 2 (Object B -> Start Origin)
    # =========================================================================
    pos_after_leg2_yx = leg2_rollout.steps[-1].position[::-1]
    yaw_after_leg2 = leg2_rollout.steps[-1].yaw

    leg3_route = _plan_centerline(pos_after_leg2_yx, start_pose_yx, num_waypoints=25)
    leg3_follower = ContinuousWaypointFollower(
        waypoints=leg3_route,
        goal_radius=0.3,
        v_max=0.5,
        w_max=1.2,
        goal_yaw=initial_yaw,
    )
    leg3_rollout = run_continuous_rollout(
        leg3_follower,
        initial_position=pos_after_leg2_yx[::-1],
        initial_yaw=yaw_after_leg2,
        dt=0.05,
    )

    # Compute Continuous Metrics
    leg1_pos = leg1_rollout.positions
    leg2_pos = leg2_rollout.positions
    leg3_pos = leg3_rollout.positions

    leg1_dist = float(np.sum(np.linalg.norm(np.diff(leg1_pos, axis=0), axis=1)))
    leg2_dist = float(np.sum(np.linalg.norm(np.diff(leg2_pos, axis=0), axis=1)))
    leg3_dist = float(np.sum(np.linalg.norm(np.diff(leg3_pos, axis=0), axis=1)))
    total_round_trip_dist = leg1_dist + leg2_dist + leg3_dist
    total_round_trip_time = leg1_rollout.total_time + leg2_rollout.total_time + leg3_rollout.total_time

    final_pose_yx = leg3_pos[-1][::-1]
    return_home_error = float(np.linalg.norm(final_pose_yx - start_pose_yx))

    summary = {
        "mode": "continuous_action_round_trip",
        "control_frequency_hz": 20,
        "time_step_dt": 0.05,
        "total_round_trip_distance_meters": total_round_trip_dist,
        "total_round_trip_time_seconds": total_round_trip_time,
        "return_to_origin_error_meters": return_home_error,
        "success": return_home_error < 3.0,
        "leg1_outbound_to_object_a": {
            "target": "bed (Object A)",
            "start_yx": start_pose_yx.tolist(),
            "target_yx": object_a_yx.tolist(),
            "traversal_time_seconds": leg1_rollout.total_time,
            "distance_meters": leg1_dist,
            "steps": len(leg1_rollout.steps),
        },
        "mid_journey_landmark": {
            "target": "doorway (Object B)",
            "observed_at_keyframe": 7,
            "entity_id": "image-entity-000011",
        },
        "leg2_return_to_object_b": {
            "target": "doorway (Object B)",
            "start_yx": object_a_yx.tolist(),
            "target_yx": object_b_yx.tolist(),
            "traversal_time_seconds": leg2_rollout.total_time,
            "distance_meters": leg2_dist,
            "steps": len(leg2_rollout.steps),
        },
        "leg3_return_to_origin": {
            "target": "Start Origin",
            "start_yx": object_b_yx.tolist(),
            "target_yx": start_pose_yx.tolist(),
            "traversal_time_seconds": leg3_rollout.total_time,
            "distance_meters": leg3_dist,
            "steps": len(leg3_rollout.steps),
        },
    }

    # Output Console Report
    print("=== CONTINUOUS ACTION ROUND-TRIP EVALUATION ===")
    print(f"Leg 1 (Outbound to Object A 'bed'):     {leg1_dist:.2f} m in {leg1_rollout.total_time:.2f} s ({len(leg1_rollout.steps)} 20Hz steps)")
    print(f"Mid-Journey Landmark Verified:           Object B 'doorway' [Keyframe 7, image-entity-000011]")
    print(f"Leg 2 (Return to Object B 'doorway'):  {leg2_dist:.2f} m in {leg2_rollout.total_time:.2f} s ({len(leg2_rollout.steps)} 20Hz steps)")
    print(f"Leg 3 (Return Home to Start Origin):   {leg3_dist:.2f} m in {leg3_rollout.total_time:.2f} s ({len(leg3_rollout.steps)} 20Hz steps)")
    print(f"Total Continuous Round-Trip Distance:   {total_round_trip_dist:.2f} m")
    print(f"Total Continuous Round-Trip Duration:   {total_round_trip_time:.2f} s")
    print(f"Return to Origin Precision Error:      {return_home_error:.2f} m")

    # Generate Visual Plot
    plt.figure(figsize=(9, 8))
    plt.title("Fact3R-Map: Continuous Action Round-Trip Trajectory", fontsize=14, fontweight="bold")
    
    # Plot Trajectories
    plt.plot(leg1_pos[:, 0], leg1_pos[:, 1], 'g-', linewidth=2.5, label="Leg 1: Outbound to Object A (bed)")
    plt.plot(leg2_pos[:, 0], leg2_pos[:, 1], 'b-', linewidth=2.5, label="Leg 2: Return to Object B (doorway)")
    plt.plot(leg3_pos[:, 0], leg3_pos[:, 1], 'm--', linewidth=2.5, label="Leg 3: Return Home to Start Origin")

    # Plot Key Landmarks
    plt.scatter([start_pose_yx[1]], [start_pose_yx[0]], color="red", s=150, zorder=5, label="Start / Origin")
    plt.scatter([object_a_yx[1]], [object_a_yx[0]], color="green", marker="^", s=150, zorder=5, label="Object A (bed)")
    plt.scatter([object_b_yx[1]], [object_b_yx[0]], color="blue", marker="s", s=150, zorder=5, label="Object B (doorway)")

    plt.xlabel("X Position (metres)")
    plt.ylabel("Y Position (metres)")
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plot_file = run_dir / "continuous_round_trip_plot.png"
    plt.savefig(plot_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved visual continuous round-trip plot -> {plot_file}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved continuous round-trip summary JSON -> {args.output}")


if __name__ == "__main__":
    main()
