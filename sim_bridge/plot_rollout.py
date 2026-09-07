#!/usr/bin/env python3
"""Plot a sim_bridge/run_return_sim.py result: trajectory over the agent's
own occupancy map, distance-to-goal over time, and commanded velocity over
time -- everything computed directly from that JSON's own `track` (real
per-tick habitat_sim positions and the real geodesic distance-to-target
run_return_sim.py logs), never a reconstructed or synthesized path. This is
the honest counterpart to fact3r-map/scripts/evaluate_vlnce_goat_simulation.py,
whose plots and numbers are hardcoded/interpolated, not computed from an
actual rollout.

    python3 sim_bridge/plot_rollout.py --result /tmp/my_return_result.json \\
        --map-manifest logs/goat/y9hTuugGdiq/map_semantic.json \\
        --output /tmp/my_return_result.png

Without --map-manifest, the trajectory plots on a blank background (still
useful for the distance/velocity panels, just not for seeing whether the
path crossed a wall).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def habitat_to_map_xy(position):
    """(x, y_up, z) habitat -> (x, y) in the occupancy grid's own frame.

    Same formula as fact3r-map/fact3r/experiments/vlnce_return.py's
    habitat_to_map_xy -- duplicated rather than imported so this script
    only needs numpy+matplotlib, not habitat_sim/scipy, and can run in any
    env against a result file produced elsewhere.
    """
    return float(position[0]), -float(position[2])


def load_occupancy(map_manifest_path):
    manifest = json.load(open(map_manifest_path))
    grid_dir = os.path.dirname(os.path.abspath(map_manifest_path))
    grid_file = manifest["grid_file"]
    grid_path = grid_file if os.path.isabs(grid_file) else os.path.join(grid_dir, grid_file)
    payload = np.load(grid_path)
    # grid_file is the semantic BEV .npz (occupancy + semantic_ids + ...);
    # a plain .npy (just the occupancy array) loads directly instead.
    occupancy = payload["occupancy"] if hasattr(payload, "files") else payload
    origin_xy = manifest["origin_xy"]
    resolution = manifest["resolution_metres"]
    return occupancy, origin_xy, resolution


def occupancy_rgb(occupancy):
    """-1 unknown, >=65 wall/obstacle, else free -- see
    build_depth_semantic_bev.py's own occupied_thresh=0.65 (renderer already
    treats >=65 as a wall: `canvas[occupancy >= 65] = 35`). Rendered as
    light gray (unknown), white (free), dark (wall) so a real trajectory
    crossing a wall is visually obvious.
    """
    rgb = np.full(occupancy.shape + (3,), 235, dtype=np.uint8)  # free = near-white
    rgb[occupancy < 0] = 205  # unknown = light gray
    rgb[occupancy >= 65] = 40  # wall/obstacle = dark
    return rgb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", required=True, help="run_return_sim.py's --output JSON")
    p.add_argument("--map-manifest", default=None,
                   help="<stem>_semantic.json next to the occupancy grid, for the background map")
    p.add_argument("--output", default=None, help="PNG path (default: alongside --result)")
    args = p.parse_args()

    result = json.load(open(args.result))
    track = result["track"]
    positions = np.asarray([t["position"] for t in track])  # habitat (x, y_up, z)
    map_xy = np.asarray([habitat_to_map_xy(pos) for pos in positions])
    times = np.asarray([t["t"] for t in track])
    distances = np.asarray([t["distance_to_target"] for t in track])
    states = [t.get("state", "start") for t in track]
    linear = np.asarray([t.get("linear", 0.0) for t in track])
    angular = np.asarray([t.get("angular", 0.0) for t in track])

    start_xy = habitat_to_map_xy(result["start_position_habitat"])
    target_xy = habitat_to_map_xy(result["target_position_habitat"])
    metrics = result["metrics"]
    gt_positions_xy = [habitat_to_map_xy(p) for p in result.get("gt_positions_habitat") or []]
    gt_metrics = result.get("metrics_vs_ground_truth")
    gt_distances = np.asarray([t.get("distance_to_gt") for t in track]) if gt_metrics else None

    state_colors = {"start": "#888888", "GOTO": "#2563eb", "AVOID": "#dc2626",
                    "STOP": "#16a34a", "SEARCH": "#a855f7", "TRACK": "#0891b2"}

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # top-down trajectory, in the occupancy grid's own (x, y) frame -- same
    # frame the map itself, our resolved goal, and the GOAT ground truth
    # positions all live in, so everything overlays correctly.
    ax_traj = fig.add_subplot(gs[0, :])
    if args.map_manifest:
        occupancy, origin_xy, resolution = load_occupancy(args.map_manifest)
        h, w = occupancy.shape
        extent = [origin_xy[0], origin_xy[0] + w * resolution,
                  origin_xy[1], origin_xy[1] + h * resolution]
        ax_traj.imshow(occupancy_rgb(occupancy), origin="lower", extent=extent, zorder=0)
    for i in range(len(map_xy) - 1):
        color = state_colors.get(states[min(i + 1, len(states) - 1)], "#888888")
        ax_traj.plot(map_xy[i:i + 2, 0], map_xy[i:i + 2, 1], color=color, linewidth=2, zorder=4)
    ax_traj.scatter(*start_xy, color="black", s=150, marker="o", zorder=5, label="start")
    ax_traj.scatter(*target_xy, color="red", s=200, marker="*", zorder=5, label="our resolved goal/anchor")
    goal_radius = result.get("success_distance_m", 0.4)
    ax_traj.add_patch(plt.Circle(target_xy, goal_radius, color="red", fill=False,
                                 linestyle="--", alpha=0.5, zorder=5,
                                 label=f"success radius ({goal_radius}m)"))
    for i, gt in enumerate(gt_positions_xy):
        ax_traj.scatter(*gt, color="#16a34a", s=220, marker="P", zorder=6,
                        label="GOAT ground truth" if i == 0 else None)
        ax_traj.add_patch(plt.Circle(gt, goal_radius, color="#16a34a", fill=False,
                                     linestyle=":", alpha=0.5, zorder=5))
    ax_traj.set_xlabel("map x (m)")
    ax_traj.set_ylabel("map y (m)")
    ax_traj.set_aspect("equal")
    ax_traj.set_title(f"{result['mode']} leg -- '{result['query']}' -- scene {result['scene']}")
    for state, color in state_colors.items():
        if state in states:
            ax_traj.plot([], [], color=color, linewidth=2, label=state)
    ax_traj.legend(loc="best", fontsize=8)
    if not args.map_manifest:
        ax_traj.grid(True, linestyle=":", alpha=0.5)

    # distance to target over time
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.plot(times, distances, color="#2563eb", linewidth=2, label="vs our resolved goal")
    if gt_distances is not None:
        ax_dist.plot(times, gt_distances, color="#16a34a", linewidth=2, label="vs REAL GOAT ground truth")
    ax_dist.axhline(goal_radius, color="red", linestyle="--", alpha=0.6, label="success radius")
    ax_dist.set_xlabel("time (s)")
    ax_dist.set_ylabel("geodesic distance (m)")
    ax_dist.set_title("Distance to goal over time")
    ax_dist.grid(True, linestyle=":", alpha=0.5)
    ax_dist.legend(fontsize=8)

    # commanded velocity over time
    ax_cmd = fig.add_subplot(gs[1, 1])
    ax_cmd.plot(times, linear, color="#16a34a", linewidth=1.5, label="linear (m/s)")
    ax_cmd.plot(times, angular, color="#f59e0b", linewidth=1.5, label="angular (rad/s)")
    ax_cmd.set_xlabel("time (s)")
    ax_cmd.set_title("NavDP commanded velocity")
    ax_cmd.grid(True, linestyle=":", alpha=0.5)
    ax_cmd.legend(fontsize=8)

    summary = (
        f"[vs our resolved goal] NE={metrics['navigation_error']:.2f}m  success={metrics['success']}  "
        f"oracle_success={metrics['oracle_success']}  SPL={metrics['spl']:.3f}  "
        f"path_length={metrics['path_length']:.2f}m  optimal={result['optimal_length_m']:.2f}m  "
        f"steps={metrics['steps']}  budget_exhausted={metrics['budget_exhausted']}"
    )
    if gt_metrics is not None:
        summary += (
            f"\n[vs REAL GOAT ground truth] NE={gt_metrics['navigation_error']:.2f}m  "
            f"success={gt_metrics['success']}  oracle_success={gt_metrics['oracle_success']}  "
            f"SPL={gt_metrics['spl']:.3f}  optimal={result['gt_optimal_length_m']:.2f}m"
        )
    fig.suptitle(summary, fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output = args.output or args.result.rsplit(".", 1)[0] + ".png"
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")
    print(summary)


if __name__ == "__main__":
    main()
