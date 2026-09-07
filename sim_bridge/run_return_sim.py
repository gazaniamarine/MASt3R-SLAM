#!/usr/bin/env python3
"""Drive an outbound A* leg or a memory-recalled return leg with NavDP, in
real Habitat simulation -- no real rover involved.

Two modes, sharing the same control loop:

  --mode outbound   A* centerline (thirdparty/safediffuser's planner, via
                    scripts/execute_vlnce_return.py's own plan_route,
                    reused not reimplemented) -> memory_nav.WaypointGoalProvider
                    -> NavDP. On arrival, the agent's ground-truth planar
                    pose is written as a return anchor (see below).

  --mode return     memory_nav.anchor_store.find_arrival_anchor recalls the
                    anchor an outbound run wrote -> memory_nav.ReturnGoalProvider
                    -> NavDP. No A*, no grid, no semantic map touched at all.

Every tick renders REAL RGB-D from habitat_sim's own sensors and steps NavDP
over a local socket (qwen3vl-2b-navdp/memory_nav/policy_server.py) -- NavDP
runs in whatever env actually has a working torch+diffusers+transformers
(here: `omnivla`, after USE_TF=0 and `pip install six`; habitat-vla itself
has no torch). Motion is genuinely continuous
(sim_bridge/continuous_action.py against the real navmesh via
pathfinder.try_step), not the discrete VLN-CE action space
execute_vlnce_return.py's own WaypointFollower uses -- that discrete
controller is reused here only for its A* route (`plan_route`), never for
stepping the agent.

Sim-only simplification, stated plainly: "pose" each tick is habitat's own
ground-truth agent state, not a real dead-reckoned odometry stream -- there
is no wheel encoder in simulation. This means the return leg here tests the
memory + NavDP-execution architecture without the real rover's odometry
drift; it is not a claim that the real-rover version will behave identically.

Optional BLIP/SAM2 landmark memory (mirrors memory_nav/outbound_node.py and
return_node.py on the real rover, see memory_nav/landmark_store.py): pass
--capture-dir on an outbound run to periodically dump frame+pose along the
route, then caption it separately (heavy models don't run inline here
either -- see landmark_store.py's docstring):

    conda run -n SAM2 python3 -m memory_nav.caption_landmarks \\
        --capture-dir logs/.../captures --anchor-file logs/.../anchors.jsonl \\
        --query "the 3d printer"

A --mode return run for the same query then narrates recalled landmarks to
the console and into the result JSON as it retraces the route -- pure
narration, never touches ReturnGoalProvider/the actual driving.

    conda run -n habitat-vla python3 sim_bridge/run_return_sim.py \\
        --mode outbound --scene 00800-TEEsavR23oF \\
        --goal logs/.../goal.json --grid logs/.../map.npy \\
        --socket /tmp/navdp_policy.sock --anchor-file logs/.../anchors.jsonl \\
        --query "the 3d printer"

    conda run -n habitat-vla python3 sim_bridge/run_return_sim.py \\
        --mode return --scene 00800-TEEsavR23oF \\
        --socket /tmp/navdp_policy.sock --anchor-file logs/.../anchors.jsonl \\
        --query "go back to the 3d printer"
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

try:
    import habitat_sim
except ImportError:
    habitat_sim = None

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)  # this file is in sim_bridge/, repo root is its parent
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, REPO_ROOT)

# NavDP's memory_nav package lives in the sibling qwen3vl-2b-navdp repo, not
# this one -- see that package's README for why the two are separate.
NAVDP_REPO = os.environ.get("NAVDP_REPO", os.path.expanduser("~/Gazania/qwen3vl-2b-navdp"))
sys.path.insert(0, NAVDP_REPO)

from memory_nav.anchor_store import append_arrival_anchor, find_arrival_anchor  # noqa: E402
from memory_nav.landmark_store import (  # noqa: E402
    append_landmark_capture, describe_landmark, find_landmarks_for_query,
    landmark_path_for, nearest_landmark,
)
from memory_nav.policy_client import PolicyClient  # noqa: E402
from memory_nav.return_goal import ReturnGoalProvider  # noqa: E402
from memory_nav.waypoint_goal import WaypointGoalProvider, decimate  # noqa: E402
from nav_pipeline.goal_utils import intrinsics_from_fov  # noqa: E402

from continuous_action import integrate  # noqa: E402


def _load_script_module(name: str):
    """Import one scripts/*.py file without running its __main__.

    Same trick execute_vlnce_return.py already uses for fact3r.experiments
    modules, applied here to execute_vlnce_return.py itself so its
    plan_route/final_pose_from_groundtruth/resolve_scene/find_dataset_config
    are reused, not re-derived.
    """
    path = os.path.join(REPO_ROOT, "scripts", f"{name}.py")
    module_name = f"sim_bridge_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_sim_rgbd(scene_glb, dataset_config, resolution, hfov, cam_height):
    """render_hm3d_traj.build_sim, plus a depth sensor -- that script never
    needed depth (pure geodesic pose tracking), NavDP does."""

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [resolution, resolution]
    rgb_spec.position = [0.0, cam_height, 0.0]
    rgb_spec.hfov = hfov

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [resolution, resolution]
    depth_spec.position = [0.0, cam_height, 0.0]
    depth_spec.hfov = hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    if dataset_config:
        sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.enable_physics = False

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _resolve_glb(args, scene_id, evr):
    glb = evr.resolve_scene(args.mp3d_root, scene_id, args.scene_dataset_config)
    if glb and os.path.isfile(glb):
        return glb, args.scene_dataset_config
    hits = sorted(glob.glob(os.path.join(
        args.hm3d_root, "*", f"*-{scene_id}", f"{scene_id}.basis.glb")))
    if not hits:
        raise SystemExit(f"no mesh found for scene {scene_id!r}")
    dataset_config = args.scene_dataset_config
    if dataset_config is None:
        render_hm3d_traj = _load_script_module("render_hm3d_traj")
        dataset_config = render_hm3d_traj.find_dataset_config(args.hm3d_root)
    return hits[0], dataset_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("outbound", "return"), required=True)
    p.add_argument("--scene", required=True, help="scene id, e.g. 00800-TEEsavR23oF or 17DRP5sb8fy")
    p.add_argument("--hm3d-root", default=os.path.join(REPO_ROOT, "datasets", "hm3d_root"))
    p.add_argument("--mp3d-root", default=os.path.join(REPO_ROOT, "datasets", "mp3d"))
    p.add_argument("--scene-dataset-config", default=None)
    p.add_argument("--start-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    p.add_argument("--start-yaw", type=float, default=0.0, help="habitat yaw, radians")
    p.add_argument(
        "--start-from-anchor", default=None,
        help="start this leg from a PREVIOUSLY recorded anchor instead of --start-position -- "
             "e.g. after outbound-ing to 'the statue', pass --start-from-anchor 'the statue' "
             "for the next leg to chase a second object FROM there, chaining "
             "start -> A -> B -> \"go back to A\" the way a real multi-object session would. "
             "Overrides --start-position/--start-yaw when given.",
    )
    p.add_argument("--socket", required=True, help="policy_server.py's Unix socket path")
    p.add_argument("--anchor-file", required=True)
    p.add_argument("--query", required=True, help="the memory key this leg is resolving/recalling")

    p.add_argument("--goal", help="[outbound] project_semantic_goal.py output (needs goal_yx)")
    p.add_argument("--grid", help="[outbound] the agent's occupancy *.npy")
    p.add_argument("--planner-root", default=os.path.join(REPO_ROOT, "thirdparty", "safediffuser"))
    p.add_argument("--robot-radius", type=float, default=0.20)
    p.add_argument("--unknown-slack", type=float, default=0.20)
    p.add_argument("--no-exclude-exterior", dest="exclude_exterior", action="store_false", default=True)
    p.add_argument("--horizon", type=int, default=256)
    p.add_argument("--waypoint-radius", type=float, default=0.4)
    p.add_argument("--capture-dir", default=None,
                   help="[outbound] periodically save frame+pose along the route for "
                        "memory_nav.caption_landmarks to caption afterward (BLIP left/right "
                        "memory). Omit to disable.")
    p.add_argument("--capture-interval-m", type=float, default=1.0,
                   help="[outbound] minimum distance travelled between captures")
    p.add_argument("--landmark-radius", type=float, default=1.5,
                   help="[return] narrate a BLIP landmark once within this many metres of "
                        "where it was recorded on the outbound leg")

    p.add_argument(
        "--gt-position", type=float, nargs=3, action="append", metavar=("X", "Y", "Z"),
        help="a GOAT ground-truth object position (habitat frame) -- see scripts/"
             "goat_ground_truth.py to look these up. Repeatable for a multi-instance "
             "category; GOAT scores success against the NEAREST instance, so this "
             "tracks distance to whichever is closest each tick. Independent of "
             "--goal/anchor -- this checks whether the real object was reached, not "
             "just whether our own pipeline's resolved point was reached.",
    )
    p.add_argument("--goal-radius", type=float, default=0.4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--predict-hz", type=float, default=3.0)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--hfov", type=float, default=90.0)
    p.add_argument("--cam-height", type=float, default=1.41)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    if habitat_sim is None:
        raise SystemExit("habitat-sim is not importable; run under conda env habitat-vla")

    evr = _load_script_module("execute_vlnce_return")
    vlnce_return = evr.vlnce_return

    glb, dataset_config = _resolve_glb(args, args.scene, evr)
    sim = _build_sim_rgbd(glb, dataset_config, args.resolution, args.hfov, args.cam_height)
    try:
        agent = sim.initialize_agent(0)
        pathfinder = sim.pathfinder
        if not pathfinder.is_loaded:
            raise SystemExit(f"no navmesh for scene {args.scene}")

        if args.start_from_anchor:
            anchor = find_arrival_anchor(args.anchor_file, args.start_from_anchor)
            if anchor is None:
                raise SystemExit(
                    f'--start-from-anchor "{args.start_from_anchor}" has no recorded anchor in '
                    f"{args.anchor_file} -- that leg must complete first."
                )
            ax, ay, atheta = anchor
            position_raw = vlnce_return.map_to_habitat(ax, ay, args.cam_height)
            position = np.asarray(pathfinder.snap_point(position_raw.astype(np.float32)), dtype=np.float64)
            if not np.all(np.isfinite(position)):
                position = position_raw
            yaw = vlnce_return.map_yaw_to_habitat(atheta)
            print(f"starting from '{args.start_from_anchor}' anchor: map=({ax:.2f}, {ay:.2f}) "
                  f"-> habitat position={position.tolist()} yaw={yaw:+.2f}")
        elif args.start_position is not None:
            position = np.asarray(args.start_position, dtype=np.float64)
            yaw = float(args.start_yaw)
        else:
            position = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
            yaw = float(args.start_yaw)

        agent_state = habitat_sim.AgentState()
        agent_state.position = position.astype(np.float32)
        agent_state.rotation = evr.yaw_to_quat(yaw)
        agent.set_state(agent_state)

        fx, fy, cx, cy = intrinsics_from_fov(args.resolution, args.resolution, args.hfov)
        intrinsics = (fx, fy, cx, cy)

        if args.mode == "outbound":
            if not (args.goal and args.grid):
                raise SystemExit("--mode outbound needs --goal and --grid")
            goal = json.load(open(args.goal))
            if "goal_yx" not in goal:
                raise SystemExit("goal file has no goal_yx; run project_semantic_goal.py")
            goal_yx = np.asarray(goal["goal_yx"], dtype=np.float64)
            target_map_xy = (float(goal_yx[1]), float(goal_yx[0]))
            start_x, start_y = vlnce_return.habitat_to_map_xy(position)
            start_yx = np.array([start_y, start_x], dtype=np.float64)
            route, problem = evr.plan_route(args.grid, start_yx, goal_yx, args)
            if route is None:
                raise SystemExit(f"no route over the agent's own map: {problem}")
            print(f"planned {len(route)} A* waypoints over the agent's own map")
            waypoints_xy = decimate(route[:, ::-1])  # (y, x) -> (x, y)
            provider = WaypointGoalProvider(waypoints_xy, waypoint_radius=args.waypoint_radius,
                                             goal_radius=args.goal_radius)
        else:
            anchor = find_arrival_anchor(args.anchor_file, args.query)
            if anchor is None:
                raise SystemExit(
                    f'no recorded return anchor for "{args.query}" in {args.anchor_file} -- '
                    "run --mode outbound for this query first"
                )
            print(f"recalled return anchor: ({anchor[0]:.2f}, {anchor[1]:.2f}, {anchor[2]:+.2f})")
            target_map_xy = (anchor[0], anchor[1])
            provider = ReturnGoalProvider(anchor, goal_radius=args.goal_radius)

        # Real target position in habitat frame, snapped to the navmesh --
        # same pattern execute_vlnce_return.py uses for its own scoring, so
        # distance is measured the same way: geodesic on the navmesh, never
        # the planner's own straight-line map distance (that can be shorter
        # than any walkable route, e.g. through a wall).
        target_habitat_raw = vlnce_return.map_to_habitat(target_map_xy[0], target_map_xy[1], position[1])
        target_habitat = np.asarray(
            pathfinder.snap_point(target_habitat_raw.astype(np.float32)), dtype=np.float64
        )
        if not np.all(np.isfinite(target_habitat)):
            target_habitat = target_habitat_raw

        client = PolicyClient(args.socket)
        client.reset()

        capture_dir = Path(args.capture_dir) if args.capture_dir else None
        if capture_dir is not None:
            capture_dir.mkdir(parents=True, exist_ok=True)
            print(f"landmark captures every {args.capture_interval_m}m -> {capture_dir} "
                  f"(run memory_nav.caption_landmarks on it afterward)")
        last_capture_xy = None
        capture_count = 0

        landmark_records = []
        landmarks_narrated = set()
        narrated_log = []
        if args.mode == "return":
            landmark_records = find_landmarks_for_query(landmark_path_for(args.anchor_file), args.query)
            print(f"{len(landmark_records)} landmark(s) recalled for '{args.query}'")

        def geodesic(a, b):
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(a, dtype=np.float32)
            path.requested_end = np.asarray(b, dtype=np.float32)
            if not pathfinder.find_path(path):
                return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
            return float(path.geodesic_distance)

        start_habitat = position.copy()
        optimal_length = geodesic(start_habitat, target_habitat)
        success_distance = args.goal_radius

        # GOAT ground truth (independent of our own resolved goal/anchor --
        # see scripts/goat_ground_truth.py). Distance to the NEAREST
        # instance, matching GOAT's own "any instance counts" scoring for
        # object goals.
        gt_positions = [np.asarray(p, dtype=np.float64) for p in (args.gt_position or [])]

        def gt_distance(pos):
            if not gt_positions:
                return None
            return min(geodesic(pos, gt) for gt in gt_positions)

        gt_optimal = min((geodesic(start_habitat, gt) for gt in gt_positions), default=None)

        track = [{"position": position.tolist(), "yaw": yaw, "t": 0.0,
                 "distance_to_target": geodesic(position, target_habitat),
                 "distance_to_gt": gt_distance(position)}]
        dt = 1.0 / args.predict_hz
        reached_step = None
        for step in range(1, args.max_steps + 1):
            obs = sim.get_sensor_observations()
            rgb = np.asarray(obs["color_sensor"])[:, :, :3].astype(np.uint8)
            depth = np.asarray(obs["depth_sensor"], dtype=np.float32)

            map_x, map_y = vlnce_return.habitat_to_map_xy(position)
            map_theta = vlnce_return.habitat_yaw_to_map(yaw)
            goal = provider.external_goal(map_x, map_y, map_theta)

            if capture_dir is not None and args.mode == "outbound":
                here = np.array([map_x, map_y])
                if (last_capture_xy is None
                        or np.linalg.norm(here - last_capture_xy) >= args.capture_interval_m):
                    frame_path = capture_dir / f"frame_{capture_count:06d}.png"
                    Image.fromarray(rgb).save(frame_path)
                    append_landmark_capture(capture_dir, frame_path, map_x, map_y, map_theta)
                    last_capture_xy = here
                    capture_count += 1

            if args.mode == "return" and landmark_records:
                hit = nearest_landmark(landmark_records, map_x, map_y, args.landmark_radius,
                                        landmarks_narrated)
                if hit is not None:
                    landmark_index, record = hit
                    landmarks_narrated.add(landmark_index)
                    description = describe_landmark(record)
                    print(f"[MEMORY] passing landmark: {description}")
                    narrated_log.append({"step": step, "map_xy": [map_x, map_y], "description": description})

            reply = client.step(rgb, depth, pose=(map_x, map_y, map_theta),
                                 external_goal=goal, intrinsics=intrinsics)

            position, yaw = integrate(position, yaw, reply["linear"], reply["angular"], dt, pathfinder)
            agent_state = habitat_sim.AgentState()
            agent_state.position = position.astype(np.float32)
            agent_state.rotation = evr.yaw_to_quat(yaw)
            agent.set_state(agent_state)
            track.append({"position": position.tolist(), "yaw": yaw, "t": step * dt,
                          "linear": reply["linear"], "angular": reply["angular"], "state": reply["state"],
                          "distance_to_target": geodesic(position, target_habitat),
                          "distance_to_gt": gt_distance(position)})

            if provider.reached:
                reached_step = step
                print(f"reached goal at step {step} ({step * dt:.1f}s)")
                break
        else:
            print(f"WARNING: did not reach goal within --max-steps={args.max_steps}")

        if args.mode == "outbound":
            append_arrival_anchor(args.anchor_file, args.query, map_x, map_y, map_theta)
            print(f"anchor recorded for '{args.query}' at ({map_x:.2f}, {map_y:.2f}, {map_theta:+.2f}) "
                  f"[habitat ground-truth pose, standing in for real odometry -- see this script's docstring] "
                  f"-> {args.anchor_file}")

        # Real metrics, reusing vlnce_return.score_rollout exactly as
        # execute_vlnce_return.py does -- computed from this rollout's own
        # positions/distances, not the fabricated constants in
        # evaluate_vlnce_goat_simulation.py.
        Rollout, RolloutStep = vlnce_return.Rollout, vlnce_return.RolloutStep
        rollout = Rollout(
            steps=[
                RolloutStep(i, entry.get("state", "start"), np.asarray(entry["position"]),
                           entry["yaw"], entry["distance_to_target"])
                for i, entry in enumerate(track)
            ],
            stopped=reached_step is not None,
            exhausted=reached_step is None,
        )
        metrics = vlnce_return.score_rollout(rollout, optimal_length, success_distance=success_distance)

        gt_metrics = None
        if gt_positions:
            gt_rollout = Rollout(
                steps=[
                    RolloutStep(i, entry.get("state", "start"), np.asarray(entry["position"]),
                               entry["yaw"], entry["distance_to_gt"])
                    for i, entry in enumerate(track)
                ],
                stopped=rollout.stopped, exhausted=rollout.exhausted,
            )
            gt_metrics = vlnce_return.score_rollout(gt_rollout, gt_optimal, success_distance=success_distance)

        payload = {
            "format": "sim-bridge-rollout",
            "version": 1,
            "mode": args.mode,
            "scene": args.scene,
            "query": args.query,
            "steps": len(track) - 1,
            "reached": reached_step is not None,
            "reached_step": reached_step,
            "start_position_habitat": start_habitat.tolist(),
            "target_position_habitat": target_habitat.tolist(),
            "target_map_xy": list(target_map_xy),
            "final_position_habitat": position.tolist(),
            "final_yaw_habitat": yaw,
            "optimal_length_m": optimal_length,
            "success_distance_m": success_distance,
            "metrics": metrics,
            "gt_positions_habitat": [p.tolist() for p in gt_positions],
            "gt_optimal_length_m": gt_optimal,
            "metrics_vs_ground_truth": gt_metrics,
            "capture_dir": str(capture_dir) if capture_dir is not None else None,
            "landmarks_narrated": narrated_log,
            "track": track,
        }
        output = args.output or f"sim_bridge_{args.mode}_{args.scene}_result.json"
        with open(output, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {output}")
        print(f"[vs our own resolved goal] navigation_error={metrics['navigation_error']:.2f}m "
              f"success={metrics['success']} spl={metrics['spl']:.3f} "
              f"path_length={metrics['path_length']:.2f}m optimal={optimal_length:.2f}m "
              f"steps={metrics['steps']}")
        if gt_metrics is not None:
            print(f"[vs REAL GOAT ground truth] navigation_error={gt_metrics['navigation_error']:.2f}m "
                  f"success={gt_metrics['success']} spl={gt_metrics['spl']:.3f} "
                  f"optimal={gt_optimal:.2f}m")
        client.close()
    finally:
        sim.close()


if __name__ == "__main__":
    main()
