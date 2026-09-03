#!/usr/bin/env python3
"""
Drive the return leg of a VLN-CE tour and score where it ends up.

Takes the goal that resolve_semantic_goal.py + project_semantic_goal.py located
in the agent's own semantic BEV, plans a route over the agent's own occupancy
grid with the vendored planner, then executes it in habitat using the VLN-CE
action space (0.25 m forward, 15 degree turns, STOP).

The simulator pathfinder is used for exactly two things, both of them scoring:
measuring geodesic distance to the true target, and resolving collisions when
the agent walks into geometry (which is the environment, not a planner). The
route itself never consults it. Planning with the pathfinder would make this
oracle navigation and the metrics meaningless.

habitat-sim and scipy both live in the `habitat-vla` env:

    conda run -n habitat-vla python3 scripts/execute_vlnce_return.py \
        --render datasets/vlnce_seqs/zsNo4HB9uLZ_t00 \
        --goal logs/vlnce_runs/zsNo4HB9uLZ_t00/goal.json \
        --grid logs/vlnce_runs/zsNo4HB9uLZ_t00/map.npy \
        --mp3d-root datasets/mp3d
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys

import numpy as np

try:
    import habitat_sim
except ImportError:
    habitat_sim = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_hm3d_traj import build_sim, find_dataset_config, yaw_to_quat  # noqa: E402
from render_vlnce_tour import resolve_scene  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PLANNER_ROOT = os.path.join(REPO_ROOT, "thirdparty", "safediffuser")
DEFAULT_MP3D_ROOT = os.path.join(REPO_ROOT, "datasets", "mp3d")


def _load_experiment_module(name):
    """Import one fact3r.experiments module without its package __init__.

    The package imports modules that need Python 3.10 syntax, and habitat-sim
    pins this environment to 3.9. These two modules are 3.9-clean on their own.
    """
    path = os.path.join(
        REPO_ROOT, "fact3r-map", "fact3r", "experiments", "%s.py" % name
    )
    module_name = "fact3r_%s" % name
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so the module has to
    # be registered before it executes or every dataclass in it raises.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


vlnce_return = _load_experiment_module("vlnce_return")
habitat_odometry = _load_experiment_module("habitat_odometry")


def final_pose_from_groundtruth(path):
    """The tour's last camera pose: where the return leg starts."""
    last = None
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                last = line.split()
    if last is None or len(last) != 8:
        raise ValueError("no usable pose in %s" % path)
    values = [float(part) for part in last]
    position = np.array(values[1:4], dtype=np.float64)
    # One implementation of this, shared with the odometry converter.
    yaw = habitat_odometry.yaw_from_quaternion(values[4:8])
    return position, yaw


def plan_route(grid_path, start_yx, goal_yx, args):
    """A route over the agent's own occupancy grid. Never the navmesh."""
    sys.path.insert(0, str(args.planner_root))
    from diffuser.hm3d.map import HM3DMap  # noqa: E402
    from diffuser.hm3d.planner import centerline  # noqa: E402

    hm3d_map = HM3DMap.load(
        grid_path,
        robot_radius=args.robot_radius,
        unknown_slack=args.unknown_slack,
        exclude_exterior=args.exclude_exterior,
    )
    try:
        # centerline returns (repaired_line, raw_astar_cells).
        route, _raw = centerline(hm3d_map, start_yx, goal_yx, args.horizon)
    except ValueError as error:
        # A* raises when start and goal sit in different components of the
        # agent's map. That is a result -- the map does not support the return
        # -- not a crash, so it is scored as a failure further up.
        return None, str(error)
    if route is None or len(np.asarray(route)) < 1:
        return None, "the planner returned an empty route"
    return np.asarray(route, dtype=np.float64), None


def main():
    if habitat_sim is None:
        print("habitat-sim is not importable; run under habitat-vla", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render",
        help="Rendered tour directory (VLN-CE). Omit when giving --scene and "
             "--start-position explicitly, as GOAT does.",
    )
    parser.add_argument("--scene", help="Scene id, when there is no render dir.")
    parser.add_argument("--hm3d-root", default=os.path.join(REPO_ROOT, "datasets", "hm3d_root"))
    parser.add_argument("--start-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--start-yaw", type=float, help="Habitat yaw, radians.")
    parser.add_argument("--target-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--success-distance", type=float)
    parser.add_argument("--cam-height", type=float, default=1.41)
    parser.add_argument("--goal", required=True, help="project_semantic_goal.py output.")
    parser.add_argument("--grid", required=True, help="The agent's occupancy *.npy.")
    parser.add_argument("--mp3d-root", default=DEFAULT_MP3D_ROOT)
    parser.add_argument("--scene-dataset-config", default=None)
    parser.add_argument("--planner-root", default=DEFAULT_PLANNER_ROOT)
    parser.add_argument("--robot-radius", type=float, default=0.20)
    parser.add_argument(
        "--unknown-slack",
        type=float,
        default=0.20,
        help="Set to the robot radius; collision_frac reads 0 otherwise.",
    )
    parser.add_argument("--no-exclude-exterior", dest="exclude_exterior",
                        action="store_false", default=True)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--goal-radius", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument(
        "--recompute-navmesh",
        action="store_true",
        help="Rebuild the navmesh from scene geometry (needed for ReplicaCAD).",
    )
    parser.add_argument("--agent-radius", type=float, default=0.20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    render = args.render
    if render:
        meta = json.load(open(os.path.join(render, "meta.json")))
    else:
        # GOAT gives the scene, a start pose and a goal position outright; there
        # is no rendered tour to read them from.
        if not (args.scene and args.start_position and args.target_position):
            print(
                "without --render you must give --scene, --start-position and "
                "--target-position",
                file=sys.stderr,
            )
            return 2
        meta = {
            "scene": args.scene,
            "cam_height_m": args.cam_height,
            "return_position": list(args.target_position),
            "return_query": args.scene,
            "success_distance_m": args.success_distance
            if args.success_distance is not None
            else 1.0,
        }
    goal = json.load(open(args.goal))
    if "goal_yx" not in goal:
        print("goal file has no goal_yx; run project_semantic_goal.py", file=sys.stderr)
        return 2
    goal_yx = np.asarray(goal["goal_yx"], dtype=np.float64)

    if render:
        start_position, start_yaw = final_pose_from_groundtruth(
            os.path.join(render, "groundtruth.txt")
        )
    else:
        start_position = np.asarray(args.start_position, dtype=np.float64)
        start_yaw = float(args.start_yaw or 0.0)
    # The grid lives in the map frame, not the odometry frame; goal_yx from
    # project_semantic_goal.py is in the same one, so the start must match it.
    start_x, start_y = vlnce_return.habitat_to_map_xy(start_position)
    start_yx = np.array([start_y, start_x], dtype=np.float64)

    cam_height = float(meta.get("cam_height_m", 1.5))
    target = np.asarray(meta["return_position"], dtype=np.float64)

    glb = resolve_scene(args.mp3d_root, meta["scene"], args.scene_dataset_config)
    if glb is None or not os.path.isfile(glb):
        # HM3D addresses scenes as <split>/<index>-<scene>/<scene>.basis.glb.
        hits = sorted(glob.glob(os.path.join(
            args.hm3d_root, "*", "*-%s" % meta["scene"], "%s.basis.glb" % meta["scene"])))
        if hits:
            glb = hits[0]
            if args.scene_dataset_config is None:
                args.scene_dataset_config = find_dataset_config(args.hm3d_root)
    if glb is None:
        print(
            "no mesh for %s and no scene dataset config given" % meta["scene"],
            file=sys.stderr,
        )
        return 1
    sim = build_sim(glb, args.scene_dataset_config, args.resolution, args.hfov,
                    cam_height)
    try:
        agent = sim.initialize_agent(0)
        pathfinder = sim.pathfinder
        if args.recompute_navmesh or not pathfinder.is_loaded:
            settings = habitat_sim.nav.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius = args.agent_radius
            settings.agent_height = cam_height
            if not sim.recompute_navmesh(pathfinder, settings):
                print("navmesh could not be built", file=sys.stderr)
                return 1
        snapped_target = np.asarray(
            pathfinder.snap_point(target.astype(np.float32)), dtype=np.float64
        )
        if not np.all(np.isfinite(snapped_target)):
            snapped_target = target

        def geodesic(a, b):
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(a, dtype=np.float32)
            path.requested_end = np.asarray(b, dtype=np.float32)
            if not pathfinder.find_path(path):
                return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
            return float(path.geodesic_distance)

        route, route_problem = plan_route(args.grid, start_yx, goal_yx, args)
        optimal = float(meta.get("return_optimal_geodesic_m", float("nan")))
        if not np.isfinite(optimal):
            optimal = geodesic(start_position, snapped_target)
        success_distance = float(meta.get("success_distance_m", 3.0))

        if route is None:
            # The agent never moves, so it fails where it stands. Recording this
            # as a scored failure keeps it in the table instead of quietly
            # dropping the hardest cases.
            standing_error = geodesic(start_position, snapped_target)
            metrics = {
                "navigation_error": standing_error,
                "success": bool(standing_error < success_distance),
                "oracle_success": bool(standing_error < success_distance),
                "path_length": 0.0,
                "optimal_length": optimal,
                "spl": 0.0,
                "steps": 0,
                "stopped": False,
                "budget_exhausted": False,
                "route_found": False,
                "route_problem": route_problem,
            }
            output = args.output or os.path.join(render or ".", "return_result.json")
            with open(output, "w") as handle:
                json.dump(
                    {
                        "format": "fact3r-vlnce-return",
                        "version": 1,
                        "scene": meta.get("scene"),
                        "return_query": meta.get("return_query"),
                        "goal_yx": goal_yx.tolist(),
                        "grid": os.path.abspath(args.grid),
                        "route_found": False,
                        "route_problem": route_problem,
                        "metrics": metrics,
                    },
                    handle,
                    indent=2,
                )
            print("no route over the agent's own map: %s" % route_problem)
            print("scored as a failure at NE %.2f m -> %s" % (standing_error, output))
            return 0

        print("planned %d waypoints over the agent's map" % len(route))
        state = {"position": start_position.copy(), "yaw": float(start_yaw)}

        def apply(action):
            if action == vlnce_return.Action.TURN_LEFT:
                state["yaw"] = vlnce_return.wrap_angle(
                    state["yaw"] + vlnce_return.TURN_ANGLE_RAD
                )
            elif action == vlnce_return.Action.TURN_RIGHT:
                state["yaw"] = vlnce_return.wrap_angle(
                    state["yaw"] - vlnce_return.TURN_ANGLE_RAD
                )
            elif action == vlnce_return.Action.MOVE_FORWARD:
                yaw = state["yaw"]
                # habitat forward at yaw is (-sin, 0, -cos).
                delta = vlnce_return.FORWARD_STEP_M * np.array(
                    [-math.sin(yaw), 0.0, -math.cos(yaw)]
                )
                desired = state["position"] + delta
                # try_step slides along walls exactly as habitat's own
                # move_forward does; this is the environment, not planning.
                moved = pathfinder.try_step(
                    state["position"].astype(np.float32), desired.astype(np.float32)
                )
                state["position"] = np.asarray(moved, dtype=np.float64)
            return (
                np.asarray(
                    vlnce_return.habitat_to_map_xy(state["position"]), dtype=np.float64
                ),
                vlnce_return.habitat_yaw_to_map(state["yaw"]),
            )

        def geodesic_to_target(_planar_position):
            return geodesic(state["position"], snapped_target)

        goal_yaw = goal.get("goal_yaw")
        if goal_yaw is not None:
            # Stored as a habitat heading; the follower drives in the map frame.
            goal_yaw = vlnce_return.habitat_yaw_to_map(float(goal_yaw))
        follower = vlnce_return.WaypointFollower(
            route, goal_radius=args.goal_radius, goal_yaw=goal_yaw
        )
        habitat_track = [state["position"].copy()]

        def step_fn(action):
            pose = apply(action)
            habitat_track.append(state["position"].copy())
            # Keep the agent's sensors on the pose we are integrating, so a
            # future visual policy sees what this controller decided.
            agent_state = habitat_sim.AgentState()
            agent_state.position = state["position"].astype(np.float32)
            agent_state.rotation = yaw_to_quat(state["yaw"])
            agent.set_state(agent_state)
            return pose

        rollout = vlnce_return.run_rollout(
            follower,
            np.array(vlnce_return.habitat_to_map_xy(start_position)),
            vlnce_return.habitat_yaw_to_map(start_yaw),
            step_fn,
            max_steps=args.max_steps,
            distance_fn=geodesic_to_target,
        )

        metrics = vlnce_return.score_rollout(
            rollout, optimal, success_distance=success_distance,
            target_yaw=goal_yaw,
        )
        metrics["route_found"] = True

        payload = {
            "format": "fact3r-vlnce-return",
            "version": 1,
            "scene": meta.get("scene"),
            "return_query": meta.get("return_query"),
            "goal_yx": goal_yx.tolist(),
            "goal_source": os.path.abspath(args.goal),
            "grid": os.path.abspath(args.grid),
            "route_found": True,
            "goal_yaw": goal.get("goal_yaw"),
            "planned_waypoints": len(route),
            "route_yx": route.tolist(),
            "start_position_habitat": start_position.tolist(),
            "start_yaw": float(start_yaw),
            "target_position_habitat": snapped_target.tolist(),
            "habitat_track": [point.tolist() for point in habitat_track],
            "metrics": metrics,
            "rollout": rollout.to_json(),
            "planner": {
                "robot_radius": args.robot_radius,
                "unknown_slack": args.unknown_slack,
                "exclude_exterior": args.exclude_exterior,
                "horizon": args.horizon,
            },
        }
        output = args.output or os.path.join(render or ".", "return_result.json")
        with open(output, "w") as handle:
            json.dump(payload, handle, indent=2)

        print(
            "return %r: NE %.2f m, success=%s, SPL %.3f over %d steps (%.1f m)"
            % (
                meta.get("return_query"),
                metrics["navigation_error"],
                metrics["success"],
                metrics["spl"],
                metrics["steps"],
                metrics["path_length"],
            )
        )
        if "heading_error_deg" in metrics:
            print(
                "  heading error %+.1f deg, pose success=%s"
                % (metrics["heading_error_deg"], metrics["pose_success"])
            )
        if metrics["budget_exhausted"]:
            print("  note: the agent never chose STOP; the step budget ran out")
        print("wrote %s" % output)
        return 0
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
