#!/usr/bin/env python3
"""
Evaluate GOAT-Bench episodes against a Fact3R map of the same scene.

One map per scene, reused across that scene's episodes -- which is the point of
GOAT: the goals arrive in sequence and the agent is never teleported, so the map
is built once and queried many times.

    python3 scripts/run_goat_eval.py \
        --episodes datasets/goat/.../val_unseen/content/y9hTuugGdiq.json.gz \
        --map logs/goat/y9hTuugGdiq/map \
        --image-goals datasets/goat_image_goals/y9hTuugGdiq \
        --out logs/goat/y9hTuugGdiq/results.json

This is a **pre-explored** variant: GOAT agents explore online, while Fact3R
needs a mapping pass first. Say so when reporting; it is not comparable to the
GOAT leaderboard.

Each subtask runs three stages, each in the environment that owns it:
  resolve  (SAM2)        text or image goal -> ranked entity
  project  (mast3r-slam) entity -> a cell the robot may stand in
  execute  (habitat-vla) plan over the agent's own map and drive there
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "fact3r-map"))

from fact3r.experiments.goat import load_scene  # noqa: E402
from fact3r.experiments.goat_runner import (  # noqa: E402
    carry_forward,
    plan_episode,
    score_request,
    skipped_result,
    summarise,
)


def run(command, label):
    """Run one stage, returning None when it fails rather than aborting the sweep."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        return None, "%s failed: %s" % (label, tail[-1] if tail else "no output")
    return result.stdout, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, help="one scene's .json.gz")
    parser.add_argument("--map", required=True, help="semantic BEV stem for this scene")
    parser.add_argument("--image-goals", help="directory of rendered goal images")
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default=None, help="scratch dir for per-subtask files")
    parser.add_argument("--hm3d-root", default=str(PROJECT_ROOT / "datasets" / "hm3d_root"))
    parser.add_argument("--sam2-env", default="SAM2")
    parser.add_argument("--planner-env", default="mast3r-slam")
    parser.add_argument("--habitat-env", default="habitat-vla")
    parser.add_argument("--device", default="0")
    parser.add_argument("--robot-radius", type=float, default=0.17)   # Stretch
    parser.add_argument("--unknown-slack", type=float, default=0.17)
    parser.add_argument("--cam-height", type=float, default=1.41)     # Stretch
    parser.add_argument("--success-distance", type=float, default=1.0)
    parser.add_argument(
        "--goal-floor-range",
        type=float,
        nargs=2,
        metavar=("MIN_Y", "MAX_Y"),
        help="skip subtasks whose goals sit outside this habitat-y band. The "
             "BEV is planar and the odometry carries no height, so a map built "
             "from a single-storey tour cannot represent goals on another "
             "floor; those subtasks are recorded as skipped rather than "
             "scored against a map that never saw them",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-subtasks", type=int, default=0)
    args = parser.parse_args()

    grid = Path(str(args.map) + ".npy")
    if not grid.is_file():
        print(f"no occupancy grid at {grid}; build the scene map first", file=sys.stderr)
        return 1

    episodes = load_scene(args.episodes)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    work = Path(args.work or (Path(args.out).parent / "work"))
    work.mkdir(parents=True, exist_ok=True)
    image_goals = Path(args.image_goals) if args.image_goals else None

    results = []
    for episode in episodes:
        requests = plan_episode(episode)
        if args.max_subtasks:
            requests = requests[: args.max_subtasks]
        print(f"episode {episode.episode_id}: {len(requests)} subtasks")
        for position, request in enumerate(requests):
            stem = work / f"ep{episode.episode_id}_s{request.index}"
            label = f"  [{request.index}] {request.modality:<11s} {request.category}"

            if args.goal_floor_range and request.goal_positions:
                low, high = args.goal_floor_range
                if not any(low <= float(g[1]) <= high for g in request.goal_positions):
                    print(f"{label} -> skipped (goal on another storey)")
                    results.append(skipped_result(request, "off-storey goal"))
                    continue

            resolve = [
                "conda", "run", "--no-capture-output", "-n", args.sam2_env, "python3",
                str(PROJECT_ROOT / "fact3r-map/scripts/resolve_semantic_goal.py"),
                "--map", str(args.map), "--output", str(stem) + "_request.json",
                "--device", args.device,
            ]
            if request.modality == "image":
                name = request.image_name(episode.scene)
                path = image_goals / name if image_goals and name else None
                if path is None or not path.is_file():
                    print(f"{label} -> skipped (no goal image)")
                    results.append(skipped_result(request, "no goal image rendered"))
                    continue
                resolve += ["--image", str(path)]
            elif request.prompt:
                resolve += ["--query", request.prompt]
            else:
                print(f"{label} -> skipped (no prompt)")
                results.append(skipped_result(request, "no text prompt"))
                continue

            _, error = run(resolve, "resolve")
            if error:
                print(f"{label} -> skipped ({error})")
                results.append(skipped_result(request, error))
                continue

            project = [
                "conda", "run", "--no-capture-output", "-n", args.planner_env, "python3",
                str(PROJECT_ROOT / "fact3r-map/scripts/project_semantic_goal.py"),
                "--request", str(stem) + "_request.json", "--grid", str(grid),
                "--output", str(stem) + "_goal.json",
                "--robot-radius", str(args.robot_radius),
                "--unknown-slack", str(args.unknown_slack),
            ]
            _, error = run(project, "project")
            if error:
                print(f"{label} -> skipped ({error})")
                results.append(skipped_result(request, error))
                continue

            target = request.goal_positions[0]
            execute = [
                "conda", "run", "--no-capture-output", "-n", args.habitat_env, "python3",
                str(PROJECT_ROOT / "scripts/execute_vlnce_return.py"),
                "--scene", episode.scene, "--hm3d-root", args.hm3d_root,
                "--goal", str(stem) + "_goal.json", "--grid", str(grid),
                "--start-position", *[str(v) for v in request.start_position],
                "--target-position", *[str(v) for v in target],
                "--success-distance", str(args.success_distance),
                "--cam-height", str(args.cam_height),
                "--unknown-slack", str(args.unknown_slack),
                "--robot-radius", str(args.robot_radius),
                "--output", str(stem) + "_return.json",
            ]
            _, error = run(execute, "execute")
            if error:
                print(f"{label} -> skipped ({error})")
                results.append(skipped_result(request, error))
                continue

            payload = json.loads((Path(str(stem) + "_return.json")).read_text())
            track = payload.get("habitat_track") or []
            final = np.asarray(track[-1] if track else request.start_position, float)
            result = score_request(
                request, final, success_distance=args.success_distance
            )
            result["route_found"] = payload.get("route_found")
            results.append(result)
            print(
                f"{label} -> {'SUCCESS' if result['success'] else 'fail   '} "
                f"{result['distance_to_goal']:.2f} m"
            )
            # GOAT never teleports: the next goal starts from here.
            carry_forward(requests, position, final)

    summary = summarise(results)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"format": "fact3r-goat-eval", "version": 1,
         "episodes": args.episodes, "map": str(args.map),
         "pre_explored": True, "summary": summary, "results": results},
        indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
