#!/usr/bin/env python3
"""Video + wheel odometry -> a collision-free trajectory toward a named object.

    python3 scripts/run_rover_pipeline.py \\
        --run logs/rover/pipeline/mpl_20260826 \\
        --video /home/nahar4/Gazania/MPL/manual_drive_20260826_180408.mp4 \\
        --odom /home/nahar4/Gazania/MPL/odom_home_session_20260826_180408.csv \\
        --time-offset 26.80 --query "a 3D printer"

Every stage already existed as a script; this is the flow between them. It
shells out per stage, the way scripts/run_depth_semantic_bev.sh does, because
the stages do not share an environment and cannot be made to:

    frontend   SAM2 + mast3r-slam   run_fact3r_real_uot.sh, which orchestrates
                                    frames -> SAM2 proposals -> SigLIP index ->
                                    MASt3R 2D matches -> UOT association, and
                                    resumes each of those itself
    fuse       SAM2                 Depth-Anything-V2 metric depth + odometry
                                    -> occupancy grid AND semantic grid, one
                                    depth pass, so one stage
    locate     SAM2                 SigLIP text query -> the winning entity's
                                    footprint
    goal       mast3r-slam          that footprint -> a reachable world (y, x)
    plan       mast3r-slam          SafeDiffuser + DSTT to that goal

SAM2 has transformers 5.12.1 and numpy 2.4.4; mast3r-slam has torch 2.5.1,
scipy 1.17.1 and numpy 1.26.4 pinned. They cannot be merged, and `locate` is
split across the boundary for exactly that reason: the SigLIP encoder is only
in SAM2, and `HM3DMap` -- whose clearance field and connected components decide
whether a goal is reachable -- imports scipy, which SAM2 does not have.

Resume is the default: a stage whose output manifest is newer than all of its
inputs is skipped. `--force` re-runs the selected stages and nothing else.

The one number this runner will not let you leave out is `--time-offset`. The
keyframe timestamps are on the video clock and the odometry is re-based to its
own start; on the 2026-08-26 capture those differ by 26.80 s, and because the
whole video interval still lands inside the odometry interval, getting it wrong
skips no frames, renders a plausible map, and leaves every pose wrong by 2.65 m
and 89 degrees of yaw. The fuse stage refuses to run without it whenever the two
stream durations disagree.
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
# Vendored so the pipeline is one clone; see thirdparty/safediffuser/README.md.
# --planner-root still accepts an external SafeDiffuser_STT checkout.
PLANNER_ROOT = REPO / "thirdparty" / "safediffuser"
DEFAULT_CKPT = (
    "logs/pretrained/maze2d-large-v1/diffusion/H384_T256/state_1920000.pt"
)

STAGES = ["frontend", "fuse", "locate", "goal", "plan"]
# The diagram in the build notes names the five front-end steps separately.
# They are steps of run_fact3r_real_uot.sh, which already resumes each of them
# on its own manifest, so they select that one stage rather than being
# reimplemented here.
STAGE_ALIASES = {
    "frames": "frontend",
    "proposals": "frontend",
    "embed": "frontend",
    "match": "frontend",
    "associate": "frontend",
}


def _slug(value):
    return "-".join(
        part for part in "".join(
            c.lower() if c.isalnum() else " " for c in value
        ).split() if part
    ) or "query"


def _newest(paths):
    """Newest mtime among existing paths, or None if none of them exist."""
    stamps = [p.stat().st_mtime for p in paths if p.exists()]
    return max(stamps) if stamps else None


def _oldest(paths):
    stamps = [p.stat().st_mtime for p in paths if p.exists()]
    return min(stamps) if stamps else None


class Stage:
    """One shelled-out step, with the files that decide whether to re-run it."""

    def __init__(self, name, env, argv, inputs, outputs, cwd=None):
        self.name = name
        self.env = env
        self.argv = [str(a) for a in argv]
        self.inputs = [pathlib.Path(p) for p in inputs]
        self.outputs = [pathlib.Path(p) for p in outputs]
        self.cwd = pathlib.Path(cwd) if cwd else REPO

    def up_to_date(self):
        """Every output exists and none is older than any input.

        Mtime, not a content hash: the artifacts here run to tens of megabytes
        and the inputs are files this pipeline wrote itself, so a rebuild is
        always cheaper to trigger than a hash is to compute.
        """
        produced = _oldest(self.outputs)
        if produced is None or any(not p.exists() for p in self.outputs):
            return False, "output missing"
        consumed = _newest(self.inputs)
        if consumed is not None and consumed > produced:
            return False, "input is newer"
        return True, "up to date"

    def command(self):
        if self.env is None:
            return self.argv
        return ["conda", "run", "--no-capture-output", "-n", self.env] + self.argv


def build_stages(args, run_dir):
    """Resolve every stage's command, environment, inputs, and outputs."""
    frontend_dir = (
        pathlib.Path(args.frontend_dir) if args.frontend_dir else run_dir / "frontend"
    )
    observations = frontend_dir / "siglip_observations" / "manifest.json"
    stem = run_dir / "map"
    slug = _slug(args.query)
    locate_dir = run_dir / "locate"
    request = locate_dir / f"{slug}_goal_request.json"
    goal = locate_dir / f"{slug}_goal.json"
    plans_dir = run_dir / "plans"
    video = pathlib.Path(args.video) if args.video else None
    odom = pathlib.Path(args.odom)

    frontend = Stage(
        "frontend",
        # The shell script picks its own environment per step: SAM2 for
        # segmentation and semantics, and the MASt3R env only for the 2.75 GB
        # matcher checkpoint. So it is invoked directly, not wrapped.
        None,
        [
            REPO / "scripts" / "run_fact3r_real_uot.sh",
            "--video", video or "",
            "--output", frontend_dir,
            "--sample-fps", args.sample_fps,
            "--sam2-env", args.sam2_env,
            "--mast3r-env", args.mast3r_env,
            "--device", args.device,
        ],
        inputs=[video] if video else [],
        outputs=[observations],
    )

    fuse = Stage(
        "fuse",
        args.sam2_env,
        [
            "python3", REPO / "fact3r-map" / "scripts" / "build_depth_semantic_bev.py",
            "--index", observations.parent,
            "--odom", odom,
            "--out", stem,
            "--fx", args.fx,
            "--pitch", args.pitch,
            "--cam-height", args.cam_height,
            "--scale", args.depth_scale,
            "--resolution", args.resolution,
            "--time-offset", args.time_offset,
            "--device", args.device,
        ],
        inputs=[observations, odom],
        outputs=[
            pathlib.Path(f"{stem}_semantic.json"),
            pathlib.Path(f"{stem}.npy"),
            pathlib.Path(f"{stem}_semantic_bev.npz"),
        ],
    )

    locate = Stage(
        "locate",
        args.sam2_env,
        [
            "python3", REPO / "fact3r-map" / "scripts" / "resolve_semantic_goal.py",
            "--map", stem,
            "--query", args.query,
            "--output", request,
            "--top-k", args.top_k,
            "--device", args.device,
        ],
        inputs=[pathlib.Path(f"{stem}_semantic.json"),
                pathlib.Path(f"{stem}_semantic_bev.npz")],
        outputs=[request],
    )

    goal_stage = Stage(
        "goal",
        args.mast3r_env,
        [
            "python3", REPO / "fact3r-map" / "scripts" / "project_semantic_goal.py",
            "--request", request,
            "--grid", pathlib.Path(f"{stem}.npy"),
            "--output", goal,
            "--planner-root", args.planner_root,
            "--robot-radius", args.robot_radius,
            "--unknown-slack", args.unknown_slack,
            "--max-projection", args.max_projection,
        ]
        + ([] if args.exclude_exterior else ["--no-exclude-exterior"])
        + (["--start-from-track"] if args.start_from_track else []),
        inputs=[request, pathlib.Path(f"{stem}.npy")],
        outputs=[goal],
    )

    # The goal is read at build time when it exists, so a resumed run plans to
    # the same point without re-deriving it. When it does not, the placeholder
    # is replaced after the goal stage runs -- see `main`.
    plan = Stage(
        "plan",
        args.mast3r_env,
        [],  # filled by `plan_argv`
        inputs=[goal, pathlib.Path(f"{stem}.npy")],
        outputs=[plans_dir / "map_dstt.json", plans_dir / "map_dstt.npz"],
        cwd=args.planner_root,
    )
    plan.argv = plan_argv(args, stem, goal, plans_dir)

    return {"frontend": frontend, "fuse": fuse, "locate": locate,
            "goal": goal_stage, "plan": plan}, {
        "frontend_dir": frontend_dir, "observations": observations,
        "stem": stem, "request": request, "goal": goal, "plans_dir": plans_dir}


def plan_argv(args, stem, goal_path, plans_dir):
    """The planner command, with endpoints if the goal stage has produced them.

    Endpoints are world (y, x) -- the order HM3DMap.cell_to_world returns, not
    the (x, y) of the grid yaml's origin. Passing them the other way round
    produces a confident plan to the wrong room and no error at all.
    """
    argv = [
        "python3", pathlib.Path(args.planner_root) / "scripts" / "plan_hm3d.py",
        "--grid", pathlib.Path(f"{stem}.npy"),
        "--ckpt", args.ckpt,
        "--n-plans", args.n_plans,
        "--seed", args.seed,
        "--unknown-slack", args.unknown_slack,
        "--robot-radius", args.robot_radius,
        "--radius-margin", args.radius_margin,
        "--min-separation", args.min_separation,
        "--out", plans_dir,
    ]
    if not args.exclude_exterior:
        argv.append("--no-exclude-exterior")
    goal_path = pathlib.Path(goal_path)
    if goal_path.exists():
        resolved = json.loads(goal_path.read_text())
        argv += ["--goal", resolved["goal_yx"][0], resolved["goal_yx"][1]]
        if resolved.get("start_yx"):
            argv += ["--start", resolved["start_yx"][0], resolved["start_yx"][1]]
    return [str(a) for a in argv]


def select(args):
    """The ordered stages this invocation should consider."""
    if args.stage:
        wanted = [STAGE_ALIASES.get(s, s) for s in args.stage]
        unknown = [s for s in wanted if s not in STAGES]
        if unknown:
            raise SystemExit(
                f"unknown stage(s) {unknown}; choose from "
                f"{STAGES + sorted(STAGE_ALIASES)}"
            )
        return [s for s in STAGES if s in set(wanted)]
    first = STAGES.index(STAGE_ALIASES.get(args.from_stage, args.from_stage)) \
        if args.from_stage else 0
    last = STAGES.index(STAGE_ALIASES.get(args.through, args.through)) \
        if args.through else len(STAGES) - 1
    if first > last:
        raise SystemExit(f"--from {args.from_stage} is after --through {args.through}")
    return STAGES[first:last + 1]


def git_sha():
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory; holds every output")
    ap.add_argument("--video", help="source video; needed only by the frontend stage")
    ap.add_argument("--odom", required=True, help="wheel odometry CSV")
    ap.add_argument("--query", default="a 3D printer",
                    help="the object to drive toward")
    ap.add_argument(
        "--time-offset", required=True, type=float,
        help="seconds added to video-clock keyframe timestamps to reach the "
             "odometry clock. Measure it with scripts/find_time_offset.py; it "
             "is required here rather than defaulted because a wrong one is "
             "silent.")
    ap.add_argument("--stage", action="append",
                    help="run only this stage; repeatable")
    ap.add_argument("--from", dest="from_stage", help="first stage to run")
    ap.add_argument("--through", help="last stage to run")
    ap.add_argument("--force", action="store_true",
                    help="re-run the selected stages even if they are current")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--frontend-dir",
                    help="adopt an existing stages 1-5 output instead of "
                         "rebuilding it in the run directory")
    ap.add_argument("--sam2-env", default="SAM2")
    ap.add_argument("--mast3r-env", default="mast3r-slam")
    ap.add_argument("--device", default="0", help="CUDA device index")
    ap.add_argument("--sample-fps", default="2")

    # Capture geometry. Defaults are the measured 2026-08-26 rover values.
    ap.add_argument("--fx", type=float, default=631.0)
    ap.add_argument("--pitch", type=float, default=2.75)
    ap.add_argument("--cam-height", type=float, default=0.50)
    ap.add_argument("--depth-scale", type=float, default=0.969)
    ap.add_argument("--resolution", type=float, default=0.05)

    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-projection", type=float, default=2.0,
                    help="metres a goal may be moved off its entity to reach "
                         "navigable space before the match is rejected")
    ap.add_argument("--start-from-track", action="store_true",
                    help="plan from the rover's last mapped pose rather than a "
                         "sampled start")

    # Planner settings, measured on this rover's grid -- see ROVER_RUNBOOK.md.
    # unknown_slack is the safety knob: above robot_radius it licenses a plan to
    # cross unobserved space while still scoring collision_frac 0.
    ap.add_argument("--unknown-slack", type=float, default=0.20)
    ap.add_argument("--robot-radius", type=float, default=0.20)
    ap.add_argument("--radius-margin", type=float, default=0.05)
    ap.add_argument("--min-separation", type=float, default=3.0)
    ap.add_argument("--n-plans", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--planner-root", default=str(PLANNER_ROOT))
    # ON for this grid. Do not carry --no-exclude-exterior over from the pilot
    # `mpl` grid, where the fill claimed 88% of the map.
    ap.add_argument("--no-exclude-exterior", dest="exclude_exterior",
                    action="store_false")
    ap.set_defaults(exclude_exterior=True)
    args = ap.parse_args()

    if shutil.which("conda") is None:
        raise SystemExit("conda is not on PATH; every stage runs in a named env")
    run_dir = pathlib.Path(args.run).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    stages, paths = build_stages(args, run_dir)
    wanted = select(args)

    print(f"run directory: {run_dir}")
    print(f"stages:        {' -> '.join(wanted)}")
    print(f"query:         {args.query!r}")
    print(f"time offset:   {args.time_offset:+.2f} s\n")

    record = {
        "format": "rover-pipeline-run",
        "version": 1,
        "git_sha": git_sha(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parameters": vars(args),
        "paths": {k: str(v) for k, v in paths.items()},
        "stages": [],
    }
    manifest_path = run_dir / "pipeline.json"
    if manifest_path.exists():
        # Keep what earlier invocations recorded: a run is usually assembled
        # over several passes, and only the stages touched now are rewritten.
        previous = json.loads(manifest_path.read_text())
        record["stages"] = [
            s for s in previous.get("stages", []) if s["stage"] not in set(wanted)
        ]

    for name in wanted:
        stage = stages[name]
        if name == "plan":
            # The goal may have been produced moments ago by this same run.
            stage.argv = plan_argv(args, paths["stem"], paths["goal"],
                                   paths["plans_dir"])
        if name == "frontend" and not args.frontend_dir and not args.video:
            raise SystemExit(
                "[frontend] needs --video to build stages 1-5, or "
                "--frontend-dir to adopt an existing one. To skip it entirely, "
                "select a later stage with --from fuse."
            )
        if name == "frontend" and args.frontend_dir:
            current, why = stage.up_to_date()
            if not current:
                raise SystemExit(
                    f"[frontend] --frontend-dir {args.frontend_dir} does not "
                    f"hold a completed observation index ({why}); expected "
                    f"{paths['observations']}"
                )
            print(f"[frontend] adopting {args.frontend_dir}")
            continue
        current, why = stage.up_to_date()
        if current and not args.force:
            print(f"[{name}] skip: {why}")
            record["stages"].append({"stage": name, "status": "skipped",
                                     "reason": why})
            continue
        printable = " ".join(stage.command())
        if args.dry_run:
            print(f"[{name}] would run ({stage.env or 'no env'}): {printable}")
            continue
        print(f"[{name}] run ({stage.env or 'no env'}): {why if not current else 'forced'}")
        started = time.time()
        result = subprocess.run(stage.command(), cwd=str(stage.cwd))
        elapsed = time.time() - started
        if result.returncode != 0:
            record["stages"].append({"stage": name, "status": "failed",
                                     "seconds": elapsed,
                                     "returncode": result.returncode,
                                     "command": printable})
            manifest_path.write_text(json.dumps(record, indent=2) + "\n")
            raise SystemExit(
                f"\n[{name}] failed with exit code {result.returncode}. "
                f"The command was:\n  {printable}\n"
                f"Run record: {manifest_path}"
            )
        missing = [str(p) for p in stage.outputs if not p.exists()]
        if missing:
            raise SystemExit(
                f"[{name}] exited 0 but did not write {missing}"
            )
        print(f"[{name}] done in {elapsed:.1f}s")
        record["stages"].append({"stage": name, "status": "ran",
                                 "seconds": elapsed, "env": stage.env,
                                 "command": printable,
                                 "outputs": [str(p) for p in stage.outputs]})

    if args.dry_run:
        return
    record["stages"].sort(key=lambda s: STAGES.index(s["stage"]))
    record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if paths["goal"].exists():
        record["goal"] = json.loads(paths["goal"].read_text())
    manifest_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nrun record: {manifest_path}")
    for entry in record["stages"]:
        seconds = entry.get("seconds")
        print(f"  {entry['stage']:10s} {entry['status']:8s} "
              f"{'' if seconds is None else f'{seconds:8.1f}s'}")


if __name__ == "__main__":
    sys.exit(main())
