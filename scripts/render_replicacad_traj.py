#!/usr/bin/env python3
"""
Render a SLAM-friendly camera tour through a *cluttered* ReplicaCAD interior.

Why this exists (and not just render_hm3d_traj.py)
--------------------------------------------------
HM3D scenes are photogrammetric reconstructions of empty real-estate scans.
Two consequences bit us:

  * MASt3R-SLAM recovers metric scale correctly on real imagery (TUM fr1_room:
    0.984) but not on HM3D renders (median 0.780, i.e. maps ~1.28x oversized).
    Whether that is caused by the reconstructed meshes or by something else is
    the open question -- rendering *clean CAD geometry* isolates it.
  * The scans are bare. An occupancy grid exists to capture obstacles, so a
    scene with no furniture barely tests the gridding stage at all.

This script therefore takes the opposite approach to render_hm3d_traj.py: it
loads a ReplicaCAD *stage* (the textured room shell -- walls, floor, stairs,
with no articulated furniture, which habitat cannot instance without Bullet)
and scatters STATIC rigid objects from the object catalog into it.

The navmesh is recomputed *after* placement with include_static_objects=True,
so the ground-truth free space reflects the clutter rather than the empty
shell. That is what makes the resulting grids scoreable.

Requires the Bullet-enabled build:

    conda run -n habitat-vla-bullet python3 scripts/render_replicacad_traj.py --stage frl_apartment_stage

Camera defaults to the same square 512x512 / 90 deg pinhole as the HM3D
pipeline, so scale and ATE are directly comparable against logs/hm3d/calib.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import habitat_sim
    import magnum as mn
except ImportError:  # pragma: no cover - import guard mirrors render_hm3d_traj
    habitat_sim = None
    mn = None

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

# The tour walker is identical to HM3D's -- geodesic legs, fixed-step resample,
# rate-limited heading changes -- so import it rather than forking it.
from render_hm3d_traj import (  # noqa: E402
    build_polyline,
    build_poses,
    largest_island,
    resample,
    yaw_to_quat,
)

DEFAULT_RC_ROOT = (
    "/media/nahar4/2e670039-b303-4e14-b517-49b4319b069d/habitat_dataget/data/replica_cad"
)
DEFAULT_CATALOG = (
    "/media/nahar4/2e670039-b303-4e14-b517-49b4319b069d/habitat_dataget/data/object_catalog.json"
)
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "datasets", "rc_seqs")

# Stage shells that load without Bullet-only articulated instances.
STAGES = [
    "frl_apartment_stage",
    "Stage_v3_sc0_staging",
    "Stage_v3_sc1_staging",
    "Stage_v3_sc2_staging",
    "Stage_v3_sc3_staging",
]


def build_sim(dataset_config, stage_id, resolution, hfov, cam_height):
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [resolution, resolution]
    rgb_spec.position = [0.0, cam_height, 0.0]
    rgb_spec.hfov = hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.scene_id = stage_id
    # Physics is required to instance rigid objects at all, even though every
    # object here is STATIC and never stepped.
    sim_cfg.enable_physics = True

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def recompute_navmesh(sim, agent_radius, agent_height, include_objects=True):
    nms = habitat_sim.NavMeshSettings()
    nms.set_defaults()
    nms.include_static_objects = include_objects
    nms.agent_radius = agent_radius
    nms.agent_height = agent_height
    if not sim.recompute_navmesh(sim.pathfinder, nms):
        raise RuntimeError("navmesh recomputation failed")


def load_catalog(path, min_extent, max_extent):
    """Catalog entries usable as clutter, split into big furniture and small props.

    Size drives *where* an object may go: a wardrobe dropped in a doorway can
    sever the navmesh, while a mug never will. Keeping the two pools separate
    lets placement spend its clearance budget only where it matters.
    """
    with open(path) as f:
        cat = json.load(f)
    big, small = [], []
    for o in cat["objects"]:
        e = float(o["max_extent"])
        if e < min_extent or e > max_extent:
            continue
        (big if e >= 0.35 else small).append(o)
    return big, small


def _floor_y(sim, x, z, fallback):
    """Height of the navmesh directly under (x, z), or fallback if off-mesh."""
    p = np.array([x, fallback, z], dtype=np.float32)
    snapped = sim.pathfinder.snap_point(p)
    if any(math.isnan(v) for v in snapped):
        return None
    return float(snapped[1])


def scatter_clutter(sim, rng, big, small, n_big, n_small, min_sep, margin, island):
    """Place STATIC objects on the floor, resting on their own bounding box.

    Objects are rejected rather than nudged when they land too close to an
    existing one: nudging biases everything toward the arena centre, which is
    exactly where the camera tour wants free space.
    """
    otm = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    placed = []
    plan = [(o, "big") for o in rng.choice(big, size=min(n_big, len(big)), replace=False)] + [
        (o, "small") for o in rng.choice(small, size=min(n_small, len(small)), replace=False)
    ]

    for entry, kind in plan:
        handles = otm.load_configs(entry["config_path"])
        if not handles:
            continue
        tmpl = otm.get_template_handles(entry["stem"])
        if not tmpl:
            continue

        radius = 0.5 * float(entry["max_extent"])
        for _ in range(40):
            pt = sim.pathfinder.get_random_navigable_point(island_index=island)
            if any(math.isnan(v) for v in pt):
                continue
            x, z = float(pt[0]), float(pt[2])
            if any(
                (x - px) ** 2 + (z - pz) ** 2 < (radius + pr + min_sep) ** 2
                for px, pz, pr in placed
            ):
                continue
            if sim.pathfinder.distance_to_closest_obstacle(pt) < radius + margin:
                continue

            obj = rom.add_object_by_template_handle(tmpl[0])
            if obj is None:
                break
            bb = obj.root_scene_node.cumulative_bb
            obj.translation = mn.Vector3(x, float(pt[1]) - bb.min[1], z)
            obj.rotation = mn.Quaternion.rotation(
                mn.Deg(float(rng.uniform(0, 360))), mn.Vector3(0.0, 1.0, 0.0)
            )
            obj.motion_type = habitat_sim.physics.MotionType.STATIC
            placed.append((x, z, radius))
            break

    return placed


def export_navmesh_gt(sim, out_path, res):
    """Top-down boolean navigability raster -- the free-space ground truth.

    Written in the same habitat world frame as groundtruth.txt so the occupancy
    evaluation can compare without any extra alignment step.
    """
    lo, hi = sim.pathfinder.get_bounds()
    xs = np.arange(lo[0], hi[0] + res, res)
    zs = np.arange(lo[2], hi[2] + res, res)
    grid = np.zeros((len(zs), len(xs)), dtype=np.uint8)
    ys = np.linspace(lo[1] + 0.05, hi[1] - 0.05, 12)
    for j, z in enumerate(zs):
        for i, x in enumerate(xs):
            for y in ys:
                if sim.pathfinder.is_navigable(np.array([x, y, z], dtype=np.float32)):
                    grid[j, i] = 1
                    break
    np.savez_compressed(
        out_path,
        nav=grid,
        origin_xz=np.array([lo[0], lo[2]], dtype=np.float64),
        res=res,
    )
    return grid


def render_scene(stage_id, dataset_config, out_dir, args, rng):
    sim = build_sim(dataset_config, stage_id, args.resolution, args.hfov, args.cam_height)
    try:
        agent = sim.initialize_agent(0)
        sim.pathfinder.seed(args.seed)

        # Empty-shell navmesh first: clutter is placed on *floor* the agent can
        # actually stand on, then the navmesh is rebuilt to account for it.
        recompute_navmesh(sim, args.agent_radius, args.agent_height, include_objects=False)
        area_empty = sim.pathfinder.navigable_area
        island = largest_island(sim.pathfinder)

        big, small = load_catalog(args.catalog, args.min_obj_extent, args.max_obj_extent)
        placed = scatter_clutter(
            sim, rng, big, small, args.num_big, args.num_small,
            args.min_separation, args.clutter_margin, island,
        )

        recompute_navmesh(sim, args.agent_radius, args.agent_height, include_objects=True)
        area_clutter = sim.pathfinder.navigable_area
        print(
            f"  {stage_id}: {len(placed)} objects | navigable "
            f"{area_empty:.1f} -> {area_clutter:.1f} m2"
        )
        if area_clutter < args.min_navigable:
            print(f"    SKIP: only {area_clutter:.1f} m2 navigable after clutter")
            return None

        island = largest_island(sim.pathfinder)
        target_length = args.num_frames * args.step
        polyline, tour_len = build_polyline(
            sim.pathfinder, target_length, args.min_leg, args.max_leg, island
        )
        positions = resample(polyline, args.step)
        frames = build_poses(positions, math.radians(args.max_turn_deg))
        if args.max_frames and len(frames) > args.max_frames:
            frames = frames[: args.max_frames]

        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith(".png"):
                    os.remove(os.path.join(out_dir, f))
        os.makedirs(out_dir, exist_ok=True)

        gt_lines = []
        for i, (pos, yaw) in enumerate(frames):
            state = habitat_sim.AgentState()
            state.position = pos.astype(np.float32)
            state.rotation = yaw_to_quat(yaw)
            agent.set_state(state)

            obs = sim.get_sensor_observations()
            Image.fromarray(obs["color_sensor"][:, :, :3], mode="RGB").save(
                os.path.join(out_dir, f"{i:06d}.png")
            )

            cam = pos + np.array([0.0, args.cam_height, 0.0])
            qx, qy, qz, qw = yaw_to_quat(yaw)
            gt_lines.append(
                f"{i / args.fps:.6f} {cam[0]:.6f} {cam[1]:.6f} {cam[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
            )
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(frames)} frames", flush=True)

        with open(os.path.join(out_dir, "groundtruth.txt"), "w") as f:
            f.write("# habitat camera poses: timestamp tx ty tz qx qy qz qw\n")
            f.write("# frame: habitat world (y-up); camera looks along local -z\n")
            f.write("\n".join(gt_lines) + "\n")

        nav = export_navmesh_gt(sim, os.path.join(out_dir, "navmesh_gt.npz"), args.gt_res)

        focal = (args.resolution / 2.0) / math.tan(math.radians(args.hfov) / 2.0)
        meta = {
            "scene": stage_id,
            "source": "replica_cad_stage",
            "num_frames": len(frames),
            "fps": args.fps,
            "step_m": args.step,
            "max_turn_deg": args.max_turn_deg,
            "cam_height_m": args.cam_height,
            "tour_length_m": tour_len,
            "resolution": [args.resolution, args.resolution],
            "hfov_deg": args.hfov,
            "intrinsics": {
                "fx": focal, "fy": focal,
                "cx": args.resolution / 2.0, "cy": args.resolution / 2.0,
            },
            "seed": args.seed,
            "num_objects": len(placed),
            "navigable_area_empty_m2": area_empty,
            "navigable_area_clutter_m2": area_clutter,
            "gt_nav_cells": int(nav.sum()),
            "gt_res_m": args.gt_res,
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  {stage_id}: {len(frames)} frames, tour {tour_len:.1f} m -> {out_dir}")
        return meta
    finally:
        sim.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rc-root", default=DEFAULT_RC_ROOT)
    p.add_argument("--catalog", default=DEFAULT_CATALOG)
    p.add_argument("--stage", default=None, help="Stage id; default renders all.")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--num-frames", type=int, default=700)
    p.add_argument("--max-frames", type=int, default=1200)
    p.add_argument("--step", type=float, default=0.04)
    p.add_argument("--max-turn-deg", type=float, default=2.0)
    p.add_argument("--min-leg", type=float, default=1.5)
    p.add_argument("--max-leg", type=float, default=12.0)
    p.add_argument("--cam-height", type=float, default=1.0)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--hfov", type=float, default=90.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-big", type=int, default=14, help="Large furniture items.")
    p.add_argument("--num-small", type=int, default=22, help="Small props.")
    p.add_argument("--min-obj-extent", type=float, default=0.08)
    p.add_argument("--max-obj-extent", type=float, default=1.6)
    p.add_argument("--min-separation", type=float, default=0.25)
    p.add_argument("--clutter-margin", type=float, default=0.12,
                   help="Keep object footprints this far off walls.")
    p.add_argument("--agent-radius", type=float, default=0.20)
    p.add_argument("--agent-height", type=float, default=1.2)
    p.add_argument("--min-navigable", type=float, default=8.0,
                   help="Abort a scene whose clutter leaves less floor than this.")
    p.add_argument("--gt-res", type=float, default=0.05)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if habitat_sim is None:
        sys.exit("habitat_sim not importable -- use the habitat-vla-bullet env")

    dataset_config = os.path.join(args.rc_root, "replicaCAD.scene_dataset_config.json")
    if not os.path.isfile(dataset_config):
        sys.exit(f"scene dataset config not found: {dataset_config}")

    stages = [args.stage] if args.stage else STAGES
    os.makedirs(args.out_root, exist_ok=True)
    for stage_id in stages:
        out_dir = os.path.join(args.out_root, stage_id)
        if os.path.isfile(os.path.join(out_dir, "meta.json")) and not args.overwrite:
            print(f"  {stage_id}: exists, skipping (use --overwrite)")
            continue
        rng = np.random.default_rng(args.seed)
        try:
            render_scene(stage_id, dataset_config, out_dir, args, rng)
        except Exception as exc:  # keep going: one bad stage should not kill the batch
            print(f"  {stage_id}: FAILED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
