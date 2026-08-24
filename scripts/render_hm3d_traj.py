#!/usr/bin/env python3
"""
Render a smooth, SLAM-friendly camera trajectory through an HM3D scene with
habitat-sim, producing an RGB image folder that MASt3R-SLAM's RGBFiles loader
can consume directly, plus the ground-truth camera trajectory in TUM format.

Unlike preview_hm3d.py (which teleports to random viewpoints), this walks a
continuous geodesic tour over the navmesh: positions are resampled at a fixed
step and heading changes are rate-limited into in-place rotations, so
consecutive frames keep the large overlap MASt3R matching needs.

habitat-sim lives in the `habitat-vla` env, so run this as:

    conda run -n habitat-vla python3 scripts/render_hm3d_traj.py --scene 00800-TEEsavR23oF

Camera is a square 512x512 pinhole at 90 deg HFOV, which makes the intrinsics
exact and aspect-ratio-free: fx = fy = cx = cy = 256, no distortion.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import habitat_sim
except ImportError:
    habitat_sim = None

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HM3D_ROOT = os.path.join(REPO_ROOT, "datasets", "hm3d_root")
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "datasets", "hm3d_seqs")


def find_scenes(hm3d_root, split):
    split_dir = os.path.join(hm3d_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split '{split}' not found at {split_dir}")
    scenes = []
    for entry in sorted(os.listdir(split_dir)):
        scene_dir = os.path.join(split_dir, entry)
        if not os.path.isdir(scene_dir):
            continue
        scene_id = entry.split("-", 1)[-1] if "-" in entry else entry
        glb = os.path.join(scene_dir, f"{scene_id}.basis.glb")
        if os.path.isfile(glb):
            scenes.append({"folder": entry, "id": scene_id, "glb": glb})
    return scenes


def find_dataset_config(hm3d_root):
    for name in (
        "hm3d_annotated_basis.scene_dataset_config.json",
        "hm3d_basis.scene_dataset_config.json",
    ):
        path = os.path.join(hm3d_root, name)
        if os.path.isfile(path):
            return path
    return None


def build_sim(scene_glb, dataset_config, resolution, hfov, cam_height):
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [resolution, resolution]
    rgb_spec.position = [0.0, cam_height, 0.0]
    rgb_spec.hfov = hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    if dataset_config:
        sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.enable_physics = False

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def geodesic_path(pathfinder, start, end):
    """Return the navmesh shortest path points, or None if unreachable."""
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = end
    if not pathfinder.find_path(path):
        return None
    return [np.array(p, dtype=np.float64) for p in path.points]


def largest_island(pathfinder):
    """Index of the biggest connected navmesh component.

    Sampling within one island guarantees every waypoint is reachable from
    every other, which a raw get_random_navigable_point() does not: HM3D
    scenes are multi-storey and their navmeshes are often fragmented.
    """
    n = pathfinder.num_islands
    if n <= 0:
        return -1
    return max(range(n), key=pathfinder.island_area)


def sample_point(pathfinder, island):
    return np.array(
        pathfinder.get_random_navigable_point(island_index=island), dtype=np.float64
    )


def build_polyline(pathfinder, target_length, min_leg, max_leg, island, max_tries=4000):
    """Chain geodesic legs between random navigable points into one polyline.

    min_leg is relaxed when a scene is too small or too fragmented to offer
    legs of the requested length, so small scenes still yield a usable tour
    rather than a single frame. The final leg returns to the start so the
    tour closes a loop.
    """
    start = sample_point(pathfinder, island)
    polyline = [start]
    cur = start
    total = 0.0
    tries = 0
    stale = 0
    while total < target_length and tries < max_tries:
        tries += 1
        cand = sample_point(pathfinder, island)
        pts = geodesic_path(pathfinder, cur, cand)
        if pts is None or len(pts) < 2:
            stale += 1
        else:
            leg = float(np.sum(np.linalg.norm(np.diff(np.stack(pts), axis=0), axis=1)))
            if min_leg <= leg <= max_leg:
                polyline.extend(pts[1:])
                total += leg
                cur = cand
                stale = 0
                continue
            stale += 1
        # Nothing is being accepted: the scene cannot offer legs this long.
        if stale >= 100 and min_leg > 0.25:
            min_leg = max(0.25, min_leg / 2.0)
            stale = 0

    # Close the loop so global optimisation has a real loop closure to find.
    pts = geodesic_path(pathfinder, cur, start)
    if pts is not None and len(pts) >= 2:
        polyline.extend(pts[1:])

    return np.stack(polyline), total


def resample(polyline, step):
    """Resample a polyline at a uniform arc-length step."""
    out = [polyline[0]]
    carry = 0.0
    for a, b in zip(polyline[:-1], polyline[1:]):
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            continue
        d = seg / seg_len
        t = step - carry
        while t <= seg_len:
            out.append(a + d * t)
            t += step
        carry = (carry + seg_len) % step
    return np.stack(out)


def yaw_from_direction(d):
    """Habitat yaw about +y such that the camera's -z forward axis follows d."""
    return math.atan2(-d[0], -d[2])


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def build_poses(positions, max_turn_rad, cam_height_is_sensor=True):
    """Turn a position sequence into (position, yaw) frames.

    When the heading changes faster than max_turn_rad per frame, the camera
    stops and rotates in place over several frames instead of snapping, which
    keeps frame-to-frame overlap high enough for matching.
    """
    frames = []
    yaw = None
    for i, p in enumerate(positions):
        if i + 1 < len(positions):
            d = positions[i + 1] - p
        else:
            d = positions[i] - positions[i - 1] if i > 0 else np.array([0.0, 0.0, -1.0])
        if np.linalg.norm(d) < 1e-9:
            d = np.array([0.0, 0.0, -1.0])
        target = yaw_from_direction(d)

        if yaw is None:
            yaw = target
            frames.append((p.copy(), yaw))
            continue

        delta = wrap(target - yaw)
        # Rotate in place until the heading is within one step of the target.
        while abs(delta) > max_turn_rad:
            yaw = wrap(yaw + math.copysign(max_turn_rad, delta))
            frames.append((p.copy(), yaw))
            delta = wrap(target - yaw)
        yaw = target
        frames.append((p.copy(), yaw))
    return frames


def yaw_to_quat(yaw):
    """habitat quaternion (x, y, z, w) for a rotation of `yaw` about +y."""
    return np.array([0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)])


def dark_fraction(sim, agent, frames, probes=30):
    """Fraction of probe views that are mostly empty background.

    HM3D meshes are reconstructions with holes; a trajectory that walks past
    one renders near-black frames with no texture to match, which is enough to
    lose tracking. Probing the tour before committing to it lets us resample.
    """
    if not frames:
        return 1.0
    idxs = np.linspace(0, len(frames) - 1, min(probes, len(frames))).astype(int)
    dark = 0
    for k in idxs:
        pos, yaw = frames[k]
        state = habitat_sim.AgentState()
        state.position = pos.astype(np.float32)
        state.rotation = yaw_to_quat(yaw)
        agent.set_state(state)
        rgb = sim.get_sensor_observations()["color_sensor"][:, :, :3]
        lum = rgb.astype(np.float32).mean(axis=2)
        if (lum < 10).mean() > 0.6:
            dark += 1
    return dark / len(idxs)


def render_scene(scene, dataset_config, out_dir, args):
    sim = build_sim(scene["glb"], dataset_config, args.resolution, args.hfov, args.cam_height)
    try:
        if not sim.pathfinder.is_loaded:
            print(f"  SKIP {scene['folder']}: no navmesh loaded")
            return None

        agent = sim.initialize_agent(0)
        # Seed after initialize_agent: it samples a navigable point of its own,
        # which would otherwise advance the pathfinder RNG past our seed.
        sim.pathfinder.seed(args.seed)

        island = largest_island(sim.pathfinder)
        target_length = args.num_frames * args.step

        best = None
        for attempt in range(args.max_attempts):
            sim.pathfinder.seed(args.seed + attempt)
            polyline, tour_len = build_polyline(
                sim.pathfinder, target_length, args.min_leg, args.max_leg, island
            )
            positions = resample(polyline, args.step)
            frames = build_poses(positions, math.radians(args.max_turn_deg))
            if args.max_frames and len(frames) > args.max_frames:
                frames = frames[: args.max_frames]

            dark = dark_fraction(sim, agent, frames)
            if best is None or dark < best[0]:
                best = (dark, frames, tour_len)
            if dark <= args.max_dark:
                break
            print(f"    attempt {attempt}: {100 * dark:.0f}% views into mesh holes, resampling")

        dark, frames, tour_len = best
        if dark > args.max_dark:
            print(f"    WARNING: best tour still {100 * dark:.0f}% empty views")

        # Clear old frames first: a shorter re-render would otherwise leave
        # stale PNGs from the previous trajectory at the tail of the sequence.
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

            # Ground truth is the camera (sensor) centre: agent position plus
            # the sensor's local y offset. Rotation is yaw about +y.
            cam = pos + np.array([0.0, args.cam_height, 0.0])
            qx, qy, qz, qw = yaw_to_quat(yaw)
            ts = i / args.fps
            gt_lines.append(
                f"{ts:.6f} {cam[0]:.6f} {cam[1]:.6f} {cam[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
            )
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(frames)} frames", flush=True)

        with open(os.path.join(out_dir, "groundtruth.txt"), "w") as f:
            f.write("# habitat camera poses: timestamp tx ty tz qx qy qz qw\n")
            f.write("# frame: habitat world (y-up); camera looks along local -z\n")
            f.write("\n".join(gt_lines) + "\n")

        focal = (args.resolution / 2.0) / math.tan(math.radians(args.hfov) / 2.0)
        meta = {
            "scene": scene["folder"],
            "scene_glb": scene["glb"],
            "num_frames": len(frames),
            "fps": args.fps,
            "step_m": args.step,
            "max_turn_deg": args.max_turn_deg,
            "cam_height_m": args.cam_height,
            "tour_length_m": tour_len,
            "resolution": [args.resolution, args.resolution],
            "hfov_deg": args.hfov,
            "intrinsics": {
                "fx": focal,
                "fy": focal,
                "cx": args.resolution / 2.0,
                "cy": args.resolution / 2.0,
            },
            "seed": args.seed,
            "dark_fraction": dark,
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(
            f"  {scene['folder']}: {len(frames)} frames, tour {tour_len:.1f} m -> {out_dir}"
        )
        return meta
    finally:
        sim.close()


def main():
    p = argparse.ArgumentParser(description="Render SLAM sequences from HM3D scenes.")
    p.add_argument("--hm3d-root", default=DEFAULT_HM3D_ROOT)
    p.add_argument("--split", default="minival")
    p.add_argument("--scene", default=None, help="Scene folder/id; default renders all in split.")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--num-frames", type=int, default=700, help="Target frame count (before turn padding).")
    p.add_argument("--max-frames", type=int, default=1200, help="Hard cap after turn padding (0 = no cap).")
    p.add_argument("--step", type=float, default=0.04, help="Metres travelled per frame.")
    p.add_argument("--max-turn-deg", type=float, default=2.0, help="Max heading change per frame.")
    p.add_argument("--min-leg", type=float, default=2.0)
    p.add_argument("--max-leg", type=float, default=20.0)
    p.add_argument("--cam-height", type=float, default=1.5)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--hfov", type=float, default=90.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=6, help="Tour resamples allowed to avoid mesh holes.")
    p.add_argument("--max-dark", type=float, default=0.05, help="Max fraction of probe views into empty space.")
    p.add_argument("--list-scenes", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    scenes = find_scenes(args.hm3d_root, args.split)
    if args.list_scenes:
        for s in scenes:
            print(s["folder"])
        return
    if habitat_sim is None:
        sys.exit("habitat_sim not importable; run with: conda run -n habitat-vla python3 ...")
    if not scenes:
        sys.exit(f"No scenes found in split '{args.split}'")
    if args.scene:
        scenes = [s for s in scenes if args.scene in (s["id"], s["folder"])]
        if not scenes:
            sys.exit(f"Scene '{args.scene}' not found in split '{args.split}'")

    dataset_config = find_dataset_config(args.hm3d_root)
    print(f"Dataset config: {dataset_config}")
    print(f"Rendering {len(scenes)} scene(s) from split '{args.split}'")

    for s in scenes:
        out_dir = os.path.join(args.out_root, s["folder"])
        if os.path.isdir(out_dir) and not args.overwrite:
            n = len([f for f in os.listdir(out_dir) if f.endswith(".png")])
            if n > 0:
                print(f"  SKIP {s['folder']}: {n} pngs already present (use --overwrite)")
                continue
        print(f"[{s['folder']}]", flush=True)
        try:
            render_scene(s, dataset_config, out_dir, args)
        except Exception as e:  # keep the batch going if one scene fails
            print(f"  FAILED {s['folder']}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
