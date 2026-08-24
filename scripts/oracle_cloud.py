#!/usr/bin/env python3
"""Render a PERFECT point cloud for an HM3D sequence, to isolate blame.

    conda run -n habitat-vla python3 scripts/oracle_cloud.py --run hm3d/oracle

The occupancy grid can be wrong for two quite different reasons: MASt3R-SLAM
reconstructed the room badly, or our gridding code turns a good reconstruction
into a bad map. Arguing about which is unproductive when it can simply be
measured.

This renders habitat's own depth along the exact ground-truth trajectory and
back-projects it, giving a cloud with zero reconstruction error: no drift, no
scale error, no pointmap noise, same camera, same viewpoints, same density.
Pushing that through the SAME gridding code isolates the two stages.

    perfect cloud + our gridding   -> gridding error alone
    MASt3R cloud  + our gridding   -> total error
    difference                     -> what the reconstruction costs us

Output mimics a SLAM run (logs/<run>/<scene>.ply and .txt) so the rest of the
pipeline consumes it unchanged.

Poses are written in the SLAM convention, not habitat's. Habitat's camera looks
along -z with +y UP; MASt3R's looks along +z with +y DOWN. Everything
downstream -- metric_scale.camera_up in particular -- assumes the latter, and
would put gravity upside down given raw habitat poses.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_hm3d_traj import find_dataset_config, find_scenes  # noqa: E402

try:
    import habitat_sim
except ImportError:
    sys.exit("needs habitat: conda run -n habitat-vla python3 ...")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# habitat camera -> SLAM camera: flip y and z.
CV_FROM_HAB = np.diag([1.0, -1.0, -1.0])

# Field layout matching mast3r_slam.evaluate.save_ply, so the oracle clouds are
# indistinguishable from real ones downstream.
PLY_DTYPE = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1"),
             ("conf", "<f4"), ("kf_id", "<u2")]
PLY_PROPS = [("float", "x"), ("float", "y"), ("float", "z"),
             ("uchar", "red"), ("uchar", "green"), ("uchar", "blue"),
             ("float", "conf"), ("ushort", "kf_id")]


def write_ply(path, arr):
    """Minimal binary little-endian PLY writer.

    Hand-rolled rather than using plyfile, which is not installed in the
    habitat env -- and adding it risks pulling a numpy that breaks habitat-sim,
    for a format that is a text header followed by a raw struct dump.
    """
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(arr)}\n".encode())
        for typ, name in PLY_PROPS:
            f.write(f"property {typ} {name}\n".encode())
        f.write(b"end_header\n")
        f.write(arr.tobytes())


def build_depth_sim(scene_glb, dataset_config, resolution, hfov, cam_height):
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "depth"
    spec.sensor_type = habitat_sim.SensorType.DEPTH
    spec.resolution = [resolution, resolution]
    spec.position = [0.0, cam_height, 0.0]
    spec.hfov = hfov
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [spec]
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    if dataset_config:
        sim_cfg.scene_dataset_config_file = dataset_config
    sim_cfg.enable_physics = False
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def read_gt(path):
    rows = np.loadtxt(path)
    return rows[:, 0], rows[:, 1:4], rows[:, 4:8]


def backproject(depth, fx, fy, cx, cy, stride):
    """Depth image -> points in the habitat camera frame (-z forward, +y up)."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[::stride, ::stride]
    ok = d > 1e-6
    u, v, d = uu[ok], vv[ok], d[ok]
    # v grows downward in the image while the camera's +y points up, hence -.
    return np.stack([(u - cx) * d / fx, -(v - cy) * d / fy, -d], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="hm3d/oracle", help="logs/<run> to write")
    p.add_argument("--seqs", default=os.path.join(REPO_ROOT, "datasets", "hm3d_seqs"))
    p.add_argument("--hm3d-root", default=os.path.join(REPO_ROOT, "datasets", "hm3d_root"))
    p.add_argument("--split", default="minival")
    p.add_argument("--scene", default=None)
    p.add_argument("--keyframes", type=int, default=60,
                   help="viewpoints to render, spread over the tour; matched to "
                        "the SLAM runs' keyframe count so density is comparable")
    p.add_argument("--pixel-stride", type=int, default=2,
                   help="subsample pixels when back-projecting, to keep the "
                        "cloud the same order of size as MASt3R's")
    args = p.parse_args()

    out_dir = os.path.join(REPO_ROOT, "logs", args.run)
    os.makedirs(out_dir, exist_ok=True)
    scenes = {s["folder"]: s for s in find_scenes(args.hm3d_root, args.split)}
    dataset_config = find_dataset_config(args.hm3d_root)

    seqs = sorted(os.listdir(args.seqs))
    if args.scene:
        seqs = [s for s in seqs if s == args.scene]

    for folder in seqs:
        seq_dir = os.path.join(args.seqs, folder)
        gt_path = os.path.join(seq_dir, "groundtruth.txt")
        meta_path = os.path.join(seq_dir, "meta.json")
        if not (os.path.isfile(gt_path) and os.path.isfile(meta_path)):
            continue
        if folder not in scenes:
            print(f"{folder}: not in split, skipping")
            continue
        meta = json.load(open(meta_path))
        K = meta["intrinsics"]
        res = meta["resolution"][0]
        cam_h = meta["cam_height_m"]

        ts, cams, quats = read_gt(gt_path)
        idx = np.unique(np.linspace(0, len(ts) - 1, args.keyframes).astype(int))
        print(f"[{folder}] {len(idx)} viewpoints of {len(ts)} frames")

        sim = build_depth_sim(scenes[folder]["glb"], dataset_config, res,
                              meta["hfov_deg"], cam_h)
        pts_all, kf_all = [], []
        try:
            agent = sim.initialize_agent(0)
            for n, i in enumerate(idx):
                st = habitat_sim.AgentState()
                # groundtruth.txt records the SENSOR centre; the agent sits
                # cam_height below it, and the sensor offset is re-applied by
                # habitat itself.
                st.position = (cams[i] - np.array([0.0, cam_h, 0.0])).astype(np.float32)
                st.rotation = quats[i].astype(np.float32)
                agent.set_state(st)
                depth = np.asarray(sim.get_sensor_observations()["depth"])
                p_cam = backproject(depth, K["fx"], K["fy"], K["cx"], K["cy"],
                                    args.pixel_stride)
                R = quat_to_R(quats[i])
                pts_all.append((R @ p_cam.T).T + cams[i])
                kf_all.append(np.full(len(p_cam), n, dtype=np.uint16))
        finally:
            sim.close()

        pts = np.concatenate(pts_all).astype(np.float32)
        kf = np.concatenate(kf_all)
        col = np.full((len(pts), 3), 200, dtype=np.uint8)
        arr = np.empty(len(pts), dtype=PLY_DTYPE)
        arr["x"], arr["y"], arr["z"] = pts.T
        arr["red"], arr["green"], arr["blue"] = col.T
        arr["conf"] = 100.0          # a perfect cloud is maximally confident
        arr["kf_id"] = kf
        write_ply(os.path.join(out_dir, f"{folder}.ply"), arr)

        # Trajectory in the SLAM convention, so downstream code sees what it
        # expects rather than habitat's y-up camera.
        with open(os.path.join(out_dir, f"{folder}.txt"), "w") as f:
            for n, i in enumerate(idx):
                q = R_to_quat(quat_to_R(quats[i]) @ CV_FROM_HAB)
                c = cams[i]
                f.write(f"{ts[i]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} "
                        f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        print(f"  {len(pts):,} points -> {out_dir}/{folder}.ply")


if __name__ == "__main__":
    main()
