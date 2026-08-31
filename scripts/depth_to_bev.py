#!/usr/bin/env python3
"""Depth-Anything-V2 over a rover video + wheel odometry -> BEV occupancy grid.

    python scripts/depth_to_bev.py --root /home/nahar4/Gazania/MPL \
           --out logs/rover/mpl_da2 --fx 631 --pitch 2.75 --scale 0.969

Writes <out>.ply and <out>.txt, then hand those to scripts/occupancy_grid.py.

What makes the depth "proper"
-----------------------------
The METRIC checkpoint is used, not the relative one, and that choice is
measured rather than assumed -- see scripts/depth_fit_eval.py, which scores
both against the RealSense frames that survived the capture. In the band this
map is built from (0.3-4 m) the two are statistically tied on median error,
50/84/157 mm against 46/84/140 mm. What separates them is generalisation:

  relative   needs a per-frame affine fit (a, b) in disparity space. Across
             the anchored frames a is stable to 21% but b scatters by 178%,
             so there is no frozen pair that works on the 74% of the run with
             no sensor depth to fit against.
  metric     needs one global scale. Measured at 0.969 with a 14% per-frame
             IQR, and a scale error is uniform -- it resizes the map without
             bending it, which an occupancy grid tolerates far better.

So the relative head is not worse here; it is unanchorable. Where sensor depth
exists, fitting it per frame is the more accurate route.

Depth past --depth-max is discarded. The error grows fast beyond the band we
validated, and a wrong far reading does not blur the map, it plants a wall
across open floor.
"""

import argparse
import csv
import pathlib

import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

METRIC_CKPT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/nahar4/Gazania/MPL")
    p.add_argument("--out", required=True, help="output stem, no extension")
    p.add_argument("--fx", type=float, default=631.0)
    p.add_argument("--pitch", type=float, default=2.75,
                   help="camera pitch below horizontal, degrees")
    p.add_argument("--cam-height", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=0.969,
                   help="global correction on the metric head's output")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--time-offset", type=float, default=0.0)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--pixel-stride", type=int, default=3)
    p.add_argument("--depth-min", type=float, default=0.3)
    p.add_argument("--depth-max", type=float, default=4.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--voxel", type=float, default=0.02)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--stationary-skip", type=float, default=0.0,
                   help="drop frames whose |v| is below this (m/s); 0 keeps all")
    return p.parse_args()


def find_inputs(root):
    root = pathlib.Path(root)
    mp4 = sorted(root.glob("*.mp4"))
    odom = sorted(root.glob("odom_*.csv"))
    if not (mp4 and odom):
        raise SystemExit(f"need an mp4 and an odom csv under {root}")
    return str(mp4[0]), str(odom[0])


def load_odom(path):
    t, x, y, th, v = [], [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            t.append(float(r["t"])); x.append(float(r["x"]))
            y.append(float(r["y"])); th.append(float(r["theta"]))
            v.append(float(r["v"]))
    t = np.array(t); t -= t[0]
    return (t, np.array(x), np.array(y), np.unwrap(np.array(th)),
            np.array(v))


def cam_to_body(pitch_deg, cam_height):
    """Camera axes in the body frame (x fwd, y left, z up), and its origin.

    Pitching the camera down by p rotates about the body's +y (left) axis:
    forward tips toward the floor and the camera's own down axis tips back.
    """
    p = np.radians(pitch_deg)
    right = np.array([0.0, -1.0, 0.0])
    down = np.array([-np.sin(p), 0.0, -np.cos(p)])
    fwd = np.array([np.cos(p), 0.0, -np.sin(p)])
    return np.stack([right, down, fwd], axis=1), np.array([0.0, 0.0, cam_height])


def voxel_downsample(pts, extra, voxel):
    if voxel <= 0:
        return pts, extra
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    return pts[idx], extra[idx]


@torch.inference_mode()
def predict(proc, model, frames_bgr, hw):
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    inp = proc(images=rgb, return_tensors="pt").to("cuda", torch.float16)
    res = proc.post_process_depth_estimation(
        model(**inp), target_sizes=[hw] * len(rgb))
    return [r["predicted_depth"].float().cpu().numpy() for r in res]


def main():
    args = parse_args()
    mp4, odom_path = find_inputs(args.root)
    ot, ox, oy, oth, ov = load_odom(odom_path)
    print(f"video {mp4}\nodom  {odom_path} ({ot[-1]:.1f} s)")

    proc = AutoImageProcessor.from_pretrained(METRIC_CKPT)
    model = AutoModelForDepthEstimation.from_pretrained(METRIC_CKPT).to(
        "cuda", torch.float16).eval()
    print(f"loaded {METRIC_CKPT}")
    print(f"fx={args.fx}  pitch={args.pitch} deg  "
          f"cam_height={args.cam_height} m  depth scale x{args.scale}")

    R_bc, t_bc = cam_to_body(args.pitch, args.cam_height)

    cap = cv2.VideoCapture(mp4)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cx, cy = W / 2.0, H / 2.0
    ps = args.pixel_stride
    vv, uu = np.mgrid[0:H:ps, 0:W:ps]
    print(f"{W}x{H}, {total} frames, stride {args.frame_stride}\n")

    chunks, kfs, poses = [], [], []
    pend_f, pend_t = [], []
    i = kept = skipped = 0
    import time
    t0 = time.time()

    def flush():
        nonlocal pend_f, pend_t
        if not pend_f:
            return
        for d, t in zip(predict(proc, model, pend_f, (H, W)), pend_t):
            z = d[::ps, ::ps] * args.scale
            m = np.isfinite(z) & (z > args.depth_min) & (z < args.depth_max)
            if m.sum() < 100:
                continue
            px, py = np.interp(t, ot, ox), np.interp(t, ot, oy)
            th = np.interp(t, ot, oth)
            zg = z[m]
            pc = np.stack([(uu[m] - cx) * zg / args.fx,
                           (vv[m] - cy) * zg / args.fx, zg], axis=1)
            pb = pc @ R_bc.T + t_bc
            c, s = np.cos(th), np.sin(th)
            wx = px + c * pb[:, 0] - s * pb[:, 1]
            wy = py + s * pb[:, 0] + c * pb[:, 1]
            # world is z-up; occupancy_grid.py wants y-down so its gravity
            # prior on +y locks onto the floor.
            chunks.append(np.stack([wx, -pb[:, 2], wy], axis=1).astype(np.float32))
            kfs.append(np.full(int(m.sum()), len(poses), dtype=np.int32))
            camx = px + c * t_bc[0] - s * t_bc[1]
            camy = py + s * t_bc[0] + c * t_bc[1]
            poses.append((t, camx, -t_bc[2], camy, th))
        pend_f, pend_t = [], []

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % args.frame_stride == 0:
            t = i / args.fps + args.time_offset
            if ot[0] <= t <= ot[-1]:
                if args.stationary_skip > 0 and \
                        abs(np.interp(t, ot, ov)) < args.stationary_skip:
                    skipped += 1
                else:
                    pend_f.append(bgr)
                    pend_t.append(t)
                    kept += 1
                    if len(pend_f) >= args.batch:
                        flush()
                        el = time.time() - t0
                        print(f"\r  frame {i}/{total}  {len(poses)} kept  "
                              f"{sum(len(c) for c in chunks):,} pts  "
                              f"{kept/max(el,1e-6):.1f} fps", end="", flush=True)
        i += 1
        if args.max_frames and kept >= args.max_frames:
            break
    flush()
    cap.release()
    del model
    torch.cuda.empty_cache()

    if not chunks:
        raise SystemExit("no frames produced points -- check --time-offset")

    pts = np.concatenate(chunks)
    kf = np.concatenate(kfs)
    print(f"\n\n{len(poses)} frames -> {len(pts):,} points"
          + (f"  ({skipped} stationary frames skipped)" if skipped else ""))
    pts, kf = voxel_downsample(pts, kf, args.voxel)
    print(f"after {args.voxel} m voxel downsample: {len(pts):,} points")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    v = np.empty(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("kf_id", "i4")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    v["kf_id"] = kf
    PlyData([PlyElement.describe(v, "vertex")]).write(str(out) + ".ply")
    with open(str(out) + ".txt", "w") as fh:
        for t, x, y, z, th in poses:
            qy, qw = np.sin(-th / 2), np.cos(-th / 2)
            fh.write(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} 0 {qy:.6f} 0 {qw:.6f}\n")

    ex = pts.max(axis=0) - pts.min(axis=0)
    print(f"\nwrote {out}.ply  (extent {ex[0]:.1f} x {ex[1]:.1f} x {ex[2]:.1f} m)")
    print(f"wrote {out}.txt  ({len(poses)} poses)")


if __name__ == "__main__":
    main()
