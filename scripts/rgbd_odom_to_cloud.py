#!/usr/bin/env python3
"""Fuse RealSense depth with wheel odometry into a cloud occupancy_grid.py can eat.

    python scripts/rgbd_odom_to_cloud.py --root /home/nahar4/Gazania/MPL \
           --out logs/rover/mpl_rgbd

Writes <out>.ply and <out>.txt, then you run:

    python scripts/occupancy_grid.py --ply <out>.ply --traj <out>.txt \
           --out logs/rover/grids/mpl_rgbd --cam-height 0.5

Why this path exists
--------------------
This is the no-neural-network baseline. Depth is measured, poses are measured,
so if the resulting map is wrong the fault is in the geometry or the clocks and
NOT in a depth network -- which is exactly the thing you want to have ruled out
before adding one. It only covers the frames whose depth survived the capture.

Frames of reference, since this is where such scripts usually go wrong
---------------------------------------------------------------------
  camera   x right, y down, z forward          (OpenCV / MASt3R convention)
  body     x forward, y left, z up             (rover, ROS convention)
  odom     x, y on the ground plane, theta CCW (from the wheel encoders + IMU)
  output   x right, y DOWN, z forward          (what occupancy_grid.py assumes,
                                                so its +y gravity prior holds)

The camera sits `--cam-height` above the ground pitched `--pitch` degrees down,
which fixes camera<-body. Odometry fixes body<-world. The output frame is the
world rotated into MASt3R's axis convention, so the floor comes out at
y = +cam_height and the RANSAC gravity prior on +y finds it immediately.

Depth beyond --depth-max is DISCARDED rather than trusted. A stereo RealSense
has error growing as Z^2; past about 4 m it invents surfaces, and an invented
surface in an occupancy grid is a wall the planner will refuse to drive
through.
"""

import argparse
import csv
import glob
import os
import pathlib

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

DEPTH_UNITS_PER_M = 1000.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/nahar4/Gazania/MPL")
    p.add_argument("--out", required=True, help="output stem, no extension")
    p.add_argument("--fx", type=float, default=None,
                   help="focal length px; omit to recover it from the floor")
    p.add_argument("--pitch", type=float, default=None,
                   help="camera pitch below horizontal, deg; omit to recover")
    p.add_argument("--cam-height", type=float, default=0.5)
    p.add_argument("--fps", type=float, default=10.0,
                   help="frame rate tying frame index to odometry time")
    p.add_argument("--time-offset", type=float, default=0.0,
                   help="seconds added to frame time before odometry lookup")
    p.add_argument("--frame-stride", type=int, default=3)
    p.add_argument("--pixel-stride", type=int, default=3)
    p.add_argument("--depth-min", type=float, default=0.3)
    p.add_argument("--depth-max", type=float, default=4.0)
    p.add_argument("--voxel", type=float, default=0.02,
                   help="pre-downsample before writing; 0 disables")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------- inputs

def find_inputs(root):
    root = pathlib.Path(root)
    odom = sorted(root.glob("odom_*.csv"))
    rgbd = [d for d in root.rglob("*") if d.is_dir() and
            list(d.glob("*_depth.png"))]
    if not (odom and rgbd):
        raise SystemExit(f"need odom csv + depth dump under {root}")
    return str(odom[0]), str(rgbd[0])


def decodable_depth(dirpath):
    """Size>0 is not enough -- the disk filled mid-write, so files exist at
    partial length and only libpng knows. Decode every candidate once."""
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "*_depth.png"))):
        if os.path.getsize(f) == 0:
            continue
        if cv2.imread(f, cv2.IMREAD_UNCHANGED) is None:
            continue
        out[int(os.path.basename(f).split("_")[1])] = f
    return out


def load_odom(path):
    t, x, y, th = [], [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            t.append(float(r["t"])); x.append(float(r["x"]))
            y.append(float(r["y"])); th.append(float(r["theta"]))
    t = np.array(t); t -= t[0]
    return t, np.array(x), np.array(y), np.unwrap(np.array(th))


# ------------------------------------------------------------- geometry

def fit_plane(pts, thresh, iters, rng):
    best = None
    for _ in range(iters):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        inl = np.abs((pts - tri[0]) @ n) < thresh
        if best is None or inl.sum() > best[1].sum():
            best = (n, inl)
    if best is None:
        return None
    q = pts[best[1]]
    c = q.mean(axis=0)
    n = np.linalg.svd(q - c)[2][2]
    n /= np.linalg.norm(n)
    return n, c, np.abs((pts - c) @ n) < thresh


def recover_fx_pitch(depths, floor_frac=0.35, lo=250.0, hi=1400.0, steps=116):
    """The fx that makes the floor flattest is the true fx.

    Unprojecting with a wrong focal length does not rescale a plane, it BENDS
    it -- the (u-cx)Z/fx term stops matching the depth ramp. So planarity is a
    shape test, and unlike a scale test it is not degenerate.
    """
    H, W = depths[0].shape
    cx, cy = W / 2.0, H / 2.0
    v0 = int(H * (1.0 - floor_frac))
    vv, uu = np.mgrid[v0:H, 0:W]
    rng = np.random.default_rng(0)
    rows = []
    for fx in np.linspace(lo, hi, steps):
        res, hgt, tlt = [], [], []
        for z in depths:
            zc = z[v0:H, :]
            good = np.isfinite(zc) & (zc > 0.3) & (zc < 8.0)
            if good.sum() < 500:
                continue
            zg = zc[good]
            pts = np.stack([(uu[good] - cx) * zg / fx,
                            (vv[good] - cy) * zg / fx, zg], axis=1)
            if len(pts) > 4000:
                pts = pts[rng.choice(len(pts), 4000, replace=False)]
            f = fit_plane(pts, 0.04, 200, rng)
            if f is None:
                continue
            n, c, inl = f
            if inl.mean() < 0.55:
                continue
            if n[1] < 0:
                n = -n
            res.append(np.abs((pts[inl] - c) @ n).mean())
            hgt.append(abs(float(c @ n)))
            tlt.append(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
        if res:
            rows.append((fx, np.median(res), np.median(hgt), np.median(tlt),
                         len(res)))
    if not rows:
        raise SystemExit("no usable floor plane found in the sensor depth")
    return np.array(rows)


def cam_to_body(pitch_deg, cam_height):
    """Camera axes expressed in the body frame, plus the camera's origin."""
    p = np.radians(pitch_deg)
    right = np.array([0.0, -1.0, 0.0])
    down = np.array([-np.sin(p), 0.0, -np.cos(p)])
    fwd = np.array([np.cos(p), 0.0, -np.sin(p)])
    R = np.stack([right, down, fwd], axis=1)      # columns = camera axes
    return R, np.array([0.0, 0.0, cam_height])


def voxel_downsample(pts, extra, voxel):
    if voxel <= 0:
        return pts, extra
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    return pts[idx], extra[idx]


# ------------------------------------------------------------------ main

def main():
    args = parse_args()
    odom_path, rgbd_dir = find_inputs(args.root)
    print(f"odom  {odom_path}\ndepth {rgbd_dir}")

    depth_files = decodable_depth(rgbd_dir)
    idxs = sorted(depth_files)
    print(f"decodable depth frames: {len(idxs)} "
          f"(index {idxs[0]}..{idxs[-1]})")

    ot, ox, oy, oth = load_odom(odom_path)
    print(f"odometry: {len(ot)} rows over {ot[-1]:.1f} s")

    use = idxs[::args.frame_stride]
    if args.max_frames:
        use = use[:args.max_frames]

    # ---- intrinsics ----
    if args.fx is None or args.pitch is None:
        probe = [cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED
                            ).astype(np.float32) / DEPTH_UNITS_PER_M
                 for i in idxs[::max(1, len(idxs) // 30)][:30]]
        arr = recover_fx_pitch(probe)
        best = arr[np.argmin(arr[:, 1])]
        fx_r, resid, height, tilt, kept = best
        print(f"\nrecovered from sensor floor over {int(kept)} frames:")
        print(f"  fx    {fx_r:.1f} px  "
              f"({2*np.degrees(np.arctan(320/fx_r)):.1f} deg HFOV)")
        print(f"  floor planarity residual  {resid*1000:.1f} mm")
        print(f"  implied camera height     {height:.3f} m "
              f"(you measured {args.cam_height:.2f})")
        print(f"  camera pitch              {tilt:.2f} deg below horizontal")
        if abs(fx_r - 250.0) < 1 or abs(fx_r - 1400.0) < 1:
            print("  WARNING: fx railed at a search bound -- widen the range.")
        fx = args.fx if args.fx is not None else float(fx_r)
        pitch = args.pitch if args.pitch is not None else float(tilt)
    else:
        fx, pitch = args.fx, args.pitch
    print(f"\nusing fx={fx:.1f}  pitch={pitch:.2f} deg  "
          f"cam_height={args.cam_height} m")

    R_bc, t_bc = cam_to_body(pitch, args.cam_height)

    # ---- accumulate ----
    d0 = cv2.imread(depth_files[use[0]], cv2.IMREAD_UNCHANGED)
    H, W = d0.shape
    cx, cy = W / 2.0, H / 2.0
    ps = args.pixel_stride
    vv, uu = np.mgrid[0:H:ps, 0:W:ps]

    chunks, kfs, poses = [], [], []
    print(f"\nunprojecting {len(use)} frames "
          f"(pixel stride {ps}, depth {args.depth_min}-{args.depth_max} m)")
    for k, i in enumerate(use):
        z = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
        if z is None:
            continue
        z = z[::ps, ::ps].astype(np.float32) / DEPTH_UNITS_PER_M
        m = np.isfinite(z) & (z > args.depth_min) & (z < args.depth_max)
        if m.sum() < 100:
            continue

        t = i / args.fps + args.time_offset
        if t < ot[0] or t > ot[-1]:
            continue
        px, py = np.interp(t, ot, ox), np.interp(t, ot, oy)
        th = np.interp(t, ot, oth)

        zg = z[m]
        pc = np.stack([(uu[m] - cx) * zg / fx,
                       (vv[m] - cy) * zg / fx, zg], axis=1)
        pb = pc @ R_bc.T + t_bc                       # body frame

        c, s = np.cos(th), np.sin(th)
        wx = px + c * pb[:, 0] - s * pb[:, 1]
        wy = py + s * pb[:, 0] + c * pb[:, 1]
        wz = pb[:, 2]                                  # world, z up

        # world (z up) -> occupancy_grid's convention (y down)
        chunks.append(np.stack([wx, -wz, wy], axis=1).astype(np.float32))
        kfs.append(np.full(m.sum(), len(poses), dtype=np.int32))

        cb = t_bc                                      # camera origin in body
        camx = px + c * cb[0] - s * cb[1]
        camy = py + s * cb[0] + c * cb[1]
        poses.append((t, camx, -cb[2], camy, th))

        if k % 50 == 0:
            print(f"\r  {k}/{len(use)}  {sum(len(c) for c in chunks):,} pts",
                  end="", flush=True)

    if not chunks:
        raise SystemExit("no frame produced points -- check --time-offset")

    pts = np.concatenate(chunks)
    kf = np.concatenate(kfs)
    print(f"\r  {len(use)}/{len(use)}  {len(pts):,} points from "
          f"{len(poses)} frames")

    pts, kf = voxel_downsample(pts, kf, args.voxel)
    print(f"  after {args.voxel} m voxel downsample: {len(pts):,} points")

    # ---- write ----
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    v = np.empty(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("kf_id", "i4")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    v["kf_id"] = kf
    PlyData([PlyElement.describe(v, "vertex")]).write(str(out) + ".ply")

    with open(str(out) + ".txt", "w") as fh:
        for t, x, y, z, th in poses:
            # Yaw about the world's down axis (+y here), as a quaternion.
            qy, qw = np.sin(-th / 2), np.cos(-th / 2)
            fh.write(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} 0 {qy:.6f} 0 {qw:.6f}\n")

    ex = pts.max(axis=0) - pts.min(axis=0)
    print(f"\nwrote {out}.ply  ({len(pts):,} pts, extent "
          f"{ex[0]:.1f} x {ex[1]:.1f} x {ex[2]:.1f} m)")
    print(f"wrote {out}.txt  ({len(poses)} poses)")
    print(f"\nnext:\n  python scripts/occupancy_grid.py --ply {out}.ply "
          f"--traj {out}.txt \\\n         --out {out}_grid "
          f"--cam-height {args.cam_height}")


if __name__ == "__main__":
    main()
