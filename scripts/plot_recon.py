#!/usr/bin/env python3
"""Render a MASt3R-SLAM point cloud + trajectory as presentable figures.

    python scripts/plot_recon.py --ply logs/rover/mpl.ply --traj logs/rover/mpl.txt \
           --cam-height 0.535 --out logs/rover/mpl_recon.png

Two views: looking straight down (the floorplan the BEV grid is built from) and
a horizontal elevation. Both are drawn in the FLOOR frame, not MASt3R's raw
camera frame, so "up" on the page is really up -- the raw frame has +y DOWN and
its origin wherever the camera happened to sit on frame 0, which makes an
unrotated plot unreadable.

Points are coloured by their own RGB, so this shows the reconstruction as it is
rather than a derived heatmap. The camera path is drawn over it.
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from occupancy_grid import (load_ply, load_traj, voxel_downsample,   # noqa: E402
                            fit_plane_ransac, plane_basis)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cam-height", type=float, default=None,
                    help="camera height above floor, metres -- recovers metric "
                         "scale the same way occupancy_grid.py does")
    ap.add_argument("--voxel", type=float, default=0.04)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--max-h", type=float, default=2.5,
                    help="drop points above this (ceiling) so the floorplan reads")
    ap.add_argument("--point-size", type=float, default=0.35)
    ap.add_argument("--floor-radius", type=float, default=4.0,
                    help="only fit the floor plane to points within this of the path")
    args = ap.parse_args()

    pts, conf, _ = load_ply(args.ply, extras=True)
    cams = load_traj(args.traj)
    print(f"{len(pts):,} points, {len(cams)} keyframe poses")

    if conf is not None and args.min_conf > 0:
        keep = conf >= args.min_conf
        pts, conf = pts[keep], conf[keep]
        print(f"  {len(pts):,} above confidence {args.min_conf}")

    rgb = None
    from plyfile import PlyData
    v = PlyData.read(str(args.ply))["vertex"].data
    if {"red", "green", "blue"} <= set(v.dtype.names):
        finite = np.isfinite(np.stack([v["x"], v["y"], v["z"]], axis=1)).all(axis=1)
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1)[finite] / 255.0
        if conf is not None and args.min_conf > 0:
            rgb = rgb[keep]

    idx = voxel_downsample(pts, args.voxel, return_index=True)
    pts, rgb = pts[idx], (rgb[idx] if rgb is not None else None)
    print(f"  {len(pts):,} after {args.voxel} m voxel downsample")

    # Scale and floor come from the SAME path occupancy_grid.py uses, so this
    # figure and the BEV grid agree about where the floor is and how big a metre
    # is. Deliberately NOT a RANSAC plane fit on this cloud: on a polished floor
    # the mirrored geometry forms a strong plane well below the real one, and a
    # fit -- global or path-restricted -- locked onto it, putting the cameras
    # 3.29 units up and deriving scale 0.163 against the anchor's 1.115.
    # The camera's own up axis has no such failure mode.
    from metric_scale import anchor_reconstruction, camera_up, load_traj_full
    _, _, quats = load_traj_full(args.traj)
    if args.cam_height:
        pts, cams, est = anchor_reconstruction(pts, cams, quats,
                                               args.cam_height,
                                               correct_drift=False)
    up = camera_up(quats)
    o = np.median(cams, axis=0) - (args.cam_height or 0.0) * up
    u, vv, nn = plane_basis(up, o, cams)

    def to_floor(p):
        d = p - o
        return np.stack([d @ u, d @ vv, d @ nn], axis=1)

    P, C = to_floor(pts), to_floor(cams)
    print(f"  camera height above fitted floor: median {np.median(C[:, 2]):.3f} m")

    band = (P[:, 2] > -0.15) & (P[:, 2] < args.max_h)
    P, C_rgb = P[band], (rgb[band] if rgb is not None else None)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))

    axes[0].scatter(P[:, 0], P[:, 1], s=args.point_size, c=C_rgb, marker=".",
                    linewidths=0)
    axes[0].plot(C[:, 0], C[:, 1], "-", color="#ff2d55", lw=2.2,
                 label="camera path", zorder=5)
    axes[0].scatter(C[0, 0], C[0, 1], s=90, marker="^", color="#00c853",
                    edgecolor="k", zorder=6, label="start")
    axes[0].scatter(C[-1, 0], C[-1, 1], s=90, marker="X", color="#d50000",
                    edgecolor="k", zorder=6, label="end")
    axes[0].set_title(f"Top-down view  ({len(P):,} points, "
                      f"{len(cams)} keyframes)", fontsize=12)
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].scatter(P[:, 0], P[:, 2], s=args.point_size, c=C_rgb, marker=".",
                    linewidths=0)
    axes[1].plot(C[:, 0], C[:, 2], "-", color="#ff2d55", lw=2.2, zorder=5)
    axes[1].axhline(0.0, color="#0060ff", ls="--", lw=1.2,
                    label="fitted floor plane")
    axes[1].set_title("Elevation (height above floor)", fontsize=12)
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("height (m)")
    axes[1].set_aspect("equal")
    axes[1].legend(loc="best", fontsize=9)

    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        args.ply).with_name(pathlib.Path(args.ply).stem + "_recon.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=145)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
