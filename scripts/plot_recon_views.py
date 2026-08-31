#!/usr/bin/env python3
"""Orbit a MASt3R-SLAM cloud: the same reconstruction from several viewpoints.

    python scripts/plot_recon_views.py --ply logs/rover/mpl.ply \
           --traj logs/rover/mpl.txt --cam-height 0.535

A single top-down projection is a poor way to judge a reconstruction -- it
collapses every surface onto one plane, so a crisp wall and a smear of noise
look identical. Rotating around the cloud shows whether surfaces hold together
as real geometry, which is the thing actually in question.

The cloud is put in the FLOOR frame first (metric, gravity-aligned), so "up" on
the page is really up and the heights on the axis are metres above the floor.
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from occupancy_grid import load_traj, plane_basis  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--cam-height", type=float, default=0.535)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--max-h", type=float, default=3.0)
    ap.add_argument("--point-size", type=float, default=0.5)
    ap.add_argument("--max-points", type=int, default=260000,
                    help="points drawn per panel")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azims", type=float, nargs="+",
                    default=[-90, -50, -10, 30, 70, 110],
                    help="one panel per azimuth, degrees")
    ap.add_argument("--no-path", dest="show_path", action="store_false",
                    help="hide the camera trajectory")
    ap.set_defaults(show_path=True)
    args = ap.parse_args()

    from plyfile import PlyData
    v = PlyData.read(str(args.ply))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1)[finite] / 255.0
    conf = np.asarray(v["conf"])[finite] if "conf" in v.dtype.names else None
    cams = load_traj(args.traj)
    print(f"{len(xyz):,} points, {len(cams)} keyframes")

    if conf is not None and args.min_conf > 0:
        sel = conf >= args.min_conf
        print(f"  confidence >= {args.min_conf}: keeping {100*sel.mean():.1f}%")
        xyz, rgb = xyz[sel], rgb[sel]

    # Metric, gravity-aligned frame from the cameras' own up axis. A RANSAC
    # plane fit is the wrong tool here: on a polished floor the mirrored
    # geometry forms a strong plane below the real one and the fit locks onto it.
    from metric_scale import anchor_reconstruction, camera_up, load_traj_full
    _, _, quats = load_traj_full(args.traj)
    xyz, cams, _ = anchor_reconstruction(xyz, cams, quats, args.cam_height,
                                         correct_drift=False)
    up = camera_up(quats)
    o = np.median(cams, axis=0) - args.cam_height * up
    u, vv, nn = plane_basis(up, o, cams)
    to_floor = lambda p: np.stack([(p - o) @ u, (p - o) @ vv, (p - o) @ nn], axis=1)
    P, C = to_floor(xyz), to_floor(cams)

    band = (P[:, 2] > -0.20) & (P[:, 2] < args.max_h)
    P, rgb = P[band], rgb[band]
    print(f"  {len(P):,} points in height band [-0.20, {args.max_h}] m")

    step = max(1, len(P) // args.max_points)
    Ps, rgbs = P[::step], rgb[::step]
    print(f"  drawing {len(Ps):,} per panel")

    n = len(args.azims)
    cols = 3 if n >= 3 else n
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(6.2 * cols, 5.4 * rows))

    spanx, spany, spanz = np.ptp(P[:, 0]), np.ptp(P[:, 1]), np.ptp(P[:, 2])
    for i, az in enumerate(args.azims):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.scatter(Ps[:, 0], Ps[:, 1], Ps[:, 2], s=args.point_size, c=rgbs,
                   marker=".", linewidths=0, depthshade=False)
        if args.show_path:
            ax.plot(C[:, 0], C[:, 1], C[:, 2], "-", color="#ff2d55", lw=2.0,
                    zorder=10)
        ax.view_init(elev=args.elev, azim=az)
        ax.set_title(f"azimuth {az:.0f}°", fontsize=11)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.set_zlabel("h (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        # Exaggerate height a little: the room is much wider than it is tall,
        # and at true aspect the vertical structure is a few pixels.
        try:
            ax.set_box_aspect((spanx, spany, 2.2 * spanz))
        except Exception:
            pass

    fig.suptitle(f"{pathlib.Path(args.ply).stem} — reconstruction from "
                 f"{n} viewpoints  ({len(P):,} points, {len(cams)} keyframes, "
                 f"{spanx:.1f} × {spany:.1f} m)", fontsize=13)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        args.ply).with_name(pathlib.Path(args.ply).stem + "_views.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=135)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
