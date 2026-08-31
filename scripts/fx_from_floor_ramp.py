#!/usr/bin/env python3
"""Solve focal length and camera pitch from the floor's depth ramp.

    python scripts/fx_from_floor_ramp.py --root /home/nahar4/Gazania/MPL \
           --cam-height 0.5

Why a third attempt at this
---------------------------
Two previous estimators were degenerate and both silently returned a search
bound rather than a measurement:

  scripts/calib_from_floor.py   minimises Cartesian distance to a fitted
                                plane. That shrinks as fx grows -- lateral
                                coordinates carry a 1/fx factor -- so the
                                metric falls monotonically and argmin is
                                fx_max. It reported fx = 800, its own limit.

  depth_fit_eval.py (v1)        same metric, same failure, reported fx = 1390
                                against a limit of 1400.

  depth_fit_eval.py (v2)        residual measured in depth instead, which
                                removes THAT degeneracy but introduces
                                another: the residual is evaluated on RANSAC
                                inliers, and the inlier set self-selects
                                whatever subset happens to be planar. It ran
                                to the lower bound instead, fx = 300.

The mistake common to all three is searching over fx at all. A planar floor
gives a closed-form relation, so there is nothing to search.

The relation
------------
Camera axes x right, y down, z forward, pitched down by p. Gravity-down in
camera coordinates is g = (0, cos p, -sin p): at p = 0 it is the image's own
down axis, at p = 90 deg it is -z, both correct. The floor is the locus
P . g = h for camera height h. A pixel's ray is r = ((u-cx)/fx, (v-cy)/fx, 1),
so Z (r . g) = h and

    1/Z(v) = (v - cy) * cos(p) / (h * fx)  -  sin(p) / h

which is LINEAR in the image row v. Regress inverse depth on row:

    slope A = cos(p) / (h * fx)      intercept B = -sin(p) / h

and with h measured on the rover, both unknowns fall straight out:

    p  = asin(-B * h)
    fx = cos(p) / (h * A)

No search range, no plane RANSAC, no bound to rail against. The floor's depth
ramp constrains fx through the SHAPE of the ramp and pitch through its offset,
and those are independent, which is exactly what the previous formulations
destroyed.

The fit is robust rather than plain least squares because the lower image is
not purely floor -- furniture legs, the rover's own chassis and the far wall
all intrude, and a single outlier row would tilt an ordinary regression.
"""

import argparse
import glob
import os
import pathlib

import cv2
import numpy as np

DEPTH_UNITS_PER_M = 1000.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/nahar4/Gazania/MPL")
    p.add_argument("--cam-height", type=float, required=True,
                   help="camera centre above the floor, metres")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--col-frac", type=float, default=0.5,
                   help="central fraction of columns used (avoids the edges, "
                        "where lens distortion is worst)")
    p.add_argument("--row-start", type=float, default=0.60,
                   help="fraction of image height where the floor search "
                        "begins; above this the floor is past the horizon")
    p.add_argument("--depth-min", type=float, default=0.3)
    p.add_argument("--depth-max", type=float, default=6.0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def find_depth_dir(root):
    root = pathlib.Path(root)
    for d in root.rglob("*"):
        if d.is_dir() and list(d.glob("*_depth.png")):
            return str(d)
    raise SystemExit(f"no depth dump under {root}")


def robust_line(x, y, iters=5):
    """Least squares reweighted toward the inliers (Tukey-ish, cheap).

    The floor is the densest structure in the band but not the only one, so
    the fit has to survive a minority of rows belonging to furniture or wall.
    """
    A = np.stack([x, np.ones_like(x)], axis=1)
    w = np.ones_like(x)
    a = b = 0.0
    for _ in range(iters):
        W = w[:, None]
        sol, *_ = np.linalg.lstsq(A * W, y * w, rcond=None)
        a, b = sol
        r = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        w = 1.0 / (1.0 + (r / (2.5 * s)) ** 2)
    return float(a), float(b), w


def main():
    args = parse_args()
    ddir = find_depth_dir(args.root)
    files = []
    for f in sorted(glob.glob(os.path.join(ddir, "*_depth.png"))):
        if os.path.getsize(f) > 0:
            files.append(f)
    print(f"depth dump {ddir}\ncandidate frames {len(files)}")

    pick = [files[i] for i in
            np.linspace(0, len(files) - 1, min(args.frames, len(files))
                        ).astype(int)]

    h = args.cam_height
    per_frame = []
    xs_all, ys_all = [], []
    for f in pick:
        z = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if z is None:
            continue
        z = z.astype(np.float32) / DEPTH_UNITS_PER_M
        H, W = z.shape
        cy = H / 2.0
        c0 = int(W * (1 - args.col_frac) / 2)
        c1 = W - c0
        r0 = int(H * args.row_start)

        band = z[r0:H, c0:c1]
        rows = np.arange(r0, H)[:, None].repeat(band.shape[1], axis=1)
        good = (np.isfinite(band) & (band > args.depth_min)
                & (band < args.depth_max))
        if good.sum() < 500:
            continue
        x = (rows[good] - cy).astype(np.float64)
        y = (1.0 / band[good]).astype(np.float64)
        xs_all.append(x)
        ys_all.append(y)

        a, b, w = robust_line(x, y)
        if a <= 0:
            continue
        sp = np.clip(-b * h, -0.999, 0.999)
        p = np.arcsin(sp)
        fx = np.cos(p) / (h * a)
        if 100 < fx < 3000:
            per_frame.append((fx, np.degrees(p), w.mean()))

    if not per_frame:
        raise SystemExit("no frame yielded a usable floor ramp")

    arr = np.array(per_frame)
    fx_med = float(np.median(arr[:, 0]))
    p_med = float(np.median(arr[:, 1]))
    q1, q3 = np.percentile(arr[:, 0], [25, 75])

    # Pooled fit over every frame at once -- less sensitive to any single
    # frame that happened to face a wall.
    x = np.concatenate(xs_all)
    y = np.concatenate(ys_all)
    a, b, w = robust_line(x, y)
    p_pool = np.arcsin(np.clip(-b * h, -0.999, 0.999))
    fx_pool = np.cos(p_pool) / (h * a)

    H0, W0 = cv2.imread(pick[0], cv2.IMREAD_UNCHANGED).shape
    print(f"\nper-frame over {len(arr)} frames:")
    print(f"  fx     median {fx_med:7.1f}   IQR [{q1:.1f}, {q3:.1f}]  "
          f"spread {100*(q3-q1)/fx_med:.1f}%")
    print(f"  pitch  median {p_med:7.2f} deg")
    print(f"\npooled over {len(x):,} floor pixels:")
    print(f"  fx     {fx_pool:7.1f} px   "
          f"({2*np.degrees(np.arctan(W0/(2*fx_pool))):.1f} deg HFOV, "
          f"{2*np.degrees(np.arctan(H0/(2*fx_pool))):.1f} deg VFOV)")
    print(f"  pitch  {np.degrees(p_pool):7.2f} deg below horizontal")
    print(f"  inlier weight {w.mean():.2f}")

    # A sanity number that does not depend on the fit: where the bottom row
    # says the floor is, given the recovered geometry.
    print(f"\ncheck: at fx={fx_pool:.0f}, the image bottom row looks "
          f"{np.degrees(np.arctan((H0/2)/fx_pool)) + np.degrees(p_pool):.1f} "
          f"deg below horizontal,")
    print(f"       so the floor there should read "
          f"{h/np.sin(np.radians(np.degrees(np.arctan((H0/2)/fx_pool)) + np.degrees(p_pool))):.2f} m "
          f"of range.")

    if args.out:
        pathlib.Path(args.out).write_text(
            f"# recovered by scripts/fx_from_floor_ramp.py\n"
            f"width: {W0}\nheight: {H0}\n"
            f"fx: {fx_pool:.2f}\nfy: {fx_pool:.2f}\n"
            f"cx: {W0/2:.2f}\ncy: {H0/2:.2f}\n"
            f"cam_height: {h}\npitch_deg: {np.degrees(p_pool):.3f}\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
