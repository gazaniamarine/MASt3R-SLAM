#!/usr/bin/env python3
"""Recover focal length and camera pitch from the floor, given the mount height.

    python scripts/calib_from_floor.py --video VIDEO.mp4 --cam-height 0.5

Why this exists
---------------
The rover's RealSense model is unknown, and the D4xx family splits into two
very different depth FOVs -- ~87 deg for the D435/D435i/D455, ~65 deg for the
D415. Guessing wrong is a ~30% error in fx, which shears every unprojected
point cloud and therefore every BEV cell. Rather than guess, measure.

The floor is the observable. Unprojecting a metric depth map with the WRONG
focal length does not merely rescale the scene -- it bends it. A true plane
comes back curved, concave for fx too small and convex for fx too large,
because the (u - cx) * Z / fx term grows the lateral spread at a rate that no
longer matches the depth ramp. So the fx that makes the floor flattest is the
right one, and unlike a scale error this is not degenerate: planarity is a
shape test, not a size test.

Height then does the second job. Once fx is fixed the floor is a genuine
plane, and the camera's perpendicular distance to it is a metric quantity we
know independently (measured on the rover). The ratio of predicted to measured
height is the scale correction for the depth network -- the same anchor
scripts/metric_scale.py applies to MASt3R clouds, reused here.

Pitch falls out of the same fit: the floor normal in camera coordinates IS the
down axis, so the angle between it and the image's own vertical is the tilt.

Assumptions, all of which the caller should sanity-check against the report:
  * square pixels (fy == fx) and a centred principal point. True to within a
    percent or so on a factory-calibrated RealSense, and not worth solving for
    when the height itself is only known to "roughly 0.5 m".
  * the lower --floor-frac of the image is mostly floor. Frames where it is
    not (a wall right in front, a person filling the view) are rejected by the
    inlier test rather than silently averaged in.
"""

import argparse
import pathlib

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

CKPT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--cam-height", type=float, required=True,
                   help="camera centre above the floor, metres")
    p.add_argument("--frames", type=int, default=40,
                   help="frames sampled evenly across the take")
    p.add_argument("--fx-min", type=float, default=200.0)
    p.add_argument("--fx-max", type=float, default=800.0)
    p.add_argument("--fx-steps", type=int, default=121)
    p.add_argument("--floor-frac", type=float, default=0.35,
                   help="bottom fraction of the image searched for floor")
    p.add_argument("--plane-thresh", type=float, default=0.04,
                   help="RANSAC inlier band, metres")
    p.add_argument("--min-inlier", type=float, default=0.55,
                   help="reject a frame whose best plane holds less than this")
    p.add_argument("--out", default=None, help="write a yaml of the result here")
    return p.parse_args()


def sample_frames(path, n):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            frames.append((int(i), bgr))
    cap.release()
    return frames


def predict_depth(model, proc, frames_bgr, hw):
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    with torch.inference_mode():
        inp = proc(images=rgb, return_tensors="pt").to("cuda", torch.float16)
        out = model(**inp)
    res = proc.post_process_depth_estimation(out, target_sizes=[hw] * len(rgb))
    return [r["predicted_depth"].float().cpu().numpy() for r in res]


def fit_plane(pts, thresh, iters=200, rng=None):
    """RANSAC plane. Returns (normal, point_on_plane, inlier_mask) or None."""
    rng = rng or np.random.default_rng(0)
    if len(pts) < 50:
        return None
    best = None
    for _ in range(iters):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = np.abs((pts - tri[0]) @ n)
        inl = d < thresh
        if best is None or inl.sum() > best[2].sum():
            best = (n, tri[0], inl)
    if best is None:
        return None
    # Refit on the inliers -- the 3-point hypothesis is only a seed.
    n, p0, inl = best
    q = pts[inl]
    c = q.mean(axis=0)
    n = np.linalg.svd(q - c)[2][2]
    n = n / np.linalg.norm(n)
    inl = np.abs((pts - c) @ n) < thresh
    return n, c, inl


def evaluate_fx(depth_maps, fx, cx, cy, floor_frac, thresh, min_inlier):
    """Planarity residual and implied camera height, pooled over frames."""
    H, W = depth_maps[0].shape
    v0 = int(H * (1.0 - floor_frac))
    vv, uu = np.mgrid[v0:H, 0:W]
    rng = np.random.default_rng(0)

    resid, heights, tilts, kept = [], [], [], 0
    for d in depth_maps:
        z = d[v0:H, :]
        good = np.isfinite(z) & (z > 0.2) & (z < 20.0)
        if good.sum() < 200:
            continue
        zg = z[good]
        x = (uu[good] - cx) * zg / fx
        y = (vv[good] - cy) * zg / fx
        pts = np.stack([x, y, zg], axis=1)
        # Subsample: RANSAC on 200k points buys nothing over 4k.
        if len(pts) > 4000:
            pts = pts[rng.choice(len(pts), 4000, replace=False)]
        fit = fit_plane(pts, thresh, rng=rng)
        if fit is None:
            continue
        n, c, inl = fit
        frac = inl.mean()
        if frac < min_inlier:
            continue
        # Floor normal should point roughly along image-down (+y).
        if n[1] < 0:
            n = -n
        resid.append(np.abs((pts[inl] - c) @ n).mean())
        heights.append(abs(float(c @ n)))       # camera at origin -> |c.n|
        tilts.append(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
        kept += 1
    if kept == 0:
        return None
    return (float(np.median(resid)), float(np.median(heights)),
            float(np.median(tilts)), kept)


def main():
    args = parse_args()
    frames = sample_frames(args.video, args.frames)
    print(f"sampled {len(frames)} frames from {pathlib.Path(args.video).name}")

    proc = AutoImageProcessor.from_pretrained(CKPT)
    model = AutoModelForDepthEstimation.from_pretrained(CKPT).to(
        "cuda", torch.float16).eval()
    print(f"loaded {CKPT}")

    H, W = frames[0][1].shape[:2]
    depths = []
    for i in range(0, len(frames), 8):
        chunk = [f for _, f in frames[i:i + 8]]
        depths.extend(predict_depth(model, proc, chunk, (H, W)))
    print(f"predicted metric depth for {len(depths)} frames ({W}x{H})")

    cx, cy = W / 2.0, H / 2.0
    grid = np.linspace(args.fx_min, args.fx_max, args.fx_steps)
    rows = []
    for fx in grid:
        r = evaluate_fx(depths, fx, cx, cy, args.floor_frac,
                        args.plane_thresh, args.min_inlier)
        if r is not None:
            rows.append((fx,) + r)
    if not rows:
        raise SystemExit("no frame produced a usable floor plane -- try "
                         "--floor-frac 0.5 or --min-inlier 0.4")

    arr = np.array(rows)
    best = arr[np.argmin(arr[:, 1])]
    fx, resid, height, tilt, kept = best
    hfov = 2 * np.degrees(np.arctan(W / (2 * fx)))
    vfov = 2 * np.degrees(np.arctan(H / (2 * fx)))
    scale = args.cam_height / height

    print("\n  fx      resid(m)  height(m)  tilt(deg)  frames")
    for r in arr[::max(1, len(arr) // 20)]:
        mark = " <-" if r[0] == fx else ""
        print(f"  {r[0]:6.1f}  {r[1]:8.4f}  {r[2]:9.3f}  {r[3]:9.2f}  {int(r[4]):6d}{mark}")

    print(f"\nbest fx = {fx:.1f} px   (fy assumed equal, cx={cx:.0f} cy={cy:.0f})")
    print(f"  implied FOV      {hfov:.1f} deg horizontal x {vfov:.1f} deg vertical")
    print(f"  planarity resid  {resid*1000:.1f} mm")
    print(f"  floor tilt       {tilt:.2f} deg  (camera pitch below horizontal)")
    print(f"  predicted height {height:.3f} m vs measured {args.cam_height:.3f} m")
    print(f"  depth scale      x{scale:.4f}")
    if not 0.75 < scale < 1.33:
        print("  WARNING: scale is far from 1.0. Either the height is wrong or "
              "the metric head is badly off on this scene -- check before use.")

    if args.out:
        pathlib.Path(args.out).write_text(
            f"# recovered by scripts/calib_from_floor.py\n"
            f"width: {W}\nheight: {H}\n"
            f"fx: {fx:.2f}\nfy: {fx:.2f}\ncx: {cx:.2f}\ncy: {cy:.2f}\n"
            f"cam_height: {args.cam_height}\n"
            f"pitch_deg: {tilt:.3f}\ndepth_scale: {scale:.4f}\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
