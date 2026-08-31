#!/usr/bin/env python3
"""Measure the lag between the video clock and the odometry clock.

    python scripts/find_time_offset.py --root /home/nahar4/Gazania/MPL

Why this exists
---------------
The 2026-08-26 session left three streams whose durations disagree: odometry
494.8 s, video 467.3 s, RGBD dump 4083 frames. Assuming they start together is
an assumption, not a fact, and it is the single most destructive thing to get
wrong -- a few seconds of lag rotates every unprojected point about the wrong
pose and turns a house into a uniform smear of obstacles. The first BEV built
here came out 95% occupied for exactly this reason.

The measurement
---------------
When the rover yaws, the picture pans. So the horizontal image shift between
consecutive frames is a proxy for angular velocity, measurable from the video
alone, and the odometry already reports w directly. Both are 1-D signals at
about 10 Hz. Cross-correlate them and the lag that maximises the correlation
is the offset between the clocks.

Yaw is the right channel to use rather than forward speed: translation changes
the picture in a way that depends on scene depth, while rotation pans every
pixel by the same amount regardless of what is in frame.
"""

import argparse
import csv
import pathlib

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/nahar4/Gazania/MPL")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--max-lag", type=float, default=40.0,
                   help="seconds of lag searched either way")
    p.add_argument("--width", type=int, default=160,
                   help="frames are downscaled to this before matching")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def find_inputs(root):
    root = pathlib.Path(root)
    mp4 = sorted(root.glob("*.mp4"))
    odom = sorted(root.glob("odom_*.csv"))
    if not (mp4 and odom):
        raise SystemExit(f"need an mp4 and an odom csv under {root}")
    return str(mp4[0]), str(odom[0])


def image_yaw_rate(path, width, max_frames):
    """Per-frame horizontal pan, in pixels, via phase correlation."""
    cap = cv2.VideoCapture(path)
    prev, out = None, []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h = int(g.shape[0] * width / g.shape[1])
        g = cv2.resize(g, (width, h), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32) / 255.0
        if prev is not None:
            (dx, _), _ = cv2.phaseCorrelate(prev, g)
            out.append(dx)
        prev = g
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    return np.asarray(out)


def load_odom(path):
    t, w = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            t.append(float(r["t"])); w.append(float(r["w"]))
    t = np.asarray(t); t -= t[0]
    return t, np.asarray(w)


def main():
    args = parse_args()
    mp4, odom_path = find_inputs(args.root)
    print(f"video {mp4}\nodom  {odom_path}")

    dx = image_yaw_rate(mp4, args.width, args.max_frames)
    vt = np.arange(len(dx)) / args.fps + 0.5 / args.fps
    print(f"video: {len(dx)} inter-frame shifts over {vt[-1]:.1f} s")

    ot, ow = load_odom(odom_path)
    print(f"odom:  {len(ot)} rows over {ot[-1]:.1f} s")

    # Resample odometry onto the video's own grid so the correlation is
    # between two signals with identical sampling.
    lags = np.arange(-args.max_lag, args.max_lag + 1e-9, 1.0 / args.fps)
    a = dx - dx.mean()
    na = np.linalg.norm(a)

    best, curve = None, []
    for lag in lags:
        w = np.interp(vt + lag, ot, ow, left=np.nan, right=np.nan)
        m = np.isfinite(w)
        if m.sum() < 0.5 * len(vt):
            curve.append(np.nan)
            continue
        b = w[m] - w[m].mean()
        nb = np.linalg.norm(b)
        c = float((a[m] @ b) / (np.linalg.norm(a[m]) * nb + 1e-12))
        curve.append(c)
        if best is None or abs(c) > abs(best[1]):
            best = (lag, c)
    curve = np.asarray(curve)

    lag, corr = best
    print(f"\nbest lag {lag:+.2f} s   correlation {corr:+.3f}")
    print("  (a negative correlation just means the image pans opposite to "
          "the sign convention of w; only |corr| matters here)")

    # Show the neighbourhood so a flat, ambiguous peak is visible as such.
    print("\n  lag(s)   corr")
    order = np.argsort(-np.abs(np.nan_to_num(curve)))
    for i in sorted(order[:9]):
        mark = " <-" if lags[i] == lag else ""
        print(f"  {lags[i]:+6.2f}  {curve[i]:+.3f}{mark}")

    sharp = np.abs(corr) - np.nanmedian(np.abs(curve))
    print(f"\npeak stands {sharp:.3f} above the median |corr| "
          f"({np.nanmedian(np.abs(curve)):.3f})")
    if abs(corr) < 0.3:
        print("WARNING: weak peak. The rover may have turned too little for "
              "yaw to identify the lag; try correlating speed instead.")
    else:
        print(f"=> pass --time-offset {lag:.2f} to scripts/depth_to_bev.py")


if __name__ == "__main__":
    main()
