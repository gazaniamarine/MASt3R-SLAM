#!/usr/bin/env python3
"""Recover the frame-index mapping between the RGBD dump, the MP4, and odometry.

    python scripts/align_streams.py --root /home/nahar4/Gazania/MPL

Why this exists
---------------
The 2026-08-26 rover session wrote three streams on three clocks that do not
agree with each other:

    odometry     494.8 s   timestamped, 9.8 Hz
    MP4          467.3 s   4673 frames, exactly 10 fps
    RGBD dump    4083 frames, no timestamps at all

If the RGBD writer had kept up, 4083 frames at 10 fps would be 408.3 s -- 59 s
short of the video. Either it dropped frames (two PNG encodes per frame is
expensive) or it started late. Those two hypotheses put the same RGBD frame in
completely different places on the odometry track, so guessing between them
would silently shear the whole map.

The RGB half of the dump and the MP4 are pictures of the same moment, so the
mapping is measurable. We match a handful of RGBD frames into the video by
image similarity, then check whether the resulting (rgbd_idx -> video_idx)
pairs fall on a straight line. A straight line through the origin with slope 1
means no drops; slope > 1 means uniform dropping; a bent line means bursty
drops and only a per-frame match will do.
"""

import argparse
import glob
import os
import pathlib

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/nahar4/Gazania/MPL")
    p.add_argument("--probes", type=int, default=12,
                   help="RGBD frames matched into the video")
    p.add_argument("--thumb", type=int, default=64,
                   help="edge of the grayscale thumbnail used for matching")
    p.add_argument("--out", default=None, help="write the mapping as npz")
    p.add_argument("--cache", default=None,
                   help="npz caching which frames decode (decoding 8k PNGs is slow)")
    return p.parse_args()


def find_inputs(root):
    root = pathlib.Path(root)
    mp4 = sorted(root.glob("*.mp4"))
    odom = sorted(root.glob("odom_*.csv"))
    rgbd = [d for d in root.rglob("*") if d.is_dir() and
            list(d.glob("*_rgb.png"))]
    if not (mp4 and odom and rgbd):
        raise SystemExit(f"missing inputs under {root}: "
                         f"mp4={bool(mp4)} odom={bool(odom)} rgbd={bool(rgbd)}")
    return mp4[0], odom[0], rgbd[0]


def usable_rgbd(dirpath, cache=None):
    """Indices where BOTH the rgb and the depth png actually DECODE.

    A nonzero file size is not enough. The capture ran the disk to 100%, and
    the frames it was mid-way through writing are left at partial length --
    they pass a size check and then blow up in libpng. Every candidate is
    decoded once, and the verdict cached, because decoding 8k PNGs is slow
    enough that you do not want it on every run.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        both = list(z["both"])
        return both, z["rgb"].item(), z["depth"].item()

    def ok(pattern, flag):
        out = {}
        for f in sorted(glob.glob(os.path.join(dirpath, pattern))):
            if os.path.getsize(f) == 0:
                continue
            if cv2.imread(f, flag) is None:
                continue
            out[int(os.path.basename(f).split("_")[1])] = f
        return out
    rgb = ok("*_rgb.png", cv2.IMREAD_COLOR)
    depth = ok("*_depth.png", cv2.IMREAD_UNCHANGED)
    both = sorted(set(rgb) & set(depth))
    if cache:
        np.savez(cache, both=np.array(both), rgb=rgb, depth=depth)
    return both, rgb, depth


def thumb_of(bgr, n):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    t -= t.mean()
    s = np.linalg.norm(t)
    return t / s if s > 1e-6 else t


def video_thumbs(path, n):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    ts = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        ts.append(thumb_of(bgr, n))
    cap.release()
    return np.stack(ts).reshape(len(ts), -1)


def main():
    args = parse_args()
    mp4, odom, rgbd_dir = find_inputs(args.root)
    print(f"mp4   {mp4}")
    print(f"odom  {odom}")
    print(f"rgbd  {rgbd_dir}")

    idxs, rgb_files, depth_files = usable_rgbd(str(rgbd_dir), args.cache)
    print(f"\nusable rgb+depth pairs: {len(idxs)} "
          f"(index {idxs[0]}..{idxs[-1]})")
    gaps = np.diff(idxs)
    if (gaps > 1).any():
        print(f"  {int((gaps > 1).sum())} gaps inside that range "
              f"({int(gaps[gaps > 1].sum() - (gaps > 1).sum())} missing frames)")

    print("building video thumbnails...")
    V = video_thumbs(str(mp4), args.thumb)
    print(f"video frames: {len(V)}")

    probe_idx = [idxs[i] for i in
                 np.linspace(0, len(idxs) - 1, args.probes).astype(int)]
    print(f"\n  rgbd_idx -> video_idx   score   (best match)")
    pairs = []
    for i in probe_idx:
        t = thumb_of(cv2.imread(rgb_files[i]), args.thumb).ravel()
        sim = V @ t
        j = int(np.argmax(sim))
        pairs.append((i, j, float(sim[j])))
        print(f"  {i:8d} -> {j:9d}   {sim[j]:.4f}")

    a = np.array(pairs, dtype=float)
    if (a[:, 2] < 0.9).any():
        print("\nWARNING: some matches score below 0.9 -- the two streams may "
              "not show the same scene, or the video is too compressed for "
              "thumbnail matching.")

    slope, icept = np.polyfit(a[:, 0], a[:, 1], 1)
    pred = slope * a[:, 0] + icept
    resid = a[:, 1] - pred
    print(f"\nlinear fit  video_idx = {slope:.5f} * rgbd_idx + {icept:.2f}")
    print(f"  max |residual| = {np.abs(resid).max():.1f} frames "
          f"({np.abs(resid).max()/10:.2f} s at 10 fps)")
    if np.abs(resid).max() < 2.0:
        print("  -> mapping is LINEAR. A constant rate + offset is exact.")
    else:
        print("  -> mapping is NOT linear (bursty drops). Every frame needs "
              "its own match; do not interpolate this fit.")

    if args.out:
        np.savez(args.out, pairs=a, slope=slope, intercept=icept,
                 usable_idx=np.array(idxs))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
