#!/usr/bin/env python3
"""Where in a frame folder is the camera actually moving?

    python scripts/motion_profile.py --frames datasets/rover/mpl --out logs/rover/motion.png

Monocular SLAM needs the camera to TRANSLATE. A stationary rover gives zero
parallax, which is the classic way to lose tracking, and a long stationary
stretch followed by an abrupt viewpoint change leaves relocalisation with no
overlap to work from. Before re-running SLAM on a long take it is worth knowing
which parts of it carry motion at all.

The measure is deliberately crude -- mean absolute difference between
consecutive frames, on a heavily downsampled greyscale image. It cannot tell
camera motion from a person walking past, so it is an upper bound on motion;
a LOW value is the reliable signal, and that is the one we act on.
"""
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def profile(paths, size=(80, 60)):
    diffs = np.zeros(len(paths), dtype=np.float32)
    prev = None
    for i, p in enumerate(paths):
        img = Image.open(p).convert("L").resize(size, Image.BILINEAR)
        cur = np.asarray(img, dtype=np.float32)
        if prev is not None:
            diffs[i] = np.abs(cur - prev).mean()
        prev = cur
    return diffs


def segments(mask, min_len):
    """Contiguous True runs of at least min_len, as (start, end) inclusive."""
    out = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask) - 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--thresh", type=float, default=2.0,
                    help="mean abs frame difference below which the camera is "
                         "treated as stationary (grey levels, 0-255)")
    ap.add_argument("--min-run", type=int, default=40,
                    help="shortest moving run worth reporting, in frames")
    args = ap.parse_args()

    folder = pathlib.Path(args.frames)
    paths = sorted(folder.glob("*.png"))
    if not paths:
        raise SystemExit(f"no PNGs in {folder}")
    print(f"{len(paths)} frames in {folder}")

    d = profile(paths)
    moving = d >= args.thresh
    runs = segments(moving, args.min_run)

    print(f"\nmean abs frame diff: median {np.median(d):.2f}, "
          f"p90 {np.percentile(d, 90):.2f}, max {d.max():.2f}")
    print(f"frames below thresh {args.thresh}: {int((~moving).sum()):,}/{len(d):,} "
          f"({100*(~moving).mean():.1f}% effectively stationary)")
    print(f"\nmoving runs of >= {args.min_run} frames:")
    total = 0
    for a, b in runs:
        total += b - a + 1
        print(f"  frames {a+1:6d} - {b+1:6d}   ({b-a+1:5d} frames, "
              f"mean diff {d[a:b+1].mean():.2f})")
    print(f"\n{len(runs)} runs covering {total:,} frames "
          f"({100*total/len(d):.1f}% of the take)")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.arange(1, len(d) + 1), d, lw=0.6, color="#3060a0")
    ax.axhline(args.thresh, color="#c04030", ls="--", lw=1,
               label=f"stationary threshold {args.thresh}")
    for a, b in runs:
        ax.axvspan(a + 1, b + 1, color="#70b070", alpha=0.25)
    ax.set_xlabel("frame")
    ax.set_ylabel("mean abs frame difference")
    ax.set_title(f"{folder.name}: motion profile "
                 f"({100*total/len(d):.0f}% of frames in usable moving runs, "
                 f"shaded)")
    ax.legend(loc="upper right", fontsize=9)
    ax.margins(x=0.01)

    out = pathlib.Path(args.out) if args.out else folder.parent / f"{folder.name}_motion.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
