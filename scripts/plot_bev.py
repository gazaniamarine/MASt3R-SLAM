#!/usr/bin/env python3
"""Visualise a rover BEV occupancy grid and the clearance field the tube is built from.

    python scripts/plot_bev.py --grid logs/rover/grids/mpl.npy --traj logs/rover/mpl.txt

Three panels, because the question "why is the safety tube so thin?" is not
answerable from the occupancy map alone:

  1. the four-way occupancy classes, as the planner sees them
  2. the same map after unknown space is split into enclosed vs exterior
  3. the clearance field -- signed distance to the nearest blocking cell, minus
     the robot radius. This is the quantity that caps the tube radius, so it is
     the one that explains the tube width.
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import distance_transform_edt, label

UNKNOWN = -1


def read_yaml(path):
    """Minimal reader for the keys occupancy_grid.py's write_pgm_yaml emits."""
    meta = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if k == "origin":
            meta[k] = [float(x) for x in v.strip("[]").split(",")]
        elif k in ("resolution", "occupied_thresh", "free_thresh"):
            meta[k] = float(v)
    return meta


def classify(prob, free_thresh, occ_thresh):
    """0 free, 1 undetermined, 2 occupied, 3 unknown."""
    cls = np.full(prob.shape, 3, dtype=np.int8)
    known = prob >= 0
    p = prob.astype(np.float32) / 100.0
    cls[known & (p < free_thresh)] = 0
    cls[known & (p >= free_thresh) & (p < occ_thresh)] = 1
    cls[known & (p >= occ_thresh)] = 2
    return cls


def split_unknown(cls):
    """Flood fill from the border through soft cells: what it reaches is outdoors.

    Mirrors HM3DMap's rule -- observed-free cells stop the fill as well as
    occupied ones, since the camera stood in them, so they are interior by
    construction.
    """
    soft = (cls == 3) | (cls == 1)
    lab, n = label(soft)
    if n == 0:
        return np.zeros(cls.shape, dtype=bool)
    border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    outside_ids = set(int(x) for x in np.unique(border) if x != 0)
    exterior = np.isin(lab, list(outside_ids)) if outside_ids else np.zeros_like(soft)
    return exterior & soft


def clearance_field(cls, exterior, res, robot_radius, unknown_slack):
    """Reproduce HM3DMap.clearance: min(sd_known, sd_soft + slack) - radius."""
    hard = (cls == 2) | exterior
    soft = ((cls == 3) | (cls == 1)) & ~exterior

    def sd(mask):
        if not mask.any():
            return np.full(mask.shape, np.inf, dtype=np.float32)
        if mask.all():
            return np.full(mask.shape, -np.inf, dtype=np.float32)
        return ((distance_transform_edt(~mask) - distance_transform_edt(mask))
                * res).astype(np.float32)

    return np.minimum(sd(hard), sd(soft) + unknown_slack) - robot_radius


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--traj", default=None,
                    help="MASt3R trajectory .txt, drawn if the grid carries a "
                         "plane basis (optional)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--robot-radius", type=float, default=0.20)
    ap.add_argument("--unknown-slack", type=float, default=0.50)
    ap.add_argument("--no-exclude-exterior", dest="exclude_exterior",
                    action="store_false",
                    help="do not treat border-connected unknown as a hard "
                         "obstacle (matches plan_hm3d.py's flag of the same name)")
    ap.set_defaults(exclude_exterior=True)
    args = ap.parse_args()

    grid_path = pathlib.Path(args.grid)
    prob = np.load(grid_path)
    meta = read_yaml(grid_path.with_suffix(".yaml"))
    res = meta["resolution"]
    ox, oy = meta["origin"][0], meta["origin"][1]
    H, W = prob.shape
    # imshow with origin="lower" puts row 0 at the bottom, which matches the
    # grid convention: row 0 is the LOW-y edge.
    extent = [ox, ox + W * res, oy, oy + H * res]

    cls = classify(prob, meta["free_thresh"], meta["occupied_thresh"])
    exterior = (split_unknown(cls) if args.exclude_exterior
                else np.zeros(cls.shape, dtype=bool))
    clear = clearance_field(cls, exterior, res, args.robot_radius,
                            args.unknown_slack)

    navigable = clear > 0
    lab, n = label(navigable)
    if n:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        largest = lab == sizes.argmax()
    else:
        largest = np.zeros_like(navigable)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))

    # ---- panel 1: raw classes
    cmap1 = ListedColormap(["#ffffff", "#f0a860", "#101010", "#c8c8c8"])
    axes[0].imshow(cls, origin="lower", extent=extent, cmap=cmap1,
                   norm=BoundaryNorm([0, 1, 2, 3, 4], 4), interpolation="nearest")
    tot = cls.size
    axes[0].set_title(
        f"BEV occupancy  ({W}x{H} @ {res} m = {W*res:.1f} x {H*res:.1f} m)\n"
        f"free {100*(cls==0).mean():.1f}%   occupied {100*(cls==2).mean():.1f}%   "
        f"unknown {100*(cls==3).mean():.1f}%", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#888") for c in
               ["#ffffff", "#f0a860", "#101010", "#c8c8c8"]]
    axes[0].legend(handles, ["observed free", "undetermined", "occupied",
                             "unknown"], loc="upper right", fontsize=8)

    # ---- panel 2: unknown split into enclosed vs exterior
    cls2 = cls.copy().astype(np.int8)
    cls2[exterior] = 4
    cmap2 = ListedColormap(["#ffffff", "#f0a860", "#101010", "#c8c8c8", "#efe3c8"])
    axes[1].imshow(cls2, origin="lower", extent=extent, cmap=cmap2,
                   norm=BoundaryNorm([0, 1, 2, 3, 4, 5], 5), interpolation="nearest")
    axes[1].set_title(
        f"unknown split: enclosed (soft) vs exterior (hard obstacle)\n"
        f"exterior {100*exterior.mean():.1f}%   enclosed unknown "
        f"{100*(((cls==3)|(cls==1)) & ~exterior).mean():.1f}%", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#888") for c in
               ["#c8c8c8", "#efe3c8"]]
    axes[1].legend(handles, ["enclosed unknown", "exterior (blocks)"],
                   loc="upper right", fontsize=8)

    # ---- panel 3: clearance, the quantity that caps the tube
    shown = np.where(clear > 0, clear, np.nan)
    im = axes[2].imshow(shown, origin="lower", extent=extent, cmap="viridis",
                        interpolation="nearest")
    axes[2].imshow(np.where(clear <= 0, 1.0, np.nan), origin="lower",
                   extent=extent, cmap=ListedColormap(["#e8e8e8"]),
                   interpolation="nearest", zorder=0)
    plt.colorbar(im, ax=axes[2], fraction=0.046, label="clearance (m)")
    pos = clear[clear > 0]
    if pos.size:
        axes[2].set_title(
            f"clearance = min(d_hard, d_soft+{args.unknown_slack}) - "
            f"{args.robot_radius} m\n"
            f"navigable {100*navigable.mean():.1f}% of grid   "
            f"max {pos.max():.2f} m   median {np.median(pos):.2f} m", fontsize=11)
    else:
        axes[2].set_title("clearance: nothing navigable", fontsize=11)

    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")

    out = pathlib.Path(args.out) if args.out else grid_path.with_name(
        grid_path.stem + "_bev.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # The numbers behind the picture, so the tube width is explainable.
    print(f"\ngrid {W}x{H} @ {res} m  = {W*res:.1f} x {H*res:.1f} m")
    print(f"  observed free   {100*(cls==0).mean():5.1f}%  "
          f"({int((cls==0).sum())} cells = {(cls==0).sum()*res*res:.1f} m2)")
    print(f"  occupied        {100*(cls==2).mean():5.1f}%")
    print(f"  unknown         {100*(cls==3).mean():5.1f}%  "
          f"(exterior {100*exterior.mean():.1f}%)")
    print(f"  navigable       {100*navigable.mean():5.1f}%  "
          f"largest component {100*largest.mean():.1f}% "
          f"({largest.sum()*res*res:.1f} m2)")
    if pos.size:
        print(f"  clearance   max {pos.max():.3f} m   median {np.median(pos):.3f} m")
        print(f"  -> tube radius is capped at clearance - margin, so this is "
              f"the ceiling on tube width")


if __name__ == "__main__":
    main()
