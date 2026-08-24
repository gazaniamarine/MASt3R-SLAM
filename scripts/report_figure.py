#!/usr/bin/env python3
"""Occupancy-grid figures: one per scene, our map beside the ground truth.

    conda run -n habitat-vla python3 scripts/report_figure.py
    conda run -n habitat-vla python3 scripts/report_figure.py --mode agreement

Writes one PNG per grid into report_for_senior/grids/ (and, in agreement mode,
report_for_senior/agreement/). Two modes, answering two different questions:

  map        The grid drawn the way an occupancy grid is normally drawn --
             white free, black occupied, grey unknown -- beside habitat's floor
             plan for the same storey. "Does the map look like the building?"

  agreement  The same cells recoloured by whether they match ground truth.
             "Where is the map wrong, and does it matter?" -- which the map
             view cannot answer, because a wall in slightly the wrong place and
             a wall that is missing entirely are both just black.

Both panels of a figure are rasterised through habitat's own grid and cropped
to the same window, so they line up cell-for-cell instead of merely resembling
each other.

The ground-truth panel is blanked outside the region the camera observed. This
matters more than it sounds: the floor plan covers a whole house while a 30 m
tour reaches part of it, so an uncropped comparison reads as the map having
lost most of the building when it was simply never shown it.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_hm3d_occupancy import (  # noqa: E402
    FREE_THRESH, OCC_THRESH, cell_centres_to_habitat, disk,
)
from render_hm3d_traj import build_sim, find_dataset_config, find_scenes  # noqa: E402

try:
    import habitat_sim  # noqa: F401
except ImportError:
    sys.exit("needs habitat: conda run -n habitat-vla python3 ...")

from scipy.ndimage import binary_dilation  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Greys as ROS map_server would draw them, plus one for "seen but undecided".
C_FREE, C_OCC, C_UNKNOWN, C_MAYBE = 255, 0, 150, 205


def rasterise(g, prob, pathfinder, res, obstacle_band=0.5, agent_radius=0.1):
    """Everything needed to draw one scene, all in habitat's raster.

    Returns (layers, metrics). `layers` holds our map, the ground truth over
    the observed region, and the agreement image; `metrics` the two numbers
    that matter for navigation.
    """
    q = cell_centres_to_habitat(g)
    topdown = np.asarray(pathfinder.get_topdown_view(res, g["habitat_floor_y"]))
    bmin = pathfinder.get_bounds()[0]

    px = np.floor((q[:, :, 0] - bmin[0]) / res).astype(int)
    pz = np.floor((q[:, :, 2] - bmin[2]) / res).astype(int)
    inside = ((px >= 0) & (px < topdown.shape[1])
              & (pz >= 0) & (pz < topdown.shape[0]))

    gt_nav = np.zeros(prob.shape, dtype=bool)
    gt_nav[inside] = topdown[pz[inside], px[inside]]

    known = (prob >= 0) & inside
    free = known & (prob < FREE_THRESH)
    occ = known & (prob >= OCC_THRESH)
    maybe = known & ~free & ~occ

    ours = np.full(topdown.shape, C_UNKNOWN, dtype=np.uint8)
    ours[pz[maybe], px[maybe]] = C_MAYBE
    ours[pz[free], px[free]] = C_FREE
    ours[pz[occ], px[occ]] = C_OCC

    seen = np.zeros(topdown.shape, dtype=bool)
    seen[pz[known], px[known]] = True
    truth = np.where(seen, np.where(topdown, C_FREE, C_OCC), C_UNKNOWN)

    # Obstacles a rover could hit: non-navigable, bounding drivable space, and
    # not merely the collar habitat erodes by the agent radius (that strip is
    # floor, and calling it free is correct).
    gt_floor = binary_dilation(gt_nav, structure=disk(agent_radius, res))
    gt_obst = (~gt_floor) & binary_dilation(gt_nav, structure=disk(obstacle_band, res))
    obst_obs = gt_obst & known

    agree = np.full(topdown.shape + (3,), 30, dtype=np.uint8)
    agree[topdown] = (72, 72, 72)
    ok_free, hit, bad = free & gt_nav, occ & obst_obs, free & obst_obs
    agree[pz[ok_free], px[ok_free]] = (70, 200, 90)
    agree[pz[hit], px[hit]] = (245, 245, 245)
    agree[pz[bad], px[bad]] = (235, 60, 60)

    n = max(1, int(obst_obs.sum()))
    return ({"ours": ours.astype(np.uint8), "truth": truth.astype(np.uint8),
             "agree": agree, "seen": seen},
            {"occ_recall": int(hit.sum()) / n, "danger": int(bad.sum()) / n})


def crop_to(seen, pad=10):
    """Window covering the observed region, shared by every panel."""
    rs, cs = np.where(seen)
    if not len(rs):
        return slice(None), slice(None)
    return (slice(max(0, rs.min() - pad), rs.max() + pad + 1),
            slice(max(0, cs.min() - pad), cs.max() + pad + 1))


def scale_bar(ax, res, ncols, metres=5.0):
    from matplotlib.patches import Rectangle
    cells = metres / res
    x0, y0 = 0.05 * ncols, 0.055 * ncols
    ax.add_patch(Rectangle((x0, y0), cells, max(1.5, 0.005 * ncols),
                           color="#c81e1e", zorder=5))
    ax.text(x0 + cells / 2, y0 - 0.018 * ncols, f"{metres:g} m", color="#c81e1e",
            fontsize=9, ha="center", va="bottom", zorder=5)


def draw(stem, layers, metrics, res, mode, out_path, plt):
    from matplotlib.patches import Patch

    sl = crop_to(layers["seen"])
    if mode == "map":
        panels = [(layers["ours"], "Our occupancy grid  —  from MASt3R-SLAM"),
                  (layers["truth"], "Ground truth  —  same area, same scale")]
        legend = [Patch(facecolor="white", edgecolor="#999999",
                        label="free / drivable"),
                  Patch(color="black", label="occupied (obstacle)"),
                  Patch(color="#969696", label="not observed by the camera")]
    else:
        panels = [(layers["agree"], "Where the map is right, and where it is wrong")]
        legend = [Patch(color="#f5f5f5", label="obstacle correctly mapped"),
                  Patch(color="#46c85a", label="free space correctly mapped"),
                  Patch(color="#eb3c3c", label="obstacle mapped as free (dangerous)"),
                  Patch(color="#484848", label="navigable but never observed")]

    h, w = layers["ours"][sl].shape
    per = 7.0
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(per * len(panels), per * (h / max(1, w)) + 2.0))
    for ax, (img, title) in zip(axes[0], panels):
        colour = img.ndim == 3
        ax.imshow(img[sl][::-1], cmap=None if colour else "gray",
                  vmin=None if colour else 0, vmax=None if colour else 255,
                  interpolation="nearest")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#bbbbbb")
        scale_bar(ax, res, w)

    fig.suptitle(
        f"{stem}\nobstacles correctly mapped {100 * metrics['occ_recall']:.0f}%"
        f"      wrongly mapped as drivable {100 * metrics['danger']:.0f}%",
        fontsize=13)
    # Four entries do not fit across a single-panel figure; wrap them instead
    # of letting the row run off both edges.
    ncol = len(legend) if len(panels) > 1 or len(legend) <= 3 else 2
    nrow = (len(legend) + ncol - 1) // ncol
    fig.legend(handles=legend, loc="lower center", ncol=ncol,
               fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.035 + 0.03 * nrow, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="map", choices=["map", "agreement"])
    p.add_argument("--run", default="hm3d/calib")
    p.add_argument("--grids-name", default="grids")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="scene folders to draw (default: every grid in the run)")
    p.add_argument("--hm3d-root", default=os.path.join(
        REPO_ROOT, "datasets", "hm3d_root"))
    p.add_argument("--split", default="minival")
    p.add_argument("--res", type=float, default=0.05)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(REPO_ROOT, "report_for_senior",
                                    "grids" if args.mode == "map" else "agreement")
    os.makedirs(args.out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gdir = os.path.join(REPO_ROOT, "logs", args.run, args.grids_name)
    with open(os.path.join(gdir, "grids.json")) as f:
        grids = json.load(f)
    if args.scenes:
        grids = [g for g in grids if g["scene"] in args.scenes]

    scenes = {s["folder"]: s for s in find_scenes(args.hm3d_root, args.split)}
    dataset_config = find_dataset_config(args.hm3d_root)

    # One simulator per scene, reused across that scene's storeys: building it
    # is by far the slow part.
    n = 0
    for scene in sorted({g["scene"] for g in grids}):
        if scene not in scenes:
            print(f"  {scene}: not in split, skipping")
            continue
        sim = build_sim(scenes[scene]["glb"], dataset_config, 128, 90.0, 1.5)
        try:
            for g in [x for x in grids if x["scene"] == scene]:
                prob = np.load(os.path.join(gdir, g["stem"] + ".npy"))
                layers, metrics = rasterise(g, prob, sim.pathfinder, args.res)
                if not layers["seen"].any():
                    print(f"  {g['stem']}: nothing observed, skipping")
                    continue
                out = os.path.join(args.out_dir, g["stem"] + ".png")
                draw(g["stem"], layers, metrics, args.res, args.mode, out, plt)
                n += 1
                print(f"  {g['stem']}: obstacles {100 * metrics['occ_recall']:.0f}% "
                      f"correct, {100 * metrics['danger']:.0f}% wrongly drivable")
        finally:
            sim.close()
    print(f"\nwrote {n} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
