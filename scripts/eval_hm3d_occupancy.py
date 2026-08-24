#!/usr/bin/env python3
"""
Score the HM3D occupancy grids against habitat's ground-truth navmesh.

    conda run -n habitat-vla python3 scripts/eval_hm3d_occupancy.py --run hm3d/calib

Each grid cell is mapped back into habitat world coordinates using the plane
basis and Sim(3) stored in grids.json, then compared against
pathfinder.get_topdown_view() sliced at that storey's floor.

Metrics are computed over OBSERVED cells only (unknown cells are excluded) --
scoring unobserved space would just measure how far the camera walked.

Metric set, and why it is this one
----------------------------------
Free-space precision/recall/IoU describe coverage. They do NOT describe safety,
and for a navigation map safety is the point, so two more are reported:

  occ_rec   Of the obstacles the camera actually observed, how many did we mark
            occupied. Missing an obstacle is what causes a collision, and until
            now nothing in this pipeline measured it.
  DANGER    Of those same obstacles, how many did we mark FREE. This is the
            number that hurts: marking an obstacle "unknown" makes a planner
            cautious, marking it "free" drives the rover into it. Read this
            column first.

"Obstacle" is not simply ~navigable -- that also covers unreachable floor, space
under furniture and everything outside the building, none of which a robot can
hit. It is defined here as non-navigable cells within `--obstacle-band` of
navigable space: the surfaces bounding the space you can actually drive in.

One systematic bias to keep in mind: habitat's "navigable" is where the agent's
CENTRE may stand, so it is eroded by the agent radius and excludes space under
furniture. Our free space is line-of-sight floor, so raw free-space precision is
capped below 1.0 even for a perfect map. `prec_f` ("fair") erodes our free space
by the same agent radius before scoring, which removes that particular handicap
and separates real error from a definitional mismatch.

Grids whose observed navmesh slice is smaller than `--min-gt` cells are ramp
artifacts rather than storeys (00803's stairwell splits produce a few hundred
navigable cells) and are excluded from the headline median, reported separately.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

try:
    import habitat_sim
except ImportError:
    habitat_sim = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_hm3d_traj import build_sim, find_dataset_config, find_scenes  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HM3D_ROOT = os.path.join(REPO_ROOT, "datasets", "hm3d_root")

FREE_THRESH, OCC_THRESH = 25, 65


def cell_centres_to_habitat(g):
    """Grid cell centres -> habitat world (x, z), plus the in-grid mask."""
    H, W = g["shape"]
    res, lo = g["res"], np.array(g["lo"])
    origin = np.array(g["origin"])
    u, v = np.array(g["u"]), np.array(g["v"])
    R, t = np.array(g["sim3_R"]), np.array(g["sim3_t"])

    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = lo[0] + (cc.ravel() + 0.5) * res
    y = lo[1] + (rr.ravel() + 0.5) * res
    # plane coords -> scaled SLAM frame -> habitat world
    p = origin[None, :] + x[:, None] * u[None, :] + y[:, None] * v[None, :]
    q = (R @ p.T).T + t
    return q.reshape(H, W, 3)


def disk(radius_m, res):
    """Boolean disk structuring element of the given radius in metres."""
    r = max(1, int(round(radius_m / res)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r ** 2


def score_grid(g, pathfinder, res, obstacle_band=0.5, agent_radius=0.1):
    prob = np.load(os.path.join(
        REPO_ROOT, "logs", g["run"], g.get("grids_name", "grids"), g["stem"] + ".npy"))
    H, W = prob.shape
    q = cell_centres_to_habitat(g)

    topdown = pathfinder.get_topdown_view(res, g["habitat_floor_y"])
    bmin = pathfinder.get_bounds()[0]

    px = np.floor((q[:, :, 0] - bmin[0]) / res).astype(int)
    pz = np.floor((q[:, :, 2] - bmin[2]) / res).astype(int)
    inside = (px >= 0) & (px < topdown.shape[1]) & (pz >= 0) & (pz < topdown.shape[0])

    gt_nav = np.zeros((H, W), dtype=bool)
    gt_nav[inside] = topdown[pz[inside], px[inside]]

    known = (prob >= 0) & inside
    free = known & (prob < FREE_THRESH)
    occ = known & (prob >= OCC_THRESH)

    tp = int((free & gt_nav).sum())
    fp = int((free & ~gt_nav).sum())
    fn = int((known & ~free & gt_nav).sum())
    union = tp + fp + fn

    # Fair precision: habitat's navigable set is already eroded by the agent
    # radius, so erode ours by the same amount before comparing. Without this
    # every cell in the one-radius band along each wall counts as a false
    # positive no matter how good the map is.
    free_er = binary_erosion(free, structure=disk(agent_radius, res)) & known
    tp_f = int((free_er & gt_nav).sum())
    fp_f = int((free_er & ~gt_nav).sum())

    # Obstacles a robot could actually hit. Two exclusions matter:
    #   * cells buried inside walls or outside the building -- hence "within
    #     obstacle_band of navigable space";
    #   * the agent-radius collar. Habitat marks the strip along every wall
    #     non-navigable because the agent's CENTRE cannot sit there, but that
    #     strip is floor, not obstacle, and calling it free is correct. Dilating
    #     the navigable set back out by the agent radius recovers the real
    #     walkable floor before the obstacle band is taken. Without this the
    #     danger rate is inflated by the whole 2-cell collar around every room.
    gt_floor = binary_dilation(gt_nav, structure=disk(agent_radius, res))
    gt_obst = (~gt_floor) & binary_dilation(gt_nav, structure=disk(obstacle_band, res))
    gt_obst_obs = gt_obst & known
    n_obst = int(gt_obst_obs.sum())
    occ_hit = int((occ & gt_obst_obs).sum())
    free_on_obst = int((free & gt_obst_obs).sum())

    # The opposite error: drivable floor marked as an obstacle. This walls off
    # space the rover could have used, and enough of it makes a map useless
    # even though it is perfectly "safe". Scored against gt_floor -- navigable
    # space dilated back out by the agent radius -- because the collar habitat
    # erodes along every wall IS floor, and occ_precision, which scores against
    # plain ~navigable, counts blocking it as correct.
    floor_obs = gt_floor & known
    n_floor = int(floor_obs.sum())
    occ_on_floor = int((occ & floor_obs).sum())

    return {
        "px": px, "pz": pz, "td_shape": topdown.shape,
        "stem": g["stem"],
        "scene": g["scene"],
        "level": g["level"],
        "known_cells": int(known.sum()),
        "free_cells": int(free.sum()),
        "occ_cells": int(occ.sum()),
        "gt_nav_observed": int((known & gt_nav).sum()),
        "free_precision": tp / max(1, tp + fp),
        "free_precision_fair": tp_f / max(1, tp_f + fp_f),
        "free_recall": tp / max(1, tp + fn),
        "free_iou": tp / max(1, union),
        "occ_precision": int((occ & ~gt_nav).sum()) / max(1, int(occ.sum())),
        "gt_obstacle_observed": n_obst,
        "occ_recall": occ_hit / max(1, n_obst),
        "danger_rate": free_on_obst / max(1, n_obst),
        "gt_floor_observed": n_floor,
        "blocked_rate": occ_on_floor / max(1, n_floor),
        "false_obstacle_rate": occ_on_floor / max(1, int(occ.sum())),
    }, prob, gt_nav, known


def save_compare(prob, gt_nav, known, row, topdown, path, title):
    """Both panels drawn in habitat's own raster so they actually line up.

    The grid lives in its own floor-plane basis, rotated arbitrarily relative
    to habitat, so plotting the two arrays side by side compares different
    frames. Scattering our cells through the same (px, pz) mapping used for
    scoring puts them in register.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    px, pz = row["px"], row["pz"]
    free = known & (prob < FREE_THRESH)
    occ = known & (prob >= OCC_THRESH)

    ours = np.full(topdown.shape, 0.5)
    inb = known
    ours[pz[inb], px[inb]] = 0.6
    ours[pz[free], px[free]] = 1.0
    ours[pz[occ], px[occ]] = 0.0

    comp = np.full(topdown.shape + (3,), 40, dtype=np.uint8)
    agree = free & gt_nav
    false_free = free & ~gt_nav
    missed = known & ~free & gt_nav
    comp[pz[agree], px[agree]] = (60, 200, 60)
    comp[pz[false_free], px[false_free]] = (220, 80, 80)
    comp[pz[missed], px[missed]] = (80, 120, 230)

    # Crop to where either map has content, so the panels are readable.
    content = (ours != 0.5) | topdown
    rs, cs = np.where(content)
    if len(rs):
        r0, r1 = max(0, rs.min() - 5), min(topdown.shape[0], rs.max() + 6)
        c0, c1 = max(0, cs.min() - 5), min(topdown.shape[1], cs.max() + 6)
    else:
        r0, r1, c0, c1 = 0, topdown.shape[0], 0, topdown.shape[1]
    sl = (slice(r0, r1), slice(c0, c1))

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    ax[0].imshow(ours[sl][::-1], cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("SLAM occupancy (white=free, black=occ, grey=unknown)")
    ax[1].imshow(topdown[sl][::-1], cmap="gray")
    ax[1].set_title("habitat navmesh ground truth (white=navigable)")
    ax[2].imshow(comp[sl][::-1])
    ax[2].set_title("green=agree  red=false free  blue=missed")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="hm3d/calib")
    p.add_argument("--hm3d-root", default=DEFAULT_HM3D_ROOT)
    p.add_argument("--split", default="minival")
    p.add_argument("--res", type=float, default=0.05)
    p.add_argument("--grids-name", default="grids",
                   help="subdirectory of logs/<run>/ holding the grids to score")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--obstacle-band", type=float, default=0.5,
                   help="how far from navigable space a non-navigable cell "
                        "still counts as a hittable obstacle, metres")
    p.add_argument("--agent-radius", type=float, default=0.1,
                   help="radius habitat's navmesh was eroded by; ours is "
                        "eroded to match for the fair-precision column")
    p.add_argument("--min-gt", type=int, default=2000,
                   help="grids observing fewer navmesh cells than this are ramp "
                        "artifacts, not storeys; excluded from the median")
    args = p.parse_args()

    if habitat_sim is None:
        sys.exit("habitat_sim not importable; run with: conda run -n habitat-vla python3 ...")

    grids_dir = os.path.join(REPO_ROOT, "logs", args.run, args.grids_name)
    with open(os.path.join(grids_dir, "grids.json")) as f:
        grids = json.load(f)
    for g in grids:
        g["run"] = args.run
        g["grids_name"] = args.grids_name

    scenes = {s["folder"]: s for s in find_scenes(args.hm3d_root, args.split)}
    dataset_config = find_dataset_config(args.hm3d_root)

    rows = []
    for scene in sorted({g["scene"] for g in grids}):
        if scene not in scenes:
            print(f"{scene}: not in split, skipping")
            continue
        sim = build_sim(scenes[scene]["glb"], dataset_config, 128, 90.0, 1.5)
        try:
            if not sim.pathfinder.is_loaded:
                print(f"{scene}: no navmesh, skipping")
                continue
            for g in [x for x in grids if x["scene"] == scene]:
                row, prob, gt_nav, known = score_grid(
                    g, sim.pathfinder, args.res,
                    obstacle_band=args.obstacle_band,
                    agent_radius=args.agent_radius)
                topdown = sim.pathfinder.get_topdown_view(args.res, g["habitat_floor_y"])
                if not args.no_plots:
                    save_compare(
                        prob, gt_nav, known, row, np.asarray(topdown),
                        os.path.join(grids_dir, g["stem"] + "_vs_gt.png"),
                        f"{g['stem']}  (IoU {row['free_iou']:.2f})")
                row.pop("px"); row.pop("pz"); row.pop("td_shape")
                rows.append(row)
        finally:
            sim.close()

    for r in rows:
        r["degenerate"] = r["gt_nav_observed"] < args.min_gt

    hdr = (f"{'grid':30s} {'prec_f':>7s} {'recall':>7s} {'IoU':>6s} "
           f"{'occ_rec':>8s} {'DANGER':>7s} {'BLOCKED':>8s} {'falseObs':>9s}")

    def line(r):
        return (f"{r['stem']:30s} "
                f"{r['free_precision_fair']:7.3f} {r['free_recall']:7.3f} "
                f"{r['free_iou']:6.3f} {r['occ_recall']:8.3f} "
                f"{r['danger_rate']:7.3f} {r['blocked_rate']:8.3f} "
                f"{r['false_obstacle_rate']:9.3f}")

    def median_row(label, rs):
        return (f"{label:30s} "
                f"{np.median([r['free_precision_fair'] for r in rs]):7.3f} "
                f"{np.median([r['free_recall'] for r in rs]):7.3f} "
                f"{np.median([r['free_iou'] for r in rs]):6.3f} "
                f"{np.median([r['occ_recall'] for r in rs]):8.3f} "
                f"{np.median([r['danger_rate'] for r in rs]):7.3f} "
                f"{np.median([r['blocked_rate'] for r in rs]):8.3f} "
                f"{np.median([r['false_obstacle_rate'] for r in rs]):9.3f}")

    real = [r for r in rows if not r["degenerate"]]
    degen = [r for r in rows if r["degenerate"]]

    print(hdr)
    print("-" * len(hdr))
    for r in real:
        print(line(r))
    if real:
        print("-" * len(hdr))
        print(median_row(f"MEDIAN ({len(real)} storeys)", real))
    if degen:
        print()
        print(f"ramp artifacts (<{args.min_gt} navmesh cells observed), "
              f"excluded from the median:")
        for r in degen:
            print(line(r))

    out = os.path.join(grids_dir, "occupancy_metrics.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
