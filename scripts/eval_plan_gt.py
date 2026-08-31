#!/usr/bin/env python3
"""Score SafeDiffuser plans against habitat's ground-truth navmesh.

    conda run -n habitat-vla python3 scripts/eval_plan_gt.py \
        --plans ../SafeDiffuser_STT/stt_results/hm3d/00807-rsggHU7g7dh_dstt.npz

The planner only ever sees the reconstructed occupancy grid, so its own
collision numbers say whether it respected the map it was given -- not whether
the map was right. This maps each planned point back into habitat world
coordinates with the Sim(3) stored in grids.json and asks the navmesh instead.

The gap between the two is the quantity of interest: it is SLAM and mapping
error expressed in the unit that matters for navigation, "how often does a plan
that looked safe on our map actually drive into something". Splitting it that
way is only meaningful because the planner scores ~0 on its own map; if it
collided there too, the two error sources would be mixed together.

One caveat on the comparison. Habitat's navigable set is where the agent's
CENTRE may stand, so it is already eroded by the agent radius and excludes
space under furniture and unreachable floor. A plan point landing outside it is
therefore not automatically a crash, which is why `off_navmesh` is reported
alongside a distance: `mean_dist_outside` says how far outside the navigable
set the offending points fall, and a few centimetres is the definitional
mismatch, while tens of centimetres is real error.
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    import habitat_sim
except ImportError:
    habitat_sim = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_hm3d_traj import build_sim, find_dataset_config, find_scenes  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HM3D_ROOT = os.path.join(REPO_ROOT, "datasets", "hm3d_root")


def plane_to_habitat(yx, g):
    """Plan points (y, x) in grid plane metres -> habitat world (x, y, z).

    Same chain as eval_hm3d_occupancy.cell_centres_to_habitat, but for
    arbitrary continuous points rather than cell centres: the plane basis lifts
    (x, y) into the reconstruction's 3D frame, then the Sim(3) rotation and
    translation carry it into habitat. No scale factor appears because the
    cloud was already made metric before the grid was built, and the Sim(3) was
    fitted with scale held at 1 for exactly that reason.
    """
    origin = np.array(g["origin"])
    u, v = np.array(g["u"]), np.array(g["v"])
    R, t = np.array(g["sim3_R"]), np.array(g["sim3_t"])
    yx = np.atleast_2d(yx)
    p = origin[None, :] + yx[:, 1:2] * u[None, :] + yx[:, 0:1] * v[None, :]
    return (R @ p.T).T + t


def score_plan(traj_yx, g, pathfinder, res):
    """Per-plan navmesh statistics for one trajectory."""
    q = plane_to_habitat(traj_yx, g)
    topdown = np.asarray(pathfinder.get_topdown_view(res, g["habitat_floor_y"]))
    bmin = pathfinder.get_bounds()[0]

    px = np.floor((q[:, 0] - bmin[0]) / res).astype(int)
    pz = np.floor((q[:, 2] - bmin[2]) / res).astype(int)
    inside = ((px >= 0) & (px < topdown.shape[1])
              & (pz >= 0) & (pz < topdown.shape[0]))

    nav = np.zeros(len(q), dtype=bool)
    nav[inside] = topdown[pz[inside], px[inside]]

    # How far outside the navigable set the bad points sit. Distance is
    # measured in the navmesh raster, so it is directly in metres.
    from scipy.ndimage import distance_transform_edt
    dist_out = distance_transform_edt(~topdown) * res
    d = np.full(len(q), np.nan)
    d[inside] = dist_out[pz[inside], px[inside]]
    bad = (~nav) & inside

    return {
        "off_navmesh": float((~nav).mean()),
        "outside_grid": float((~inside).mean()),
        "mean_dist_outside_m": float(d[bad].mean()) if bad.any() else 0.0,
        "max_dist_outside_m": float(np.nanmax(d)) if inside.any() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True, help="*.npz written by plan_hm3d.py")
    ap.add_argument("--run", default="hm3d/calib")
    ap.add_argument("--grids-name", default="grids")
    ap.add_argument("--hm3d-root", default=DEFAULT_HM3D_ROOT)
    ap.add_argument("--split", default="val")
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--json-out", default=None,
                    help="per-plan navmesh scores; defaults to <run>/plans/<stem>_gt.json")
    args = ap.parse_args()

    if habitat_sim is None:
        sys.exit("habitat_sim not importable; run with: conda run -n habitat-vla python3 ...")

    data = np.load(args.plans, allow_pickle=True)
    stem = str(data["stem"])
    traj = data["traj"]
    est = data["est_collision_frac"]

    grids_dir = os.path.join(REPO_ROOT, "logs", args.run, args.grids_name)
    with open(os.path.join(grids_dir, "grids.json")) as f:
        grids = json.load(f)
    match = [g for g in grids if g["stem"] == stem]
    if not match:
        sys.exit(f"{stem} not found in {grids_dir}/grids.json")
    g = match[0]

    scenes = {s["folder"]: s for s in find_scenes(args.hm3d_root, args.split)}
    if g["scene"] not in scenes:
        sys.exit(f"{g['scene']} not in split {args.split}")
    dataset_config = find_dataset_config(args.hm3d_root)
    sim = build_sim(scenes[g["scene"]]["glb"], dataset_config, 128, 90.0, 1.5)
    try:
        if not sim.pathfinder.is_loaded:
            sys.exit(f"{g['scene']}: no navmesh")
        print(f"{stem}: scoring {len(traj)} plans against the navmesh\n")
        print("%5s %14s %14s %18s %16s"
              % ("plan", "est_collision", "off_navmesh", "mean_dist_out(m)", "max_dist_out(m)"))
        rows = []
        for i, t in enumerate(traj):
            r = score_plan(t, g, sim.pathfinder, args.res)
            rows.append(r)
            print("%5d %13.1f%% %13.1f%% %18.3f %16.3f"
                  % (i, 100 * est[i], 100 * r["off_navmesh"],
                     r["mean_dist_outside_m"], r["max_dist_outside_m"]))
        print("\n" + "=" * 74)
        print("mean over %d plans:" % len(rows))
        print("  collision on OUR map      %.1f%%" % (100 * est.mean()))
        print("  off habitat navmesh       %.1f%%"
              % (100 * np.mean([r["off_navmesh"] for r in rows])))
        print("  mean distance outside     %.3f m"
              % np.mean([r["mean_dist_outside_m"] for r in rows]))
        print("  max distance outside      %.3f m"
              % np.max([r["max_dist_outside_m"] for r in rows]))
        json_out = args.json_out
        if json_out is None:
            # Beside the plans being scored, in this repo's logs tree.
            plans_dir = os.path.join(REPO_ROOT, "logs", args.run, "plans")
            os.makedirs(plans_dir, exist_ok=True)
            json_out = os.path.join(plans_dir, stem + "_gt.json")
        if json_out:
            with open(json_out, "w") as f:
                json.dump({"stem": stem, "plans": rows,
                           "est_collision_frac": est.tolist()}, f, indent=2)
            print("wrote", json_out)
    finally:
        sim.close()


if __name__ == "__main__":
    main()
