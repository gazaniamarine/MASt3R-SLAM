#!/usr/bin/env python3
"""
Build 2D occupancy grids (BEV maps) from the HM3D MASt3R-SLAM reconstructions.

    python3 scripts/hm3d_occupancy.py --run hm3d/calib

Two things this adds over calling occupancy_grid.py directly:

Scale correction
    MASt3R's metric head does not recover real-world scale -- on these scenes
    the reconstruction is off by a median 27%, and by 123% at worst, so a raw
    grid's cell size is not the metres it claims. Scale is measured from the
    known 1.5 m render camera height by scripts/metric_scale.py, which needs no
    ground truth and so is the same path the rover uses.

    Using the habitat ground truth instead was tried and is WORSE -- see
    --scale-mode. Ground truth gives the scale that best aligns the trajectory,
    which is not the scale that makes local geometry metric once the
    reconstruction is warped, and the grid depends on the latter.

Per-storey splitting
    Four of the ten tours climb stairs. Flattening a two-storey tour into one
    grid prints the upper floor's walls on top of the lower floor's free space,
    and it wrecks the floor-plane fit (00809 fitted with 6% inliers and reported
    camera heights spanning 0.83-3.94 m). Camera heights are clustered into
    levels and each level is gridded separately.

Writes, per level, the ROS map_server pair (.pgm/.yaml), the raw int8 grid
(.npy), a preview .png, and grids.json -- which carries the plane basis and the
Sim(3) rotation/translation needed to map cells back into habitat world
coordinates for validation against the ground-truth navmesh.
"""
import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from occupancy_grid import (  # noqa: E402
    build_occupancy,
    load_ply,
    load_traj,
    write_pgm_yaml,
)
from eval_hm3d import associate, read_tum, umeyama  # noqa: E402
from metric_scale import (  # noqa: E402
    anchor_reconstruction,
    camera_up,
    deform,
    load_traj_full,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CAM_HEIGHT = 1.5  # metres above the floor, as rendered by render_hm3d_traj.py


def cluster_levels(cam_h, sep=1.5, band=0.5, min_count=5):
    """Group camera heights into storeys by greedy density-peak picking.

    A gap-based split does not work here: stairs put keyframes at every height
    in between, so the distribution is continuous and everything collapses into
    one cluster. What separates storeys is density, not gaps -- the camera
    dwells on flat floors and only passes through the stairs. So: take the
    densest height, claim everything within `sep` of it as that storey, repeat.

    `sep` at 1.5 m keeps a real storey apart from the next while treating a
    short flight of steps within one floor (00807 spans 1.2 m) as a single
    level.
    """
    remaining = np.sort(np.asarray(cam_h, dtype=float))
    levels = []
    while len(remaining) >= min_count:
        # Density of each candidate height = neighbours within +/- band.
        counts = np.array([np.sum(np.abs(remaining - h) < band) for h in remaining])
        peak = remaining[int(np.argmax(counts))]
        core = remaining[np.abs(remaining - peak) < band]
        levels.append(float(core.mean()))
        remaining = remaining[np.abs(remaining - levels[-1]) >= sep]
    return sorted(levels)


def gt_scale_profile(traj_path, gt_path, n_kf, window=8):
    """Per-keyframe scale read off ground truth. Oracle only.

    Scale at keyframe k is the ratio of ground-truth to estimated path length
    over a window of keyframes centred on it -- a local quantity, unlike the
    single Sim(3) scale, so it tracks a reconstruction whose scale drifts.
    Values are interpolated out to every keyframe, since only the timestamped
    subset associates with ground truth.
    """
    t_est, x_est = read_tum(traj_path)
    t_gt, x_gt = read_tum(gt_path)
    i, j = associate(t_est, t_gt)
    e, g = x_est[i], x_gt[j]

    s = np.ones(len(e))
    for m in range(len(e)):
        a, b = max(0, m - window), min(len(e), m + window + 1)
        le = np.linalg.norm(np.diff(e[a:b], axis=0), axis=1).sum()
        lg = np.linalg.norm(np.diff(g[a:b], axis=0), axis=1).sum()
        if le > 1e-6 and lg > 1e-6:
            s[m] = lg / le
    return np.interp(np.arange(n_kf), i, s)


def save_preview(prob, path):
    """Grey preview: white = free, black = occupied, mid-grey = unknown."""
    from PIL import Image

    img = np.full(prob.shape, 128, dtype=np.uint8)
    img[(prob >= 0) & (prob < 25)] = 255
    img[prob >= 65] = 0
    between = (prob >= 25) & (prob < 65)
    img[between] = 190
    # Grid row 0 is the low-y edge; flip so the preview reads like a map.
    Image.fromarray(img[::-1]).save(path)


def process_scene(scene, run_dir, seq_dir, args):
    ply_path = run_dir / f"{scene}.ply"
    traj_path = run_dir / f"{scene}.txt"
    gt_path = seq_dir / scene / "groundtruth.txt"
    if not (ply_path.exists() and traj_path.exists() and gt_path.exists()):
        print(f"  skipping {scene}: missing inputs")
        return []

    pts, conf, kf_id = load_ply(ply_path, extras=True)
    cams = load_traj(traj_path)

    if conf is None and args.min_conf > 0:
        print(f"  NOTE: {ply_path.name} predates per-point confidence; "
              "--min-conf ignored. Re-run scripts/eval_hm3d.sh to add it.")
    elif conf is not None and args.min_conf > 0:
        sel = conf >= args.min_conf
        print(f"  confidence >= {args.min_conf}: keeping {100 * sel.mean():.1f}% "
              f"of {len(pts):,} points")
        pts, conf = pts[sel], conf[sel]
        if kf_id is not None:
            kf_id = kf_id[sel]

    # `up` is needed by every mode, not just the anchored one: it pins the floor
    # plane and splits the storeys. Deriving it from the cameras costs nothing
    # and is unrelated to where the SCALE comes from, so all three modes get it.
    # (An earlier version computed it only on the anchored path, which made the
    # ground-truth comparison unfair -- it was measuring scale source AND floor
    # source at once.)
    _, _, quats = load_traj_full(traj_path)
    up = camera_up(quats)

    if args.scale_mode == "gt-global":
        # One Sim(3) scale for the whole run, fitted against ground truth.
        t_est, x_est = read_tum(traj_path)
        t_gt, x_gt = read_tum(gt_path)
        i, j = associate(t_est, t_gt)
        s = umeyama(x_est[i], x_gt[j], with_scale=True)[0]
        pts, cams = pts * s, cams * s
        print(f"  {len(pts):,} points, {len(cams)} poses, GT global scale {s:.3f}")
    elif args.scale_mode == "gt-profile":
        # Oracle: a per-keyframe scale read straight off ground truth, then the
        # same deformation the anchor would apply. Not a shippable mode -- it is
        # the ceiling on what any scale correction could achieve, which is the
        # number that says whether scale is still the thing holding the map back.
        prof = gt_scale_profile(traj_path, gt_path, len(cams))
        pts, cams = deform(pts, cams, prof)
        print(f"  {len(pts):,} points, {len(cams)} poses, GT profile scale "
              f"{prof.min():.3f}-{prof.max():.3f} (median {np.median(prof):.3f})")
    else:
        # Measure the scale from the camera height. This is the path the rover
        # uses, where no ground truth exists.
        pts, cams, est = anchor_reconstruction(pts, cams, quats, CAM_HEIGHT,
                                               correct_drift=args.correct_drift)

    # Sim(3) to habitat world, for validation only: with the cloud already
    # metric this is a rigid transform, so it is fitted with scale fixed at 1.
    # Letting it float would quietly re-absorb any scale error the anchor left
    # behind and make the map look better than it is.
    t_est, x_est = read_tum(traj_path)
    t_gt, x_gt = read_tum(gt_path)
    i, j = associate(t_est, t_gt)
    s_res = np.linalg.norm(cams[i][1:] - cams[i][:-1], axis=1).sum()
    s_res /= max(np.linalg.norm(x_est[i][1:] - x_est[i][:-1], axis=1).sum(), 1e-9)
    _, R, t = umeyama(x_est[i] * s_res, x_gt[j], with_scale=False)
    s = s_res

    # Storey heights come from a known-good up axis, never from a plane fit on
    # the merged cloud. Fitting a global plane to pick out storeys is
    # unreliable: RANSAC can lock onto an upper floor or a ceiling, and
    # plane_basis's "cameras sit above the floor" heuristic then orients up
    # backwards -- measured against ground truth, 00809 came out at corr -0.998
    # (fully inverted) and 00800 at -0.495. Each level's own floor plane is
    # still fitted locally inside build_occupancy, where all the cameras really
    # are ~1.5 m above that one floor.
    # Heights are offsets along the reconstruction's own up axis rather than
    # habitat's, which is fine -- only differences between storeys matter here.
    cam_h, pt_h = cams @ up, pts @ up
    levels = cluster_levels(cam_h)
    print(f"  storeys detected: {len(levels)} at reconstruction heights "
          f"{[round(l, 2) for l in levels]}")

    results = []
    for k, lv in enumerate(levels):
        # Cameras belonging to this storey, and everything from its floor up to
        # its ceiling. The floor must be included or the per-level plane fit has
        # nothing to lock onto.
        cam_mask = np.abs(cam_h - lv) < args.level_tol
        if cam_mask.sum() < 5:
            print(f"  level {k}: only {int(cam_mask.sum())} poses, skipping")
            continue
        floor_h = lv - CAM_HEIGHT
        pt_mask = (pt_h > floor_h - 0.35) & (pt_h < floor_h + args.ceiling)

        # Where this storey's floor sits in habitat's frame, which is what the
        # navmesh slice in eval_hm3d_occupancy.py needs. Derived from this
        # level's own cameras rather than from `lv`, because `lv` is measured
        # along the reconstruction's up axis whenever the scale was anchored
        # rather than taken from ground truth.
        lvl_cams_hab = (R @ cams[cam_mask].T).T + t
        habitat_floor_y = float(np.median(lvl_cams_hab[:, 1]) - CAM_HEIGHT)

        sub_pts, sub_cams = pts[pt_mask], cams[cam_mask]
        if len(sub_pts) < 5000:
            print(f"  level {k}: only {len(sub_pts)} points, skipping")
            continue

        # build_occupancy indexes `sub_cams`, so global keyframe ids have to be
        # renumbered into that subset. Points observed from another storey get
        # -1 and are carved from nothing rather than from an unrelated pose.
        sub_conf = conf[pt_mask] if conf is not None else None
        sub_kf = None
        if kf_id is not None:
            lut = np.full(len(cams), -1, dtype=np.int32)
            lut[np.flatnonzero(cam_mask)] = np.arange(int(cam_mask.sum()))
            sub_kf = lut[np.clip(kf_id[pt_mask], 0, len(cams) - 1)]

        stem = scene if len(levels) == 1 else f"{scene}_level{k}"
        print(f"  level {k} (floor {floor_h:+.2f} m): "
              f"{len(sub_pts):,} pts, {len(sub_cams)} poses -> {stem}")
        # This storey's floor: CAM_HEIGHT below its own cameras along the
        # cameras' up axis. Valid in every scale mode, since a correctly scaled
        # reconstruction puts the camera CAM_HEIGHT above the floor by
        # definition -- that is what "correctly scaled" means here.
        level_floor = (up, np.median(sub_cams, axis=0) - CAM_HEIGHT * up)

        try:
            prob, lo, info = build_occupancy(
                sub_pts, sub_cams, res=args.res, voxel=args.voxel,
                min_h=args.min_h, max_h=args.max_h,
                floor_radius=args.floor_radius, plane_thresh=args.plane_thresh,
                ransac_iters=args.ransac_iters, gravity_tol=args.gravity_tol,
                seed=args.seed, verbose=args.verbose,
                max_ray=args.max_ray, floor_support=args.floor_support,
                floor_plane=level_floor, occlusion=args.occlusion,
                kf_id=sub_kf, max_observers=args.max_observers,
                min_cell_points=args.min_cell_points,
                min_obstacle_top=args.min_obstacle_top,
                conf=sub_conf, conf_ref=args.conf_ref,
                require_ground_contact=args.ground_contact,
                ground_band=args.ground_band)
        except (ValueError, SystemExit) as e:
            print(f"    FAILED: {e}")
            continue

        out = run_dir / args.grids_name / stem
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out.with_suffix(".npy"), prob)
        write_pgm_yaml(prob, out, args.res, lo, 0.65, 0.25)
        save_preview(prob, out.with_suffix(".png"))

        H, W = prob.shape
        known = int((prob >= 0).sum())
        # Habitat height of this storey's floor, for the navmesh slice.
        results.append({
            "scene": scene,
            "stem": stem,
            "level": k,
            "shape": [int(H), int(W)],
            "res": args.res,
            "lo": [float(lo[0]), float(lo[1])],
            "origin": info["origin"].tolist(),
            "u": info["u"].tolist(),
            "v": info["v"].tolist(),
            "n": info["n"].tolist(),
            "sim3_scale": float(s),
            "sim3_R": R.tolist(),
            "sim3_t": t.tolist(),
            "habitat_floor_y": float(habitat_floor_y),
            "known_cells": known,
            "known_frac": float(known / (H * W)),
            "occupied_cells": int((prob >= 65).sum()),
            "free_cells": int(((prob >= 0) & (prob < 25)).sum()),
            "inlier_frac": float(info["inlier_frac"]),
            "tilt_deg": float(info["tilt_deg"]),
        })
        print(f"    grid {W}x{H}, known {100*known/(H*W):.1f}%, "
              f"floor inliers {info['inlier_frac']:.1f}%")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="hm3d/calib")
    p.add_argument("--seqs", default=str(REPO_ROOT / "datasets" / "hm3d_seqs"))
    p.add_argument("--scene", default=None, help="Single scene (default: all).")
    p.add_argument("--res", type=float, default=0.05)
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--min-h", type=float, default=0.10)
    p.add_argument("--max-h", type=float, default=1.50)
    p.add_argument("--ceiling", type=float, default=2.60,
                   help="metres above a storey's floor to include in its slab")
    p.add_argument("--level-tol", type=float, default=1.00,
                   help="max distance from a level centre for a pose to belong to it")
    p.add_argument("--floor-radius", type=float, default=4.0)
    p.add_argument("--plane-thresh", type=float, default=0.03)
    p.add_argument("--ransac-iters", type=int, default=400)
    p.add_argument("--gravity-tol", type=float, default=35.0)
    p.add_argument("--max-ray", type=float, default=6.0,
                   help="cap free-space ray length, metres")
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="drop points whose MASt3R confidence is below this. "
                        "Needs a .ply written after per-point confidence was "
                        "added; ignored with a note on older clouds.")
    p.add_argument("--grids-name", default="grids",
                   help="subdirectory of logs/<run>/ to write into. Use a "
                        "different name to build a comparison set without "
                        "overwriting the grids you already trust.")
    p.add_argument("--conf-ref", type=float, default=0.0,
                   help="confidence counting as one full unit of evidence. "
                        "0 disables weighting, which is the default: measured "
                        "against a plain count at matched DANGER, weighting "
                        "loses (see occupancy_grid.build_occupancy).")
    p.add_argument("--ground-contact", action="store_true",
                   help="require obstacle cells to have points near the floor. "
                        "Valid when every obstacle rests on flat ground; would "
                        "discard genuine overhangs such as tabletops.")
    p.add_argument("--ground-band", type=float, default=0.25,
                   help="how far above --min-h still counts as ground contact")
    p.add_argument("--min-obstacle-top", type=float, default=0.0,
                   help="a cell counts as an obstacle only if something in it "
                        "reaches this high above the floor. Rejects floor "
                        "smear without clipping the base off real obstacles. "
                        "0 disables the test.")
    p.add_argument("--min-cell-points", type=int, default=4,
                   help="points a cell needs before it counts as an obstacle. "
                        "Above 1 this suppresses single-point phantoms, which "
                        "matter doubly because a phantom also blocks rays.")
    p.add_argument("--max-observers", type=int, default=4,
                   help="cap on free-space rays cast per occupied cell, taking "
                        "the keyframes that saw it most. 0 falls back to one "
                        "ray from the nearest pose, ignoring kf_id.")
    p.add_argument("--no-occlusion", dest="occlusion", action="store_false",
                   help="let free-space rays pass through obstacles (the old "
                        "behaviour). For comparison only.")
    p.set_defaults(occlusion=True)
    p.add_argument("--no-floor-support", dest="floor_support",
                   action="store_false",
                   help="clear space on line of sight alone, without requiring "
                        "observed floor underneath")
    p.set_defaults(floor_support=True)
    p.add_argument("--scale-mode", default="anchor",
                   choices=["anchor", "gt-global", "gt-profile"],
                   help="where scale comes from. 'anchor' measures it from the "
                        "camera height (the only one available on a real run); "
                        "'gt-global' takes one Sim(3) scale from ground truth; "
                        "'gt-profile' takes a per-keyframe scale from ground "
                        "truth, which is the oracle ceiling. The two gt modes "
                        "are for comparison only.")
    p.add_argument("--correct-drift", action="store_true",
                   help="also apply the EXPERIMENTAL per-keyframe drift "
                        "correction (see metric_scale.deform)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    run_dir = REPO_ROOT / "logs" / args.run
    seq_dir = pathlib.Path(args.seqs)
    if not run_dir.is_dir():
        raise SystemExit(f"no such run: {run_dir}")

    scenes = sorted(f.stem for f in run_dir.glob("*.ply"))
    if args.scene:
        scenes = [s for s in scenes if s == args.scene]
        if not scenes:
            raise SystemExit(f"scene {args.scene} not found in {run_dir}")

    all_results = []
    for scene in scenes:
        print(f"[{scene}]")
        all_results.extend(process_scene(scene, run_dir, seq_dir, args))

    out_json = run_dir / args.grids_name / "grids.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{len(all_results)} grids from {len(scenes)} scenes -> {out_json}")


if __name__ == "__main__":
    main()
