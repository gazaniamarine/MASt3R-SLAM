#!/usr/bin/env python3
"""Turn a MASt3R-SLAM point cloud + trajectory into a 2D occupancy grid.

    python scripts/occupancy_grid.py --ply logs/run1.ply --traj logs/run1.txt \
           --out logs/grid_run1

Writes <out>.pgm + <out>.yaml (the ROS map_server pair, ready for nav2's
map_server / AMCL) and <out>.npy (raw int8 occupancy, ROS convention:
0-100 = probability, -1 = unknown).

Why the plane fit is not optional
--------------------------------
MASt3R's world frame is the FIRST CAMERA, not gravity: +x right, +y DOWN,
+z forward, origin wherever the camera sat on frame 0. Nothing in the system
knows which way is up, so the floor is NOT z=0 -- it is an arbitrary plane.
A camera pitched a few degrees down makes the floor appear sloped, which
pushes distant floor points above any fixed height threshold and paints
phantom obstacles at the map edges. Fitting the plane is what removes that.

Deliberately depends only on numpy + plyfile, both already in the
mast3r-slam env. open3d is NOT required -- installing it risks dragging in a
numpy 2.x that breaks this repo's dataset loaders.
"""

import argparse
import pathlib
import sys

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

try:
    from plyfile import PlyData
except ImportError:
    sys.exit("plyfile missing. It ships with the mast3r-slam env -- "
             "check you are running that env's python.")


# ---------------------------------------------------------------- loading

def load_ply(path, extras=False):
    """Points, and optionally the per-point confidence and source keyframe.

    `extras` returns (pts, conf, kf_id); either may be None on a cloud written
    before those fields existed, so callers must handle their absence rather
    than assume a re-run has happened.
    """
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    finite = np.isfinite(pts).all(axis=1)
    if not extras:
        return pts[finite]
    names = v.dtype.names
    conf = np.asarray(v["conf"])[finite] if "conf" in names else None
    kf_id = np.asarray(v["kf_id"])[finite].astype(np.int32) if "kf_id" in names else None
    return pts[finite], conf, kf_id


def load_traj(path):
    """MASt3R writes: timestamp x y z qx qy qz qw (world<-camera)."""
    rows = np.loadtxt(str(path))
    if rows.ndim == 1:
        rows = rows[None]
    return rows[:, 1:4].astype(np.float64)


def voxel_downsample(pts, voxel, return_index=False):
    """Keep one point per occupied voxel. Pure numpy: quantise, then unique.

    `return_index` gives the surviving rows instead of the points, so per-point
    attributes (confidence, source keyframe) can be subset the same way.
    """
    if voxel <= 0:
        keep = np.arange(len(pts))
        return keep if return_index else pts
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(idx)
    return keep if return_index else pts[keep]


# ------------------------------------------------------------ plane fitting

def fit_plane_ransac(pts, thresh, iters, rng, up_axis=None, tol_deg=35.0):
    """Return (normal, point_on_plane, n_inliers) of the dominant FLOOR plane.

    Plain "largest plane" is not good enough. In a room the biggest planar
    surface is often a wall or a desktop, and picking one silently tilts the
    whole map. up_axis is a gravity prior -- MASt3R's world frame is the first
    camera with +y DOWN, so a level-ish mount makes +y a good guess at the
    floor normal. Hypotheses more than tol_deg off it are discarded outright
    rather than merely down-weighted.
    """
    n_pts = len(pts)
    if n_pts < 3:
        sys.exit("not enough points to fit a floor plane")

    cos_tol = np.cos(np.deg2rad(tol_deg))
    best_inliers, best, rejected = -1, None, 0
    for _ in range(iters):
        tri = pts[rng.choice(n_pts, 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        if up_axis is not None and abs(float(normal @ up_axis)) < cos_tol:
            rejected += 1
            continue          # sign is arbitrary here; orientation fixed later
        dist = np.abs((pts - tri[0]) @ normal)
        count = int((dist < thresh).sum())
        if count > best_inliers:
            best_inliers, best = count, (normal, tri[0])

    if best is None:
        sys.exit(f"no plane within {tol_deg} deg of the gravity prior "
                 f"({rejected}/{iters} hypotheses rejected). Either the camera "
                 f"was mounted far from level, or the floor is barely visible "
                 f"-- raise --gravity-tol, or check the cloud.")

    # Refit on the inlier set: the 3-point hypothesis is only a seed.
    normal, origin = best
    inliers = pts[np.abs((pts - origin) @ normal) < thresh]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid)
    return vh[2] / np.linalg.norm(vh[2]), centroid, len(inliers)


def plane_basis(normal, origin, cam_positions):
    """Orthonormal (u, v, n) with n pointing UP -- away from the floor."""
    # Cameras are always above the floor they drove on; use them to orient.
    if np.mean((cam_positions - origin) @ normal) < 0:
        normal = -normal
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ normal) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(seed, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v, normal


# ------------------------------------------------------------- ray casting

def bresenham_free(grid_logodds, r0, c0, r1, c1, l_free, stop_short,
                   max_steps=None, occ_mask=None):
    """March from (r0,c0) toward (r1,c1), decrementing log-odds along the way.

    stop_short leaves the last few cells untouched so the ray does not erase
    the very obstacle that terminated it. max_steps caps the range, which
    limits how far a ray that slips through a gap in the geometry can carve.

    occ_mask stops the ray at the first occupied cell it meets. Without it a ray
    aimed at a far wall marches straight through every obstacle in between and
    clears them all -- and since each ray subtracts l_free while a cell's
    obstacle points only add once each, a genuine wall crossed by enough rays is
    voted empty. Measured on HM3D, that mechanism was marking 32% of observed
    obstacles as free space. Light does not pass through walls and neither
    should these rays.
    """
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    steps = max(dr, dc) - stop_short
    if max_steps is not None:
        steps = min(steps, max_steps)
    r, c, n = r0, c0, 0
    H, W = grid_logodds.shape
    while n < steps:
        if 0 <= r < H and 0 <= c < W:
            # Stop before clearing an obstacle: everything past it is occluded
            # and this ray is no evidence about it either way.
            if occ_mask is not None and n > 0 and occ_mask[r, c]:
                return
            grid_logodds[r, c] += l_free
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        n += 1


# -------------------------------------------------------------------- output

def write_pgm_yaml(prob, out, res, origin_xy, occ_thresh, free_thresh):
    """ROS map_server pair. PGM: 254=free, 0=occupied, 205=unknown."""
    img = np.full(prob.shape, 205, dtype=np.uint8)
    img[prob >= 0] = 254
    img[(prob >= 0) & (prob >= occ_thresh * 100)] = 0
    known_free = (prob >= 0) & (prob < free_thresh * 100)
    img[known_free] = 254

    # PGM row 0 is the TOP of the image; ROS origin is the BOTTOM-left cell.
    img = np.flipud(img)
    pgm = out.with_suffix(".pgm")
    with open(pgm, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (img.shape[1], img.shape[0]))
        f.write(img.tobytes())

    with open(out.with_suffix(".yaml"), "w") as f:
        f.write(
            f"image: {pgm.name}\n"
            f"resolution: {res}\n"
            f"origin: [{origin_xy[0]:.4f}, {origin_xy[1]:.4f}, 0.0]\n"
            f"negate: 0\n"
            f"occupied_thresh: {occ_thresh}\n"
            f"free_thresh: {free_thresh}\n"
        )
    return pgm


def build_occupancy(pts, cams, *, res=0.05, voxel=0.03, min_h=0.10, max_h=1.50,
                    floor_radius=4.0, plane_thresh=0.03, ransac_iters=400,
                    gravity_tol=35.0, gravity_prior=True, seed=0, verbose=True,
                    max_ray=None, floor_support=False, floor_tol=0.12,
                    support_radius=0.30, floor_plane=None, occlusion=True,
                    kf_id=None, max_observers=4, min_cell_points=4,
                    min_obstacle_top=0.0, conf=None, conf_ref=0.0,
                    require_ground_contact=False, ground_band=0.25):
    """Point cloud + camera path -> int8 occupancy grid (ROS convention).

    Returns (prob, lo, info). `lo` is the bottom-left corner of the grid in
    plane coordinates; `info` carries the plane basis and floor origin so a
    caller can map cells back into the cloud's own frame.

    `kf_id` is the index of the keyframe that observed each point. When given,
    free space is carved from the poses that actually saw a surface instead of
    from whichever pose happens to lie nearest it.

    IMPORTANT: `kf_id` must index into `cams` as passed, not into the original
    trajectory. A caller handing in a subset of poses (hm3d_occupancy.py splits
    them per storey) has to remap first, using -1 for points whose observing
    pose is not in the subset. Silently indexing a subset with global ids would
    cast rays from arbitrary wrong positions.
    """
    rng = np.random.default_rng(seed)

    ds_idx = voxel_downsample(pts, voxel, return_index=True)
    pts = pts[ds_idx]
    if kf_id is not None:
        kf_id = np.asarray(kf_id)[ds_idx]
    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)[ds_idx]
    if verbose:
        print(f"voxel downsample @ {voxel} m -> {len(pts):,} points")

    # Fit the floor only near where the rover actually drove. Distant geometry
    # (walls, far clutter) otherwise dominates and can win the RANSAC vote.
    # KD-tree rather than a broadcast difference: the latter would allocate
    # len(pts) x len(cams) x 3 floats, which is gigabytes on a real cloud.
    tree = cKDTree(cams)
    near = pts[tree.query(pts, k=1)[0] < floor_radius]
    if len(near) < 1000:
        near = pts
    sample = near[rng.choice(len(near), min(len(near), 60000), replace=False)]

    up_axis = np.array([0.0, 1.0, 0.0])   # MASt3R world: +y is DOWN
    if floor_plane is not None:
        # Floor handed to us by the metric anchor: normal from the cameras' own
        # orientation, offset from the known camera mount height. Strictly
        # better than voting for it here -- on 00801 this RANSAC locks onto a
        # surface 0.95 m below the camera when the floor is 1.72 m down, on 3.9%
        # inliers, which silently lifts the whole obstacle height band by 0.55 m
        # and reclassifies every knee-high obstacle as floor.
        normal, origin = floor_plane
        origin = np.asarray(origin, dtype=float)
        u, v, n = plane_basis(np.asarray(normal, dtype=float), origin, cams)
        n_in = int((np.abs((sample - origin) @ n) < plane_thresh).sum())
    else:
        normal, origin, n_in = fit_plane_ransac(
            sample, plane_thresh, ransac_iters, rng,
            up_axis=up_axis if gravity_prior else None,
            tol_deg=gravity_tol)
        u, v, n = plane_basis(normal, origin, cams)
    tilt = np.rad2deg(np.arccos(min(1.0, abs(float(n @ up_axis)))))
    frac = 100.0 * n_in / len(sample)
    if verbose:
        print(f"floor plane: {n_in:,}/{len(sample):,} inliers ({frac:.1f}%), "
              f"normal={n.round(3)}, tilt vs gravity prior {tilt:.1f} deg")
        if frac < 15.0:
            print("  WARNING: low inlier fraction -- the floor may be poorly "
                  "observed. Check the .ply before trusting this grid.")

    rel = pts - origin
    height = rel @ n
    cam_h = (cams - origin) @ n
    if verbose:
        print(f"camera height above floor: mean {cam_h.mean():.2f} m "
              f"(min {cam_h.min():.2f}, max {cam_h.max():.2f})")

    keep = (height > min_h) & (height < max_h)
    obst = pts[keep]
    obst_h = height[keep]
    obst_kf = kf_id[keep] if kf_id is not None else None
    obst_conf = conf[keep] if conf is not None else None
    if verbose:
        print(f"height slice [{min_h}, {max_h}] m -> {len(obst):,} obstacle points")
    if len(obst) == 0:
        raise ValueError("no points in the obstacle height band")

    def to_plane(q):
        r = q - origin
        return np.stack([r @ u, r @ v], axis=1)

    obst2d = to_plane(obst)
    cam2d = to_plane(cams)

    allxy = np.vstack([obst2d, cam2d])
    lo = allxy.min(axis=0) - 5 * res
    hi = allxy.max(axis=0) + 5 * res
    W = int(np.ceil((hi[0] - lo[0]) / res))
    H = int(np.ceil((hi[1] - lo[1]) / res))
    if verbose:
        print(f"grid {W} x {H} cells @ {res} m ({W*res:.1f} x {H*res:.1f} m)")

    def to_cell(xy):
        c = np.floor((xy[:, 0] - lo[0]) / res).astype(int)
        r = np.floor((xy[:, 1] - lo[1]) / res).astype(int)
        return np.clip(r, 0, H - 1), np.clip(c, 0, W - 1)

    L = np.zeros((H, W), dtype=np.float32)
    L_FREE, L_OCC = -0.4, 0.85
    L_MIN, L_MAX = -2.0, 3.5

    orr, occ = to_cell(obst2d)
    crr, ccc = to_cell(cam2d)

    cam_cells = np.stack([crr, ccc], axis=1)
    stop_short = max(1, int(round(0.10 / res)))
    max_steps = None if max_ray is None else max(1, int(round(max_ray / res)))

    # Occupancy is known before any ray is cast, so rays can be occluded by it.
    # This has to be built up front rather than accumulated as we go: the result
    # would otherwise depend on the order cells happen to be visited in.
    # A cell becomes an obstacle only on enough evidence. One stray point used
    # to be enough, which matters more than it sounds: a phantom cell is also an
    # OCCLUDER, so a single bad point stops every ray behind it and costs the
    # free space in its shadow. Requiring min_cell_points separates a surface
    # from a speck without throwing away the low-confidence points that give
    # genuine surfaces their density.
    # Evidence per cell. Confidence weighting is available but OFF by default
    # (conf_ref <= 0), because it was measured and it loses. Compared against a
    # plain count at matched DANGER it is strictly worse -- at BLOCKED 0.32 it
    # costs 0.145 danger against 0.091 -- for the same reason filtering on
    # --min-conf loses: low-confidence points are still real geometry, and their
    # value here is as OCCLUDERS that stop rays punching through a thinly
    # reconstructed wall. Down-weighting them throws that away and buys nothing.
    if obst_conf is not None and conf_ref > 0:
        w = np.clip(obst_conf / conf_ref, 0.05, 1.0)
    else:
        w = np.ones(len(obst), dtype=np.float32)
    evidence = np.zeros((H, W), dtype=np.float32)
    np.add.at(evidence, (orr, occ), w)
    solid = evidence >= min_cell_points

    # On flat ground every obstacle rests on the floor, so an obstacle cell
    # should have something in its lowest slice; a floating blob is then noise.
    #
    # OFF by default, and NOT validated: HM3D is the wrong place to test it.
    # Those are furnished houses full of tables, counters and beds whose tops
    # have no ground contact in their own cell column, so the test deletes real
    # obstacles there and measures worse than a plain count (at BLOCKED 0.30 it
    # costs 0.178 danger against 0.091). That result says HM3D violates the
    # precondition, not that the idea is wrong. It needs a flat environment
    # where every obstacle really does rest on the ground -- which is exactly
    # the rover case, and where it should be measured before being trusted.
    if require_ground_contact:
        low = obst_h <= (min_h + ground_band)
        grounded = np.zeros((H, W), dtype=bool)
        grounded[orr[low], occ[low]] = True
        if verbose:
            print(f"ground-contact test (points within {ground_band} m of the "
                  f"floor): {int((solid & grounded).sum()):,}/{int(solid.sum()):,} "
                  "obstacle cells kept")
        solid &= grounded

    # A real obstacle has vertical extent; floor smear does not. Points lifted
    # over min_h by plane tilt or reconstruction noise form a thin sheet a few
    # centimetres above the floor, and they were blocking 40% of the drivable
    # area. Requiring a cell to contain something that reaches min_obstacle_top
    # discards the sheet while keeping the whole of a genuine obstacle -- base
    # included, which simply raising min_h would not do.
    if min_obstacle_top > min_h:
        tallest = np.zeros((H, W), dtype=np.float32)
        np.maximum.at(tallest, (orr, occ), obst_h.astype(np.float32))
        tall_enough = tallest >= min_obstacle_top
        if verbose:
            before = int(solid.sum())
            print(f"vertical-extent test (>={min_obstacle_top} m): "
                  f"{int((solid & tall_enough).sum()):,}/{before:,} obstacle "
                  "cells kept")
        solid &= tall_enough
    if verbose and min_cell_points > 1:
        kept = int(solid.sum())
        total = int((counts > 0).sum())
        print(f"obstacle cells with >={min_cell_points} points: "
              f"{kept:,}/{total:,} ({100 * kept / max(1, total):.1f}%)")

    # Points belonging to a cell that cleared the evidence bar. Needed by the
    # ray loops below as well as the occupancy accumulation further down.
    keep_pt = solid[orr, occ]

    occ_mask = solid
    mask_arg = occ_mask if occlusion else None

    if max_observers <= 0:
        obst_kf = None       # explicit request for the nearest-pose fallback

    if obst_kf is None:
        # Fallback: one ray per occupied cell from the nearest pose. Nearest is
        # not the same as observing -- the closest pose may be on the far side
        # of the wall -- but without kf_id in the .ply there is nothing better.
        occupied_cells = np.unique(np.stack([orr[keep_pt], occ[keep_pt]], axis=1),
                                   axis=0)
        if verbose:
            print(f"carving {len(occupied_cells):,} rays from nearest pose "
                  "(no kf_id in cloud)")
        for (r1, c1) in occupied_cells:
            d = (cam_cells[:, 0] - r1) ** 2 + (cam_cells[:, 1] - c1) ** 2
            r0, c0 = cam_cells[int(np.argmin(d))]
            bresenham_free(L, int(r0), int(c0), int(r1), int(c1), L_FREE,
                           stop_short, max_steps, mask_arg)
    else:
        # One ray per (occupied cell, observing keyframe). This is the line of
        # sight that actually existed, so the space it clears is space some
        # camera really looked through. Capped at max_observers per cell, taking
        # the keyframes that contributed the most points to it: a surface seen
        # by thirty poses does not need thirty nearly-identical rays.
        pairs, pair_n = np.unique(
            np.stack([orr[keep_pt], occ[keep_pt], obst_kf[keep_pt]], axis=1),
            axis=0, return_counts=True)
        order = np.lexsort((-pair_n, pairs[:, 1], pairs[:, 0]))
        pairs, pair_n = pairs[order], pair_n[order]
        cell_id = pairs[:, 0].astype(np.int64) * W + pairs[:, 1]
        # Rank within each cell, exploiting the sort above: first occurrence of
        # a cell resets the counter.
        first = np.concatenate([[True], cell_id[1:] != cell_id[:-1]])
        starts = np.flatnonzero(first)
        rank = np.arange(len(pairs)) - np.repeat(starts, np.diff(
            np.concatenate([starts, [len(pairs)]])))
        pairs = pairs[rank < max_observers]
        if verbose:
            print(f"carving {len(pairs):,} rays from observing keyframes "
                  f"(<={max_observers} per cell)")
        n_cams = len(cam_cells)
        for r1, c1, k in pairs:
            if k < 0 or k >= n_cams:   # observer not among the poses given
                continue
            r0, c0 = cam_cells[k]
            bresenham_free(L, int(r0), int(c0), int(r1), int(c1), L_FREE,
                           stop_short, max_steps, mask_arg)

    # Free space backed by observed floor, not just by a line drawn through a
    # gap. A ray that slips through a hole in the reconstruction sweeps a wide
    # wedge of "free" space beyond the wall; requiring floor support underneath
    # removes those fans, which are the dominant false-free error on HM3D.
    if floor_support:
        fmask = (height > -floor_tol) & (height <= min_h)
        if fmask.any():
            frr, fcc = to_cell(to_plane(pts[fmask]))
            sup = np.zeros((H, W), dtype=bool)
            sup[frr, fcc] = True
            rad = max(1, int(round(support_radius / res)))
            yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
            sup = binary_dilation(sup, structure=(xx**2 + yy**2 <= rad**2))
            L[(L < 0) & ~sup] = 0.0     # back to unknown, obstacles untouched
        elif verbose:
            print("  WARNING: no floor points found; skipping support gating")

    np.add.at(L, (orr[keep_pt], occ[keep_pt]), L_OCC)
    np.clip(L, L_MIN, L_MAX, out=L)

    prob = np.full((H, W), -1, dtype=np.int8)
    touched = L != 0
    p_occ = 1.0 - 1.0 / (1.0 + np.exp(L[touched]))
    prob[touched] = np.round(p_occ * 100).astype(np.int8)

    info = {"origin": origin, "u": u, "v": v, "n": n, "res": res,
            "inlier_frac": frac, "tilt_deg": tilt,
            "cam_h_mean": float(cam_h.mean()), "n_obstacle": int(len(obst))}
    return prob, lo, info


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ply", required=True)
    p.add_argument("--traj", required=True)
    p.add_argument("--out", required=True, help="output stem, no extension")
    p.add_argument("--voxel", type=float, default=0.03,
                   help="downsample size, metres (default 0.03)")
    p.add_argument("--res", type=float, default=0.05,
                   help="grid cell size, metres (default 0.05)")
    p.add_argument("--min-h", type=float, default=0.10,
                   help="floor clearance: below this is floor, not obstacle")
    p.add_argument("--max-h", type=float, default=1.50,
                   help="above this the rover drives underneath")
    p.add_argument("--scale", type=float, default=1.0,
                   help="global scale correction applied to cloud and path. "
                        "Prefer --cam-height, which measures it.")
    p.add_argument("--cam-height", type=float, default=None,
                   help="measured height of the camera above the floor, in "
                        "metres. Recovers metric scale from the reconstruction "
                        "itself -- no ground truth needed. MASt3R-SLAM's own "
                        "scale is off by a median 27%%, so without this the "
                        "grid's cell size is not the metres it claims.")
    p.add_argument("--floor-radius", type=float, default=4.0,
                   help="only fit the plane to points within this of the path")
    p.add_argument("--plane-thresh", type=float, default=0.03)
    p.add_argument("--ransac-iters", type=int, default=400)
    p.add_argument("--gravity-tol", type=float, default=35.0,
                   help="max degrees the floor normal may deviate from +y")
    p.add_argument("--no-gravity-prior", action="store_true",
                   help="accept the largest plane regardless of orientation")
    p.add_argument("--max-ray", type=float, default=None,
                   help="cap free-space ray length, metres (default: uncapped)")
    p.add_argument("--no-occlusion", dest="occlusion", action="store_false",
                   help="let free-space rays pass through obstacles (the old "
                        "behaviour). Kept for comparison only.")
    p.set_defaults(occlusion=True)
    p.add_argument("--no-floor-support", dest="floor_support",
                   action="store_false",
                   help="clear space on line of sight alone, without requiring "
                        "observed floor underneath. Raises recall and lowers "
                        "precision -- the wrong trade for navigation.")
    p.set_defaults(floor_support=True)
    p.add_argument("--min-conf", type=float, default=0.0,
                   help="drop points whose MASt3R confidence is below this")
    p.add_argument("--max-observers", type=int, default=4,
                   help="cap on free-space rays cast per occupied cell")
    p.add_argument("--min-cell-points", type=int, default=4,
                   help="points a cell needs before it counts as an obstacle")
    p.add_argument("--min-obstacle-top", type=float, default=0.0,
                   help="a cell is an obstacle only if something in it reaches "
                        "this high above the floor (0 disables)")
    p.add_argument("--conf-ref", type=float, default=0.0,
                   help="confidence counting as one full unit of evidence. "
                        "0 disables weighting (default; weighting measured worse)")
    p.add_argument("--ground-contact", action="store_true",
                   help="require obstacle cells to have points near the floor "
                        "(valid only where obstacles rest on flat ground)")
    p.add_argument("--ground-band", type=float, default=0.25,
                   help="how far above --min-h still counts as ground contact")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    pts, conf, kf_id = load_ply(args.ply, extras=True)
    cams = load_traj(args.traj)
    print(f"loaded {len(pts):,} points, {len(cams)} keyframe poses"
          f"{'' if conf is None else ' (with per-point confidence)'}")

    if args.min_conf > 0:
        if conf is None:
            print("  NOTE: this .ply predates per-point confidence; "
                  "--min-conf ignored.")
        else:
            sel = conf >= args.min_conf
            print(f"  confidence >= {args.min_conf}: keeping "
                  f"{100 * sel.mean():.1f}% of points")
            pts = pts[sel]
            kf_id = None if kf_id is None else kf_id[sel]

    if args.cam_height is not None:
        # Measure the scale off the known camera height rather than trusting
        # MASt3R's metric head, which is off by a median 27% (and by 123% at
        # worst) on these sequences.
        from metric_scale import anchor_reconstruction, camera_up, load_traj_full
        _, _, quats = load_traj_full(args.traj)
        pts, cams, est = anchor_reconstruction(
            pts, cams, quats, args.cam_height, correct_drift=False)
        if args.scale != 1.0:
            pts, cams = pts * args.scale, cams * args.scale
            print(f"  then multiplied by --scale {args.scale}")
        # The cloud is metric now, so the floor is a known plane: `cam_height`
        # below the cameras along their own up axis. Hand it to build_occupancy
        # rather than letting RANSAC hunt for it again.
        up = camera_up(quats)
        floor_plane = (up, np.median(cams, axis=0) - args.cam_height * up)
    else:
        pts, cams = pts * args.scale, cams * args.scale
        floor_plane = None
        if args.scale == 1.0:
            print("  NOTE: no --cam-height given, so this grid is in "
                  "MASt3R's arbitrary units, not metres.")

    prob, lo, _ = build_occupancy(
        pts, cams, res=args.res, voxel=args.voxel, min_h=args.min_h,
        max_h=args.max_h, floor_radius=args.floor_radius,
        plane_thresh=args.plane_thresh, ransac_iters=args.ransac_iters,
        gravity_tol=args.gravity_tol, gravity_prior=not args.no_gravity_prior,
        seed=args.seed, max_ray=args.max_ray, floor_support=args.floor_support,
        floor_plane=floor_plane, occlusion=args.occlusion,
        kf_id=kf_id, max_observers=args.max_observers,
        min_cell_points=args.min_cell_points,
        min_obstacle_top=args.min_obstacle_top,
        conf=conf, conf_ref=args.conf_ref,
        require_ground_contact=args.ground_contact,
        ground_band=args.ground_band)

    np.save(out.with_suffix(".npy"), prob)
    pgm = write_pgm_yaml(prob, out, args.res, lo, 0.65, 0.25)

    H, W = prob.shape
    known = int((prob >= 0).sum())
    occ_n = int((prob >= 65).sum())
    print(f"\nwrote {pgm}, {out.with_suffix('.yaml')}, {out.with_suffix('.npy')}")
    print(f"  known {known:,}/{H*W:,} cells ({100*known/(H*W):.1f}%), "
          f"{occ_n:,} occupied")
    print(f"  origin (bottom-left) = ({lo[0]:.3f}, {lo[1]:.3f}) m, res {args.res} m")


if __name__ == "__main__":
    main()
