#!/usr/bin/env python3
"""Recover metric scale for a MASt3R-SLAM reconstruction without ground truth.

Why this exists
---------------
MASt3R-SLAM's metric head does not produce reliable real-world scale. Measured
over three dataset families (report_for_senior/1_scale_finding.png) the
recovered scale scatters from 0.45 to 2.05, and it scatters on real camera
footage just as much as on renders. A reconstruction 30% too small makes a
0.8 m doorway measure 0.56 m, which is the difference between a rover planning
through a gap and wedging itself in it.

There are two separate errors and they are not equally fixable:

  Constant bias   The map is internally consistent but the wrong size. One
                  scalar fixes it completely. This is what real footage does:
                  across nine TUM sequences the local scale ratio holds to
                  within 1-5% while sitting anywhere between 0.81 and 1.49.

  Warp            The trajectory scale drifts along the run -- up to 2.3x
                  end-to-end on the long HM3D house tours. No scalar fixes
                  this; correcting the trajectory leaves local geometry wrong
                  and vice versa.

This module solves the first and does not pretend to solve the second. The
floor anchor below fixes the size of the map reliably; it cannot see the warp,
and everything tried here that claimed to -- trend fits, spread of windowed
heights -- turned out to have no usable correlation with the real thing. What
would work is a second metric observable that measures distance travelled
directly, which on the rover means wheel odometry.

For navigation that split is the right one to ship. A uniformly scaled map that
is 5% out is usable; a map bent by a warp correction that guessed the sign
wrong is worse than the uncorrected one.

The anchor
----------
The camera sits at a known fixed height above the floor: 1.5 m for the HM3D
renders, whatever you measured on the rover mast. That is a metric quantity
observable in every frame, so it recovers scale with no ground truth at all.

Measured against the habitat ground truth over the ten HM3D scenes, the
recovered scale is within a median 5.4% of the value that makes local geometry
metric, with no catastrophic failures. For comparison the uncorrected
reconstructions are off by a median 27%.

Three details, each of which cost a wrong answer to find:

  * up comes from the cameras, not from a plane vote. RANSAC over the merged
    cloud returns a camera height of 0.13 m on 00803 and 5.67 m on 00808 -- it
    locks onto a stairwell landing and onto another storey's ceiling. Averaging
    the cameras' own image-down axis cannot do that, and a rover camera is near
    enough level for it to hold.

  * the neighbourhood is a horizontal cylinder, not a sphere. A 2 m ball around
    a camera 1.5 m up reaches only 1.3 m of floor around the wheels, so most of
    what it captures is wall.

  * the floor evidence is pooled across keyframes before the peak is taken, not
    after. The floor is the one surface every keyframe sees at the SAME offset
    below itself, so pooling makes it a sharp spike while furniture, sitting at
    a different offset under each keyframe, smears out. Taking a per-keyframe
    mode first and combining afterwards throws that away, and read 1.96 m on
    00800 where the answer was 1.61 m.
"""

import numpy as np
from scipy.spatial import cKDTree


# ------------------------------------------------------------------- up axis

def quat_to_R(q):
    """TUM order: qx qy qz qw."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_traj_full(path):
    """MASt3R writes: timestamp x y z qx qy qz qw (world<-camera)."""
    rows = np.loadtxt(str(path))
    if rows.ndim == 1:
        rows = rows[None]
    return rows[:, 0], rows[:, 1:4].astype(np.float64), rows[:, 4:8].astype(np.float64)


def camera_up(quats):
    """Gravity-up in the world frame, from the cameras' own orientation.

    +y is down in a camera frame, so R_wc @ [0,1,0] is world-down for that
    keyframe. Averaging over the run cancels per-frame pitch and roll of a
    camera that is level on average; normalising afterwards is what makes the
    result a direction rather than a shrunken mean.
    """
    down = np.mean([quat_to_R(q) @ np.array([0.0, 1.0, 0.0]) for q in quats], axis=0)
    n = np.linalg.norm(down)
    if n < 1e-6:
        raise ValueError("camera orientations average to nothing -- is the "
                         "trajectory file missing its quaternion columns?")
    return -down / n


def _horizontal_basis(up):
    """Two unit vectors spanning the plane perpendicular to `up`."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(seed @ up) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = seed - (seed @ up) * up
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1)])


# ------------------------------------------------------------- floor evidence

def voxel_downsample(pts, voxel):
    """One point per occupied voxel. Quantise, then unique."""
    if voxel <= 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int64)
    return pts[np.sort(np.unique(keys, axis=0, return_index=True)[1])]


def floor_drops(pts, cams, up, radius=2.0, min_drop=0.30, max_drop=4.0):
    """Per keyframe, how far below the camera each nearby point sits.

    `min_drop` discards the rover's own chassis and clutter it is nosing into;
    `max_drop` stops a mezzanine edge from finding the storey below.

    Feed this a VOXEL-DOWNSAMPLED cloud. MASt3R's per-keyframe pointmaps
    oversample the near field enormously, so on a raw cloud the drop histogram
    peaks in its very first bin and the anchor reports a camera 0.36 m off the
    ground on eight of the ten HM3D scenes. Downsampling is what makes the
    histogram count surface area instead of counting pixels.
    """
    up = np.asarray(up, float)
    basis = _horizontal_basis(up)
    h_pts = pts @ up
    tree = cKDTree(pts @ basis.T)
    cams_xy, h_cams = cams @ basis.T, cams @ up

    out = []
    for cxy, ch in zip(cams_xy, h_cams):
        d = ch - h_pts[tree.query_ball_point(cxy, r=radius)]
        out.append(d[(d > min_drop) & (d < max_drop)])
    return out


def aggregate_peak(drops, idxs=None, min_drop=0.30, max_drop=4.0, bin_w=0.04,
                   min_pts=50, min_kf=3):
    """Pooled floor offset over a set of keyframes, or NaN if unmeasurable.

    Each keyframe's histogram is normalised before summing so that a densely
    reconstructed corner cannot outvote the rest of the run.
    """
    nb = int(round((max_drop - min_drop) / bin_w))
    acc = np.zeros(nb)
    used = 0
    for k in (range(len(drops)) if idxs is None else idxs):
        d = drops[k]
        if len(d) < min_pts:
            continue
        c, _ = np.histogram(d, bins=nb, range=(min_drop, max_drop))
        tot = c.sum()
        if tot:
            acc += c / tot
            used += 1
    if used < min_kf:
        return np.nan, used
    # Smooth over three bins so a spike split across a boundary still wins.
    acc = np.convolve(acc, np.ones(3) / 3.0, mode="same")
    return min_drop + (int(np.argmax(acc)) + 0.5) * bin_w, used


# ------------------------------------------------------------------- profile

def path_coord(cams):
    """Cumulative distance travelled, normalised to [0, 1].

    Drift accumulates with distance covered, not with keyframe index --
    keyframes bunch up where the camera turns on the spot, which would give
    those moments undue leverage on a fit against index.
    """
    d = np.linalg.norm(np.diff(np.asarray(cams, float), axis=0), axis=1)
    x = np.concatenate([[0.0], np.cumsum(d)])
    return x / x[-1] if x[-1] > 0 else x


# A windowed height within this fraction of the pooled one counts as agreeing.
AGREE_TOL = 0.20
# Below this fraction of windows agreeing, the floor measurement is shaky and
# the scale it produced is provisional.
AGREE_LIMIT = 0.60
# Below this fraction of keyframes finding any floor, there is nothing to read.
COVERAGE_LIMIT = 0.50


def estimate_scale(drops, cams, true_height, n_windows=8):
    """Metric scale, plus a verdict on how well the floor could be measured.

    `verdict` grades the MEASUREMENT, not the map:
      "confident"       the windows agree with the pooled height, so the scale
                        is as good as this method gets (~5% on HM3D).
      "low-confidence"  the windows disagree. The scale is still the best
                        estimate available, but look at the map. On HM3D this
                        catches the worst case -- 00808, 27% out -- at the cost
                        of two false alarms.
      "unreliable"      most keyframes found no floor at all.

    It deliberately says nothing about whether the map is warped. Warp is not
    measurable from this signal: the windowed heights are far too noisy, and
    across the ten HM3D scenes their spread has no usable correlation with the
    true end-to-end warp (00807 spreads 6.1x on a map warped 1.03x, while 00804
    spreads 1.6x on one warped 1.54x). Reporting a warp number from them would
    be inventing precision. See the module docstring for what would work.
    """
    h, used = aggregate_peak(drops)
    n = len(drops)
    if not np.isfinite(h):
        raise ValueError(
            f"no floor found: only {used} of {n} keyframes had enough points "
            "beneath them. The cloud is too sparse under the camera, or `up` "
            "is wrong -- check the reported tilt before trusting anything.")

    # Windowed heights, to see whether that one number holds along the run.
    x = path_coord(cams)
    edges = np.linspace(0, n, n_windows + 1).astype(int)
    wx, wh = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 3:
            continue
        hw, u = aggregate_peak(drops, idxs=range(a, b))
        if np.isfinite(hw):
            wx.append(float(np.mean(x[a:b])))
            wh.append(hw)
    wx, wh = np.asarray(wx), np.asarray(wh)

    agreement = float(np.mean(np.abs(wh / h - 1.0) < AGREE_TOL)) if len(wh) else 0.0

    coverage = used / n
    if coverage < COVERAGE_LIMIT:
        verdict = "unreliable"
    elif agreement < AGREE_LIMIT:
        verdict = "low-confidence"
    else:
        verdict = "confident"

    scale = true_height / h
    # Per-keyframe profile, for the optional deformation. Interpolated in log
    # space between window centres and held flat outside them.
    if len(wh) >= 4:
        prof = np.exp(np.interp(x, wx, np.log(wh)))
    else:
        prof = np.full(n, h)

    return {
        "scale": scale,
        "height": h,
        "coverage": float(coverage),
        "agreement": agreement,
        "verdict": verdict,
        "window_heights": wh.tolist(),
        "scale_profile": true_height / prof,
    }


# ---------------------------------------------------------------- deformation

def deform(pts, cams, scale_profile, chunk=200_000):
    """Non-rigid scale correction. EXPERIMENTAL -- off by default, and here is why.

    The idea is sound: rebuild the trajectory by integrating each inter-keyframe
    step at that step's own scale, and carry every point along with the
    keyframe nearest it, so local geometry stays metric while the path stops
    stretching.

    The idea does not survive contact with the measurement. Validated against
    ground truth on the ten HM3D scenes, the windowed floor height gives a
    drift estimate with the WRONG SIGN on three of them -- 00800 reads 0.66
    where the truth is 1.19. Applying that bends a map which was within 8% of
    metric to 17% out. The pooled level is trustworthy; its trend is not.

    So this runs only when you ask for it, and the honest default is to take
    the reliable scalar and report the warp rather than pretend to fix it.
    Turn it on if you have a better scale profile from somewhere else --
    wheel odometry on the rover would supply exactly that, and would not have
    the sign problem, because odometry measures distance travelled directly.
    """
    s = np.asarray(scale_profile, float)
    cams = np.asarray(cams, float)
    if len(s) != len(cams):
        raise ValueError(f"profile has {len(s)} entries for {len(cams)} poses")

    # Midpoint scale per step, so a step is not credited to whichever of its
    # two endpoints happened to come first.
    steps = np.diff(cams, axis=0)
    s_step = 0.5 * (s[:-1] + s[1:])
    cams_new = np.vstack([cams[0] * s[0],
                          cams[0] * s[0] + np.cumsum(steps * s_step[:, None], axis=0)])

    # Scale each point about its nearest keyframe, then move it to where that
    # keyframe now is. Nearest-keyframe assignment is an approximation -- the
    # .ply carries no point-to-keyframe association -- but it is the same one
    # the free-space carving in occupancy_grid.py already relies on. Chunked
    # because the cloud runs to tens of millions of points.
    tree = cKDTree(cams)
    out = np.empty_like(pts)
    for i in range(0, len(pts), chunk):
        blk = pts[i:i + chunk]
        a = tree.query(blk, k=1)[1]
        out[i:i + chunk] = cams_new[a] + s[a][:, None] * (blk - cams[a])
    return out, cams_new


def anchor_reconstruction(pts, cams, quats, true_height, *, radius=2.0,
                          voxel=0.04, correct_drift=False, verbose=True):
    """Measure the floor, report, and return a metric cloud and path.

    The measurement runs on a downsampled copy; the cloud returned is the full
    one, scaled.
    """
    up = camera_up(quats)
    drops = floor_drops(voxel_downsample(pts, voxel), cams, up, radius=radius)
    est = estimate_scale(drops, cams, true_height)

    apply_deform = correct_drift and est["verdict"] != "unreliable"
    if verbose:
        print(f"  metric anchor [{est['verdict']}]: floor {est['height']:.2f} m "
              f"below camera under {100*est['coverage']:.0f}% of keyframes "
              f"-> scale {est['scale']:.3f}")
        if est["verdict"] == "low-confidence":
            print(f"    WARNING: only {100*est['agreement']:.0f}% of windows "
                  "agree with that height, so the scale is provisional. "
                  "Check this map before navigating on it.")
        elif est["verdict"] == "unreliable":
            print("    WARNING: the floor could not be measured under half the "
                  "keyframes. Applying the scalar as a best guess -- inspect "
                  "this map before navigating on it.")
        if apply_deform:
            print("    applying EXPERIMENTAL per-keyframe drift correction")

    if apply_deform:
        pts_m, cams_m = deform(pts, cams, est["scale_profile"])
    else:
        pts_m, cams_m = pts * est["scale"], cams * est["scale"]
    est["deformed"] = bool(apply_deform)
    return pts_m, cams_m, est
