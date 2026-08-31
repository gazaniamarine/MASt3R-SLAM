#!/usr/bin/env python3
"""Fit Depth-Anything-V2 to RealSense depth and report the error in millimetres.

    python scripts/depth_fit_eval.py --root /home/nahar4/Gazania/MPL --frames 120

Why this exists
---------------
Two questions have to be answered with numbers before a BEV can be built from
predicted depth, and neither is answerable by looking at a colourised depth
video:

  1. Which checkpoint?  The METRIC head outputs metres directly and needs one
     global scale. The RELATIVE head outputs affine-invariant inverse depth --
     unusable on its own, but with per-frame sensor depth to fit against, both
     of its unknowns are directly observed, and it is trained on far more data.
     Which one actually wins here is an empirical question.

  2. Where does the sensor stop being the better source?  A stereo RealSense
     has error growing as Z^2. Somewhere past a few metres the network becomes
     more trustworthy than the sensor it is being scored against, and beyond
     that point these residuals measure the SENSOR's error, not the network's.
     Reporting a single RMSE hides this completely, so everything below is
     split by distance band.

The focal length gets recovered here too, from the sensor depth rather than
from the network's own floor. Unprojecting with the wrong fx bends a plane
instead of merely resizing it, so the fx that makes the floor flattest is the
right one -- the same shape test scripts/calib_from_floor.py uses, but anchored
on measured depth instead of predicted depth.
"""

import argparse
import glob
import os
import pathlib

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

METRIC_CKPT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
RELATIVE_CKPT = "depth-anything/Depth-Anything-V2-Large-hf"

# RealSense units. The dump is uint16; 1000 => millimetres.
DEPTH_UNITS_PER_M = 1000.0

# Bands are chosen around where a D4xx stereo pair stops being credible, not
# on round numbers: inside 1 m it is excellent, by 4 m the Z^2 error term
# dominates, past 6 m it is mostly noise.
BANDS = [(0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 10.0)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/nahar4/Gazania/MPL")
    p.add_argument("--frames", type=int, default=120,
                   help="frames sampled evenly across the usable window")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--cam-height", type=float, default=0.5)
    p.add_argument("--fx-min", type=float, default=250.0)
    p.add_argument("--fx-max", type=float, default=1400.0)
    p.add_argument("--fx-steps", type=int, default=116)
    p.add_argument("--floor-frac", type=float, default=0.35)
    p.add_argument("--gt-min", type=float, default=0.3, help="metres")
    p.add_argument("--gt-max", type=float, default=10.0, help="metres")
    p.add_argument("--fit-max", type=float, default=4.0,
                   help="only GT closer than this is used to FIT (beyond it "
                        "the sensor is not trustworthy enough to fit against)")
    p.add_argument("--out", default=None, help="write recovered params as yaml")
    return p.parse_args()


# --------------------------------------------------------------------------
# inputs

def find_inputs(root):
    root = pathlib.Path(root)
    mp4 = sorted(root.glob("*.mp4"))
    rgbd = [d for d in root.rglob("*") if d.is_dir() and
            list(d.glob("*_depth.png"))]
    if not (mp4 and rgbd):
        raise SystemExit(f"need an mp4 and a depth dump under {root}")
    return str(mp4[0]), str(rgbd[0])


def decodable_depth(dirpath, cache=None):
    """Depth frames that actually decode, keyed by frame index."""
    if cache and os.path.exists(cache):
        return np.load(cache, allow_pickle=True)["depth"].item()
    out = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "*_depth.png"))):
        if os.path.getsize(f) == 0:
            continue
        if cv2.imread(f, cv2.IMREAD_UNCHANGED) is None:
            continue
        out[int(os.path.basename(f).split("_")[1])] = f
    return out


def read_video_frames(path, indices):
    """Pull specific frame indices out of the mp4, in one sequential pass.

    Seeking per frame with CAP_PROP_POS_FRAMES is unreliable on inter-frame
    compressed video; reading straight through and keeping what we want is
    slower to write but correct.
    """
    want = set(indices)
    cap = cv2.VideoCapture(path)
    got, i = {}, 0
    while want:
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = bgr
            want.discard(i)
        i += 1
    cap.release()
    return got


# --------------------------------------------------------------------------
# geometry

def fit_plane(pts, thresh, iters=200, rng=None):
    """RANSAC plane, refit on inliers. Returns (normal, centroid, mask)."""
    rng = rng or np.random.default_rng(0)
    if len(pts) < 50:
        return None
    best = None
    for _ in range(iters):
        tri = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        inl = np.abs((pts - tri[0]) @ n) < thresh
        if best is None or inl.sum() > best[1].sum():
            best = (n, inl)
    if best is None:
        return None
    q = pts[best[1]]
    c = q.mean(axis=0)
    n = np.linalg.svd(q - c)[2][2]
    n /= np.linalg.norm(n)
    return n, c, np.abs((pts - c) @ n) < thresh


def recover_fx(depth_m, args):
    """Sweep fx; the value whose floor best PREDICTS the measured depth wins.

    The obvious version of this test -- fit a plane, report the mean Cartesian
    distance from it -- is broken, and broken in a way that silently returns
    whatever the top of your search range is. As fx grows the unprojection
    tends to orthographic, every lateral coordinate (u-cx)Z/fx shrinks toward
    zero, and so does any residual measured in metres of Cartesian distance.
    The metric therefore falls monotonically with fx and its argmin is the
    search bound. Both scripts/calib_from_floor.py and the first version of
    this one railed at their upper limits for exactly this reason.

    Measuring the residual in DEPTH instead removes the degeneracy. Fit the
    plane, then for each pixel intersect its ray with that plane and compare
    the predicted Z against the Z the sensor measured:

        Z_pred(u,v) = (n . c) / (n . r),   r = ((u-cx)/fx, (v-cy)/fx, 1)

    Z is an observable the unprojection cannot shrink. A too-large fx flattens
    every ray toward (0,0,1), which forces Z_pred to be near-constant across
    the image -- and a real floor's depth ramps from 1 m at the bottom of the
    frame to many metres at the top, so the error explodes rather than
    vanishes. The metric now has a genuine interior minimum.
    """
    H, W = depth_m[0].shape
    cx, cy = W / 2.0, H / 2.0
    v0 = int(H * (1.0 - args.floor_frac))
    vv, uu = np.mgrid[v0:H, 0:W]
    rng = np.random.default_rng(0)

    rows = []
    for fx in np.linspace(args.fx_min, args.fx_max, args.fx_steps):
        resid, heights, tilts = [], [], []
        for z in depth_m:
            zc = z[v0:H, :]
            good = np.isfinite(zc) & (zc > args.gt_min) & (zc < 8.0)
            if good.sum() < 500:
                continue
            zg = zc[good]
            ru = (uu[good] - cx) / fx
            rv = (vv[good] - cy) / fx
            pts = np.stack([ru * zg, rv * zg, zg], axis=1)
            if len(pts) > 4000:
                sel = rng.choice(len(pts), 4000, replace=False)
                pts, ru, rv, zg = pts[sel], ru[sel], rv[sel], zg[sel]
            fit = fit_plane(pts, 0.04, rng=rng)
            if fit is None:
                continue
            n, c, inl = fit
            if inl.mean() < 0.55:
                continue
            if n[1] < 0:
                n = -n
            r = np.stack([ru[inl], rv[inl], np.ones(int(inl.sum()))], axis=1)
            denom = r @ n
            ok = np.abs(denom) > 1e-6
            if ok.sum() < 100:
                continue
            zpred = (c @ n) / denom[ok]
            # A plane behind the camera is not a floor; reject rather than
            # let a negative prediction average away against a positive error.
            valid = ok.copy()
            valid[ok] = zpred > 0
            if valid.sum() < 100:
                continue
            resid.append(float(np.median(
                np.abs(zpred[zpred > 0] - zg[inl][valid]))))
            heights.append(abs(float(c @ n)))
            tilts.append(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
        if resid:
            rows.append((fx, np.median(resid), np.median(heights),
                         np.median(tilts), len(resid)))
    if not rows:
        raise SystemExit("no usable floor plane in the sensor depth")
    return np.array(rows)


# --------------------------------------------------------------------------
# prediction

def load_model(ckpt):
    proc = AutoImageProcessor.from_pretrained(ckpt)
    model = AutoModelForDepthEstimation.from_pretrained(ckpt).to(
        "cuda", torch.float16).eval()
    return proc, model


@torch.inference_mode()
def predict(proc, model, frames_bgr, hw, batch):
    out = []
    for i in range(0, len(frames_bgr), batch):
        chunk = frames_bgr[i:i + batch]
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in chunk]
        inp = proc(images=rgb, return_tensors="pt").to("cuda", torch.float16)
        res = proc.post_process_depth_estimation(
            model(**inp), target_sizes=[hw] * len(rgb))
        out.extend(r["predicted_depth"].float().cpu().numpy() for r in res)
    return out


def fit_metric(pred, gt, mask):
    """One scale. The metric head claims metres, so only its size is wrong."""
    r = gt[mask] / np.clip(pred[mask], 1e-6, None)
    s = float(np.median(r))
    return s * pred, {"scale": s}


def fit_relative(pred, gt, mask):
    """Affine fit in DISPARITY space -- that is the space the head is
    affine-invariant in, so a linear solve there is the right model. Fitting
    in depth space instead would be solving the wrong equation."""
    d = pred[mask].astype(np.float64)
    disp_gt = 1.0 / gt[mask]
    A = np.stack([d, np.ones_like(d)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, disp_gt, rcond=None)
    disp = a * pred + b
    # Behind the camera or at infinity: mark invalid rather than wrap to
    # negative depth, which would put points behind the rover.
    z = np.where(disp > 1e-4, 1.0 / np.maximum(disp, 1e-4), np.nan)
    return z, {"a": float(a), "b": float(b)}


def band_stats(errs_by_band):
    out = []
    for (lo, hi), e in errs_by_band.items():
        if not e:
            out.append(((lo, hi), None))
            continue
        e = np.concatenate(e)
        out.append(((lo, hi), (np.median(np.abs(e)) * 1000,
                               np.sqrt(np.mean(e ** 2)) * 1000,
                               len(e))))
    return out


# --------------------------------------------------------------------------

def main():
    args = parse_args()
    mp4, rgbd_dir = find_inputs(args.root)
    depth_files = decodable_depth(rgbd_dir)
    idxs = sorted(depth_files)
    print(f"mp4         {mp4}")
    print(f"depth dump  {rgbd_dir}")
    print(f"usable depth frames: {len(idxs)} (index {idxs[0]}..{idxs[-1]})")

    pick = [idxs[i] for i in
            np.linspace(0, len(idxs) - 1, args.frames).astype(int)]
    pick = sorted(set(pick))
    print(f"sampling {len(pick)} of them\n")

    vid = read_video_frames(mp4, pick)
    pick = [i for i in pick if i in vid]
    gt = [cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED).astype(np.float32)
          / DEPTH_UNITS_PER_M for i in pick]
    rgb = [vid[i] for i in pick]
    H, W = gt[0].shape
    print(f"loaded {len(pick)} rgb+depth pairs at {W}x{H}")

    # ---- alignment sanity: is the depth registered to the colour frame? ----
    # Compare edge maps at a few integer shifts. If the dump were unaligned
    # depth-camera output, the best shift would sit tens of pixels off zero.
    ge = cv2.Sobel(cv2.cvtColor(rgb[len(rgb) // 2], cv2.COLOR_BGR2GRAY),
                   cv2.CV_32F, 1, 1, ksize=3)
    de = cv2.Sobel(gt[len(gt) // 2], cv2.CV_32F, 1, 1, ksize=3)
    ge, de = np.abs(ge), np.abs(de)
    ge[~np.isfinite(ge)] = 0
    de[~np.isfinite(de)] = 0
    best = None
    for dx in range(-24, 25, 4):
        a = ge[:, max(0, dx):W + min(0, dx)]
        b = de[:, max(0, -dx):W + min(0, -dx)]
        c = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if best is None or c > best[1]:
            best = (dx, c)
    print(f"depth/rgb edge alignment: best x-shift {best[0]:+d} px "
          f"(corr {best[1]:.3f})"
          + ("  -> registered to colour" if abs(best[0]) <= 4 else
             "  -> WARNING: depth looks UNREGISTERED"))

    # ---- focal length from the sensor's own floor ----
    print("\nrecovering fx from sensor depth...")
    arr = recover_fx(gt[::max(1, len(gt) // 40)], args)
    bestrow = arr[np.argmin(arr[:, 1])]
    fx, resid, height, tilt, kept = bestrow
    print("    fx    resid(mm)  height(m)  tilt(deg)  frames")
    for r in arr[::max(1, len(arr) // 14)]:
        mark = " <-" if r[0] == fx else ""
        print(f"  {r[0]:6.1f}  {r[1]*1000:8.1f}  {r[2]:9.3f}  "
              f"{r[3]:8.2f}  {int(r[4]):6d}{mark}")
    if abs(fx - args.fx_min) < 1 or abs(fx - args.fx_max) < 1:
        print("  WARNING: fx hit the edge of the search range -- widen it.")
    hfov = 2 * np.degrees(np.arctan(W / (2 * fx)))
    print(f"\n  fx = {fx:.1f} px  ({hfov:.1f} deg HFOV)   "
          f"floor resid {resid*1000:.1f} mm")
    print(f"  implied camera height {height:.3f} m "
          f"(measured {args.cam_height:.3f} m -> x{args.cam_height/height:.3f})")
    print(f"  camera pitch {tilt:.2f} deg below horizontal")

    # ---- the two checkpoints ----
    results = {}
    for name, ckpt, fitter in [("metric", METRIC_CKPT, fit_metric),
                               ("relative", RELATIVE_CKPT, fit_relative)]:
        print(f"\n{'='*66}\n{name}: {ckpt}")
        proc, model = load_model(ckpt)
        preds = predict(proc, model, rgb, (H, W), args.batch)
        del model
        torch.cuda.empty_cache()

        errs = {b: [] for b in BANDS}
        params = []
        for p, g in zip(preds, gt):
            valid = np.isfinite(g) & (g > args.gt_min) & (g < args.gt_max)
            fitm = valid & (g < args.fit_max)
            if fitm.sum() < 2000:
                continue
            z, par = fitter(p, g, fitm)
            params.append(par)
            for lo, hi in BANDS:
                m = valid & (g >= lo) & (g < hi) & np.isfinite(z)
                if m.sum() > 50:
                    errs[(lo, hi)].append((z[m] - g[m]).astype(np.float32))
        results[name] = (band_stats(errs), params)

        print(f"  fitted on {len(params)} frames "
              f"(GT < {args.fit_max:g} m only)")
        if name == "metric":
            s = np.array([q["scale"] for q in params])
            print(f"  per-frame scale: median {np.median(s):.4f}  "
                  f"IQR [{np.percentile(s,25):.4f}, {np.percentile(s,75):.4f}]"
                  f"  spread {100*(np.percentile(s,75)-np.percentile(s,25))/np.median(s):.1f}%")
        else:
            # The question that decides whether this generalises past frame
            # 1216: are a and b stable enough to freeze as constants where
            # there is no sensor depth to fit against?
            A = np.array([q["a"] for q in params])
            B = np.array([q["b"] for q in params])
            for nm, v in (("a", A), ("b", B)):
                iqr = np.percentile(v, 75) - np.percentile(v, 25)
                print(f"  per-frame {nm}: median {np.median(v):.5f}  "
                      f"IQR [{np.percentile(v,25):.5f}, "
                      f"{np.percentile(v,75):.5f}]  "
                      f"spread {100*iqr/abs(np.median(v)):.1f}%")
            print(f"  frozen (a,b) would give depth = 1/({np.median(A):.5f}*d "
                  f"+ {np.median(B):.5f})")

    # ---- the table that decides it ----
    print(f"\n{'='*66}\nerror vs RealSense, by true distance (median |err| / RMSE, mm)")
    print(f"{'band (m)':>12}  {'metric':>18}  {'relative':>18}  {'pixels':>10}")
    for i, (band, _) in enumerate(results["metric"][0]):
        row = f"{band[0]:5.1f}-{band[1]:<5.1f} "
        cells, npx = [], 0
        for name in ("metric", "relative"):
            st = results[name][0][i][1]
            cells.append("        --        " if st is None else
                         f"{st[0]:8.0f} /{st[1]:8.0f}")
            if st:
                npx = st[2]
        print(f"{row:>12}  {cells[0]:>18}  {cells[1]:>18}  {npx:10,d}")
    print("\nPast ~4 m these numbers score the SENSOR as much as the network.")

    if args.out:
        pathlib.Path(args.out).write_text(
            f"# recovered by scripts/depth_fit_eval.py from sensor depth\n"
            f"width: {W}\nheight: {H}\n"
            f"fx: {fx:.2f}\nfy: {fx:.2f}\ncx: {W/2:.2f}\ncy: {H/2:.2f}\n"
            f"cam_height: {args.cam_height}\n"
            f"pitch_deg: {tilt:.3f}\n"
            f"floor_resid_mm: {resid*1000:.2f}\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
