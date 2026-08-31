#!/usr/bin/env python3
"""Per-frame Depth-Anything-V2 BEV, rendered as a video to spot missed obstacles.

    python scripts/bev_video.py --out /media/nahar4/<disk>/bev_video/mpl

Four panels per frame: RGB, DA2 depth, the floor/obstacle classification, and
the BEV built from it. Watch the third and fourth together -- a red blob in the
classification with no black cell under it in the BEV is an obstacle the grid
failed to place, which is the thing this video exists to find.

Two stages, because no conda env has both halves
------------------------------------------------
transformers lives in qwen_vlm; plyfile/scipy live in mast3r-slam. Stage
`depth` runs the network out of process and writes one float16 memmap; stage
`render` does the geometry and the video. Re-rendering with different heights,
resolution or fx costs seconds instead of re-running 4673 forward passes.

RGB comes from the MP4, not the dump
------------------------------------
Only 104 of the dump's RGB PNGs survive -- the capture filled the disk. The MP4
is intact and its frame index matches the dump exactly (correlation 0.9997 on
probes), so every frame is available through it.

--depth-max is the setting that decides whether you see anything
---------------------------------------------------------------
The 4.0 m default is inherited from the RealSense pipeline, where error grows as
Z^2 and a far reading plants a wall across open floor. Depth-Anything has no
such wall -- it returns up to 19.8 m on this session -- so in the open hall this
run drives through, a 4 m cap discards 60-69% of every frame and the BEV comes
out completely empty on half the frames. Raise it (8.0 is what these figures
were made with) whenever the scene is bigger than a corridor, and understand
that you are trading the validated accuracy band for coverage.

The floor plane is PINNED, not fitted per frame
-----------------------------------------------
Fitting per frame was measured on this session: over the 104 frames with usable
RGB the fitted pitch scatters with an 11.6 deg IQR, collapsing to 0.92 deg once
restricted to frames that actually show floor. The scatter is an unconstrained
fit on floor-poor frames -- three latched onto a wall at +31 deg -- not a moving
camera. Letting that drive the video would make the BEV flicker and you would be
watching the plane fit rather than the obstacles. --fit-per-frame if you want to
see that failure mode on purpose.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

METRIC_CKPT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
DA2_PYTHON = "/home/nahar4/miniconda3/envs/qwen_vlm/bin/python"
VIDEO = "/home/nahar4/Gazania/MPL/manual_drive_20260826_180408.mp4"
CALIB = "/home/nahar4/Gazania/MPL/rover_intrinsics_ramp.yaml"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=VIDEO)
    p.add_argument("--out", required=True,
                   help="output stem on the external disk, no extension")
    p.add_argument("--stage", choices=["all", "depth", "render"], default="all")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--da2-python", default=DA2_PYTHON)

    p.add_argument("--calib", default=CALIB)
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--scale", type=float, default=0.969,
                   help="global correction on the metric head")
    p.add_argument("--pitch", type=float, default=-1.5,
                   help="camera pitch below horizontal, deg (measured -1.5)")
    p.add_argument("--cam-height", type=float, default=0.5)
    p.add_argument("--fit-per-frame", action="store_true",
                   help="RANSAC the floor on every frame instead of pinning it")

    p.add_argument("--depth-min", type=float, default=0.3)
    p.add_argument("--depth-max", type=float, default=4.0)
    p.add_argument("--res", type=float, default=0.05)
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--min-h", type=float, default=0.10)
    p.add_argument("--max-h", type=float, default=1.50)
    p.add_argument("--min-cell-points", type=int, default=4)
    p.add_argument("--min-obstacle-top", type=float, default=0.0)
    p.add_argument("--pixel-stride", type=int, default=2)
    p.add_argument("--bev-forward", type=float, default=5.0)
    p.add_argument("--bev-width", type=float, default=6.0)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--no-h264", action="store_true",
                   help="skip the ffmpeg re-encode and leave cv2's mp4v output")
    return p.parse_args()


# ------------------------------------------------------------ stage 1: depth

DEPTH_WORKER = r'''
import sys, cv2, numpy as np, torch, time
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
ckpt, video, dst, stride, maxf, batch = sys.argv[1:7]
stride, maxf, batch = int(stride), int(maxf), int(batch)

cap = cv2.VideoCapture(video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idx = list(range(0, total, stride))
if maxf:
    idx = idx[:maxf]
n = len(idx)
print(f"{n} frames of {total}, {W}x{H} -> {dst}", flush=True)

proc = AutoImageProcessor.from_pretrained(ckpt)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.float32
model = AutoModelForDepthEstimation.from_pretrained(ckpt).to(dev, dt).eval()
print(f"model on {dev}", flush=True)

mm = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16, shape=(n, H, W))
np.save(dst.replace(".npy", "_index.npy"), np.array(idx, dtype=np.int32))

want = set(idx)
pend, pend_slot, k, t0 = [], [], 0, time.time()
def flush():
    global pend, pend_slot
    if not pend:
        return
    with torch.inference_mode():
        inp = proc(images=[cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in pend],
                   return_tensors="pt").to(dev, dt)
        res = proc.post_process_depth_estimation(model(**inp),
                                                 target_sizes=[(H, W)] * len(pend))
    for slot, r in zip(pend_slot, res):
        mm[slot] = r["predicted_depth"].float().cpu().numpy().astype(np.float16)
    pend, pend_slot = [], []

i = 0
while True:
    ok, bgr = cap.read()
    if not ok:
        break
    if i in want:
        pend.append(bgr); pend_slot.append(k); k += 1
        if len(pend) >= batch:
            flush()
            el = time.time() - t0
            print(f"\r  {k}/{n}  {k/max(el,1e-6):.1f} fps", end="", flush=True)
    i += 1
    if k >= n and not pend:
        break
flush()
mm.flush()
print(f"\ndone: {k} depth maps in {time.time()-t0:.0f} s", flush=True)
'''


def run_depth_stage(args, depth_npy):
    if not os.path.exists(args.da2_python):
        raise SystemExit(f"{args.da2_python} not found -- pass --da2-python")
    print(f"stage 1: Depth-Anything-V2 -> {depth_npy}")
    r = subprocess.run(
        [args.da2_python, "-c", DEPTH_WORKER, METRIC_CKPT, args.video,
         depth_npy, str(args.stride), str(args.max_frames), str(args.batch)])
    if r.returncode != 0 or not os.path.exists(depth_npy):
        raise SystemExit("depth stage failed")


# ----------------------------------------------------------- stage 2: render

def load_calib(path, W, H, fx_override):
    fx, cx, cy = None, W / 2.0, H / 2.0
    if path and os.path.exists(path):
        import yaml
        with open(path) as fh:
            k = yaml.safe_load(fh) or {}
        fx, cx, cy = k.get("fx"), k.get("cx", cx), k.get("cy", cy)
    if fx_override is not None:
        fx = fx_override
    if fx is None:
        raise SystemExit("no fx: pass --fx or a --calib yaml that has one")
    return float(fx), float(cx), float(cy)


def pinned_floor(pitch_deg, cam_height):
    """Floor plane implied by a known mount, in camera coordinates."""
    p = np.radians(pitch_deg)
    n_up = np.array([0.0, -np.cos(p), np.sin(p)])
    n_up /= np.linalg.norm(n_up)
    return n_up, -cam_height * n_up


def depth_colour(z, dmin, dmax):
    v = np.clip((z - dmin) / (dmax - dmin), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~np.isfinite(z)] = (40, 40, 40)
    return img


CLS_COLOUR = np.array([[40, 40, 40],        # 0 no depth
                       [200, 120, 40],      # 1 floor        (BGR blue-ish)
                       [40, 60, 220],       # 2 obstacle     (BGR red)
                       [40, 150, 150]], np.uint8)   # 3 overhead


def label(img, text, org=(8, 22), scale=0.6, colour=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                cv2.LINE_AA)


def render(args, depth_npy, index_npy, out_mp4):
    from occupancy_grid import (build_occupancy, fit_plane_ransac,
                                plane_basis)

    depths = np.load(depth_npy, mmap_mode="r")
    index = np.load(index_npy)
    n, H, W = depths.shape
    fx, cx, cy = load_calib(args.calib, W, H, args.fx)
    print(f"stage 2: {n} frames, {W}x{H}, fx={fx:.1f}")

    cap = cv2.VideoCapture(args.video)
    ps = args.pixel_stride
    vv, uu = np.mgrid[0:H:ps, 0:W:ps]

    n_up0, org0 = pinned_floor(args.pitch, args.cam_height)
    cams = np.zeros((1, 3))

    # Fixed BEV canvas so the panel does not jump around between frames.
    bw = int(round(args.bev_width / args.res))
    bf = int(round(args.bev_forward / args.res))
    back = int(round(1.0 / args.res))              # a metre behind the camera
    bh = bf + back
    panel = (W, H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, args.fps, (W * 2, H * 2))
    if not vw.isOpened():
        raise SystemExit(f"cannot open {out_mp4} for writing")

    stats = []
    t0 = time.time()
    # Sequential reads when the stride allows it. cap.set() per frame forces a
    # seek and costs more than the whole geometry pipeline on a 4673-frame run.
    pos = -1
    for k in range(n):
        want = int(index[k])
        if want != pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, want)
        ok, bgr = cap.read()
        pos = want
        if not ok:
            continue
        z = depths[k].astype(np.float32) * args.scale

        zs = z[::ps, ::ps]
        m = np.isfinite(zs) & (zs > args.depth_min) & (zs < args.depth_max)
        pts = np.empty((0, 3))
        if m.sum() > 500:
            zg = zs[m]
            pts = np.stack([(uu[m] - cx) * zg / fx,
                            (vv[m] - cy) * zg / fx, zg], axis=1).astype(np.float64)

        n_up, org = n_up0, org0
        if args.fit_per_frame and len(pts) > 2000:
            rng = np.random.default_rng(0)
            s = pts[rng.choice(len(pts), min(len(pts), 20000), replace=False)]
            try:
                nn, oo, _ = fit_plane_ransac(s, 0.03, 300, rng,
                                             up_axis=np.array([0.0, 1.0, 0.0]),
                                             tol_deg=35.0)
                _, _, n_up = plane_basis(nn, oo, cams)
                org = oo
            except SystemExit:
                pass

        # --- classification panel, at full image resolution ---
        cls = np.zeros((H, W), np.uint8)
        if len(pts):
            h_all = (pts - org) @ n_up
            lab = np.where(h_all <= args.min_h, 1,
                           np.where(h_all < args.max_h, 2, 3)).astype(np.uint8)
            sub = np.zeros(m.shape, np.uint8)
            sub[m] = lab
            cls = cv2.resize(sub, (W, H), interpolation=cv2.INTER_NEAREST)
        cls_img = CLS_COLOUR[cls]

        # --- BEV ---
        bev = np.full((bh, bw, 3), 158, np.uint8)
        n_obst_pts = n_cells = n_rejected = 0
        if len(pts):
            hh = (pts - org) @ n_up
            n_obst_pts = int(((hh > args.min_h) & (hh < args.max_h)).sum())
        if n_obst_pts > 0:
            try:
                prob, lo, info = build_occupancy(
                    pts, cams, res=args.res, voxel=args.voxel,
                    min_h=args.min_h, max_h=args.max_h,
                    floor_plane=(n_up, org), floor_support=False,
                    kf_id=np.zeros(len(pts), np.int32),
                    min_cell_points=args.min_cell_points,
                    min_obstacle_top=args.min_obstacle_top, verbose=False)
                bev, n_cells = paste_bev(prob, lo, info, args, bw, bh, back)
                n_rejected = count_rejected_cells(pts, n_up, org, args)
            except (ValueError, SystemExit):
                pass
        bev_img = cv2.resize(bev, panel, interpolation=cv2.INTER_NEAREST)
        draw_bev_axes(bev_img, args, bw, bh, back)

        rgb_img = bgr.copy()
        label(rgb_img, f"frame {int(index[k])}")
        d_img = depth_colour(z, args.depth_min, args.depth_max)
        label(d_img, f"DA2 depth  {args.depth_min}-{args.depth_max} m")
        label(cls_img, f"floor <{args.min_h} m | obstacle {args.min_h}-{args.max_h} m")
        label(cls_img, f"{n_obst_pts:,} obstacle pts", (8, 46), 0.5)
        label(bev_img, f"BEV @ {args.res} m")
        label(bev_img, f"{n_cells} cells drawn, {n_rejected} rejected "
                       f"(<{args.min_cell_points} pts)", (8, 46), 0.5)

        top = np.hstack([rgb_img, d_img])
        bot = np.hstack([cls_img, bev_img])
        vw.write(np.vstack([top, bot]))
        stats.append((int(index[k]), n_obst_pts, n_cells, n_rejected))

        if k % 50 == 0:
            el = time.time() - t0
            print(f"\r  {k}/{n}  {k/max(el,1e-6):.1f} fps", end="", flush=True)

    vw.release()
    cap.release()
    print(f"\nwrote {out_mp4}  ({len(stats)} frames, {time.time()-t0:.0f} s)")
    if not args.no_h264:
        transcode_h264(out_mp4, args.fps)

    st = np.array(stats)
    csv = out_mp4.replace(".mp4", "_stats.csv")
    np.savetxt(csv, st, fmt="%d", delimiter=",",
               header="frame,obstacle_points,cells_drawn,cells_rejected",
               comments="")
    blind = int((st[:, 2] == 0).sum())
    print(f"wrote {csv}")
    print(f"  frames with NO obstacle cell at all: {blind}/{len(st)} "
          f"({100*blind/len(st):.1f}%)")
    print(f"  median cells drawn {np.median(st[:,2]):.0f}, "
          f"median rejected {np.median(st[:,3]):.0f}")


def transcode_h264(path, fps):
    """Re-encode OpenCV's mp4v output to H.264.

    Worth the extra minute: cv2's `mp4v` is MPEG-4 Part 2, which VLC plays but
    browsers, QuickTime and most preview panes do not. -pix_fmt yuv420p is the
    part that actually matters for compatibility, and +faststart puts the index
    at the front so the file streams instead of needing a full download.
    """
    if not shutil.which("ffmpeg"):
        print("  ffmpeg not found; leaving the mp4v file as-is")
        return
    tmp = path.replace(".mp4", "_h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-r", str(fps), "-i", path,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    print("  transcoding to H.264 ...", flush=True)
    r = subprocess.run(cmd)
    if r.returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, path)
        mb = os.path.getsize(path) / 1e6
        print(f"  {path} is now H.264, {mb:.0f} MB")
    else:
        print("  ffmpeg failed; the mp4v file is still there")


def paste_bev(prob, lo, info, args, bw, bh, back):
    """Drop build_occupancy's variable-extent grid into a fixed canvas.

    The pinned plane makes the basis identical every frame, so this is an
    integer index shift -- no resampling, no interpolation error.
    """
    u, v = info["u"], info["v"]
    res = args.res
    H, W = prob.shape

    r2d = np.array([np.array([1.0, 0, 0]) @ u, np.array([1.0, 0, 0]) @ v])
    f2d = np.array([np.array([0, 0, 1.0]) @ u, np.array([0, 0, 1.0]) @ v])
    M = np.stack([r2d / (np.linalg.norm(r2d) + 1e-9),
                  f2d / (np.linalg.norm(f2d) + 1e-9)])
    P = np.round(M)

    rr, cc = np.nonzero(prob >= 0)
    px = lo[0] + (cc + 0.5) * res
    py = lo[1] + (rr + 0.5) * res
    d = np.stack([px, py], 1) @ P.T                      # display metres

    ci = np.floor(d[:, 0] / res).astype(int) + bw // 2
    ri = np.floor(d[:, 1] / res).astype(int) + back
    ok = (ci >= 0) & (ci < bw) & (ri >= 0) & (ri < bh)

    canvas = np.full((bh, bw, 3), 158, np.uint8)
    val = prob[rr, cc]
    free = ok & (val < 25)
    occ = ok & (val >= 65)
    canvas[ri[free], ci[free]] = 255
    canvas[ri[occ], ci[occ]] = 0

    return np.flipud(canvas), int(occ.sum())


def count_rejected_cells(pts, n_up, org, args):
    """Cells holding obstacle points that did not clear --min-cell-points.

    These never reach `prob` at all, so they are invisible in the BEV panel --
    which is exactly the failure this video is meant to expose. Mirrors
    build_occupancy's own order of operations (voxel downsample FIRST, then
    count per cell) or the number would not describe the same grid.
    """
    from occupancy_grid import plane_basis, voxel_downsample

    keep = voxel_downsample(pts, args.voxel, return_index=True)
    q = pts[keep]
    h = (q - org) @ n_up
    q = q[(h > args.min_h) & (h < args.max_h)]
    if len(q) == 0:
        return 0
    u, v, _ = plane_basis(n_up, org, np.zeros((1, 3)))
    rel = q - org
    xy = np.stack([rel @ u, rel @ v], axis=1)
    cell = np.floor(xy / args.res).astype(np.int64)
    _, counts = np.unique(cell, axis=0, return_counts=True)
    return int(((counts >= 1) & (counts < args.min_cell_points)).sum())


def draw_bev_axes(img, args, bw, bh, back):
    """Camera marker, heading arrow, and 1 m range rings."""
    h, w = img.shape[:2]
    sx, sy = w / bw, h / bh
    cx = int((bw // 2) * sx)
    cy = int(h - back * sy)
    for m_ in range(1, int(args.bev_forward) + 1):
        y = int(cy - m_ / args.res * sy)
        if 0 <= y < h:
            cv2.line(img, (0, y), (w, y), (120, 120, 120), 1)
            label(img, f"{m_} m", (4, y - 4), 0.4, (90, 90, 90))
    cv2.arrowedLine(img, (cx, cy), (cx, cy - int(0.8 / args.res * sy)),
                    (200, 60, 60), 2, tipLength=0.3)
    cv2.circle(img, (cx, cy), 5, (200, 60, 60), -1)


def main():
    args = parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    depth_npy = str(out) + "_da2_depth.npy"
    index_npy = str(out) + "_da2_depth_index.npy"

    if args.stage in ("all", "depth"):
        run_depth_stage(args, depth_npy)
    if args.stage in ("all", "render"):
        if not os.path.exists(depth_npy):
            raise SystemExit(f"{depth_npy} missing -- run --stage depth first")
        render(args, depth_npy, index_npy, str(out) + ".mp4")


if __name__ == "__main__":
    main()
