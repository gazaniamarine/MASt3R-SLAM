#!/usr/bin/env python3
"""One RGB frame -> Depth-Anything-V2 -> BEV occupancy. No odometry, no SLAM.

    python scripts/single_frame_bev.py --frame 442 --out logs/rover/sf442

Why a single frame is worth running
-----------------------------------
Every multi-frame map on this rover confounds four things: the depth network,
the intrinsics, the camera mount, and the odometry/clock alignment. One frame
with the pose pinned to the identity removes two of them. Whatever is wrong in
the output here is wrong in the geometry -- fx, pitch, cam height, or the depth
itself -- and cannot be blamed on drift or on the +26.8 s video/odom offset.

The frame IS the world. The camera sits at the origin looking down +z, which is
already the frame occupancy_grid.py expects (x right, y DOWN, z forward), so its
gravity prior on +y finds the floor with no rotation applied.

Floor removal, since that is the thing being checked
----------------------------------------------------
Not by depth -- the floor spans the whole depth range, from 0.3 m underfoot to
the far wall, so no depth cut separates it from anything. It is removed by
HEIGHT ABOVE A FITTED PLANE: RANSAC the floor (gravity prior rejects walls and
table tops), then keep only points in [--min-h, --max-h] above it. --show-floor
renders that classification back onto the image so you can see which pixels the
grid called floor.

--source picks what gets unprojected: the Depth-Anything metric head (default),
the RealSense depth recorded alongside, or both side by side, which is the
honest way to see what the network costs you on this exact frame.
"""

import argparse
import os
import pathlib
import sys

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from occupancy_grid import (build_occupancy, fit_plane_ransac, plane_basis,
                            write_pgm_yaml)

METRIC_CKPT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
RGBD_DIR = ("/home/nahar4/Gazania/MPL/rgbd_20260826_180408/"
            "rgbd_20260826_180408")
DEPTH_UNITS_PER_M = 1000.0
# transformers lives in qwen_vlm; plyfile/scipy live in mast3r-slam. Neither has
# both, so the network is invoked out of process rather than by forcing an
# install into an env this repo depends on.
DA2_PYTHON = "/home/nahar4/miniconda3/envs/qwen_vlm/bin/python"

DA2_WORKER = r'''
import sys, cv2, numpy as np, torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
ckpt, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
bgr = cv2.imread(src)
proc = AutoImageProcessor.from_pretrained(ckpt)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if dev == "cuda" else torch.float32
model = AutoModelForDepthEstimation.from_pretrained(ckpt).to(dev, dt).eval()
with torch.inference_mode():
    inp = proc(images=[cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)],
               return_tensors="pt").to(dev, dt)
    res = proc.post_process_depth_estimation(model(**inp),
                                             target_sizes=[bgr.shape[:2]])
np.save(dst, res[0]["predicted_depth"].float().cpu().numpy())
print(f"DA2 on {dev} -> {dst}")
'''


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=RGBD_DIR, help="rgbd dump directory")
    p.add_argument("--frame", type=int, default=442,
                   help="frame index; must have a DECODABLE rgb png")
    p.add_argument("--rgb", default=None, help="explicit rgb path, overrides --frame")
    p.add_argument("--depth", default=None, help="explicit sensor depth path")
    p.add_argument("--out", required=True, help="output stem, no extension")
    p.add_argument("--source", choices=["da2", "sensor", "both"], default="da2",
                   help="which depth to unproject (default: da2)")
    p.add_argument("--list-frames", action="store_true",
                   help="print the frames whose rgb decodes, then exit")
    p.add_argument("--da2-python", default=DA2_PYTHON,
                   help="interpreter that has transformers+torch. No env here "
                        "has both that and plyfile/scipy, so the network runs "
                        "out of process and its output is cached as .npy")
    p.add_argument("--da2-npy", default=None,
                   help="use this precomputed metre-valued depth instead of "
                        "running the network")
    p.add_argument("--refresh-da2", action="store_true",
                   help="ignore the cached .npy and re-run the network")

    p.add_argument("--calib", default="/home/nahar4/Gazania/MPL/"
                                      "rover_intrinsics_ramp.yaml",
                   help="yaml with fx/cx/cy; --fx overrides it")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--scale", type=float, default=0.969,
                   help="global correction on the metric head (measured 0.969)")
    p.add_argument("--cam-height", type=float, default=None,
                   help="known camera height; with --pitch it PINS the floor "
                        "plane instead of fitting it, which is the check worth "
                        "running against the fitted answer")
    p.add_argument("--pitch", type=float, default=None,
                   help="camera pitch below horizontal, degrees")

    p.add_argument("--pixel-stride", type=int, default=1)
    p.add_argument("--depth-min", type=float, default=0.3)
    p.add_argument("--depth-max", type=float, default=4.0)
    p.add_argument("--res", type=float, default=0.05)
    p.add_argument("--voxel", type=float, default=0.03)
    p.add_argument("--min-h", type=float, default=0.10)
    p.add_argument("--max-h", type=float, default=1.50)
    p.add_argument("--min-obstacle-top", type=float, default=0.0)
    p.add_argument("--min-cell-points", type=int, default=4)
    p.add_argument("--max-ray", type=float, default=None)
    p.add_argument("--gravity-tol", type=float, default=35.0)
    p.add_argument("--no-floor-support", dest="floor_support",
                   action="store_false")
    p.set_defaults(floor_support=True)
    p.add_argument("--show-floor", action="store_true",
                   help="extra panel: per-pixel floor/obstacle classification")
    return p.parse_args()


# ---------------------------------------------------------------- inputs

def decodable_rgb_frames(dirpath):
    """Size>0 is not enough. The disk filled mid-capture so files exist at
    partial length and only libpng knows which ones survived."""
    out = []
    for name in sorted(os.listdir(dirpath)):
        if not name.endswith("_rgb.png"):
            continue
        f = os.path.join(dirpath, name)
        if os.path.getsize(f) == 0 or cv2.imread(f) is None:
            continue
        out.append(int(name.split("_")[1]))
    return out


def load_calib(path, W, H):
    fx, cx, cy = None, W / 2.0, H / 2.0
    if path and os.path.exists(path):
        import yaml
        with open(path) as fh:
            k = yaml.safe_load(fh) or {}
        fx = k.get("fx")
        cx, cy = k.get("cx", cx), k.get("cy", cy)
    return fx, cx, cy


def da2_depth(rgb_path, cache, interpreter, refresh):
    """Depth-Anything-V2 metric head, run once and cached to `cache`.

    Cached deliberately: the geometry below is the part being checked, and
    re-running a 300M-parameter network every time fx changes would make that
    check slow enough that you stop doing it.
    """
    if os.path.exists(cache) and not refresh:
        print(f"  reusing cached {cache} (--refresh-da2 to redo)")
        return np.load(cache)
    if not os.path.exists(interpreter):
        raise SystemExit(f"{interpreter} not found -- pass --da2-python "
                         f"pointing at an env with transformers + torch")
    import subprocess
    print(f"  running {METRIC_CKPT}\n  via {interpreter}")
    r = subprocess.run([interpreter, "-c", DA2_WORKER, METRIC_CKPT,
                        rgb_path, cache], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(cache):
        raise SystemExit(f"DA2 worker failed:\n{r.stdout}\n{r.stderr}")
    print("  " + r.stdout.strip().splitlines()[-1])
    return np.load(cache)


# ---------------------------------------------------------------- geometry

def unproject(z, fx, cx, cy, stride, dmin, dmax):
    """Depth map -> points in the camera frame (x right, y DOWN, z forward).

    That frame is already occupancy_grid.py's world convention, which is the
    whole reason a single frame needs no pose at all.
    """
    H, W = z.shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    zs = z[::stride, ::stride]
    m = np.isfinite(zs) & (zs > dmin) & (zs < dmax)
    zg = zs[m]
    pts = np.stack([(uu[m] - cx) * zg / fx,
                    (vv[m] - cy) * zg / fx, zg], axis=1)
    return pts.astype(np.float64), m


def pinned_floor(pitch_deg, cam_height):
    """Floor plane implied by a known mount, in camera coordinates.

    Camera up is -y. Pitching the camera down by p tips that up axis forward,
    so the floor normal gains a +z component: the floor is not parallel to the
    image's bottom edge and treating it as if it were is what plants phantom
    obstacles at the far end of the map.
    """
    p = np.radians(pitch_deg)
    n_up = np.array([0.0, -np.cos(p), np.sin(p)])
    n_up /= np.linalg.norm(n_up)
    return n_up, -cam_height * n_up          # camera at origin, floor below


def describe_plane(n_up, origin):
    """Report a floor plane as the mount it implies: pitch and height."""
    fwd = np.array([0.0, 0.0, 1.0])
    pitch = np.degrees(np.arcsin(np.clip(-float(fwd @ n_up), -1, 1)))
    height = float((np.zeros(3) - origin) @ n_up)
    roll = np.degrees(np.arcsin(np.clip(float(np.array([1.0, 0, 0]) @ n_up), -1, 1)))
    return pitch, height, roll


def write_cloud(pts, stem):
    v = np.empty(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("kf_id", "i4")])
    v["x"], v["y"], v["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    v["kf_id"] = 0
    PlyData([PlyElement.describe(v, "vertex")]).write(stem + ".ply")
    with open(stem + ".txt", "w") as fh:
        fh.write("0.000000 0.000000 0.000000 0.000000 0 0 0 1\n")


# ------------------------------------------------------------------- run one

def run_one(tag, z, args, fx, cx, cy, out_stem):
    print(f"\n=== {tag} ===")
    band = np.isfinite(z) & (z > args.depth_min) & (z < args.depth_max)
    print(f"depth in [{args.depth_min}, {args.depth_max}] m: "
          f"{100 * band.mean():.1f}% of pixels, "
          f"median {np.median(z[band]):.2f} m" if band.any() else "no valid depth")
    pts, mask = unproject(z, fx, cx, cy, args.pixel_stride,
                          args.depth_min, args.depth_max)
    if len(pts) < 1000:
        print("  too few points, skipping")
        return None
    print(f"unprojected {len(pts):,} points")

    cams = np.zeros((1, 3))
    if args.pitch is not None and args.cam_height is not None:
        n_up, origin = pinned_floor(args.pitch, args.cam_height)
        print(f"floor PINNED from the mount: pitch {args.pitch:.2f} deg, "
              f"height {args.cam_height:.2f} m")
        inl = int((np.abs((pts - origin) @ n_up) < 0.03).sum())
        print(f"  points within 3 cm of that plane: {inl:,} "
              f"({100 * inl / len(pts):.1f}%)")
    else:
        rng = np.random.default_rng(0)
        s = pts[rng.choice(len(pts), min(len(pts), 60000), replace=False)]
        normal, origin, n_in = fit_plane_ransac(
            s, 0.03, 600, rng, up_axis=np.array([0.0, 1.0, 0.0]),
            tol_deg=args.gravity_tol)
        _, _, n_up = plane_basis(normal, origin, cams)
        pitch, height, roll = describe_plane(n_up, origin)
        print(f"floor FITTED from this frame: {n_in:,}/{len(s):,} inliers "
              f"({100 * n_in / len(s):.1f}%)")
        print(f"  implies pitch {pitch:+.2f} deg below horizontal, "
              f"camera height {height:.3f} m, roll {roll:+.2f} deg")
        if n_in / len(s) < 0.15:
            print("  WARNING: low inlier fraction -- the floor is barely "
                  "visible here, so the height band below is unreliable.")

    height_above = (pts - origin) @ n_up
    n_floor = int((height_above <= args.min_h).sum())
    n_obst = int(((height_above > args.min_h) &
                  (height_above < args.max_h)).sum())
    print(f"height slice: {n_floor:,} floor/below ({100*n_floor/len(pts):.1f}%), "
          f"{n_obst:,} obstacle ({100*n_obst/len(pts):.1f}%), "
          f"{len(pts)-n_floor-n_obst:,} above {args.max_h} m")
    if n_obst == 0:
        print("  no obstacle points -- nothing to map")
        return None

    write_cloud(pts, out_stem)
    prob, lo, info = build_occupancy(
        pts, cams, res=args.res, voxel=args.voxel, min_h=args.min_h,
        max_h=args.max_h, floor_plane=(n_up, origin), max_ray=args.max_ray,
        floor_support=args.floor_support, kf_id=np.zeros(len(pts), np.int32),
        min_cell_points=args.min_cell_points,
        min_obstacle_top=args.min_obstacle_top, verbose=True)

    out = pathlib.Path(out_stem)
    np.save(out.with_suffix(".npy"), prob)
    write_pgm_yaml(prob, out, args.res, lo, 0.65, 0.25)
    H, W = prob.shape
    known = int((prob >= 0).sum())
    print(f"grid {W}x{H}: known {known:,} ({100*known/(H*W):.1f}%), "
          f"occupied {int((prob >= 65).sum()):,}")
    print(f"wrote {out_stem}.ply/.txt/.npy/.pgm/.yaml")

    return {"prob": prob, "lo": lo, "info": info, "z": z, "mask": mask,
            "height": height_above, "n_up": n_up, "origin": origin, "tag": tag}


# ---------------------------------------------------------------- plotting

def bev_rgb(prob):
    """int8 occupancy -> RGB: grey unknown, white free, black occupied."""
    img = np.full(prob.shape + (3,), 0.62, dtype=np.float32)
    known = prob >= 0
    img[known] = 1.0
    img[known & (prob >= 65)] = 0.0
    mid = known & (prob >= 25) & (prob < 65)
    img[mid] = 0.8
    return img


def display_transform(u, v):
    """2x2 taking plane coordinates to a display frame: x right, y forward.

    plane_basis picks its in-plane axes from an arbitrary seed vector, so the
    grid comes out rotated by whatever that seed gave -- on this frame forward
    lands on -x and the map reads sideways. Returns None if the basis is not
    axis-aligned (only possible with a badly tilted floor), in which case the
    raster is shown as-is rather than silently sheared.
    """
    r2d = np.array([np.array([1.0, 0, 0]) @ u, np.array([1.0, 0, 0]) @ v])
    f2d = np.array([np.array([0, 0, 1.0]) @ u, np.array([0, 0, 1.0]) @ v])
    M = np.stack([r2d / (np.linalg.norm(r2d) + 1e-9),
                  f2d / (np.linalg.norm(f2d) + 1e-9)])
    P = np.round(M)
    if not np.allclose(M, P, atol=0.25) or abs(abs(np.linalg.det(P)) - 1) > 1e-6:
        return None
    return P


def orient_forward_up(img, lo, res, P):
    """Apply `P` to a plane raster indexed [row = plane v, col = plane u]."""
    H, W = img.shape[:2]
    if abs(P[0, 0]) < 0.5:                 # display-x follows plane v
        img = img.transpose(1, 0, 2)
        sx, sy = P[0, 1], P[1, 0]
    else:
        sx, sy = P[0, 0], P[1, 1]
    if sx < 0:
        img = img[:, ::-1]
    if sy < 0:
        img = img[::-1, :]

    hi = lo + np.array([W * res, H * res])
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                    [lo[0], hi[1]], [hi[0], hi[1]]]) @ P.T
    return img, [box[:, 0].min(), box[:, 0].max(),
                 box[:, 1].min(), box[:, 1].max()]


def plot(results, bgr, args, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 2 + len(results) + (len(results) if args.show_floor else 0)
    fig, ax = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.4))
    ax = np.atleast_1d(ax)
    k = 0
    ax[k].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax[k].set_title(f"rgb  frame {args.frame}")
    ax[k].axis("off"); k += 1

    r0 = results[0]
    im = ax[k].imshow(np.where(np.isfinite(r0["z"]) & (r0["z"] > 0), r0["z"], np.nan),
                      cmap="turbo", vmin=args.depth_min, vmax=args.depth_max)
    ax[k].set_title(f"depth  {r0['tag']}")
    ax[k].axis("off")
    fig.colorbar(im, ax=ax[k], fraction=0.046, label="m"); k += 1

    for r in results:
        if args.show_floor:
            cls = np.full(r["mask"].shape, 0, np.uint8)
            h = r["height"]
            cls[r["mask"]] = np.where(h <= args.min_h, 1,
                                      np.where(h < args.max_h, 2, 3))
            pal = np.array([[0.15, 0.15, 0.15], [0.20, 0.45, 0.85],
                            [0.90, 0.30, 0.20], [0.55, 0.55, 0.20]])
            ax[k].imshow(pal[cls])
            ax[k].set_title(f"{r['tag']}: floor(blue) / obstacle(red)\n"
                            f"band [{args.min_h}, {args.max_h}] m above plane")
            ax[k].axis("off"); k += 1

        prob, lo = r["prob"], r["lo"]
        H, W = prob.shape
        u, v = r["info"]["u"], r["info"]["v"]
        img = bev_rgb(prob)
        cam2d = np.array([(-r["origin"]) @ u, (-r["origin"]) @ v])
        P = display_transform(u, v)
        if P is None:
            ext = [lo[0], lo[0] + W * args.res, lo[1], lo[1] + H * args.res]
            head = np.array([np.array([0, 0, 1.0]) @ u,
                             np.array([0, 0, 1.0]) @ v])
            note = " (plane basis not axis-aligned)"
        else:
            img, ext = orient_forward_up(img, lo, args.res, P)
            cam2d = P @ cam2d
            head = np.array([0.0, 1.0])
            note = ""
        ax[k].imshow(img, extent=ext, origin="lower")
        ax[k].plot(*cam2d, "o", color="#1f77b4", ms=7)
        ax[k].arrow(cam2d[0], cam2d[1], head[0] * 0.6, head[1] * 0.6,
                    head_width=0.15, color="#1f77b4", zorder=5)
        ax[k].set_title(f"{r['tag']}: BEV @ {args.res} m{note}\n"
                        "white free, black occupied, grey unknown")
        ax[k].set_aspect("equal")
        ax[k].set_xlabel("m right"); ax[k].set_ylabel("m forward"); k += 1

    fig.suptitle(f"single frame {args.frame}, no odometry "
                 f"(pose = identity, floor by plane fit)", y=0.99)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"\nwrote {out_png}")


# -------------------------------------------------------------------- main

def main():
    args = parse_args()

    if args.list_frames:
        fr = decodable_rgb_frames(args.dir)
        print(f"{len(fr)} decodable rgb frames:\n{fr}")
        return

    rgb_path = args.rgb or os.path.join(args.dir,
                                        f"frame_{args.frame:06d}_rgb.png")
    dep_path = args.depth or os.path.join(args.dir,
                                          f"frame_{args.frame:06d}_depth.png")
    bgr = cv2.imread(rgb_path)
    if bgr is None:
        fr = decodable_rgb_frames(args.dir)
        raise SystemExit(
            f"{rgb_path} does not decode -- the capture truncated it.\n"
            f"{len(fr)} frames have a usable rgb, e.g. {fr[:12]}\n"
            f"run with --list-frames for all of them.")
    H, W = bgr.shape[:2]
    print(f"rgb {rgb_path}  {W}x{H}")

    fx, cx, cy = load_calib(args.calib, W, H)
    if args.fx is not None:
        fx = args.fx
    if fx is None:
        raise SystemExit("no fx: pass --fx or a --calib yaml that has one")
    print(f"fx={fx:.1f}  cx={cx:.1f}  cy={cy:.1f}  "
          f"({2*np.degrees(np.arctan(W/2/fx)):.1f} deg HFOV)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if args.source in ("da2", "both"):
        print("\nDepth-Anything-V2 metric on one frame")
        cache = args.da2_npy or (str(out) + "_da2_raw.npy")
        z = da2_depth(rgb_path, cache, args.da2_python, args.refresh_da2)
        z = z * args.scale
        print(f"  metric head x{args.scale} -> "
              f"{z.min():.2f}..{z.max():.2f} m")
        r = run_one("DA2", z, args, fx, cx, cy,
                    str(out) + ("_da2" if args.source == "both" else ""))
        if r:
            results.append(r)

    if args.source in ("sensor", "both"):
        d = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED)
        if d is None:
            print(f"\nsensor depth {dep_path} does not decode -- skipping")
        else:
            z = d.astype(np.float32) / DEPTH_UNITS_PER_M
            z[z <= 0] = np.nan
            r = run_one("sensor", z, args, fx, cx, cy,
                        str(out) + ("_sensor" if args.source == "both" else ""))
            if r:
                results.append(r)

    if len(results) == 2:
        a, b = results[0]["z"], results[1]["z"]
        m = (np.isfinite(a) & np.isfinite(b) & (b > args.depth_min) &
             (b < args.depth_max))
        if m.sum() > 100:
            e = a[m] - b[m]
            print(f"\nDA2 vs sensor over {m.sum():,} shared pixels: "
                  f"median error {np.median(e)*1000:+.0f} mm, "
                  f"abs median {np.median(np.abs(e))*1000:.0f} mm, "
                  f"ratio {np.median(a[m]/b[m]):.3f}")

    if not results:
        raise SystemExit("nothing mapped")
    plot(results, bgr, args, str(out) + ".png")


if __name__ == "__main__":
    main()
