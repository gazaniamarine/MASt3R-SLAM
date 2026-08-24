#!/usr/bin/env python3
"""
Evaluate MASt3R-SLAM runs on the rendered HM3D sequences.

Associates the estimated keyframe trajectory (logs/<run>/<scene>.txt) with the
habitat ground truth (datasets/hm3d_seqs/<scene>/groundtruth.txt) by timestamp
and reports absolute trajectory error under both SE(3) alignment (rotation +
translation, scale fixed at 1 -- meaningful because the metric MASt3R
checkpoint predicts real-world scale) and Sim(3) alignment (scale free, which
also reveals the scale MASt3R actually recovered).

Only translation is compared: habitat's camera axes differ from the SLAM
convention by a constant right-multiplied rotation, which leaves camera
positions untouched but would corrupt a naive rotation comparison.

    python3 scripts/eval_hm3d.py --run hm3d
"""
import argparse
import json
import os
import struct

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_tum(path):
    ts, xyz = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ts.append(float(parts[0]))
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(ts), np.array(xyz, dtype=np.float64)


def associate(t_est, t_gt, max_diff=0.02):
    """Match each estimated stamp to the nearest ground-truth stamp."""
    idx_est, idx_gt = [], []
    for i, t in enumerate(t_est):
        j = int(np.argmin(np.abs(t_gt - t)))
        if abs(t_gt[j] - t) <= max_diff:
            idx_est.append(i)
            idx_gt.append(j)
    return np.array(idx_est, dtype=int), np.array(idx_gt, dtype=int)


def umeyama(src, dst, with_scale):
    """Rigid/similarity transform mapping src onto dst (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    if with_scale:
        var_s = (s0**2).sum() / src.shape[0]
        scale = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    else:
        scale = 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def ate(src, dst, with_scale):
    scale, R, t = umeyama(src, dst, with_scale)
    aligned = (scale * (R @ src.T)).T + t
    err = np.linalg.norm(aligned - dst, axis=1)
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "max": float(err.max()),
        "scale": scale,
    }


def ply_stats(path):
    """Vertex count and bounding box of a binary little-endian PLY."""
    # Stride is summed from the header's property lines rather than assumed.
    # It used to be hard-coded to 15 (xyz float32 + rgb uchar), which silently
    # misreads every vertex the moment save_ply gains a field -- as it did when
    # per-point confidence and keyframe id were added.
    sizes = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1,
             "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
             "int": 4, "uint": 4, "int32": 4, "uint32": 4,
             "float": 4, "float32": 4,
             "double": 8, "float64": 8}
    with open(path, "rb") as f:
        n, stride, in_vertex = None, 0, False
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element "):
                # Only the vertex element's properties contribute to its stride.
                in_vertex = line.split()[1] == "vertex"
                if in_vertex:
                    n = int(line.split()[-1])
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties not supported")
                stride += sizes[parts[1]]
            if line == "end_header":
                break
        data = f.read(n * stride)
    if len(data) < n * stride:
        n = len(data) // stride
    xyz = np.frombuffer(
        np.frombuffer(data[: n * stride], dtype=np.uint8).reshape(n, stride)[:, :12].tobytes(),
        dtype="<f4",
    ).reshape(n, 3)
    finite = xyz[np.isfinite(xyz).all(1)]
    return {
        "points": int(n),
        "bbox_min": finite.min(0).tolist() if len(finite) else None,
        "bbox_max": finite.max(0).tolist() if len(finite) else None,
        "extent_m": (finite.max(0) - finite.min(0)).tolist() if len(finite) else None,
    }


def path_length(xyz):
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="hm3d", help="logs/<run> subdirectory")
    p.add_argument("--seqs", default=os.path.join(REPO_ROOT, "datasets", "hm3d_seqs"))
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    log_dir = os.path.join(REPO_ROOT, "logs", args.run)
    if not os.path.isdir(log_dir):
        raise SystemExit(f"No such run directory: {log_dir}")

    rows = []
    for fn in sorted(os.listdir(log_dir)):
        if not fn.endswith(".txt"):
            continue
        scene = fn[:-4]
        est_path = os.path.join(log_dir, fn)
        gt_path = os.path.join(args.seqs, scene, "groundtruth.txt")
        if not os.path.isfile(gt_path):
            print(f"{scene}: no ground truth, skipping")
            continue

        t_est, x_est = read_tum(est_path)
        t_gt, x_gt = read_tum(gt_path)
        i, j = associate(t_est, t_gt)
        if len(i) < 3:
            print(f"{scene}: only {len(i)} associated poses, skipping")
            continue

        # Fraction of the sequence the estimate actually spans. Without this,
        # a run that loses tracking early scores a flatteringly low ATE
        # because only the tracked prefix is ever compared.
        coverage = float(t_est[-1] / t_gt[-1]) if t_gt[-1] > 0 else 0.0

        row = {
            "scene": scene,
            "keyframes": int(len(t_est)),
            "associated": int(len(i)),
            "gt_frames": int(len(t_gt)),
            "coverage": coverage,
            "gt_path_len_m": path_length(x_gt),
            "se3": ate(x_est[i], x_gt[j], with_scale=False),
            "sim3": ate(x_est[i], x_gt[j], with_scale=True),
        }
        ply = os.path.join(log_dir, f"{scene}.ply")
        if os.path.isfile(ply):
            row["recon"] = ply_stats(ply)
        rows.append(row)

    hdr = (
        f"{'scene':22s} {'kf':>4s} {'cover':>6s} {'gt_len_m':>9s} "
        f"{'ATE_se3':>9s} {'ATE_sim3':>9s} {'scale':>7s} {'points':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        pts = r.get("recon", {}).get("points", 0)
        flag = "" if r["coverage"] >= 0.9 else "  <-- lost tracking"
        print(
            f"{r['scene']:22s} {r['keyframes']:4d} {100 * r['coverage']:5.1f}% "
            f"{r['gt_path_len_m']:9.1f} {r['se3']['rmse']:9.3f} {r['sim3']['rmse']:9.3f} "
            f"{r['sim3']['scale']:7.3f} {pts:10d}{flag}"
        )

    # Only sequences tracked end-to-end are comparable: a truncated run's ATE
    # is computed over the tracked prefix alone.
    full = [r for r in rows if r["coverage"] >= 0.9]
    if full:
        print("-" * len(hdr))
        print(
            f"{'MEDIAN (' + str(len(full)) + '/' + str(len(rows)) + ' full-coverage)':22s} "
            f"{'':4s} {'':6s} {'':9s} "
            f"{np.median([r['se3']['rmse'] for r in full]):9.3f} "
            f"{np.median([r['sim3']['rmse'] for r in full]):9.3f} "
            f"{np.median([r['sim3']['scale'] for r in full]):7.3f}"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
