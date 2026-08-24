#!/usr/bin/env python3
"""
Visual check for an HM3D run: a top-down view of the reconstructed point cloud
with the estimated trajectory drawn over it, next to the estimated vs
ground-truth trajectory after Sim(3) alignment.

    python3 scripts/plot_hm3d_recon.py --run hm3d/calib --scene 00805-SUHsP6z2gcJ
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_hm3d import read_tum, associate, umeyama

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_ply_xyzrgb(path, max_points=400000, seed=0):
    with open(path, "rb") as f:
        n = None
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        raw = np.frombuffer(f.read(n * 15), dtype=np.uint8)
    n = raw.size // 15
    raw = raw[: n * 15].reshape(n, 15)
    xyz = np.frombuffer(raw[:, :12].tobytes(), dtype="<f4").reshape(n, 3).astype(np.float64)
    rgb = raw[:, 12:15].astype(np.float32) / 255.0
    ok = np.isfinite(xyz).all(1)
    xyz, rgb = xyz[ok], rgb[ok]
    if len(xyz) > max_points:
        idx = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="hm3d/calib")
    p.add_argument("--scene", required=True)
    p.add_argument("--seqs", default=os.path.join(REPO_ROOT, "datasets", "hm3d_seqs"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    log_dir = os.path.join(REPO_ROOT, "logs", args.run)
    ply = os.path.join(log_dir, f"{args.scene}.ply")
    est_path = os.path.join(log_dir, f"{args.scene}.txt")
    gt_path = os.path.join(args.seqs, args.scene, "groundtruth.txt")

    xyz, rgb = read_ply_xyzrgb(ply)
    t_est, x_est = read_tum(est_path)
    t_gt, x_gt = read_tum(gt_path)
    i, j = associate(t_est, t_gt)
    scale, R, t = umeyama(x_est[i], x_gt[j], with_scale=True)
    est_aligned = (scale * (R @ x_est.T)).T + t

    # Trim the sparse outlier tail so the top-down view isn't all empty space.
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    keep = ((xyz >= lo) & (xyz <= hi)).all(1)
    xyz_v, rgb_v = xyz[keep], rgb[keep]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax = axes[0]
    # Height-sorted so floor points do not paint over furniture.
    order = np.argsort(-xyz_v[:, 1])
    ax.scatter(xyz_v[order, 0], xyz_v[order, 2], c=rgb_v[order], s=0.3, marker=".", linewidths=0)
    ax.plot(x_est[:, 0], x_est[:, 2], "-", color="red", lw=1.5, label="estimated trajectory")
    ax.set_aspect("equal")
    ax.set_title(f"{args.scene}: reconstruction (top-down), {len(xyz):,} pts shown")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(x_gt[:, 0], x_gt[:, 2], "-", color="black", lw=2, label="ground truth (habitat)")
    ax.plot(est_aligned[:, 0], est_aligned[:, 2], "-", color="red", lw=1.5, label="MASt3R-SLAM (Sim3 aligned)")
    ax.scatter(x_gt[0, 0], x_gt[0, 2], c="green", s=60, zorder=5, label="start")
    ax.set_aspect("equal")
    err = np.linalg.norm(est_aligned[i] - x_gt[j], axis=1)
    ax.set_title(f"trajectory: ATE rmse {np.sqrt((err**2).mean()):.3f} m, scale {scale:.3f}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="best")

    fig.tight_layout()
    out = args.out or os.path.join(log_dir, f"{args.scene}_view.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
