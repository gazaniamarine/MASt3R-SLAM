#!/usr/bin/env python3
"""Check that lifted SAM2 proposals agree across keyframes.

Three questions, in the order that makes a failure interpretable:

1. Is the lift arithmetic right? Recompute world points from the keyframe
   pointmap and pose and compare against what was stored.
2. Is the *reconstruction* co-registered at all? Consecutive raw keyframe
   pointmaps are compared without involving SAM2. If this is low, nothing about
   the proposals can be concluded.
3. Do proposals re-observe the same surface? Reported only after splitting out
   proposals whose points left the next keyframe's frustum, because those are
   not failures -- the object simply went out of view.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _voxels(points: np.ndarray, size: float) -> set[tuple[int, int, int]]:
    return set(map(tuple, np.floor(points / size).astype(np.int64)))


def _world(pointmap: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return pointmap @ pose[:3, :3].T + pose[:3, 3]


def _visible_fraction(points: np.ndarray, pose, intrinsics, shape) -> float:
    """Fraction of ``points`` that project inside the keyframe image."""

    height, width = shape
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    depth = camera[:, 2]
    in_front = depth > 1e-6
    if not in_front.any():
        return 0.0
    safe = np.where(in_front, depth, 1.0)
    u = intrinsics[0, 0] * camera[:, 0] / safe + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[:, 1] / safe + intrinsics[1, 2]
    inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return float(inside.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.10)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=1.0,
        help="confidence floor for the raw-pointmap control",
    )
    args = parser.parse_args()

    run = json.loads((args.proposals / "manifest.json").read_text())
    keyframe_index = {
        k["frame_id"]: k
        for k in json.loads((args.keyframes / "manifest.json").read_text())["keyframes"]
    }
    print(f"backend={run['backend']}  model={run['model']}  frames={run['frame_count']}")

    frames = []
    for entry in run["frames"]:
        manifest = json.loads((args.proposals / entry["manifest"]).read_text())
        directory = args.proposals / f"frame_{manifest['frame_id']:06d}"
        points = []
        for proposal in manifest["proposals"]:
            with np.load(directory / proposal["file"], allow_pickle=False) as data:
                if "points_world" in data.files:
                    points.append(np.array(data["points_world"], copy=True))
        frames.append((manifest["frame_id"], points))

    # --- 1. lift arithmetic ---
    worst = 0.0
    for frame_id, points in frames[:5]:
        data = np.load(args.keyframes / keyframe_index[frame_id]["file"])
        manifest = json.loads(
            (args.proposals / f"frame_{frame_id:06d}" / "manifest.json").read_text()
        )
        directory = args.proposals / f"frame_{frame_id:06d}"
        for entry in manifest["proposals"][:5]:
            stored = np.load(directory / entry["file"])
            if "points_world" not in stored.files:
                continue
            rc = stored["pixel_rc"]
            recomputed = _world(
                data["pointmap_camera"][rc[:, 0], rc[:, 1]],
                data["pose_world_from_camera"],
            )
            worst = max(worst, float(np.abs(recomputed - stored["points_world"]).max()))
    print(f"\n[1] lift reproduces stored world points to {worst:.2e} m "
          f"-> {'OK' if worst < 1e-4 else 'MISMATCH'}")

    # --- 2. raw pointmap control ---
    controls = []
    previous = None
    for frame_id, _ in frames:
        data = np.load(args.keyframes / keyframe_index[frame_id]["file"])
        points = _world(
            data["pointmap_camera"].reshape(-1, 3), data["pose_world_from_camera"]
        )
        keep = (
            data["geometry_confidence"].reshape(-1) > args.min_confidence
        ) & np.isfinite(points).all(axis=1)
        current = _voxels(points[keep], args.voxel)
        if previous is not None:
            controls.append(
                len(previous & current) / max(1, min(len(previous), len(current)))
            )
        previous = current
    print(f"[2] raw keyframe pointmaps share "
          f"{np.median(controls):.1%} of {args.voxel*100:.0f}cm voxels (median) "
          f"-> {'co-registered' if np.median(controls) > 0.2 else 'CHECK POSES'}")

    # --- 3. proposal re-observation ---
    reobserved, visible = [], []
    for (frame_a, points_a), (frame_b, points_b) in zip(frames, frames[1:]):
        data = np.load(args.keyframes / keyframe_index[frame_b]["file"])
        pose = data["pose_world_from_camera"]
        intrinsics = data["intrinsics"]
        shape = data["geometry_confidence"].shape
        union: set = set().union(*[_voxels(p, args.voxel) for p in points_b]) if points_b else set()
        for points in points_a:
            cells = _voxels(points, args.voxel)
            reobserved.append(len(cells & union) / len(cells))
            visible.append(_visible_fraction(points, pose, intrinsics, shape))
    reobserved, visible = np.array(reobserved), np.array(visible)
    zero = reobserved == 0
    missed = zero & (visible > 0.5)

    print(f"[3] proposals compared: {len(reobserved)}")
    print(f"    matched proposals keep median {np.median(visible[~zero]):.1%} of "
          f"points in the next frame")
    print(f"    zero-overlap: {zero.mean():.1%}, of which "
          f"{(visible[zero] == 0).mean():.1%} left the frustum entirely")
    print(f"    GENUINE MISSES (in view, no overlap): {missed.sum()} "
          f"({missed.mean():.1%} of proposals)")


if __name__ == "__main__":
    main()
