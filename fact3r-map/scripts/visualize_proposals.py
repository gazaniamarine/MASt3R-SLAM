#!/usr/bin/env python3
"""Inspection output for a proposal run: 2D mask overlays and one merged cloud.

The per-frame ``alignment.ply`` written by the build script shows a single
keyframe. This merges every frame so the proposals can be seen against the whole
reconstruction, which is where mask instability across viewpoints shows up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _write_ply(path: Path, points: np.ndarray, colours: np.ndarray) -> None:
    vertex = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colours.T
    with open(path, "wb") as handle:
        handle.write(b"ply\nformat binary_little_endian 1.0\n")
        handle.write(f"element vertex {len(points)}\n".encode())
        handle.write(b"property float x\nproperty float y\nproperty float z\n")
        handle.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(b"end_header\n")
        handle.write(vertex.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-every", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=3_000_000)
    args = parser.parse_args()

    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=True)
    run = json.loads((args.proposals / "manifest.json").read_text())
    keyframe_index = {
        k["frame_id"]: k
        for k in json.loads((args.keyframes / "manifest.json").read_text())["keyframes"]
    }
    # Fixed seed: proposal colours should not change between runs, or two
    # inspection images of the same scene cannot be compared.
    rng = np.random.default_rng(0)

    all_points, all_colours, overlays = [], [], 0
    for index, entry in enumerate(run["frames"]):
        manifest = json.loads((args.proposals / entry["manifest"]).read_text())
        frame_id = manifest["frame_id"]
        directory = args.proposals / f"frame_{frame_id:06d}"
        rgb = np.load(args.keyframes / keyframe_index[frame_id]["file"])["rgb"].astype(np.float32)
        overlay = rgb.copy()
        for proposal in manifest["proposals"]:
            data = np.load(directory / proposal["file"])
            colour = rng.integers(40, 255, 3).astype(np.float32)
            mask = data["mask"].astype(bool)
            overlay[mask] = 0.45 * overlay[mask] + 0.55 * colour
            points = data["points_world"]
            all_points.append(points)
            all_colours.append(np.repeat(colour[None, :], len(points), axis=0))
        if index % args.overlay_every == 0:
            side = np.concatenate([rgb, overlay], axis=1).clip(0, 255).astype(np.uint8)
            Image.fromarray(side).save(args.output / f"masks_frame_{frame_id:06d}.png")
            overlays += 1

    points = np.concatenate(all_points).astype(np.float32)
    colours = np.concatenate(all_colours).clip(0, 255).astype(np.uint8)
    step = max(1, len(points) // args.max_points)
    _write_ply(args.output / "scene_proposals.ply", points[::step], colours[::step])
    print(f"{overlays} overlays and scene_proposals.ply "
          f"({len(points[::step]):,} of {len(points):,} points) -> {args.output}")


if __name__ == "__main__":
    main()
