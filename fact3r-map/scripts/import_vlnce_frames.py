#!/usr/bin/env python3
"""Turn a rendered habitat tour into a Fact3R frame export.

`scripts/render_vlnce_tour.py` writes a PNG sequence; the mapping pipeline
expects the `fact3r-mast3r-keyframes` layout that
`export_real_video_frames.py` produces from video. This bridges the two so
`run_fact3r_real_uot.sh` can reuse an existing frames directory and skip its
own sampling stage.

Keyframes keep the image-only convention of the video path: identity poses and
NaN pointmaps. Habitat ground truth is real and available, but writing it into
the keyframes would give the outbound map geometry the rover runs never had,
and the two would stop being comparable. Ground-truth poses are carried in the
manifest instead, for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_real_video_frames import _mast3r_raster  # noqa: E402


def _read_groundtruth(path: Path) -> list[dict]:
    """Parse the TUM-style pose file the renderer writes."""

    if not path.is_file():
        return []
    poses = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            continue
        values = [float(part) for part in parts]
        poses.append(
            {
                "timestamp": values[0],
                "position": values[1:4],
                "quaternion_xyzw": values[4:8],
            }
        )
    return poses


def _cropped_intrinsics(source, sample_image, size, image_shape):
    """Re-derive intrinsics after the MASt3R resize and centre crop.

    `_mast3r_raster` rescales to `size` on the long edge and then centre-crops
    a square render to 4:3, so the source intrinsics no longer describe the
    saved images. Both operations are centred, which keeps the principal point
    at the image centre and leaves the focal length scaled by the resize alone.
    """

    if not source:
        return None
    width, height = Image.open(sample_image).size
    scale = size / max(width, height)
    out_height, out_width = image_shape
    return {
        "fx": float(source["fx"]) * scale,
        "fy": float(source["fy"]) * scale,
        "cx": out_width / 2.0,
        "cy": out_height / 2.0,
        "width": out_width,
        "height": out_height,
    }


def _frame_stride(render_fps: float, sample_fps: float) -> int:
    if sample_fps <= 0.0:
        raise ValueError("sample_fps must be positive")
    if sample_fps >= render_fps:
        return 1
    return max(1, int(round(render_fps / sample_fps)))


def convert(
    render_directory: Path,
    output: Path,
    *,
    sample_fps: float,
    max_frames: int | None,
    size: int,
) -> Path:
    images = sorted(render_directory.glob("*.png"))
    if not images:
        raise ValueError(f"no PNG frames in {render_directory}")

    meta_path = render_directory / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    render_fps = float(meta.get("fps", 30.0))
    poses = _read_groundtruth(render_directory / "groundtruth.txt")

    stride = _frame_stride(render_fps, sample_fps)
    selected = images[::stride]
    if max_frames:
        selected = selected[:max_frames]

    rgb_directory = output / "rgb"
    rgb_directory.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for emitted, image_path in enumerate(selected):
        source_frame = int(image_path.stem)
        rgb = _mast3r_raster(np.asarray(Image.open(image_path).convert("RGB")), size)
        image_name = f"frame_{emitted:06d}.jpg"
        Image.fromarray(rgb).save(rgb_directory / image_name, quality=96, subsampling=0)
        height, width = rgb.shape[:2]
        keyframe_name = f"keyframe_{emitted:06d}_frame_{emitted:06d}.npz"
        np.savez_compressed(
            output / keyframe_name,
            frame_id=np.asarray(emitted, dtype=np.int64),
            rgb=rgb,
            pointmap_camera=np.full((height, width, 3), np.nan, np.float32),
            geometry_confidence=np.zeros((height, width), np.float32),
            pose_world_from_camera=np.eye(4, dtype=np.float32),
        )
        entry = {
            "keyframe_index": emitted,
            "frame_id": emitted,
            "source_frame_id": source_frame,
            "timestamp": source_frame / render_fps,
            "file": keyframe_name,
            "rgb_file": f"rgb/{image_name}",
            "image_shape": [height, width],
            "has_mast3r_descriptors": False,
        }
        if source_frame < len(poses):
            entry["groundtruth_pose"] = poses[source_frame]
        entries.append(entry)

    if not entries:
        raise ValueError("no frames selected")

    keyframe_intrinsics = _cropped_intrinsics(meta.get("intrinsics"), selected[0], size,
                                              entries[0]["image_shape"])

    manifest = {
        "format": "fact3r-mast3r-keyframes",
        "version": 1,
        "coordinate_convention": "image_only_no_geometry",
        "mode": "real_world_pairwise_matching",
        "source_video": str(render_directory.resolve()),
        "source_fps": render_fps,
        "sample_fps": min(sample_fps, render_fps),
        "keyframes": entries,
        # Everything the return leg and the scorer need, carried forward so the
        # evaluation never has to re-open the render directory.
        "vlnce": {
            "scene": meta.get("scene"),
            "legs": meta.get("legs"),
            "return_query": meta.get("return_query"),
            "return_leg_index": meta.get("return_leg_index"),
            "return_position": meta.get("return_position"),
            "return_optimal_geodesic_m": meta.get("return_optimal_geodesic_m"),
            "tour_start_position": meta.get("tour_start_position"),
            "tour_final_position": meta.get("tour_final_position"),
            "success_distance_m": meta.get("success_distance_m"),
            "source_intrinsics": meta.get("intrinsics"),
            "keyframe_intrinsics": keyframe_intrinsics,
            "frame_stride": stride,
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if (render_directory / "groundtruth.txt").is_file():
        shutil.copy2(render_directory / "groundtruth.txt", output / "groundtruth.txt")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", required=True, help="A rendered tour directory.")
    parser.add_argument("--output", required=True, help="Frames export directory.")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Frames per second to keep; matches the rover pipeline default.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    manifest = convert(
        Path(args.render),
        Path(args.output),
        sample_fps=args.sample_fps,
        max_frames=args.max_frames or None,
        size=args.size,
    )
    payload = json.loads(manifest.read_text())
    print(
        f"wrote {len(payload['keyframes'])} keyframes (stride "
        f"{payload['vlnce']['frame_stride']}) -> {manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
