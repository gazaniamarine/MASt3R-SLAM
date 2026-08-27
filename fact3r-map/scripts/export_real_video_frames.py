#!/usr/bin/env python3
"""Sample a real video into MASt3R-aligned, geometry-free RGB keyframes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _mast3r_raster(rgb: np.ndarray, size: int = 512) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    scale = size / max(image.size)
    image = image.resize(
        tuple(max(1, round(value * scale)) for value in image.size),
        Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.BICUBIC,
    )
    width, height = image.size
    half_width = ((2 * (width // 2)) // 16) * 8
    half_height = ((2 * (height // 2)) // 16) * 8
    if width == height:
        half_height = 3 * half_width // 4
    centre_x, centre_y = width // 2, height // 2
    image = image.crop(
        (
            centre_x - half_width,
            centre_y - half_height,
            centre_x + half_width,
            centre_y + half_height,
        )
    )
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def export_video(
    video: Path,
    output: Path,
    *,
    sample_fps: float = 2.0,
    max_frames: int | None = None,
) -> Path:
    try:
        import cv2
    except ImportError as error:
        raise ImportError("install opencv-python to read real-world video") from error
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = sample_fps
    output.mkdir(parents=True, exist_ok=True)
    rgb_directory = output / "rgb"
    rgb_directory.mkdir(exist_ok=True)
    entries: list[dict[str, object]] = []
    source_frame = 0
    emitted = 0
    next_timestamp = 0.0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            timestamp = source_frame / source_fps
            if timestamp + 1e-9 < next_timestamp:
                source_frame += 1
                continue
            rgb = _mast3r_raster(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            # Association uses this contiguous sampled-frame ID for temporal
            # gaps; preserve the original video frame separately for audit.
            frame_id = emitted
            image_name = f"frame_{frame_id:06d}.jpg"
            Image.fromarray(rgb).save(
                rgb_directory / image_name, quality=96, subsampling=0
            )
            height, width = rgb.shape[:2]
            keyframe_name = f"keyframe_{emitted:06d}_frame_{frame_id:06d}.npz"
            np.savez_compressed(
                output / keyframe_name,
                frame_id=np.asarray(frame_id, dtype=np.int64),
                rgb=rgb,
                pointmap_camera=np.full((height, width, 3), np.nan, np.float32),
                geometry_confidence=np.zeros((height, width), np.float32),
                pose_world_from_camera=np.eye(4, dtype=np.float32),
            )
            entries.append(
                {
                    "keyframe_index": emitted,
                    "frame_id": frame_id,
                    "source_frame_id": source_frame,
                    "timestamp": timestamp,
                    "file": keyframe_name,
                    "rgb_file": f"rgb/{image_name}",
                    "image_shape": [height, width],
                    "has_mast3r_descriptors": False,
                }
            )
            emitted += 1
            next_timestamp = emitted / sample_fps
            source_frame += 1
            if max_frames is not None and emitted >= max_frames:
                break
    finally:
        capture.release()
    if not entries:
        raise ValueError("video produced no frames")
    manifest = {
        "format": "fact3r-mast3r-keyframes",
        "version": 1,
        "coordinate_convention": "image_only_no_geometry",
        "mode": "real_world_pairwise_matching",
        "source_video": str(video.resolve()),
        "source_fps": source_fps,
        "sample_fps": sample_fps,
        "keyframes": entries,
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    path = export_video(
        args.video,
        args.output,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
    )
    count = len(json.loads(path.read_text(encoding="utf-8"))["keyframes"])
    print(f"Exported {count} geometry-free RGB frames to {path}")


if __name__ == "__main__":
    main()
