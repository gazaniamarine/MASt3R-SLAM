#!/usr/bin/env python3
"""Fuse metric monocular depth, rover odometry, and Fact3R entities into BEV.

This consumes the exact RGB keyframes and masks referenced by a completed
SigLIP/Qwen observation index. That keeps depth pixels and semantic masks
aligned and avoids trying to synchronize two independently decoded videos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from fact3r.integrations.mast3r_slam import iter_exported_keyframes  # noqa: E402
from fact3r.semantics.observation_index import load_observation_index  # noqa: E402
from fact3r.semantics.semantic_bev import (  # noqa: E402
    aggregate_group_embeddings,
    backproject_depth,
    build_semantic_grid,
    camera_points_to_rover_map,
    camera_to_body,
)
METRIC_CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def _sidecar(stem: Path, suffix: str) -> Path:
    return Path(f"{stem}{suffix}")


def _load_odometry(path: Path) -> tuple[np.ndarray, ...]:
    timestamp, x, y, theta, velocity = [], [], [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            timestamp.append(float(row["t"]))
            x.append(float(row["x"]))
            y.append(float(row["y"]))
            theta.append(float(row["theta"]))
            velocity.append(float(row.get("v", 0.0)))
    if len(timestamp) < 2:
        raise ValueError(f"odometry needs at least two rows: {path}")
    time = np.asarray(timestamp, dtype=np.float64)
    time -= time[0]
    if np.any(np.diff(time) <= 0):
        raise ValueError("odometry timestamps must be strictly increasing")
    return (
        time,
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.unwrap(np.asarray(theta, dtype=np.float64)),
        np.asarray(velocity, dtype=np.float64),
    )


def _find_odometry(root: Path) -> Path:
    matches = sorted(root.glob("odom_*.csv"))
    if not matches:
        raise ValueError(f"no odom_*.csv found under {root}")
    return matches[0]


def _mast3r_intrinsics(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    raster_size: int = 512,
) -> tuple[float, float, float, float]:
    """Apply the exporter resize and centered crop to source intrinsics."""

    scale = raster_size / max(source_width, source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    crop_x = (resized_width - target_width) / 2.0
    crop_y = (resized_height - target_height) / 2.0
    return (
        fx * scale,
        fy * scale,
        cx * scale - crop_x,
        cy * scale - crop_y,
    )


def _source_image_shape(keyframe_manifest: dict[str, object]) -> tuple[int, int] | None:
    source = keyframe_manifest.get("source_video")
    if source is None:
        return None
    path = Path(str(source))
    if not path.exists():
        return None
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return (width, height) if width > 0 and height > 0 else None


def _predict_depth_batch(
    processor, model, frames: list[np.ndarray], *, device: str, dtype
) -> list[np.ndarray]:
    import cv2
    import torch

    if not frames:
        return []
    shapes = [frame.shape[:2] for frame in frames]
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    inputs = processor(images=rgb, return_tensors="pt").to(device, dtype)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_depth_estimation(outputs, target_sizes=shapes)
    return [item["predicted_depth"].float().cpu().numpy() for item in results]


def _group_metadata(
    observations: list[dict[str, object]], group_ids: list[str]
) -> list[dict[str, object]]:
    metadata = []
    for index, group_id in enumerate(group_ids):
        members = [item for item in observations if str(item["group_id"]) == group_id]
        first = members[0]
        metadata.append(
            {
                "semantic_id": index,
                "group_id": group_id,
                "entity_id": first.get("entity_id"),
                "track_id": first.get("track_id"),
                "observation_count": len(members),
                "prototype_row": index,
            }
        )
    return metadata


def _colour(group_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return 55 + digest[0] % 190, 55 + digest[1] % 190, 55 + digest[2] % 190


def _render_semantic_map(
    occupancy: np.ndarray,
    semantic_ids: np.ndarray,
    groups: list[dict[str, object]],
    output: Path,
) -> None:
    height, width = occupancy.shape
    canvas = np.full((height, width, 3), 128, dtype=np.uint8)
    known = occupancy >= 0
    canvas[known] = 245
    canvas[occupancy >= 65] = 35
    for group in groups:
        semantic_id = int(group["semantic_id"])
        mask = semantic_ids == semantic_id
        if mask.any():
            colour = np.asarray(_colour(str(group["group_id"])), dtype=np.float32)
            canvas[mask] = (0.25 * canvas[mask] + 0.75 * colour).astype(np.uint8)
    # Grid row zero is the lower edge in ROS/occupancy coordinates.
    map_image = Image.fromarray(canvas[::-1])
    visible_groups = [
        group for group in groups if np.any(semantic_ids == int(group["semantic_id"]))
    ]
    legend_width = 330 if visible_groups else 0
    image = Image.new("RGB", (width + legend_width, height), (24, 24, 24))
    image.paste(map_image, (0, 0))
    if visible_groups:
        draw = ImageDraw.Draw(image)
        draw.text((width + 12, 10), "Persistent semantic entities", fill="white")
        for row, group in enumerate(visible_groups[:30]):
            y = 34 + row * 18
            colour = _colour(str(group["group_id"]))
            draw.rectangle((width + 12, y, width + 24, y + 12), fill=colour)
            label = str(group["group_id"])
            draw.text((width + 31, y - 1), label[:43], fill=(235, 235, 235))
        if len(visible_groups) > 30:
            draw.text(
                (width + 12, 34 + 30 * 18),
                f"+ {len(visible_groups) - 30} more (see JSON)",
                fill=(190, 190, 190),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _write_ply(path: Path, points: np.ndarray, keyframes: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("kf_id", "<i4")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["kf_id"] = keyframes
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property int kf_id\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def _write_semantic_ply(
    path: Path, points: np.ndarray, semantic_ids: np.ndarray, weights: np.ndarray
) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("semantic_id", "<i4"),
            ("weight", "<f4"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["semantic_id"] = semantic_ids
    vertices["weight"] = weights
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property int semantic_id\nproperty float weight\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def _keyframe_timestamps(
    keyframes, keyframe_entries: dict[int, dict[str, object]], frame_fps: float
) -> np.ndarray:
    """Keyframe times on the video clock, before any offset is applied."""

    stamps = []
    for keyframe in keyframes:
        entry = keyframe_entries.get(keyframe.frame_id, {})
        raw = entry.get("timestamp", keyframe.timestamp)
        try:
            stamps.append(float(raw))
        except (TypeError, ValueError):
            stamps.append(keyframe.frame_id / frame_fps)
    return np.asarray(stamps, dtype=np.float64)


def _resolve_time_offset(
    video_t: np.ndarray, odom_t: np.ndarray, args: argparse.Namespace
) -> tuple[float, dict[str, object]]:
    """Settle the video->odometry clock offset, refusing to guess it silently.

    The keyframe timestamps are on the video clock and `_load_odometry` re-bases
    odometry to its own first row, so the two only agree if the camera and the
    logger started together. On the 2026-08-26 capture they did not -- logging
    ran 27 s early -- and because the whole video interval still sits inside the
    odometry interval, nothing downstream notices: no frame is skipped, the PNG
    renders, and every pose is wrong by 2.65 m and 89 degrees.

    So a duration disagreement larger than `--clock-tolerance` makes an explicit
    `--time-offset` mandatory. The number itself comes from
    scripts/find_time_offset.py, which correlates image pan against logged yaw
    rate; it is not something this script can recover from durations alone,
    because equal durations do not imply equal start times either.
    """

    video_span = float(video_t[-1] - video_t[0])
    odom_span = float(odom_t[-1] - odom_t[0])
    mismatch = abs(odom_span - video_span)
    check = {
        "video_span_seconds": video_span,
        "odometry_span_seconds": odom_span,
        "duration_mismatch_seconds": mismatch,
        "tolerance_seconds": args.clock_tolerance,
    }
    if args.time_offset is None:
        if mismatch > args.clock_tolerance and not args.assume_synchronised:
            raise SystemExit(
                f"refusing to fuse: the streams disagree by {mismatch:.1f}s "
                f"(video {video_span:.1f}s, odometry {odom_span:.1f}s) and no "
                "--time-offset was given. Every pose would be silently wrong "
                "while skipped_frames stayed 0. Measure the offset with\n"
                "  python scripts/find_time_offset.py --root <capture dir>\n"
                "and pass it as --time-offset SECONDS, or override with "
                "--assume-synchronised if the clocks really are shared."
            )
        offset = 0.0
        check["source"] = "assumed-synchronised"
    else:
        offset = float(args.time_offset)
        check["source"] = "explicit"
    check["time_offset_seconds"] = offset
    # After the shift the video interval should land inside the odometry one;
    # anything sticking out is time the rover was not being logged for.
    check["outside_odometry_seconds"] = float(
        max(0.0, odom_t[0] - (video_t[0] + offset))
        + max(0.0, (video_t[-1] + offset) - odom_t[-1])
    )
    print(
        f"clock check: video {video_span:.1f}s, odometry {odom_span:.1f}s, "
        f"offset {offset:+.2f}s ({check['source']}), "
        f"{check['outside_odometry_seconds']:.1f}s of video outside odometry"
    )
    return offset, check


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    odometry = parser.add_mutually_exclusive_group(required=True)
    odometry.add_argument("--odom", type=Path)
    odometry.add_argument("--root", type=Path, help="directory containing odom_*.csv")
    parser.add_argument("--out", type=Path, required=True, help="output stem")
    parser.add_argument("--fx", type=float, default=631.0)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument(
        "--source-width",
        type=int,
        help="original video width when its recorded path is unavailable",
    )
    parser.add_argument(
        "--source-height",
        type=int,
        help="original video height when its recorded path is unavailable",
    )
    parser.add_argument(
        "--intrinsics-are-keyframe",
        action="store_true",
        help="fx/fy/cx/cy already describe the saved resized Fact3R frames",
    )
    parser.add_argument("--pitch", type=float, default=2.75)
    parser.add_argument("--cam-height", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=0.969)
    parser.add_argument(
        "--time-offset",
        type=float,
        help="seconds added to the keyframe (video-clock) timestamps to reach "
        "the odometry clock; measure it with scripts/find_time_offset.py",
    )
    parser.add_argument(
        "--clock-tolerance",
        type=float,
        default=2.0,
        help="seconds the two stream durations may disagree by before an "
        "explicit --time-offset becomes mandatory",
    )
    parser.add_argument(
        "--assume-synchronised",
        action="store_true",
        help="accept a zero offset despite a duration mismatch",
    )
    parser.add_argument("--frame-fps", type=float, default=2.0)
    parser.add_argument("--pixel-stride", type=int, default=3)
    parser.add_argument("--semantic-pixel-stride", type=int, default=3)
    parser.add_argument("--depth-min", type=float, default=0.3)
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--min-height", type=float, default=0.10)
    parser.add_argument("--max-height", type=float, default=1.50)
    parser.add_argument("--min-cell-points", type=int, default=4)
    parser.add_argument("--max-ray", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--stationary-skip", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-model", default=METRIC_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from occupancy_grid import build_occupancy, voxel_downsample, write_pgm_yaml

    if args.batch <= 0 or args.pixel_stride <= 0 or args.semantic_pixel_stride <= 0:
        raise ValueError("batch and pixel strides must be positive")
    if (args.source_width is None) != (args.source_height is None):
        raise ValueError("source-width and source-height must be provided together")
    odometry_path = args.odom or _find_odometry(args.root)
    odom_t, odom_x, odom_y, odom_yaw, odom_velocity = _load_odometry(
        odometry_path
    )
    index_path, index_manifest, embeddings = load_observation_index(args.index)
    observations = [dict(item) for item in index_manifest["observations"]]
    if not observations:
        raise ValueError("semantic observation index is empty")
    keyframe_directory = Path(str(index_manifest["source_keyframes"]))
    proposal_directory = Path(str(index_manifest["source_proposals"]))
    keyframe_manifest = json.loads(
        (keyframe_directory / "manifest.json").read_text(encoding="utf-8")
    )
    keyframe_entries = {
        int(item["frame_id"]): item for item in keyframe_manifest["keyframes"]
    }
    keyframes = list(iter_exported_keyframes(keyframe_directory))
    if args.max_frames:
        keyframes = keyframes[: args.max_frames]
    if not keyframes:
        raise ValueError("source keyframe export is empty")
    time_offset, clock_check = _resolve_time_offset(
        _keyframe_timestamps(keyframes, keyframe_entries, args.frame_fps),
        odom_t,
        args,
    )

    first_height, first_width = keyframes[0].image_shape
    source_shape = (
        (args.source_width, args.source_height)
        if args.source_width is not None
        else _source_image_shape(keyframe_manifest)
    )
    fy = args.fx if args.fy is None else args.fy
    calibration_width, calibration_height = (
        (first_width, first_height)
        if args.intrinsics_are_keyframe or source_shape is None
        else source_shape
    )
    cx = calibration_width / 2.0 if args.cx is None else args.cx
    cy = calibration_height / 2.0 if args.cy is None else args.cy
    if not args.intrinsics_are_keyframe and source_shape is not None:
        fx, fy, cx, cy = _mast3r_intrinsics(
            source_width=source_shape[0],
            source_height=source_shape[1],
            target_width=first_width,
            target_height=first_height,
            fx=args.fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        print(
            f"transformed source-video intrinsics to semantic raster: "
            f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
        )
    else:
        fx = args.fx
        if source_shape is None and not args.intrinsics_are_keyframe:
            print(
                "WARNING: source video is unavailable; treating intrinsics as "
                "already calibrated for the saved semantic keyframes"
            )

    observations_by_frame: dict[int, list[tuple[int, dict[str, object]]]] = {}
    for row, observation in enumerate(observations):
        observations_by_frame.setdefault(int(observation["frame_id"]), []).append(
            (row, observation)
        )
    group_ids = sorted({str(item["group_id"]) for item in observations})
    group_lookup = {group_id: index for index, group_id in enumerate(group_ids)}
    groups = _group_metadata(observations, group_ids)
    prototypes = aggregate_group_embeddings(embeddings, observations, group_ids)
    rotation_body_from_camera, translation_body_from_camera = camera_to_body(
        args.pitch, args.cam_height
    )

    import cv2
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device)
    model_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"loading metric depth model {args.depth_model}...")
    processor = AutoImageProcessor.from_pretrained(args.depth_model)
    model = AutoModelForDepthEstimation.from_pretrained(args.depth_model).to(
        device, model_dtype
    ).eval()

    cloud_chunks: list[np.ndarray] = []
    cloud_keyframes: list[np.ndarray] = []
    semantic_chunks: list[np.ndarray] = []
    semantic_group_chunks: list[np.ndarray] = []
    semantic_weight_chunks: list[np.ndarray] = []
    poses: list[tuple[float, float, float, float, float]] = []
    pending: list[tuple[object, float, np.ndarray]] = []
    skipped = 0
    started = perf_counter()

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        bgr_frames = [item[2] for item in pending]
        depths = _predict_depth_batch(
            processor, model, bgr_frames, device=device, dtype=model_dtype
        )
        for (keyframe, timestamp, _), depth in zip(pending, depths):
            depth = np.asarray(depth, dtype=np.float32) * args.scale
            rover_x = float(np.interp(timestamp, odom_t, odom_x))
            rover_y = float(np.interp(timestamp, odom_t, odom_y))
            rover_yaw = float(np.interp(timestamp, odom_t, odom_yaw))
            points_camera, _, _ = backproject_depth(
                depth,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                pixel_stride=args.pixel_stride,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
            )
            if len(points_camera) < 100:
                continue
            points_world = camera_points_to_rover_map(
                points_camera,
                rover_x=rover_x,
                rover_y=rover_y,
                rover_yaw=rover_yaw,
                rotation_body_from_camera=rotation_body_from_camera,
                translation_body_from_camera=translation_body_from_camera,
            )
            pose_index = len(poses)
            cloud_chunks.append(points_world)
            cloud_keyframes.append(
                np.full(len(points_world), pose_index, dtype=np.int32)
            )
            camera_x = rover_x + np.cos(rover_yaw) * translation_body_from_camera[0] - np.sin(rover_yaw) * translation_body_from_camera[1]
            camera_y = rover_y + np.sin(rover_yaw) * translation_body_from_camera[0] + np.cos(rover_yaw) * translation_body_from_camera[1]
            poses.append((timestamp, camera_x, -args.cam_height, camera_y, rover_yaw))

            frame_observations = observations_by_frame.get(keyframe.frame_id, [])
            by_group: dict[str, list[dict[str, object]]] = {}
            for _, observation in frame_observations:
                by_group.setdefault(str(observation["group_id"]), []).append(
                    observation
                )
            for group_id, members in by_group.items():
                union = np.zeros(depth.shape, dtype=bool)
                qualities = []
                for observation in members:
                    with np.load(
                        proposal_directory / str(observation["mask_file"]),
                        allow_pickle=False,
                    ) as payload:
                        mask = np.asarray(payload["mask"], dtype=np.uint8)
                    if mask.shape != depth.shape:
                        mask = cv2.resize(
                            mask,
                            (depth.shape[1], depth.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    union |= mask > 0
                    qualities.append(
                        float(observation.get("proposal_score", 1.0))
                        * float(observation.get("association_confidence", 1.0))
                    )
                semantic_camera, _, _ = backproject_depth(
                    depth,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    pixel_stride=args.semantic_pixel_stride,
                    depth_min=args.depth_min,
                    depth_max=args.depth_max,
                    mask=union,
                )
                if not len(semantic_camera):
                    continue
                semantic_world = camera_points_to_rover_map(
                    semantic_camera,
                    rover_x=rover_x,
                    rover_y=rover_y,
                    rover_yaw=rover_yaw,
                    rotation_body_from_camera=rotation_body_from_camera,
                    translation_body_from_camera=translation_body_from_camera,
                )
                # One entity contributes at most one vote per BEV cell in one
                # frame. This makes repeated views useful without allowing a
                # large mask (or a denser image region) to dominate the map.
                horizontal_cells = np.floor(
                    semantic_world[:, [0, 2]] / args.resolution
                ).astype(np.int64)
                _, representative = np.unique(
                    horizontal_cells, axis=0, return_index=True
                )
                semantic_world = semantic_world[np.sort(representative)]
                quality = float(np.clip(max(qualities, default=1.0), 0.05, 1.0))
                semantic_chunks.append(semantic_world)
                semantic_group_chunks.append(
                    np.full(
                        len(semantic_world), group_lookup[group_id], dtype=np.int32
                    )
                )
                semantic_weight_chunks.append(
                    np.full(len(semantic_world), quality, dtype=np.float32)
                )
        pending = []

    for keyframe in keyframes:
        entry = keyframe_entries.get(keyframe.frame_id, {})
        raw_timestamp = entry.get("timestamp", keyframe.timestamp)
        try:
            timestamp = float(raw_timestamp) + time_offset
        except (TypeError, ValueError):
            timestamp = keyframe.frame_id / args.frame_fps + time_offset
        if timestamp < odom_t[0] or timestamp > odom_t[-1]:
            skipped += 1
            continue
        if args.stationary_skip > 0 and abs(
            float(np.interp(timestamp, odom_t, odom_velocity))
        ) < args.stationary_skip:
            skipped += 1
            continue
        bgr = cv2.cvtColor(np.asarray(keyframe.rgb), cv2.COLOR_RGB2BGR)
        pending.append((keyframe, timestamp, bgr))
        if len(pending) >= args.batch:
            flush()
            print(
                f"processed {len(poses)}/{len(keyframes)} frames, "
                f"{sum(map(len, cloud_chunks)):,} depth points, "
                f"{sum(map(len, semantic_chunks)):,} semantic points"
            )
    flush()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    if not cloud_chunks or not poses:
        raise ValueError("no depth points were produced; check odometry/time offset")

    points = np.concatenate(cloud_chunks)
    keyframe_ids = np.concatenate(cloud_keyframes)
    keep = voxel_downsample(points, args.voxel, return_index=True)
    points, keyframe_ids = points[keep], keyframe_ids[keep]
    semantic_points = (
        np.concatenate(semantic_chunks)
        if semantic_chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    semantic_groups = (
        np.concatenate(semantic_group_chunks)
        if semantic_group_chunks
        else np.empty(0, dtype=np.int32)
    )
    semantic_weights = (
        np.concatenate(semantic_weight_chunks)
        if semantic_weight_chunks
        else np.empty(0, dtype=np.float32)
    )
    camera_positions = np.asarray(
        [[item[1], item[2], item[3]] for item in poses], dtype=np.float64
    )
    occupancy, lower_xy, floor = build_occupancy(
        points,
        camera_positions,
        res=args.resolution,
        voxel=args.voxel,
        min_h=args.min_height,
        max_h=args.max_height,
        floor_plane=(np.asarray([0.0, 1.0, 0.0]), np.zeros(3)),
        max_ray=args.max_ray,
        floor_support=True,
        occlusion=True,
        kf_id=keyframe_ids,
        max_observers=4,
        min_cell_points=args.min_cell_points,
    )
    semantic_grid = build_semantic_grid(
        semantic_points,
        semantic_groups,
        semantic_weights,
        shape=occupancy.shape,
        lower_xy=lower_xy,
        resolution=args.resolution,
        floor_origin=floor["origin"],
        floor_u=floor["u"],
        floor_v=floor["v"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(_sidecar(args.out, ".npy"), occupancy)
    write_pgm_yaml(occupancy, args.out, args.resolution, lower_xy, 0.65, 0.25)
    np.save(_sidecar(args.out, "_entity_embeddings.npy"), prototypes)
    np.savez_compressed(
        _sidecar(args.out, "_semantic_bev.npz"),
        occupancy=occupancy,
        semantic_ids=semantic_grid.entity_ids,
        semantic_confidence=semantic_grid.confidence,
        semantic_support=semantic_grid.support,
        origin_xy=np.asarray(lower_xy, dtype=np.float64),
        resolution=np.asarray(args.resolution, dtype=np.float64),
        floor_origin=np.asarray(floor["origin"], dtype=np.float64),
        floor_u=np.asarray(floor["u"], dtype=np.float64),
        floor_v=np.asarray(floor["v"], dtype=np.float64),
    )
    _write_ply(_sidecar(args.out, ".ply"), points, keyframe_ids)
    _write_semantic_ply(
        _sidecar(args.out, "_semantic.ply"),
        semantic_points,
        semantic_groups,
        semantic_weights,
    )
    with _sidecar(args.out, ".txt").open("w", encoding="utf-8") as handle:
        for timestamp, x, y, z, yaw in poses:
            qy, qw = np.sin(-yaw / 2.0), np.cos(-yaw / 2.0)
            handle.write(
                f"{timestamp:.6f} {x:.6f} {y:.6f} {z:.6f} "
                f"0 {qy:.6f} 0 {qw:.6f}\n"
            )
    preview_path = _sidecar(args.out, "_semantic.png")
    _render_semantic_map(occupancy, semantic_grid.entity_ids, groups, preview_path)
    manifest = {
        "format": "fact3r-depth-semantic-bev",
        "version": 1,
        "source_observation_index": str(index_path.resolve()),
        "source_odometry": str(odometry_path.resolve()),
        "depth_model": args.depth_model,
        "semantic_model": index_manifest["model"],
        "coordinate_convention": "metric rover odometry; ROS bottom-left BEV",
        "resolution_metres": args.resolution,
        "origin_xy": np.asarray(lower_xy).tolist(),
        "shape": list(occupancy.shape),
        "intrinsics_on_keyframes": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "depth_scale": args.scale,
        "time_offset_seconds": time_offset,
        "clock_check": clock_check,
        "camera_pitch_degrees": args.pitch,
        "camera_height_metres": args.cam_height,
        "processed_frames": len(poses),
        "skipped_frames": skipped,
        "depth_point_count": len(points),
        "semantic_point_count": len(semantic_points),
        "semantic_cell_count": int((semantic_grid.entity_ids >= 0).sum()),
        "entity_embedding_file": _sidecar(
            args.out, "_entity_embeddings.npy"
        ).name,
        "grid_file": _sidecar(args.out, "_semantic_bev.npz").name,
        "preview_file": preview_path.name,
        "groups": groups,
        "elapsed_seconds": perf_counter() - started,
    }
    manifest_path = _sidecar(args.out, "_semantic.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\nDepth + semantic BEV complete")
    print(f"occupancy map: {args.out}.pgm / {args.out}.yaml")
    print(f"semantic map:  {preview_path}")
    print(f"semantic grid: {_sidecar(args.out, '_semantic_bev.npz')}")
    print(f"entity legend: {manifest_path}")
    print(f"frames={len(poses)} entities={len(groups)} semantic_cells={manifest['semantic_cell_count']}")


if __name__ == "__main__":
    main()
