"""Disk boundary between a MASt3R-SLAM run and offline Fact3R mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from fact3r.reconstruction.keyframes import KeyframeRecord
from fact3r.reconstruction.pointmap_adapter import keyframe_record_from_mast3r


EXPORT_VERSION = 1


def _json_scalar(value: object) -> object:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb)
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(np.nanmax(values)) <= 1.0:
            values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def _constrain_pointmap_to_camera_rays(record: KeyframeRecord) -> KeyframeRecord:
    """Match MASt3R-SLAM's calibrated reconstruction export convention."""

    if record.intrinsics is None:
        return record
    height, width = record.image_shape
    rows, columns = np.indices((height, width), dtype=np.float32)
    depth = record.pointmap_camera[..., 2]
    intrinsics = record.intrinsics
    pointmap = np.empty_like(record.pointmap_camera)
    pointmap[..., 0] = (columns - intrinsics[0, 2]) / intrinsics[0, 0] * depth
    pointmap[..., 1] = (rows - intrinsics[1, 2]) / intrinsics[1, 1] * depth
    pointmap[..., 2] = depth
    return KeyframeRecord(
        frame_id=record.frame_id,
        timestamp=record.timestamp,
        rgb=record.rgb,
        pointmap_camera=pointmap,
        geometry_confidence=record.geometry_confidence,
        pose_world_from_camera=record.pose_world_from_camera,
        mast3r_descriptors=record.mast3r_descriptors,
        descriptor_confidence=record.descriptor_confidence,
        intrinsics=record.intrinsics,
    )


def export_mast3r_keyframes(
    output_directory: str | Path,
    timestamps: Sequence[object],
    keyframes: object,
) -> Path:
    """Export the final posed keyframes without importing SAM2 into SLAM.

    One compressed NPZ is written per keyframe so downstream segmentation can
    stream a long sequence without keeping every dense pointmap in memory.
    """

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for keyframe_index in range(len(keyframes)):
        frame = keyframes[keyframe_index]
        timestamp = _json_scalar(timestamps[frame.frame_id])
        record = keyframe_record_from_mast3r(frame, timestamp=timestamp)
        record = _constrain_pointmap_to_camera_rays(record)
        filename = f"keyframe_{keyframe_index:06d}_frame_{record.frame_id:06d}.npz"
        payload: dict[str, np.ndarray] = {
            "frame_id": np.asarray(record.frame_id, dtype=np.int64),
            "rgb": _rgb_uint8(record.rgb),
            "pointmap_camera": record.pointmap_camera.astype(np.float32, copy=False),
            "geometry_confidence": record.geometry_confidence.astype(
                np.float32, copy=False
            ),
            "pose_world_from_camera": record.pose_world_from_camera.astype(
                np.float32, copy=False
            ),
        }
        if record.intrinsics is not None:
            payload["intrinsics"] = record.intrinsics.astype(np.float32, copy=False)
        if record.mast3r_descriptors is not None:
            payload["mast3r_descriptors"] = record.mast3r_descriptors.astype(
                np.float32, copy=False
            )
        if record.descriptor_confidence is not None:
            payload["descriptor_confidence"] = record.descriptor_confidence.astype(
                np.float32, copy=False
            )
        np.savez_compressed(output / filename, **payload)
        entries.append(
            {
                "keyframe_index": keyframe_index,
                "frame_id": record.frame_id,
                "timestamp": timestamp,
                "file": filename,
                "image_shape": list(record.image_shape),
                "has_mast3r_descriptors": record.mast3r_descriptors is not None,
            }
        )

    manifest = {
        "format": "fact3r-mast3r-keyframes",
        "version": EXPORT_VERSION,
        "coordinate_convention": "pointmap_camera + pose_world_from_camera",
        "keyframes": entries,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def iter_exported_keyframes(
    export_directory: str | Path,
) -> Iterator[KeyframeRecord]:
    """Stream keyframes written by :func:`export_mast3r_keyframes`."""

    directory = Path(export_directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "fact3r-mast3r-keyframes":
        raise ValueError(f"unsupported keyframe export in {manifest_path}")
    if manifest.get("version") != EXPORT_VERSION:
        raise ValueError(
            f"unsupported keyframe export version {manifest.get('version')}"
        )

    for entry in manifest["keyframes"]:
        with np.load(directory / entry["file"], allow_pickle=False) as payload:
            descriptors = (
                payload["mast3r_descriptors"]
                if "mast3r_descriptors" in payload.files
                else None
            )
            descriptor_confidence = (
                payload["descriptor_confidence"]
                if "descriptor_confidence" in payload.files
                else None
            )
            intrinsics = payload["intrinsics"] if "intrinsics" in payload.files else None
            yield KeyframeRecord(
                frame_id=int(payload["frame_id"]),
                timestamp=entry.get("timestamp"),
                rgb=payload["rgb"],
                pointmap_camera=payload["pointmap_camera"],
                geometry_confidence=payload["geometry_confidence"],
                pose_world_from_camera=payload["pose_world_from_camera"],
                mast3r_descriptors=descriptors,
                descriptor_confidence=descriptor_confidence,
                intrinsics=intrinsics,
            )
