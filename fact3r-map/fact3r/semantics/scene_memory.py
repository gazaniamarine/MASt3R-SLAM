"""Full-frame semantic areas for functional place retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from fact3r.integrations.mast3r_slam import iter_exported_keyframes
from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder


def _normalise_rows(values: object) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if array.ndim != 2 or not np.all(np.isfinite(array)) or np.any(norms <= 1e-12):
        raise ValueError("scene embeddings must contain finite nonzero rows")
    return np.ascontiguousarray(array / norms)


def _rgb_image(values: object) -> Image.Image:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating) and array.size:
        if float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
    return Image.fromarray(
        np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8)), mode="RGB"
    )


def group_scene_areas(
    embeddings: object,
    timestamps: Sequence[float | str | None],
    *,
    area_seconds: float = 5.0,
    visual_split_similarity: float = 0.65,
) -> tuple[NDArray[np.float32], list[dict[str, object]]]:
    """Segment a causal frame stream into compact semantic areas."""

    if area_seconds <= 0:
        raise ValueError("area_seconds must be positive")
    if not -1.0 <= visual_split_similarity <= 1.0:
        raise ValueError("visual_split_similarity must be in [-1, 1]")
    vectors = _normalise_rows(embeddings)
    if len(vectors) != len(timestamps) or len(vectors) == 0:
        raise ValueError("scene vectors and timestamps must be nonempty and aligned")

    def numeric_timestamp(index: int) -> float | None:
        value = timestamps[index]
        try:
            result = float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
        return result if result is not None and np.isfinite(result) else None

    groups: list[list[int]] = [[0]]
    for index in range(1, len(vectors)):
        current = groups[-1]
        prototype = _normalise_rows(
            np.mean(vectors[current], axis=0, keepdims=True)
        )[0]
        start_time = numeric_timestamp(current[0])
        current_time = numeric_timestamp(index)
        temporal_split = (
            start_time is not None
            and current_time is not None
            and current_time - start_time >= area_seconds
        )
        visual_split = (
            len(current) >= 2
            and float(vectors[index] @ prototype) < visual_split_similarity
        )
        if temporal_split or visual_split:
            groups.append([index])
        else:
            current.append(index)

    prototypes = _normalise_rows(
        np.stack([np.mean(vectors[indices], axis=0) for indices in groups])
    )
    areas = []
    for area_index, indices in enumerate(groups):
        similarities = vectors[indices] @ prototypes[area_index]
        representative = indices[int(np.argmax(similarities))]
        areas.append(
            {
                "area_id": f"area-{area_index:06d}",
                "frame_indices": indices,
                "representative_frame_index": representative,
            }
        )
    return prototypes, areas


def build_scene_memory(
    *,
    keyframes: str | Path,
    output: str | Path,
    encoder: Qwen3VLEmbeddingEncoder,
    batch_size: int = 4,
    area_seconds: float = 5.0,
    visual_split_similarity: float = 0.65,
    max_frames: int | None = None,
) -> Path:
    """Encode complete frames once and persist temporally coherent areas."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    keyframe_directory = Path(keyframes)
    output_directory = Path(output)
    output_directory.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(
        (keyframe_directory / "manifest.json").read_text(encoding="utf-8")
    )
    source_entries = {
        int(entry["frame_id"]): entry for entry in source_manifest["keyframes"]
    }
    navigation_available = (
        source_manifest.get("coordinate_convention") != "image_only_no_geometry"
    )

    frame_records: list[dict[str, object]] = []
    image_batch: list[Image.Image] = []
    chunks: list[NDArray[np.float32]] = []
    encoding_seconds = 0.0

    def flush() -> None:
        nonlocal encoding_seconds
        if not image_batch:
            return
        started = perf_counter()
        chunks.append(encoder.encode_images(image_batch))
        encoding_seconds += perf_counter() - started
        image_batch.clear()

    for keyframe in iter_exported_keyframes(keyframe_directory):
        if max_frames is not None and len(frame_records) >= max_frames:
            break
        image = _rgb_image(keyframe.rgb)
        source_entry = source_entries[keyframe.frame_id]
        rgb_file = source_entry.get("rgb_file")
        if rgb_file is None:
            frame_directory = output_directory / "frames"
            frame_directory.mkdir(exist_ok=True)
            image_path = frame_directory / f"frame_{keyframe.frame_id:06d}.jpg"
            image.save(image_path, quality=92)
        else:
            image_path = keyframe_directory / str(rgb_file)
        frame_records.append(
            {
                "frame_id": keyframe.frame_id,
                "timestamp": keyframe.timestamp,
                "image_file": str(image_path.resolve()),
                "camera_pose_world_from_camera": (
                    keyframe.pose_world_from_camera.tolist()
                    if navigation_available
                    else None
                ),
            }
        )
        image_batch.append(image)
        if len(image_batch) >= batch_size:
            flush()
    flush()
    if not frame_records:
        raise ValueError("cannot build scene memory from an empty frame export")
    embeddings = _normalise_rows(np.concatenate(chunks, axis=0))
    area_embeddings, areas = group_scene_areas(
        embeddings,
        [record["timestamp"] for record in frame_records],
        area_seconds=area_seconds,
        visual_split_similarity=visual_split_similarity,
    )
    for area, prototype in zip(areas, area_embeddings, strict=True):
        indices = [int(index) for index in area["frame_indices"]]
        representative = int(area["representative_frame_index"])
        area.update(
            {
                "start_frame_id": frame_records[indices[0]]["frame_id"],
                "end_frame_id": frame_records[indices[-1]]["frame_id"],
                "start_timestamp": frame_records[indices[0]]["timestamp"],
                "end_timestamp": frame_records[indices[-1]]["timestamp"],
                "representative_frame_id": frame_records[representative]["frame_id"],
                "revisit_pose_world_from_camera": frame_records[representative][
                    "camera_pose_world_from_camera"
                ],
            }
        )

    np.save(output_directory / "frame_embeddings.npy", embeddings)
    np.save(output_directory / "area_embeddings.npy", area_embeddings)
    manifest = {
        "format": "fact3r-qwen-scene-memory",
        "version": 1,
        "model": encoder.model_name,
        "source_keyframes": str(keyframe_directory.resolve()),
        "navigation_pose_available": navigation_available,
        "frame_embedding_file": "frame_embeddings.npy",
        "area_embedding_file": "area_embeddings.npy",
        "frame_count": len(frame_records),
        "area_count": len(areas),
        "area_config": {
            "area_seconds": area_seconds,
            "visual_split_similarity": visual_split_similarity,
        },
        "timing": {
            "model_load_seconds": encoder.load_seconds,
            "frame_encoding_seconds": encoding_seconds,
            "frames_per_second": len(frame_records) / max(encoding_seconds, 1e-12),
        },
        "frames": frame_records,
        "areas": areas,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path
