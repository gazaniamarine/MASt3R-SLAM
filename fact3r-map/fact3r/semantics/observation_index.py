"""SigLIP-backed semantic retrieval over persistent mask observations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from fact3r.association.tracklets import load_tracklet_run
from fact3r.integrations.mast3r_slam import iter_exported_keyframes
from fact3r.proposals.storage import load_proposal_run_manifest
from fact3r.visualization.association import mask_boundary


FloatArray = NDArray[np.floating]


class VisionLanguageEncoder(Protocol):
    """Small interface that keeps storage and retrieval model-independent."""

    @property
    def model_name(self) -> str: ...

    @property
    def device_name(self) -> str: ...

    @property
    def load_seconds(self) -> float: ...

    def encode_images(self, images: Sequence[Image.Image]) -> FloatArray: ...

    def encode_text(self, texts: Sequence[str]) -> FloatArray: ...


def _normalise_rows(values: object) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("embeddings must be a finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings cannot contain zero-length rows")
    return np.ascontiguousarray(array / norms)


def _pooled_feature_value(output: object) -> object:
    """Unwrap feature tensors across Transformers 4.x and 5.x APIs."""

    for name in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)):
        for value in reversed(output):
            if getattr(value, "ndim", None) == 2:
                return value
    return output


class Siglip2Encoder:
    """Lazy optional-dependency adapter for Hugging Face SigLIP2."""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        *,
        device: str = "auto",
    ) -> None:
        started = perf_counter()
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "SigLIP indexing requires torch and transformers; install the "
                "fact3r-map semantic optional dependencies"
            ) from error

        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        elif device.isdigit():
            resolved_device = f"cuda:{device}"
        else:
            resolved_device = device

        self._torch = torch
        self._model_name = model_name
        self._device_name = resolved_device
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval().to(resolved_device)
        self._load_seconds = perf_counter() - started

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    def _move(self, inputs: Mapping[str, object]) -> dict[str, object]:
        return {
            name: (
                value.to(self._device_name)
                if hasattr(value, "to")
                else value
            )
            for name, value in inputs.items()
        }

    def encode_images(self, images: Sequence[Image.Image]) -> FloatArray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        inputs = self._move(
            self._processor(images=list(images), return_tensors="pt")
        )
        with self._torch.inference_mode():
            features = _pooled_feature_value(
                self._model.get_image_features(**inputs)
            )
        if not self._torch.is_tensor(features) or features.ndim != 2:
            raise TypeError(
                "SigLIP image encoder did not return a pooled feature matrix"
            )
        return _normalise_rows(features.detach().float().cpu().numpy())

    def encode_text(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        inputs = self._move(
            self._processor(
                text=list(texts),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        )
        with self._torch.inference_mode():
            features = _pooled_feature_value(
                self._model.get_text_features(**inputs)
            )
        if not self._torch.is_tensor(features) or features.ndim != 2:
            raise TypeError(
                "SigLIP text encoder did not return a pooled feature matrix"
            )
        return _normalise_rows(features.detach().float().cpu().numpy())


def _rgb_uint8(rgb: object) -> NDArray[np.uint8]:
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(np.nanmax(values)) <= 1.0:
            values = values * 255.0
    return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))


def masked_context_crop(
    rgb: object,
    mask: object,
    *,
    context_fraction: float = 0.15,
    outside_mask_alpha: float = 0.20,
) -> Image.Image:
    """Crop one proposal while retaining dim context around its mask."""

    if context_fraction < 0.0:
        raise ValueError("context_fraction cannot be negative")
    if not 0.0 <= outside_mask_alpha <= 1.0:
        raise ValueError("outside_mask_alpha must be in [0, 1]")
    image = _rgb_uint8(rgb)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != image.shape[:2]:
        raise ValueError("mask and RGB image shapes do not match")
    rows, columns = np.nonzero(selected)
    if len(rows) == 0:
        raise ValueError("cannot encode an empty mask")

    height, width = image.shape[:2]
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    padding = int(round(context_fraction * max(y1 - y0, x1 - x0)))
    y0, y1 = max(0, y0 - padding), min(height, y1 + padding)
    x0, x1 = max(0, x0 - padding), min(width, x1 + padding)

    crop = image[y0:y1, x0:x1].astype(np.float32)
    crop_mask = selected[y0:y1, x0:x1]
    neutral = np.full_like(crop, 127.0)
    crop[~crop_mask] = (
        outside_mask_alpha * crop[~crop_mask]
        + (1.0 - outside_mask_alpha) * neutral[~crop_mask]
    )
    return Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8), mode="RGB")


@dataclass(frozen=True, slots=True)
class MappingAssignment:
    entity_id: str | None
    track_id: str | None
    status: str
    association_confidence: float = 1.0


def _association_confidence(
    evidence: Mapping[str, object], *, default: float = 1.0
) -> float:
    for name in (
        "conditional_probability",
        "row_probability",
        "retained_ratio",
        "pending_mean_birth_residual_ratio",
        "birth_residual_ratio",
    ):
        value = evidence.get(name)
        if value is not None and np.isfinite(float(value)):
            return float(np.clip(float(value), 0.0, 1.0))
    cost = evidence.get("cost", evidence.get("best_cost"))
    if cost is not None and np.isfinite(float(cost)):
        return float(np.clip(1.0 - float(cost), 0.0, 1.0))
    return float(np.clip(default, 0.0, 1.0))


def _manifest_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate / "manifest.json" if candidate.is_dir() else candidate


def load_mapping_assignments(
    mapping: str | Path | None,
) -> dict[tuple[int, str], MappingAssignment]:
    """Resolve proposal identities, including retrospectively committed tracks."""

    if mapping is None:
        return {}
    manifest_path = _manifest_path(mapping)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") not in {
        "fact3r-hungarian-baseline",
        "fact3r-balanced-sinkhorn",
        "fact3r-visibility-residual-transport",
    }:
        raise ValueError(f"unsupported mapping manifest {manifest_path}")
    committed_tracks = {
        str(track_id): str(entity_id)
        for track_id, entity_id in payload.get(
            "committed_track_entities", {}
        ).items()
    }
    assignments: dict[tuple[int, str], MappingAssignment] = {}
    for frame in payload.get("frames", []):
        frame_id = int(frame["frame_id"])
        for match in frame.get("matches", []):
            proposal_id = str(match["proposal_id"])
            tracklet = match.get("tracklet") or {}
            track_id = tracklet.get("track_id")
            assignments[(frame_id, proposal_id)] = MappingAssignment(
                entity_id=str(match["entity_id"]),
                track_id=None if track_id is None else str(track_id),
                status="matched",
                association_confidence=_association_confidence(match),
            )
        for unmatched in frame.get("unmatched_proposals", []):
            proposal_id = str(unmatched["proposal_id"])
            tracklet = unmatched.get("tracklet") or {}
            track_id = unmatched.get("track_id", tracklet.get("track_id"))
            track_id = None if track_id is None else str(track_id)
            entity_id = unmatched.get("resolved_entity_id")
            if entity_id is None:
                entity_id = unmatched.get("created_entity_id")
            if entity_id is None and track_id is not None:
                entity_id = committed_tracks.get(track_id)
            status = unmatched.get("commitment_status")
            if status is None:
                status = "created" if entity_id is not None else "unassigned"
            assignments[(frame_id, proposal_id)] = MappingAssignment(
                entity_id=None if entity_id is None else str(entity_id),
                track_id=track_id,
                status=str(status),
                association_confidence=_association_confidence(unmatched),
            )
    return assignments


def _flush_image_batch(
    encoder: VisionLanguageEncoder,
    images: list[Image.Image],
    chunks: list[NDArray[np.float32]],
) -> float:
    if not images:
        return 0.0
    started = perf_counter()
    encoded = _normalise_rows(encoder.encode_images(images))
    elapsed = perf_counter() - started
    if len(encoded) != len(images):
        raise ValueError("image encoder returned an unexpected batch size")
    chunks.append(encoded)
    images.clear()
    return elapsed


def build_observation_index(
    *,
    keyframes: str | Path,
    proposals: str | Path,
    output: str | Path,
    encoder: VisionLanguageEncoder,
    mapping: str | Path | None = None,
    tracklets: str | Path | None = None,
    batch_size: int = 32,
    context_fraction: float = 0.15,
    outside_mask_alpha: float = 0.20,
) -> Path:
    """Encode every saved SAM proposal and write a reusable observation index."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    started = perf_counter()
    keyframe_directory = Path(keyframes)
    proposal_directory = Path(proposals)
    output_directory = Path(output)
    output_directory.mkdir(parents=True, exist_ok=True)
    proposal_run = load_proposal_run_manifest(proposal_directory)
    proposal_frames = {
        int(entry["frame_id"]): entry for entry in proposal_run["frames"]
    }
    assignments = load_mapping_assignments(mapping)
    tracklet_run = None if tracklets is None else load_tracklet_run(tracklets)
    if (
        tracklet_run is not None
        and Path(tracklet_run.source_proposals).resolve()
        != proposal_directory.resolve()
    ):
        raise ValueError("tracklets and observation index use different proposals")

    observations: list[dict[str, object]] = []
    image_batch: list[Image.Image] = []
    embedding_chunks: list[NDArray[np.float32]] = []
    image_encoding_seconds = 0.0
    indexed_frames = 0
    for keyframe in iter_exported_keyframes(keyframe_directory):
        run_entry = proposal_frames.get(keyframe.frame_id)
        if run_entry is None:
            continue
        indexed_frames += 1
        frame_manifest_path = proposal_directory / str(run_entry["manifest"])
        frame_manifest = json.loads(
            frame_manifest_path.read_text(encoding="utf-8")
        )
        frame_tracklets = {
            item.proposal_id: item
            for item in (
                ()
                if tracklet_run is None
                else tracklet_run.observations_by_frame.get(keyframe.frame_id, ())
            )
        }
        for proposal in frame_manifest["proposals"]:
            proposal_id = str(proposal["proposal_id"])
            proposal_file = frame_manifest_path.parent / str(proposal["file"])
            with np.load(proposal_file, allow_pickle=False) as evidence:
                mask = np.array(evidence["mask"], dtype=bool, copy=True)
            crop = masked_context_crop(
                keyframe.rgb,
                mask,
                context_fraction=context_fraction,
                outside_mask_alpha=outside_mask_alpha,
            )
            assignment = assignments.get(
                (keyframe.frame_id, proposal_id),
                MappingAssignment(None, None, "unassigned"),
            )
            tracklet = frame_tracklets.get(proposal_id)
            track_id = assignment.track_id or (
                None if tracklet is None else tracklet.track_id
            )
            geometry_status = str(
                proposal.get("geometry_status", "anchored_3d")
            )
            assignment_status = assignment.status
            association_confidence = assignment.association_confidence
            if assignment.entity_id is None and geometry_status == "unanchored_2d":
                assignment_status = "unanchored_2d"
                association_confidence = (
                    0.5
                    if tracklet is None or tracklet.link_iou is None
                    else float(tracklet.link_iou)
                )
            observation_index = len(observations)
            group_id = (
                assignment.entity_id
                or track_id
                or f"observation-{observation_index:06d}"
            )
            mask_rows, mask_columns = np.nonzero(mask)
            mask_center_rc = [
                float(np.mean(mask_rows)),
                float(np.mean(mask_columns)),
            ]
            camera_origin = np.asarray(
                keyframe.pose_world_from_camera[:3, 3], dtype=np.float64
            )
            view_ray_world = None
            if keyframe.intrinsics is not None:
                pixel = np.asarray(
                    [mask_center_rc[1], mask_center_rc[0], 1.0],
                    dtype=np.float64,
                )
                ray_camera = np.linalg.solve(
                    np.asarray(keyframe.intrinsics, dtype=np.float64), pixel
                )
                ray_world = (
                    np.asarray(
                        keyframe.pose_world_from_camera[:3, :3],
                        dtype=np.float64,
                    )
                    @ ray_camera
                )
                ray_norm = float(np.linalg.norm(ray_world))
                if np.isfinite(ray_norm) and ray_norm > 1e-12:
                    view_ray_world = (ray_world / ray_norm).tolist()
            observations.append(
                {
                    "index": observation_index,
                    "proposal_id": proposal_id,
                    "frame_id": keyframe.frame_id,
                    "timestamp": keyframe.timestamp,
                    "entity_id": assignment.entity_id,
                    "track_id": track_id,
                    "group_id": group_id,
                    "assignment_status": assignment_status,
                    "association_confidence": association_confidence,
                    "proposal_score": float(proposal["score"]),
                    "mask_area": int(proposal["mask_area"]),
                    "geometry_status": geometry_status,
                    "geometry_coverage": float(
                        proposal.get("geometry_coverage", 1.0)
                    ),
                    "lifted_point_count": int(
                        proposal.get("lifted_point_count", proposal["mask_area"])
                    ),
                    "bounding_box_xyxy": proposal.get("bounding_box_xyxy"),
                    "mask_center_rc": mask_center_rc,
                    "camera_pose_world_from_camera": np.asarray(
                        keyframe.pose_world_from_camera
                    ).tolist(),
                    "camera_origin_world": camera_origin.tolist(),
                    "view_ray_world": view_ray_world,
                    "track_link_iou": (
                        None if tracklet is None else tracklet.link_iou
                    ),
                    "mask_file": str(proposal_file.relative_to(proposal_directory)),
                }
            )
            image_batch.append(crop)
            if len(image_batch) >= batch_size:
                image_encoding_seconds += _flush_image_batch(
                    encoder, image_batch, embedding_chunks
                )

    image_encoding_seconds += _flush_image_batch(
        encoder, image_batch, embedding_chunks
    )
    if not observations:
        raise ValueError("no common keyframe and proposal observations found")
    embeddings = np.concatenate(embedding_chunks, axis=0)
    if len(embeddings) != len(observations):
        raise RuntimeError("observation and embedding counts diverged")
    np.save(output_directory / "embeddings.npy", embeddings)

    elapsed = perf_counter() - started
    manifest = {
        "format": "fact3r-siglip-observation-index",
        "version": 1,
        "model": encoder.model_name,
        "device": encoder.device_name,
        "source_keyframes": str(keyframe_directory.resolve()),
        "source_proposals": str(proposal_directory.resolve()),
        "source_tracklets": (
            None if tracklets is None else str(_manifest_path(tracklets).resolve())
        ),
        "source_mapping": (
            None if mapping is None else str(_manifest_path(mapping).resolve())
        ),
        "embedding_file": "embeddings.npy",
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "frame_count": indexed_frames,
        "observation_count": len(observations),
        "assigned_observation_count": sum(
            observation["entity_id"] is not None for observation in observations
        ),
        "unanchored_observation_count": sum(
            observation.get("geometry_status") == "unanchored_2d"
            for observation in observations
        ),
        "track_only_observation_count": sum(
            observation["entity_id"] is None
            and observation.get("track_id") is not None
            for observation in observations
        ),
        "crop_config": {
            "context_fraction": context_fraction,
            "outside_mask_alpha": outside_mask_alpha,
        },
        "timing": {
            "model_load_seconds": float(encoder.load_seconds),
            "image_encoding_seconds": image_encoding_seconds,
            "total_index_seconds_excluding_model_load": elapsed,
            "observations_per_encoding_second": (
                0.0
                if image_encoding_seconds <= 0.0
                else len(observations) / image_encoding_seconds
            ),
        },
        "observations": observations,
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest_path


def load_observation_index(
    index: str | Path,
) -> tuple[Path, dict[str, object], NDArray[np.float32]]:
    manifest_path = _manifest_path(index)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") not in {
        "fact3r-siglip-observation-index",
        "fact3r-vla-visual-observation-index",
    }:
        raise ValueError(f"unsupported observation index {manifest_path}")
    if manifest.get("version") != 1:
        raise ValueError(
            f"unsupported observation-index version {manifest.get('version')}"
        )
    embeddings = _normalise_rows(
        np.load(
            manifest_path.parent / str(manifest["embedding_file"]),
            allow_pickle=False,
        )
    )
    if len(embeddings) != len(manifest["observations"]):
        raise ValueError("observation index contains inconsistent row counts")
    return manifest_path, manifest, embeddings


def attach_mapping_to_observation_index(
    *,
    index: str | Path,
    mapping: str | Path,
    output: str | Path,
) -> Path:
    """Reuse pre-UOT embeddings and attach final entity/track assignments."""

    source_path, source_manifest, embeddings = load_observation_index(index)
    if source_manifest.get("source_mapping") is not None:
        raise ValueError(
            "mapping attachment requires a pre-UOT index built without --mapping"
        )
    mapping_path = _manifest_path(mapping)
    mapping_manifest = json.loads(mapping_path.read_text(encoding="utf-8"))
    indexed_proposals = Path(str(source_manifest["source_proposals"])).resolve()
    mapped_proposals = mapping_manifest.get("source_proposals")
    if (
        mapped_proposals is not None
        and Path(str(mapped_proposals)).resolve() != indexed_proposals
    ):
        raise ValueError("index and mapping were built from different proposals")
    committed_tracks = {
        str(track_id): str(entity_id)
        for track_id, entity_id in mapping_manifest.get(
            "committed_track_entities", {}
        ).items()
    }
    assignments = load_mapping_assignments(mapping)
    observations: list[dict[str, object]] = []
    for source in source_manifest["observations"]:
        observation = dict(source)
        key = (int(observation["frame_id"]), str(observation["proposal_id"]))
        assignment = assignments.get(key)
        source_track_id = observation.get("track_id")
        track_id = (
            assignment.track_id
            if assignment is not None and assignment.track_id is not None
            else (
                None if source_track_id is None else str(source_track_id)
            )
        )
        entity_id = None if assignment is None else assignment.entity_id
        if entity_id is None and track_id is not None:
            entity_id = committed_tracks.get(track_id)
        if assignment is not None:
            status = assignment.status
            confidence = assignment.association_confidence
        elif entity_id is not None:
            status = "retrospectively_anchored"
            confidence = float(observation.get("association_confidence", 0.5))
        else:
            status = str(observation.get("assignment_status", "unassigned"))
            confidence = float(observation.get("association_confidence", 0.5))
        observation["entity_id"] = entity_id
        observation["track_id"] = track_id
        observation["assignment_status"] = status
        observation["association_confidence"] = confidence
        observation["group_id"] = (
            entity_id
            or track_id
            or f"observation-{int(observation['index']):06d}"
        )
        observations.append(observation)

    output_directory = Path(output)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.save(output_directory / "embeddings.npy", embeddings)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "source_mapping": str(mapping_path.resolve()),
            "embedding_file": "embeddings.npy",
            "assigned_observation_count": sum(
                observation["entity_id"] is not None
                for observation in observations
            ),
            "unanchored_observation_count": sum(
                observation.get("geometry_status") == "unanchored_2d"
                for observation in observations
            ),
            "track_only_observation_count": sum(
                observation["entity_id"] is None
                and observation.get("track_id") is not None
                for observation in observations
            ),
            "reused_embedding_source": str(source_path.resolve()),
            "observations": observations,
        }
    )
    timing = dict(source_manifest.get("timing", {}))
    timing["reused_pre_uot_embeddings"] = True
    timing["additional_image_encoding_seconds"] = 0.0
    manifest["timing"] = timing
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def rank_observation_groups(
    embeddings: FloatArray,
    observations: Sequence[Mapping[str, object]],
    query_embedding: FloatArray,
    *,
    top_views: int = 2,
) -> tuple[NDArray[np.float32], list[dict[str, object]]]:
    """Rank persistent entities/tracks using their strongest independent views."""

    if top_views <= 0:
        raise ValueError("top_views must be positive")
    vectors = _normalise_rows(embeddings)
    query = _normalise_rows(query_embedding)
    if query.shape != (1, vectors.shape[1]):
        raise ValueError("query and observation embedding dimensions differ")
    scores = np.asarray(vectors @ query[0], dtype=np.float32)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, observation in enumerate(observations):
        grouped[str(observation["group_id"])].append(index)

    groups: list[dict[str, object]] = []
    for group_id, indices in grouped.items():
        ordered = sorted(indices, key=lambda item: float(scores[item]), reverse=True)
        strongest = ordered[: min(top_views, len(ordered))]
        first = observations[indices[0]]
        groups.append(
            {
                "group_id": group_id,
                "entity_id": first.get("entity_id"),
                "track_id": first.get("track_id"),
                "score": float(np.mean(scores[strongest])),
                "observation_indices": indices,
                "ranked_observation_indices": ordered,
            }
        )
    groups.sort(key=lambda item: float(item["score"]), reverse=True)
    return scores, groups


def default_positive_prompts(query: str) -> tuple[str, ...]:
    subject = query.strip()
    return tuple(
        dict.fromkeys(
            (
                subject,
                f"a photo of {subject}",
                f"a close-up photo of {subject}",
                f"{subject} in an indoor scene",
            )
        )
    )


def default_negative_prompts() -> tuple[str, ...]:
    return (
        "an unrelated indoor object",
        "background with no distinct object",
        "an unrecognizable partial object fragment",
    )


def _retrievable_group_id(
    observation: Mapping[str, object],
    *,
    confirmed_only: bool,
    include_unanchored_tracks: bool,
) -> str | None:
    entity_id = observation.get("entity_id")
    if entity_id is not None:
        return str(entity_id)
    track_id = observation.get("track_id")
    if include_unanchored_tracks and track_id is not None:
        return str(track_id)
    if confirmed_only:
        return None
    group_id = track_id or observation.get("group_id")
    return None if group_id is None else str(group_id)


def map_derived_hard_negative_scores(
    embeddings: FloatArray,
    observations: Sequence[Mapping[str, object]],
    query_embeddings: FloatArray,
    *,
    neighbors: int = 3,
    confirmed_only: bool = True,
    include_unanchored_tracks: bool = True,
) -> tuple[NDArray[np.float32], dict[str, list[dict[str, object]]]]:
    """Use visually nearest competing map entities as query-time negatives."""

    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    vectors = _normalise_rows(embeddings)
    queries = _normalise_rows(query_embeddings)
    if vectors.shape[1] != queries.shape[1]:
        raise ValueError("query and observation embedding dimensions differ")
    query = _normalise_rows(np.mean(queries, axis=0, keepdims=True))[0]
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, observation in enumerate(observations):
        group_id = _retrievable_group_id(
            observation,
            confirmed_only=confirmed_only,
            include_unanchored_tracks=include_unanchored_tracks,
        )
        if group_id is None:
            continue
        grouped[group_id].append(index)
    scores = np.full(len(observations), -1.0, dtype=np.float32)
    diagnostics: dict[str, list[dict[str, object]]] = {
        group_id: [] for group_id in grouped
    }
    if len(grouped) < 2:
        return scores, diagnostics
    group_ids = sorted(grouped)
    prototypes = _normalise_rows(
        np.stack(
            [np.mean(vectors[grouped[group_id]], axis=0) for group_id in group_ids]
        )
    )
    prototype_similarity = prototypes @ prototypes.T
    query_similarity = prototypes @ query
    for group_index, group_id in enumerate(group_ids):
        order = np.argsort(-prototype_similarity[group_index])
        competitor_indices = [
            int(index)
            for index in order
            if int(index) != group_index
        ][:neighbors]
        if not competitor_indices:
            continue
        strongest = float(np.max(query_similarity[competitor_indices]))
        scores[grouped[group_id]] = strongest
        diagnostics[group_id] = [
            {
                "group_id": group_ids[index],
                "prototype_similarity": float(
                    prototype_similarity[group_index, index]
                ),
                "query_similarity": float(query_similarity[index]),
            }
            for index in competitor_indices
        ]
    return scores, diagnostics


def rank_semantic_entity_groups(
    embeddings: FloatArray,
    observations: Sequence[Mapping[str, object]],
    positive_embeddings: FloatArray,
    negative_embeddings: FloatArray,
    *,
    top_views: int = 3,
    min_supporting_views: int = 2,
    min_view_margin: float = 0.02,
    min_entity_margin: float = 0.02,
    reference_mask_area: float = 4096.0,
    confirmed_only: bool = True,
    include_unanchored_tracks: bool = True,
    automatic_map_negatives: bool = True,
    map_negative_neighbors: int = 3,
    map_negative_weight: float = 1.0,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    list[dict[str, object]],
]:
    """Rank entities with contrastive, quality-weighted multi-view evidence."""

    if top_views <= 0:
        raise ValueError("top_views must be positive")
    if min_supporting_views <= 0:
        raise ValueError("min_supporting_views must be positive")
    if reference_mask_area <= 0.0:
        raise ValueError("reference_mask_area must be positive")
    if map_negative_neighbors <= 0:
        raise ValueError("map_negative_neighbors must be positive")
    if map_negative_weight < 0.0:
        raise ValueError("map_negative_weight cannot be negative")
    vectors = _normalise_rows(embeddings)
    positives = _normalise_rows(positive_embeddings)
    negatives = _normalise_rows(negative_embeddings)
    if positives.shape[1] != vectors.shape[1]:
        raise ValueError("positive prompt and observation dimensions differ")
    if negatives.shape[1] != vectors.shape[1]:
        raise ValueError("negative prompt and observation dimensions differ")

    positive_scores = np.asarray(
        np.mean(vectors @ positives.T, axis=1), dtype=np.float32
    )
    text_negative_scores = np.asarray(
        np.max(vectors @ negatives.T, axis=1), dtype=np.float32
    )
    map_negative_scores = np.full(len(vectors), -1.0, dtype=np.float32)
    map_negative_diagnostics: dict[str, list[dict[str, object]]] = {}
    if automatic_map_negatives:
        map_negative_scores, map_negative_diagnostics = (
            map_derived_hard_negative_scores(
                vectors,
                observations,
                positives,
                neighbors=map_negative_neighbors,
                confirmed_only=confirmed_only,
                include_unanchored_tracks=include_unanchored_tracks,
            )
        )
    negative_scores = np.maximum(
        text_negative_scores,
        map_negative_weight * map_negative_scores,
    ).astype(np.float32)
    margins = np.asarray(positive_scores - negative_scores, dtype=np.float32)
    qualities = np.empty(len(observations), dtype=np.float32)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, observation in enumerate(observations):
        group_id = _retrievable_group_id(
            observation,
            confirmed_only=confirmed_only,
            include_unanchored_tracks=include_unanchored_tracks,
        )
        if group_id is None:
            qualities[index] = 0.0
            continue
        sam_confidence = float(
            np.clip(float(observation.get("proposal_score", 1.0)), 0.0, 1.0)
        )
        association_confidence = float(
            np.clip(
                float(observation.get("association_confidence", 1.0)),
                0.0,
                1.0,
            )
        )
        mask_area = max(0.0, float(observation.get("mask_area", 0.0)))
        area_quality = min(1.0, np.sqrt(mask_area / reference_mask_area))
        qualities[index] = sam_confidence * association_confidence * area_quality
        grouped[group_id].append(index)

    groups: list[dict[str, object]] = []
    for group_id, indices in grouped.items():
        ordered = sorted(indices, key=lambda item: float(margins[item]), reverse=True)
        supporting = [
            index for index in ordered if float(margins[index]) >= min_view_margin
        ]
        evidence = (supporting or ordered)[: min(top_views, len(indices))]
        evidence_weights = qualities[evidence].astype(np.float64)
        if float(evidence_weights.sum()) <= 1e-12:
            entity_margin = float(np.mean(margins[evidence]))
        else:
            entity_margin = float(
                np.average(margins[evidence], weights=evidence_weights)
            )
        rejection_reasons: list[str] = []
        if len(supporting) < min_supporting_views:
            rejection_reasons.append("insufficient_supporting_views")
        if entity_margin < min_entity_margin:
            rejection_reasons.append("entity_margin_below_threshold")
        first = observations[indices[0]]
        groups.append(
            {
                "group_id": group_id,
                "entity_id": first.get("entity_id"),
                "track_id": first.get("track_id"),
                "score": entity_margin,
                "entity_margin": entity_margin,
                "observation_count": len(indices),
                "supporting_view_count": len(supporting),
                "observation_indices": indices,
                "supporting_observation_indices": supporting,
                "ranked_observation_indices": ordered,
                "mean_observation_quality": float(np.mean(qualities[indices])),
                "map_hard_negative_score": float(map_negative_scores[indices[0]]),
                "map_hard_negatives": map_negative_diagnostics.get(group_id, []),
                "accepted": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            }
        )
    groups.sort(key=lambda item: float(item["entity_margin"]), reverse=True)
    return positive_scores, negative_scores, margins, qualities, groups


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return compact[:80] or "query"


def _render_match(
    rgb: object,
    mask: object,
    *,
    query: str,
    group_id: str,
    frame_id: int,
    positive_score: float,
    negative_score: float,
    semantic_margin: float,
    observation_quality: float,
    entity_margin: float,
) -> Image.Image:
    image = _rgb_uint8(rgb)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != image.shape[:2]:
        raise ValueError("stored mask and keyframe shape do not match")
    canvas = image.astype(np.float32)
    highlight = np.asarray([40.0, 240.0, 100.0], dtype=np.float32)
    canvas[selected] = 0.55 * canvas[selected] + 0.45 * highlight
    canvas[mask_boundary(selected)] = [40.0, 255.0, 80.0]
    header_height = 78
    rendered = Image.new(
        "RGB", (image.shape[1], image.shape[0] + header_height), (20, 20, 20)
    )
    rendered.paste(Image.fromarray(canvas.astype(np.uint8)), (0, header_height))
    draw = ImageDraw.Draw(rendered)
    draw.text(
        (8, 7),
        f'query "{query}" | {group_id} | frame {frame_id}',
        fill=(245, 245, 245),
    )
    draw.text(
        (8, 31),
        f"entity margin={entity_margin:.3f} | view margin={semantic_margin:.3f} "
        f"| quality={observation_quality:.3f}",
        fill=(190, 230, 200),
    )
    draw.text(
        (8, 53),
        f"positive={positive_score:.3f} | strongest confounder={negative_score:.3f}",
        fill=(185, 200, 230),
    )
    return rendered


def _resize_to_width(image: Image.Image, maximum_width: int) -> Image.Image:
    if image.width <= maximum_width:
        return image.copy()
    height = max(1, round(image.height * maximum_width / image.width))
    return image.resize((maximum_width, height), Image.Resampling.LANCZOS)


def _write_gallery_outputs(
    rendered: Sequence[tuple[Path, Mapping[str, object]]],
    output: Path,
    *,
    query: str,
    gif_width: int,
    gif_duration_ms: int,
) -> None:
    if not rendered:
        return
    gif_frames: list[Image.Image] = []
    thumbs: list[Image.Image] = []
    for path, _ in rendered:
        with Image.open(path) as source:
            image = source.convert("RGB")
            gif_frames.append(_resize_to_width(image, gif_width))
            thumbs.append(_resize_to_width(image, 420))
    gif_frames[0].save(
        output / "matches.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=gif_duration_ms,
        loop=0,
    )
    columns = 3
    tile_height = max(tile.height for tile in thumbs)
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 420, rows * tile_height), (25, 25, 25))
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, ((index % columns) * 420, (index // columns) * tile_height))
    sheet.save(output / "contact_sheet.jpg", quality=90)

    cards = "\n".join(
        "<figure><img src=\"{}\"><figcaption>{}</figcaption></figure>".format(
            escape(str(path.relative_to(output))),
            escape(
                f"{entry['group_id']} · frame {entry['frame_id']} · "
                f"margin {float(entry['semantic_margin']):.3f}"
            ),
        )
        for path, entry in rendered
    )
    (output / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Fact3R query</title>"
        "<style>body{background:#161616;color:#eee;font-family:sans-serif}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}"
        "figure{margin:0;background:#222;padding:8px}img{width:100%;height:auto}"
        "figcaption{padding-top:6px}</style>"
        f"<h1>Query: {escape(query)}</h1><main>{cards}</main>",
        encoding="utf-8",
    )


def _write_no_match_html(
    output: Path,
    *,
    query: str,
    rejected_groups: Sequence[Mapping[str, object]],
) -> None:
    rows = "".join(
        "<tr><td>{}</td><td>{:.3f}</td><td>{}</td><td>{}</td></tr>".format(
            escape(str(group["group_id"])),
            float(group["entity_margin"]),
            int(group["supporting_view_count"]),
            escape(", ".join(str(item) for item in group["rejection_reasons"])),
        )
        for group in rejected_groups[:10]
    )
    (output / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>No confident match</title>"
        "<style>body{background:#161616;color:#eee;font-family:sans-serif;max-width:1000px;"
        "margin:40px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #555;"
        "padding:8px;text-align:left}</style>"
        f"<h1>No confident match for: {escape(query)}</h1>"
        "<p>No confirmed entity passed both the view-support and contrastive-margin gates.</p>"
        "<h2>Highest rejected candidates</h2><table><tr><th>Entity</th><th>Margin</th>"
        f"<th>Supporting views</th><th>Reason</th></tr>{rows}</table>",
        encoding="utf-8",
    )


def query_observation_index(
    *,
    index: str | Path,
    query: str,
    output: str | Path,
    encoder: VisionLanguageEncoder,
    max_entities: int = 3,
    min_entity_score: float | None = None,
    top_views: int = 3,
    positive_prompts: Sequence[str] | None = None,
    negative_prompts: Sequence[str] | None = None,
    confirmed_only: bool = True,
    include_unanchored_tracks: bool = True,
    min_supporting_views: int = 2,
    min_view_margin: float = 0.02,
    min_entity_margin: float = 0.02,
    reference_mask_area: float = 4096.0,
    automatic_map_negatives: bool = True,
    map_negative_neighbors: int = 3,
    map_negative_weight: float = 1.0,
    max_observations_per_entity: int | None = None,
    gif_width: int = 1000,
    gif_duration_ms: int = 400,
) -> Path:
    """Retrieve verified semantic entities and render every stored view."""

    if not query.strip():
        raise ValueError("query cannot be empty")
    if max_entities <= 0:
        raise ValueError("max_entities must be positive")
    if min_entity_score is not None:
        min_entity_margin = min_entity_score
    if max_observations_per_entity is not None and max_observations_per_entity <= 0:
        raise ValueError("max_observations_per_entity must be positive")
    if gif_width <= 0 or gif_duration_ms <= 0:
        raise ValueError("GIF dimensions and duration must be positive")
    started = perf_counter()
    manifest_path, manifest, embeddings = load_observation_index(index)
    if encoder.model_name != manifest["model"]:
        raise ValueError(
            f"query encoder {encoder.model_name!r} does not match index model "
            f"{manifest['model']!r}"
        )
    positive_prompts = tuple(
        prompt.strip()
        for prompt in (
            default_positive_prompts(query)
            if positive_prompts is None
            else positive_prompts
        )
        if prompt.strip()
    )
    negative_prompts = tuple(
        prompt.strip()
        for prompt in (
            default_negative_prompts()
            if negative_prompts is None
            else negative_prompts
        )
        if prompt.strip()
    )
    if not positive_prompts:
        raise ValueError("at least one positive prompt is required")
    if not negative_prompts:
        raise ValueError("at least one confounder prompt is required")
    text_started = perf_counter()
    text_embeddings = _normalise_rows(
        encoder.encode_text([*positive_prompts, *negative_prompts])
    )
    text_seconds = perf_counter() - text_started
    positive_scores, negative_scores, margins, qualities, groups = (
        rank_semantic_entity_groups(
            embeddings,
            manifest["observations"],
            text_embeddings[: len(positive_prompts)],
            text_embeddings[len(positive_prompts) :],
            top_views=top_views,
            min_supporting_views=min_supporting_views,
            min_view_margin=min_view_margin,
            min_entity_margin=min_entity_margin,
            reference_mask_area=reference_mask_area,
            confirmed_only=confirmed_only,
            include_unanchored_tracks=include_unanchored_tracks,
            automatic_map_negatives=automatic_map_negatives,
            map_negative_neighbors=map_negative_neighbors,
            map_negative_weight=map_negative_weight,
        )
    )
    accepted_groups = [group for group in groups if bool(group["accepted"])]
    rejected_groups = [group for group in groups if not bool(group["accepted"])]
    selected_groups = accepted_groups[:max_entities]

    output_directory = Path(output)
    output_directory.mkdir(parents=True, exist_ok=True)
    observations = manifest["observations"]
    result_groups: list[dict[str, object]] = []
    result_group_lookup: dict[str, dict[str, object]] = {}
    for rank, group in enumerate(selected_groups, start=1):
        best_observation = observations[
            int(group["ranked_observation_indices"][0])
        ]
        result_group = {
            "rank": rank,
            "group_id": group["group_id"],
            "entity_id": group["entity_id"],
            "track_id": group["track_id"],
            "entity_margin": group["entity_margin"],
            "observation_count": group["observation_count"],
            "supporting_view_count": group["supporting_view_count"],
            "mean_observation_quality": group["mean_observation_quality"],
            "map_hard_negative_score": group["map_hard_negative_score"],
            "map_hard_negatives": group["map_hard_negatives"],
            "memory_type": (
                "anchored_3d_entity"
                if group["entity_id"] is not None
                else "unanchored_2d_track"
            ),
            "navigation_target_available": group["entity_id"] is not None,
            "best_revisit_view": {
                "frame_id": best_observation["frame_id"],
                "timestamp": best_observation.get("timestamp"),
                "camera_pose_world_from_camera": best_observation.get(
                    "camera_pose_world_from_camera"
                ),
                "camera_origin_world": best_observation.get(
                    "camera_origin_world"
                ),
                "view_ray_world": best_observation.get("view_ray_world"),
                "mask_center_rc": best_observation.get("mask_center_rc"),
            },
            "observations": [],
        }
        result_groups.append(result_group)
        result_group_lookup[str(group["group_id"])] = result_group

    selected_rows: list[tuple[dict[str, object], dict[str, object]]] = []
    for group in selected_groups:
        indices = list(group["ranked_observation_indices"])
        if max_observations_per_entity is not None:
            indices = indices[:max_observations_per_entity]
        indices.sort(key=lambda index_value: int(observations[index_value]["frame_id"]))
        for observation_index in indices:
            selected_rows.append((group, observations[observation_index]))

    rendered: list[tuple[Path, dict[str, object]]] = []
    if selected_rows:
        needed_frames = {int(row[1]["frame_id"]) for row in selected_rows}
        keyframe_images = {
            keyframe.frame_id: np.array(keyframe.rgb, copy=True)
            for keyframe in iter_exported_keyframes(manifest["source_keyframes"])
            if keyframe.frame_id in needed_frames
        }
        proposal_directory = Path(str(manifest["source_proposals"]))
        frame_output = output_directory / "frames"
        frame_output.mkdir(parents=True, exist_ok=True)
        for render_index, (group, observation) in enumerate(selected_rows):
            frame_id = int(observation["frame_id"])
            rgb = keyframe_images.get(frame_id)
            if rgb is None:
                raise ValueError(
                    f"keyframe {frame_id} is missing from the source export"
                )
            with np.load(
                proposal_directory / str(observation["mask_file"]),
                allow_pickle=False,
            ) as evidence:
                mask = np.array(evidence["mask"], dtype=bool, copy=True)
            observation_index = int(observation["index"])
            rendered_image = _render_match(
                rgb,
                mask,
                query=query,
                group_id=str(group["group_id"]),
                frame_id=frame_id,
                positive_score=float(positive_scores[observation_index]),
                negative_score=float(negative_scores[observation_index]),
                semantic_margin=float(margins[observation_index]),
                observation_quality=float(qualities[observation_index]),
                entity_margin=float(group["entity_margin"]),
            )
            filename = (
                f"{render_index:04d}_{_slug(str(group['group_id']))}_"
                f"frame_{frame_id:06d}.jpg"
            )
            path = frame_output / filename
            rendered_image.save(path, quality=92)
            result_observation = {
                "proposal_id": observation["proposal_id"],
                "frame_id": frame_id,
                "timestamp": observation.get("timestamp"),
                "positive_score": float(positive_scores[observation_index]),
                "strongest_confounder_score": float(
                    negative_scores[observation_index]
                ),
                "semantic_margin": float(margins[observation_index]),
                "observation_quality": float(qualities[observation_index]),
                "geometry_status": observation.get(
                    "geometry_status", "anchored_3d"
                ),
                "geometry_coverage": observation.get("geometry_coverage", 1.0),
                "camera_pose_world_from_camera": observation.get(
                    "camera_pose_world_from_camera"
                ),
                "view_ray_world": observation.get("view_ray_world"),
                "supports_query": bool(
                    margins[observation_index] >= min_view_margin
                ),
                "image": str(path.relative_to(output_directory)),
            }
            result_group_lookup[str(group["group_id"])]["observations"].append(
                result_observation
            )
            rendered.append(
                (
                    path,
                    {
                        "group_id": group["group_id"],
                        **result_observation,
                    },
                )
            )

        _write_gallery_outputs(
            rendered,
            output_directory,
            query=query,
            gif_width=gif_width,
            gif_duration_ms=gif_duration_ms,
        )
    else:
        _write_no_match_html(
            output_directory,
            query=query,
            rejected_groups=rejected_groups,
        )

    rejected_diagnostics = [
        {
            "group_id": group["group_id"],
            "entity_id": group["entity_id"],
            "entity_margin": group["entity_margin"],
            "observation_count": group["observation_count"],
            "supporting_view_count": group["supporting_view_count"],
            "mean_observation_quality": group["mean_observation_quality"],
            "map_hard_negative_score": group["map_hard_negative_score"],
            "map_hard_negatives": group["map_hard_negatives"],
            "rejection_reasons": group["rejection_reasons"],
        }
        for group in rejected_groups[:20]
    ]
    result = {
        "format": "fact3r-semantic-query-results",
        "version": 2,
        "query": query,
        "source_index": str(manifest_path.resolve()),
        "model": encoder.model_name,
        "positive_prompts": list(positive_prompts),
        "confounder_prompts": list(negative_prompts),
        "ranking_config": {
            "confirmed_only": confirmed_only,
            "include_unanchored_tracks": include_unanchored_tracks,
            "entity_top_views": top_views,
            "min_supporting_views": min_supporting_views,
            "min_view_margin": min_view_margin,
            "min_entity_margin": min_entity_margin,
            "reference_mask_area": reference_mask_area,
            "automatic_map_negatives": automatic_map_negatives,
            "map_negative_neighbors": map_negative_neighbors,
            "map_negative_weight": map_negative_weight,
        },
        "confident_match_found": bool(result_groups),
        "selected_entity_count": len(result_groups),
        "rejected_entity_count": len(rejected_groups),
        "rendered_observation_count": len(rendered),
        "timing": {
            "model_load_seconds": float(encoder.load_seconds),
            "text_encoding_seconds": text_seconds,
            "total_query_seconds_excluding_model_load": perf_counter() - started,
        },
        "entities": result_groups,
        "highest_rejected_entities": rejected_diagnostics,
        "gallery": "index.html",
        "gif": "matches.gif" if rendered else None,
        "contact_sheet": "contact_sheet.jpg" if rendered else None,
    }
    result_path = output_directory / "results.json"
    result_path.write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result_path
