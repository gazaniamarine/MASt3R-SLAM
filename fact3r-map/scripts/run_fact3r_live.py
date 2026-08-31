#!/usr/bin/env python3
"""Causal real-time Fact3R mapping from a camera, stream, or paced video."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter, sleep

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.association.live_image_uot import (  # noqa: E402
    LiveImageUOTMapper,
    LiveTrackletEvidence,
)
from fact3r.association.tracklets import (  # noqa: E402
    TRACKLET_FORMAT,
    TRACKLET_VERSION,
    link_propagated_masks,
)
from fact3r.proposals.mask_filter import (  # noqa: E402
    MaskFilterConfig,
    filter_image_mask_proposals,
    mask_iou,
)
from fact3r.proposals.mask_generator import MaskProposal2D  # noqa: E402
from fact3r.proposals.sam2_official_generator import (  # noqa: E402
    SAM2OfficialMaskGenerator,
)
from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    attach_mapping_to_observation_index,
    masked_context_crop,
)


def _capture_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


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


def _box(mask: np.ndarray) -> np.ndarray | None:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return np.asarray(
        [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1],
        dtype=np.float32,
    )


def _propagate_masks(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    masks: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    if not masks:
        return ()
    import cv2

    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = estimator.calc(target_gray, source_gray, None)
    height, width = source_gray.shape
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = np.ascontiguousarray(columns + flow[..., 0], dtype=np.float32)
    map_y = np.ascontiguousarray(rows + flow[..., 1], dtype=np.float32)
    return tuple(
        np.ascontiguousarray(
            cv2.remap(
                np.asarray(mask, dtype=np.uint8),
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        )
        for mask in masks
    )


def _propagated_proposals(
    masks: tuple[np.ndarray, ...],
    sources: tuple[MaskProposal2D, ...],
    *,
    frame_id: int,
    config: MaskFilterConfig,
) -> tuple[MaskProposal2D, ...]:
    raw = [
        MaskProposal2D(
            proposal_id=f"frame-{frame_id:06d}-flow-{index:04d}",
            frame_id=frame_id,
            mask=mask,
            score=max(config.min_score, source.score * 0.995),
            bounding_box_xyxy=_box(mask),
            source="optical-flow-memory",
            metadata={"source_proposal_id": source.proposal_id},
        )
        for index, (mask, source) in enumerate(zip(masks, sources))
    ]
    return tuple(
        filter_image_mask_proposals(
            raw, raw[0].mask.shape, config, frame_id=frame_id
        )
        if raw
        else ()
    )


def _save_keyframe(
    directory: Path,
    *,
    frame_id: int,
    source_frame_id: int,
    timestamp: float,
    rgb: np.ndarray,
) -> dict[str, object]:
    rgb_directory = directory / "rgb"
    rgb_directory.mkdir(parents=True, exist_ok=True)
    image_name = f"frame_{frame_id:06d}.jpg"
    Image.fromarray(rgb).save(rgb_directory / image_name, quality=94, subsampling=0)
    height, width = rgb.shape[:2]
    filename = f"keyframe_{frame_id:06d}_frame_{frame_id:06d}.npz"
    np.savez_compressed(
        directory / filename,
        frame_id=np.asarray(frame_id, dtype=np.int64),
        rgb=rgb,
        pointmap_camera=np.full((height, width, 3), np.nan, np.float32),
        geometry_confidence=np.zeros((height, width), np.float32),
        pose_world_from_camera=np.eye(4, dtype=np.float32),
    )
    return {
        "keyframe_index": frame_id,
        "frame_id": frame_id,
        "source_frame_id": source_frame_id,
        "timestamp": timestamp,
        "file": filename,
        "rgb_file": f"rgb/{image_name}",
        "image_shape": [height, width],
        "has_mast3r_descriptors": False,
    }


def _save_proposals(
    directory: Path,
    *,
    frame_id: int,
    timestamp: float,
    image_shape: tuple[int, int],
    proposals: tuple[MaskProposal2D, ...],
) -> dict[str, object]:
    frame_directory = directory / f"frame_{frame_id:06d}"
    frame_directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, proposal in enumerate(proposals):
        filename = f"proposal_{index:04d}.npz"
        np.savez_compressed(frame_directory / filename, mask=proposal.mask)
        entries.append(
            {
                "proposal_id": proposal.proposal_id,
                "file": filename,
                "source": proposal.source,
                "score": proposal.score,
                "mask_area": proposal.area,
                "geometry_status": "unanchored_2d",
                "geometry_coverage": 0.0,
                "lifted_point_count": 0,
                "centroid_xyz": None,
                "bounding_box_xyz": None,
                "bounding_box_xyxy": (
                    None
                    if proposal.bounding_box_xyxy is None
                    else proposal.bounding_box_xyxy.tolist()
                ),
            }
        )
    manifest = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "image_shape": list(image_shape),
        "proposal_count": len(proposals),
        "lifted_proposal_count": 0,
        "unanchored_proposal_count": len(proposals),
        "visualization": None,
        "proposals": entries,
    }
    (frame_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _colour(entity_id: str) -> np.ndarray:
    value = abs(hash(entity_id))
    return np.asarray(
        [60 + value % 170, 60 + (value // 173) % 170, 60 + (value // 29929) % 170],
        dtype=np.float32,
    )


def _preview(
    rgb: np.ndarray,
    proposals: tuple[MaskProposal2D, ...],
    assignments: dict[str, object],
    *,
    frame_id: int,
    processed_fps: float,
    target_fps: float,
) -> np.ndarray:
    import cv2

    canvas = rgb.astype(np.float32)
    for proposal in proposals:
        assignment = assignments[proposal.proposal_id]
        colour = _colour(assignment.entity_id)
        canvas[proposal.mask] = 0.55 * canvas[proposal.mask] + 0.45 * colour
        if proposal.bounding_box_xyxy is not None:
            x0, y0, x1, y1 = proposal.bounding_box_xyxy.astype(int)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour.tolist(), 1)
            cv2.putText(
                canvas,
                assignment.entity_id.replace("image-entity-", "E"),
                (x0, max(12, y0 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                colour.tolist(),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        f"frame={frame_id} processed={processed_fps:.2f} FPS target={target_fps:g}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _write_artifacts(
    *,
    output: Path,
    source: str,
    sample_fps: float,
    discovery_model: str,
    filter_config: MaskFilterConfig,
    refresh_frames: int,
    frame_entries: list[dict[str, object]],
    proposal_frames: list[dict[str, object]],
    tracklet_frames: list[dict[str, object]],
    observations: list[dict[str, object]],
    embeddings: list[np.ndarray],
    mapper: LiveImageUOTMapper,
    encoder: Siglip2Encoder,
    encoded_count: int,
    semantic_seconds: float,
    started: float,
    processing_seconds: float,
    dropped_frames: int,
) -> None:
    frames = output / "frames"
    proposals = output / "sam2_proposals"
    tracklets = output / "sam2_tracklets"
    appearance = output / "siglip_pre_uot"
    mapping = output / "image_uot"
    final_index = output / "siglip_observations"
    for directory in (frames, proposals, tracklets, appearance, mapping):
        directory.mkdir(parents=True, exist_ok=True)
    frame_manifest = {
        "format": "fact3r-mast3r-keyframes",
        "version": 1,
        "coordinate_convention": "image_only_no_geometry",
        "mode": "live_real_world",
        "source_video": source,
        "source_fps": None,
        "sample_fps": sample_fps,
        "keyframes": frame_entries,
    }
    (frames / "manifest.json").write_text(
        json.dumps(frame_manifest, indent=2) + "\n", encoding="utf-8"
    )
    proposal_manifest = {
        "format": "fact3r-sam2-proposals",
        "version": 2,
        "backend": "official-live",
        "model": discovery_model,
        "tracking_model": "opencv-dis-fast",
        "propagation_backend": "optical-flow",
        "keyframe_export": str(frames.resolve()),
        "refresh_frames": refresh_frames,
        "filter_config": asdict(filter_config),
        "frame_count": len(proposal_frames),
        "proposal_count": sum(item["proposal_count"] for item in proposal_frames),
        "lifted_proposal_count": 0,
        "unanchored_proposal_count": sum(
            item["proposal_count"] for item in proposal_frames
        ),
        "frames": [
            {
                "frame_id": item["frame_id"],
                "proposal_count": item["proposal_count"],
                "lifted_proposal_count": 0,
                "unanchored_proposal_count": item["proposal_count"],
                "manifest": f"frame_{int(item['frame_id']):06d}/manifest.json",
            }
            for item in proposal_frames
        ],
    }
    (proposals / "manifest.json").write_text(
        json.dumps(proposal_manifest, indent=2) + "\n", encoding="utf-8"
    )
    all_track_ids = {
        str(item["track_id"])
        for frame in tracklet_frames
        for item in frame["observations"]
    }
    link_count = sum(
        item["source_proposal_id"] is not None
        for frame in tracklet_frames
        for item in frame["observations"]
    )
    tracklet_manifest = {
        "format": TRACKLET_FORMAT,
        "version": TRACKLET_VERSION,
        "source_proposals": str(proposals.resolve()),
        "keyframe_export": str(frames.resolve()),
        "model": "opencv-dis-fast",
        "propagation_backend": "optical-flow",
        "min_link_iou": 0.30,
        "max_seeds_per_batch": 0,
        "frame_count": len(tracklet_frames),
        "track_count": len(all_track_ids),
        "link_count": link_count,
        "frames": tracklet_frames,
    }
    (tracklets / "manifest.json").write_text(
        json.dumps(tracklet_manifest, indent=2) + "\n", encoding="utf-8"
    )
    embedding_matrix = (
        np.stack(embeddings).astype(np.float32, copy=False)
        if embeddings
        else np.empty((0, 0), dtype=np.float32)
    )
    np.save(appearance / "embeddings.npy", embedding_matrix)
    elapsed = perf_counter() - started
    appearance_manifest = {
        "format": "fact3r-siglip-observation-index",
        "version": 1,
        "semantic_query_capable": True,
        "model": encoder.model_name,
        "device": encoder.device_name,
        "source_keyframes": str(frames.resolve()),
        "source_proposals": str(proposals.resolve()),
        "source_tracklets": str((tracklets / "manifest.json").resolve()),
        "source_mapping": None,
        "embedding_file": "embeddings.npy",
        "embedding_dimension": 0 if not embeddings else int(embedding_matrix.shape[1]),
        "embedding_dtype": str(embedding_matrix.dtype),
        "frame_count": len(frame_entries),
        "observation_count": len(observations),
        "assigned_observation_count": 0,
        "unanchored_observation_count": len(observations),
        "track_only_observation_count": len(observations),
        "crop_config": {
            "context_fraction": 0.02,
            "outside_mask_alpha": 0.0,
            "reuse_propagated_track_embeddings": True,
        },
        "encoded_observation_count": encoded_count,
        "reused_embedding_count": len(observations) - encoded_count,
        "timing": {
            "model_load_seconds": float(encoder.load_seconds),
            "image_encoding_seconds": semantic_seconds,
            "total_index_seconds_excluding_model_load": elapsed,
        },
        "observations": observations,
    }
    (appearance / "manifest.json").write_text(
        json.dumps(appearance_manifest, indent=2) + "\n", encoding="utf-8"
    )
    mapping_manifest = {
        "format": "fact3r-visibility-residual-transport",
        "version": 1,
        "mode": "live_image_uot_no_dense_reconstruction",
        "source_proposals": str(proposals.resolve()),
        "source_tracklets": str((tracklets / "manifest.json").resolve()),
        "source_appearance_index": str((appearance / "manifest.json").resolve()),
        "source_mast3r_pair_matches": None,
        "appearance_model": encoder.model_name,
        "association_cues": ["appearance", "optical_flow_temporal_continuity"],
        "entity_count": mapper.entity_count,
        "matched_total": mapper.total_matches,
        "created_total": mapper.total_births,
        "committed_track_entities": mapper.committed_track_entities,
        "timing": {
            "total_seconds": processing_seconds,
            "frames_per_second": (
                len(frame_entries) / processing_seconds
                if processing_seconds > 0
                else None
            ),
        },
        "frames": mapper.frames,
    }
    (mapping / "manifest.json").write_text(
        json.dumps(mapping_manifest, indent=2) + "\n", encoding="utf-8"
    )
    if observations:
        attach_mapping_to_observation_index(
            index=appearance,
            mapping=mapping,
            output=final_index,
        )
    status = {
        "format": "fact3r-live-status",
        "version": 1,
        "source": source,
        "sample_fps": sample_fps,
        "processed_frames": len(frame_entries),
        "dropped_frames": dropped_frames,
        "entities": mapper.entity_count,
        "wall_seconds": elapsed,
        "processing_seconds": processing_seconds,
        "processed_fps": (
            len(frame_entries) / processing_seconds if processing_seconds > 0 else 0.0
        ),
        "real_time_factor": (
            len(frame_entries) / processing_seconds / sample_fps
            if processing_seconds > 0
            else 0.0
        ),
        "mast3r_live_matching": False,
    }
    (output / "live_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="camera index, RTSP/HTTP URL, or video path",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-realtime-pacing", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--refresh-seconds", type=float, default=10.0)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--discovery-model", default="facebook/sam2-hiera-large")
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    args = parser.parse_args()
    if args.sample_fps <= 0 or args.refresh_seconds <= 0:
        raise ValueError("sample FPS and refresh seconds must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")
    if (args.output / "live_status.json").exists():
        raise ValueError(
            f"output already contains a completed/checkpointed live run: {args.output}; "
            "choose a new output directory"
        )

    import cv2

    source_value = _capture_source(args.source)
    capture = cv2.VideoCapture(source_value)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise ValueError(f"cannot open live source: {args.source}")
    is_file = not isinstance(source_value, int) and Path(args.source).is_file()
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = args.sample_fps
    refresh_frames = max(1, round(args.sample_fps * args.refresh_seconds))
    filter_config = MaskFilterConfig(
        min_score=0.70,
        min_area_pixels=20,
        min_area_fraction=0.0001,
        max_area_fraction=0.8,
        erosion_pixels=0,
        min_component_pixels=10,
        min_geometry_confidence=1.0,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frames_directory = args.output / "frames"
    proposals_directory = args.output / "sam2_proposals"
    frames_directory.mkdir(parents=True, exist_ok=True)
    proposals_directory.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM2 discovery model {args.discovery_model}...")
    discovery = SAM2OfficialMaskGenerator(
        args.discovery_model,
        device=args.device,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_threshold=filter_config.min_score,
        stability_score_threshold=0.80,
    )
    print(f"Loading SigLIP model {args.siglip_model}...")
    encoder = Siglip2Encoder(args.siglip_model, device=args.device)
    mapper = LiveImageUOTMapper()

    frame_entries: list[dict[str, object]] = []
    proposal_frames: list[dict[str, object]] = []
    tracklet_frames: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    observation_embeddings: list[np.ndarray] = []
    observation_index_by_proposal: dict[str, int] = {}
    source_proposals: tuple[MaskProposal2D, ...] = ()
    source_tracks: dict[str, str] = {}
    source_vectors: dict[str, np.ndarray] = {}
    source_rgb: np.ndarray | None = None
    next_track = 0
    source_frame_id = 0
    processed = 0
    dropped = 0
    next_sample_timestamp = 0.0
    started = perf_counter()
    capture_started = started
    processing_seconds = 0.0
    semantic_seconds = 0.0
    encoded_count = 0

    def checkpoint() -> None:
        _write_artifacts(
            output=args.output,
            source=args.source,
            sample_fps=args.sample_fps,
            discovery_model=args.discovery_model,
            filter_config=filter_config,
            refresh_frames=refresh_frames,
            frame_entries=frame_entries,
            proposal_frames=proposal_frames,
            tracklet_frames=tracklet_frames,
            observations=observations,
            embeddings=observation_embeddings,
            mapper=mapper,
            encoder=encoder,
            encoded_count=encoded_count,
            semantic_seconds=semantic_seconds,
            started=started,
            processing_seconds=processing_seconds,
            dropped_frames=dropped,
        )

    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if is_file:
                timestamp = source_frame_id / source_fps
            else:
                timestamp = perf_counter() - capture_started
            source_frame_id += 1
            if timestamp + 1e-9 < next_sample_timestamp:
                dropped += 1
                continue
            if is_file and not args.no_realtime_pacing:
                due = capture_started + timestamp
                now = perf_counter()
                if now + 1e-9 < due:
                    sleep(due - now)
                elif now - due > 1.0 / args.sample_fps:
                    dropped += 1
                    next_sample_timestamp += 1.0 / args.sample_fps
                    continue
            next_sample_timestamp += 1.0 / args.sample_fps
            frame_started = perf_counter()
            rgb = _mast3r_raster(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frame_id = processed
            propagated_masks = (
                ()
                if source_rgb is None or not source_proposals
                else _propagate_masks(
                    source_rgb,
                    rgb,
                    tuple(item.mask for item in source_proposals),
                )
            )
            refresh = (
                frame_id == 0
                or frame_id % refresh_frames == 0
                or not source_proposals
            )
            if refresh:
                proposals = tuple(
                    filter_image_mask_proposals(
                        list(discovery.generate(rgb, frame_id=frame_id)),
                        rgb.shape[:2],
                        filter_config,
                        frame_id=frame_id,
                    )
                )
            else:
                proposals = _propagated_proposals(
                    propagated_masks,
                    source_proposals,
                    frame_id=frame_id,
                    config=filter_config,
                )

            links: dict[str, tuple[str, float]] = {}
            if source_proposals and proposals:
                if refresh:
                    linked = link_propagated_masks(
                        tuple(item.proposal_id for item in source_proposals),
                        propagated_masks,
                        tuple(item.proposal_id for item in proposals),
                        tuple(item.mask for item in proposals),
                        min_mask_iou=0.30,
                    )
                    links = {
                        item.target_proposal_id: (
                            item.source_proposal_id,
                            item.mask_iou,
                        )
                        for item in linked
                    }
                else:
                    propagated_by_source = {
                        source.proposal_id: propagated_masks[index]
                        for index, source in enumerate(source_proposals)
                    }
                    for proposal in proposals:
                        source_id = str(proposal.metadata["source_proposal_id"])
                        links[proposal.proposal_id] = (
                            source_id,
                            mask_iou(proposal.mask, propagated_by_source[source_id]),
                        )

            current_tracks: dict[str, str] = {}
            live_tracklets: dict[str, LiveTrackletEvidence] = {}
            tracklet_observations = []
            for proposal in proposals:
                link = links.get(proposal.proposal_id)
                if link is None or link[0] not in source_tracks:
                    track_id = f"track-{next_track:06d}"
                    next_track += 1
                    source_id = None
                    link_iou = None
                else:
                    source_id, link_iou = link
                    track_id = source_tracks[source_id]
                current_tracks[proposal.proposal_id] = track_id
                live_tracklets[proposal.proposal_id] = LiveTrackletEvidence(
                    track_id, source_id, link_iou
                )
                tracklet_observations.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "track_id": track_id,
                        "source_proposal_id": source_id,
                        "link_iou": link_iou,
                    }
                )

            vectors: list[np.ndarray | None] = [None] * len(proposals)
            encode_indices = []
            crops = []
            for index, proposal in enumerate(proposals):
                source_id = live_tracklets[proposal.proposal_id].source_proposal_id
                if not refresh and source_id in source_vectors:
                    vectors[index] = source_vectors[str(source_id)]
                else:
                    encode_indices.append(index)
                    crops.append(
                        masked_context_crop(
                            rgb,
                            proposal.mask,
                            context_fraction=0.02,
                            outside_mask_alpha=0.0,
                        )
                    )
            if crops:
                semantic_started = perf_counter()
                encoded = encoder.encode_images(crops)
                semantic_seconds += perf_counter() - semantic_started
                encoded_count += len(encoded)
                for index, vector in zip(encode_indices, encoded):
                    vectors[index] = np.asarray(vector, dtype=np.float32)
            if any(vector is None for vector in vectors):
                raise RuntimeError("live semantic embedding assignment is incomplete")
            vector_matrix = (
                np.stack(vectors).astype(np.float32, copy=False)
                if vectors
                else np.empty((0, 1), dtype=np.float32)
            )
            assignments, frame_mapping = mapper.update(
                frame_id=frame_id,
                proposals=proposals,
                embeddings=vector_matrix,
                tracklets=live_tracklets,
            )

            frame_entries.append(
                _save_keyframe(
                    frames_directory,
                    frame_id=frame_id,
                    source_frame_id=source_frame_id - 1,
                    timestamp=timestamp,
                    rgb=rgb,
                )
            )
            proposal_manifest = _save_proposals(
                proposals_directory,
                frame_id=frame_id,
                timestamp=timestamp,
                image_shape=rgb.shape[:2],
                proposals=proposals,
            )
            proposal_frames.append(proposal_manifest)
            tracklet_frames.append(
                {"frame_id": frame_id, "observations": tracklet_observations}
            )
            for proposal_index, (proposal, proposal_entry) in enumerate(
                zip(proposals, proposal_manifest["proposals"])
            ):
                rows, columns = np.nonzero(proposal.mask)
                tracklet = live_tracklets[proposal.proposal_id]
                reference = (
                    None
                    if tracklet.source_proposal_id is None
                    else observation_index_by_proposal.get(
                        tracklet.source_proposal_id
                    )
                )
                observation_index = len(observations)
                observations.append(
                    {
                        "index": observation_index,
                        "proposal_id": proposal.proposal_id,
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "entity_id": None,
                        "track_id": tracklet.track_id,
                        "group_id": tracklet.track_id,
                        "assignment_status": "unanchored_2d",
                        "association_confidence": assignments[
                            proposal.proposal_id
                        ].confidence,
                        "proposal_score": proposal.score,
                        "mask_area": proposal.area,
                        "geometry_status": "unanchored_2d",
                        "geometry_coverage": 0.0,
                        "lifted_point_count": 0,
                        "bounding_box_xyxy": proposal_entry["bounding_box_xyxy"],
                        "mask_center_rc": [
                            float(np.mean(rows)),
                            float(np.mean(columns)),
                        ],
                        "camera_pose_world_from_camera": np.eye(4).tolist(),
                        "camera_origin_world": [0.0, 0.0, 0.0],
                        "view_ray_world": None,
                        "track_link_iou": tracklet.link_iou,
                        "mask_file": (
                            f"frame_{frame_id:06d}/{proposal_entry['file']}"
                        ),
                        "embedding_reused_from_observation": reference,
                    }
                )
                observation_embeddings.append(vector_matrix[proposal_index])
                observation_index_by_proposal[proposal.proposal_id] = observation_index

            processed += 1
            processing_seconds += perf_counter() - frame_started
            rolling_fps = processed / max(processing_seconds, 1e-9)
            preview = _preview(
                rgb,
                proposals,
                assignments,
                frame_id=frame_id,
                processed_fps=rolling_fps,
                target_fps=args.sample_fps,
            )
            cv2.imwrite(
                str(args.output / "latest_preview.jpg"),
                cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
            )
            print(
                f"frame {frame_id}: mode={'discover' if refresh else 'track'} "
                f"proposals={len(proposals)} matched={len(frame_mapping['matches'])} "
                f"entities={mapper.entity_count} frame_time={perf_counter() - frame_started:.3f}s "
                f"processed_fps={rolling_fps:.3f} "
                f"realtime_factor={rolling_fps / args.sample_fps:.3f}x"
            )
            if args.display:
                cv2.imshow(
                    "Fact3R live",
                    cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
                )
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break
            source_proposals = proposals
            source_tracks = current_tracks
            source_vectors = {
                proposal.proposal_id: vector_matrix[index]
                for index, proposal in enumerate(proposals)
            }
            source_rgb = rgb
            if processed % args.checkpoint_every == 0:
                checkpoint_started = perf_counter()
                checkpoint()
                processing_seconds += perf_counter() - checkpoint_started
            if args.max_frames is not None and processed >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("Stopping live capture and finalizing the semantic memory...")
    finally:
        capture.release()
        if args.display:
            cv2.destroyAllWindows()

    if not frame_entries:
        raise ValueError("live source produced no sampled frames")
    checkpoint()
    status = json.loads((args.output / "live_status.json").read_text())
    print("Live Fact3R map complete")
    print(f"processed frames: {status['processed_frames']}")
    print(f"processed FPS:    {status['processed_fps']:.3f}")
    print(f"real-time factor: {status['real_time_factor']:.3f}x")
    print(f"entities:         {status['entities']}")
    print(f"preview:          {args.output / 'latest_preview.jpg'}")
    query_index = args.output / "siglip_observations"
    if query_index.exists():
        print(f"query index:      {query_index}")
    else:
        print("query index:      not created because no searchable masks were found")
    print("Live mode uses appearance + temporal continuity; MASt3R is not run live.")


if __name__ == "__main__":
    main()
