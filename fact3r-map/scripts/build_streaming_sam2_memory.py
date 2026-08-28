#!/usr/bin/env python3
"""Causal SAM2 memory with periodic dense discovery and inter-frame propagation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.association.tracklets import (  # noqa: E402
    TRACKLET_FORMAT,
    TRACKLET_VERSION,
    link_propagated_masks,
)
from fact3r.integrations.mast3r_slam import iter_exported_keyframes  # noqa: E402
from fact3r.proposals.mask_filter import (  # noqa: E402
    MaskFilterConfig,
    filter_image_mask_proposals,
    mask_iou,
)
from fact3r.proposals.mask_generator import MaskProposal2D  # noqa: E402
from fact3r.proposals.proposal_pipeline import (  # noqa: E402
    GeneratedProposal,
    GeometryStatus,
    generate_lifted_proposals,
)
from fact3r.proposals.sam2_official_generator import (  # noqa: E402
    SAM2OfficialMaskGenerator,
)
from fact3r.proposals.sam2_video_tracker import SAM2OfficialVideoTracker  # noqa: E402
from fact3r.proposals.storage import save_frame_proposals  # noqa: E402


def _rgb_uint8(rgb: object) -> np.ndarray:
    values = np.asarray(rgb)
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(np.nanmax(values)) <= 1.0:
            values = values * 255.0
    return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))


def _box(mask: np.ndarray) -> np.ndarray | None:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return np.asarray(
        [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1],
        dtype=np.float32,
    )


def _propagated_proposals(
    masks: tuple[np.ndarray, ...],
    sources: tuple[GeneratedProposal, ...],
    *,
    frame_id: int,
    config: MaskFilterConfig,
    source_name: str = "sam2-video-memory",
) -> tuple[GeneratedProposal, ...]:
    raw = [
        MaskProposal2D(
            proposal_id=f"frame-{frame_id:06d}-sam2-memory-{index:04d}",
            frame_id=frame_id,
            mask=mask,
            score=max(config.min_score, source.mask_2d.score * 0.995),
            bounding_box_xyxy=_box(mask),
            source=source_name,
            metadata={"source_proposal_id": source.mask_2d.proposal_id},
        )
        for index, (mask, source) in enumerate(zip(masks, sources))
    ]
    if not raw:
        return ()
    filtered = filter_image_mask_proposals(
        raw,
        raw[0].mask.shape,
        config,
        frame_id=frame_id,
    )
    return tuple(
        GeneratedProposal(
            mask_2d=proposal,
            lifted_3d=None,
            geometry_status=GeometryStatus.UNANCHORED_2D,
            geometry_coverage=0.0,
        )
        for proposal in filtered
    )


def _optical_flow_propagate(
    source_rgb: object,
    target_rgb: object,
    masks: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    """Warp every source mask with one shared backward optical-flow field."""

    if not masks:
        return ()
    try:
        import cv2
    except ImportError as error:
        raise ImportError(
            "optical-flow propagation requires opencv-python in the SAM2 environment"
        ) from error
    source = _rgb_uint8(source_rgb)
    target = _rgb_uint8(target_rgb)
    if source.shape != target.shape:
        raise ValueError("optical-flow frames must share one image shape")
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    # Backward flow tells each target pixel where to sample the source mask.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--proposals-output", type=Path, required=True)
    parser.add_argument("--tracklets-output", type=Path, required=True)
    parser.add_argument("--discovery-model", default="facebook/sam2-hiera-large")
    parser.add_argument("--tracking-model", default="facebook/sam2-hiera-small")
    parser.add_argument("--device", default="0")
    parser.add_argument("--refresh-frames", type=int, default=10)
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--points-per-batch", type=int, default=32)
    parser.add_argument("--max-seeds-per-batch", type=int, default=16)
    parser.add_argument(
        "--propagation-backend",
        choices=("sam2", "optical-flow"),
        default="sam2",
        help="SAM2 video memory or one shared DIS optical-flow warp",
    )
    parser.add_argument("--min-link-iou", type=float, default=0.30)
    parser.add_argument("--pred-iou-threshold", type=float, default=0.75)
    parser.add_argument("--stability-score-threshold", type=float, default=0.85)
    parser.add_argument("--min-area-pixels", type=int, default=40)
    parser.add_argument("--min-area-fraction", type=float, default=0.0002)
    parser.add_argument("--max-area-fraction", type=float, default=0.8)
    parser.add_argument("--min-component-pixels", type=int, default=20)
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    args = parser.parse_args()
    if args.refresh_frames <= 0:
        raise ValueError("refresh_frames must be positive")
    if args.propagation_backend == "optical-flow":
        try:
            import cv2  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "--propagation-backend optical-flow requires opencv-python "
                "in the SAM2 environment"
            ) from error

    keyframes = tuple(iter_exported_keyframes(args.keyframes))
    if not keyframes:
        raise ValueError("cannot stream an empty keyframe export")
    image_shapes = {item.image_shape for item in keyframes}
    if len(image_shapes) != 1:
        raise ValueError("all streaming frames must share one image shape")
    args.proposals_output.mkdir(parents=True, exist_ok=True)
    args.tracklets_output.mkdir(parents=True, exist_ok=True)
    filter_config = MaskFilterConfig(
        min_score=args.pred_iou_threshold,
        min_area_pixels=args.min_area_pixels,
        min_area_fraction=args.min_area_fraction,
        max_area_fraction=args.max_area_fraction,
        erosion_pixels=0,
        min_component_pixels=args.min_component_pixels,
        min_geometry_confidence=1.0,
    )
    started = perf_counter()
    discovery = SAM2OfficialMaskGenerator(
        args.discovery_model,
        device=args.device,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_threshold=args.pred_iou_threshold,
        stability_score_threshold=args.stability_score_threshold,
    )
    tracker = (
        SAM2OfficialVideoTracker(args.tracking_model, device=args.device)
        if args.propagation_backend == "sam2"
        else None
    )

    proposal_summaries: list[dict[str, object]] = []
    tracklet_frames: list[dict[str, object]] = []
    next_track_index = 0
    link_count = 0
    discovery_frames = 0
    discovery_seconds = 0.0
    propagation_seconds = 0.0
    propagated_filter_seconds = 0.0
    linking_seconds = 0.0
    storage_seconds = 0.0
    initialization_seconds = perf_counter() - started

    temporary_context = (
        tempfile.TemporaryDirectory(prefix="fact3r_streaming_sam2_")
        if tracker is not None
        else nullcontext(None)
    )
    with temporary_context as temporary:
        if tracker is not None:
            video_directory = Path(str(temporary))
            for index, keyframe in enumerate(keyframes):
                Image.fromarray(_rgb_uint8(keyframe.rgb)).save(
                    video_directory / f"{index:06d}.jpg", quality=95, subsampling=0
                )
            state = tracker.initialize(
                str(video_directory),
                offload_video_to_cpu=True,
                offload_state_to_cpu=args.offload_state_to_cpu,
            )
        else:
            state = None

        source_generated: tuple[GeneratedProposal, ...] = ()
        source_tracks: dict[str, str] = {}
        source_rgb: object | None = None
        for frame_index, keyframe in enumerate(keyframes):
            propagated_masks: tuple[np.ndarray, ...] = ()
            if frame_index > 0 and source_generated:
                stage_started = perf_counter()
                source_masks = tuple(
                    item.mask_2d.mask for item in source_generated
                )
                if tracker is None:
                    if source_rgb is None:
                        raise RuntimeError("missing source RGB for optical flow")
                    propagated_masks = _optical_flow_propagate(
                        source_rgb, keyframe.rgb, source_masks
                    )
                else:
                    propagated_masks = tracker.propagate_one_step(
                        state,
                        source_frame_index=frame_index - 1,
                        source_masks=source_masks,
                        max_seeds_per_batch=args.max_seeds_per_batch,
                    )
                propagation_seconds += perf_counter() - stage_started

            refresh = frame_index == 0 or frame_index % args.refresh_frames == 0
            if refresh:
                stage_started = perf_counter()
                target_generated = tuple(
                    generate_lifted_proposals(keyframe, discovery, filter_config)
                )
                discovery_seconds += perf_counter() - stage_started
                discovery_frames += 1
            else:
                stage_started = perf_counter()
                target_generated = _propagated_proposals(
                    propagated_masks,
                    source_generated,
                    frame_id=keyframe.frame_id,
                    config=filter_config,
                    source_name=(
                        "sam2-video-memory"
                        if tracker is not None
                        else "optical-flow-memory"
                    ),
                )
                propagated_filter_seconds += perf_counter() - stage_started

            target_ids = tuple(item.mask_2d.proposal_id for item in target_generated)
            target_masks = tuple(item.mask_2d.mask for item in target_generated)
            observations = []
            target_tracks: dict[str, str] = {}
            link_started = perf_counter()
            link_payload_by_target: dict[str, tuple[str, float]] = {}
            if frame_index == 0:
                pass
            elif not refresh:
                source_mask_by_id = {
                    item.mask_2d.proposal_id: propagated_masks[index]
                    for index, item in enumerate(source_generated)
                }
                for target in target_generated:
                    source_proposal_id = target.mask_2d.metadata.get(
                        "source_proposal_id"
                    )
                    if (
                        source_proposal_id is None
                        or str(source_proposal_id) not in source_tracks
                        or str(source_proposal_id) not in source_mask_by_id
                    ):
                        continue
                    source_proposal_id = str(source_proposal_id)
                    link_payload_by_target[target.mask_2d.proposal_id] = (
                        source_proposal_id,
                        mask_iou(
                            target.mask_2d.mask,
                            source_mask_by_id[source_proposal_id],
                        ),
                    )
                link_count += len(link_payload_by_target)
            else:
                links = (
                    ()
                    if not propagated_masks or not target_masks
                    else link_propagated_masks(
                        tuple(
                            item.mask_2d.proposal_id for item in source_generated
                        ),
                        propagated_masks,
                        target_ids,
                        target_masks,
                        min_mask_iou=args.min_link_iou,
                    )
                )
                link_payload_by_target = {
                    item.target_proposal_id: (
                        item.source_proposal_id,
                        item.mask_iou,
                    )
                    for item in links
                }
                link_count += len(links)
            linking_seconds += perf_counter() - link_started
            for proposal_id in target_ids:
                link_payload = link_payload_by_target.get(proposal_id)
                if link_payload is None:
                    track_id = f"track-{next_track_index:06d}"
                    next_track_index += 1
                    source_proposal_id = None
                    link_iou = None
                else:
                    source_proposal_id, link_iou = link_payload
                    track_id = source_tracks[source_proposal_id]
                target_tracks[proposal_id] = track_id
                observations.append(
                    {
                        "proposal_id": proposal_id,
                        "track_id": track_id,
                        "source_proposal_id": source_proposal_id,
                        "link_iou": link_iou,
                    }
                )
            storage_started = perf_counter()
            summary = save_frame_proposals(
                args.proposals_output, keyframe, target_generated
            )
            storage_seconds += perf_counter() - storage_started
            proposal_summaries.append(summary)
            tracklet_frames.append(
                {"frame_id": keyframe.frame_id, "observations": observations}
            )
            print(
                f"frame {keyframe.frame_id}: mode={'discover' if refresh else 'track'} "
                f"proposals={len(target_generated)} "
                f"linked={sum(item['source_proposal_id'] is not None for item in observations)}"
            )
            source_generated = target_generated
            source_tracks = target_tracks
            source_rgb = keyframe.rgb

    proposal_manifest = {
        "format": "fact3r-sam2-proposals",
        "version": 2,
        "backend": "official-streaming",
        "model": args.discovery_model,
        "tracking_model": args.tracking_model,
        "propagation_backend": args.propagation_backend,
        "keyframe_export": str(args.keyframes.resolve()),
        "refresh_frames": args.refresh_frames,
        "filter_config": {
            name: getattr(filter_config, name)
            for name in filter_config.__dataclass_fields__
        },
        "frame_count": len(proposal_summaries),
        "proposal_count": sum(item["proposal_count"] for item in proposal_summaries),
        "lifted_proposal_count": 0,
        "unanchored_proposal_count": sum(
            item["proposal_count"] for item in proposal_summaries
        ),
        "timing": {
            "discovery_frames": discovery_frames,
            "discovery_seconds": discovery_seconds,
            "propagation_seconds": propagation_seconds,
            "propagated_filter_seconds": propagated_filter_seconds,
            "linking_seconds": linking_seconds,
            "storage_seconds": storage_seconds,
            "initialization_seconds": initialization_seconds,
            "total_seconds": perf_counter() - started,
        },
        "frames": [
            {
                "frame_id": item["frame_id"],
                "proposal_count": item["proposal_count"],
                "lifted_proposal_count": 0,
                "unanchored_proposal_count": item["proposal_count"],
                "manifest": f"frame_{item['frame_id']:06d}/manifest.json",
            }
            for item in proposal_summaries
        ],
    }
    (args.proposals_output / "manifest.json").write_text(
        json.dumps(proposal_manifest, indent=2) + "\n", encoding="utf-8"
    )
    tracklet_manifest = {
        "format": TRACKLET_FORMAT,
        "version": TRACKLET_VERSION,
        "source_proposals": str(args.proposals_output.resolve()),
        "keyframe_export": str(args.keyframes.resolve()),
        "model": (
            args.tracking_model
            if args.propagation_backend == "sam2"
            else "opencv-dis-fast"
        ),
        "propagation_backend": args.propagation_backend,
        "min_link_iou": args.min_link_iou,
        "max_seeds_per_batch": args.max_seeds_per_batch,
        "frame_count": len(tracklet_frames),
        "track_count": next_track_index,
        "link_count": link_count,
        "frames": tracklet_frames,
    }
    (args.tracklets_output / "manifest.json").write_text(
        json.dumps(tracklet_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Streaming SAM2: {len(keyframes)} frames, {discovery_frames} dense "
        f"refreshes, discovery={discovery_seconds:.1f}s, "
        f"propagation={propagation_seconds:.1f}s, "
        f"filter={propagated_filter_seconds:.1f}s, "
        f"link={linking_seconds:.1f}s, storage={storage_seconds:.1f}s, "
        f"initialization={initialization_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
