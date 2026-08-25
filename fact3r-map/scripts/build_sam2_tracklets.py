#!/usr/bin/env python3
"""Build re-anchored short-term tracklets with Meta's SAM2 video predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

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
from fact3r.proposals.sam2_video_tracker import (  # noqa: E402
    SAM2OfficialVideoTracker,
)
from fact3r.proposals.storage import load_proposal_run_manifest  # noqa: E402


def _default_output(proposal_directory: Path) -> Path:
    return (
        proposal_directory.parent.parent
        / "fact3r_sam2_tracklets"
        / proposal_directory.name
    )


def _rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb)
    if np.issubdtype(values.dtype, np.floating) and values.size:
        if float(np.nanmax(values)) <= 1.0:
            values = values * 255.0
    return np.ascontiguousarray(np.clip(values, 0, 255).astype(np.uint8))


def _write_video_frames(
    keyframe_directory: Path, video_directory: Path
) -> tuple[int, ...]:
    frame_ids: list[int] = []
    image_shape: tuple[int, int] | None = None
    for keyframe_index, keyframe in enumerate(
        iter_exported_keyframes(keyframe_directory)
    ):
        if image_shape is None:
            image_shape = keyframe.image_shape
        elif keyframe.image_shape != image_shape:
            raise ValueError("SAM2 video keyframes must all share one image shape")
        Image.fromarray(_rgb_uint8(keyframe.rgb), mode="RGB").save(
            video_directory / f"{keyframe_index:06d}.jpg",
            quality=95,
            subsampling=0,
        )
        frame_ids.append(keyframe.frame_id)
    return tuple(frame_ids)


def _load_mask_frame(
    proposal_directory: Path, run_entry: dict[str, object]
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    frame_manifest_path = proposal_directory / str(run_entry["manifest"])
    frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    proposal_ids: list[str] = []
    masks: list[np.ndarray] = []
    for entry in frame_manifest["proposals"]:
        with np.load(
            frame_manifest_path.parent / entry["file"], allow_pickle=False
        ) as payload:
            mask = np.array(payload["mask"], dtype=bool, copy=True)
        proposal_ids.append(str(entry["proposal_id"]))
        masks.append(mask)
    return tuple(proposal_ids), tuple(masks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-link-iou", type=float, default=0.30)
    parser.add_argument("--max-seeds-per-batch", type=int, default=16)
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    args = parser.parse_args()

    proposal_run = load_proposal_run_manifest(args.proposals)
    if proposal_run.get("backend") != "official":
        raise ValueError("SAM2 video tracklets require official-SAM2 proposals")
    model = args.model or str(proposal_run["model"])
    proposal_frames = proposal_run["frames"]
    output = args.output or _default_output(args.proposals)
    output.mkdir(parents=True, exist_ok=True)

    frame_payloads: list[dict[str, object]] = []
    next_track_index = 0
    link_count = 0
    with tempfile.TemporaryDirectory(prefix="fact3r_sam2_tracklets_") as temporary:
        video_directory = Path(temporary)
        keyframe_ids = _write_video_frames(args.keyframes, video_directory)
        proposal_frame_ids = tuple(
            int(entry["frame_id"]) for entry in proposal_frames
        )
        if keyframe_ids != proposal_frame_ids:
            raise ValueError(
                "keyframe export and proposal run contain different frame IDs"
            )
        if not proposal_frames:
            raise ValueError("cannot build tracklets for an empty proposal run")

        tracker = SAM2OfficialVideoTracker(model, device=args.device)
        state = tracker.initialize(
            str(video_directory),
            offload_video_to_cpu=True,
            offload_state_to_cpu=args.offload_state_to_cpu,
        )

        source_ids, source_masks = _load_mask_frame(
            args.proposals, proposal_frames[0]
        )
        source_tracks: dict[str, str] = {}
        first_observations = []
        for proposal_id in source_ids:
            track_id = f"track-{next_track_index:06d}"
            next_track_index += 1
            source_tracks[proposal_id] = track_id
            first_observations.append(
                {
                    "proposal_id": proposal_id,
                    "track_id": track_id,
                    "source_proposal_id": None,
                    "link_iou": None,
                }
            )
        frame_payloads.append(
            {
                "frame_id": proposal_frame_ids[0],
                "observations": first_observations,
            }
        )

        for source_index in range(len(proposal_frames) - 1):
            target_ids, target_masks = _load_mask_frame(
                args.proposals, proposal_frames[source_index + 1]
            )
            propagated_masks = tracker.propagate_one_step(
                state,
                source_frame_index=source_index,
                source_masks=source_masks,
                max_seeds_per_batch=args.max_seeds_per_batch,
            )
            links = link_propagated_masks(
                source_ids,
                propagated_masks,
                target_ids,
                target_masks,
                min_mask_iou=args.min_link_iou,
            )
            links_by_target = {link.target_proposal_id: link for link in links}
            target_tracks: dict[str, str] = {}
            observations = []
            for proposal_id in target_ids:
                link = links_by_target.get(proposal_id)
                if link is None:
                    track_id = f"track-{next_track_index:06d}"
                    next_track_index += 1
                    source_proposal_id = None
                    link_iou = None
                else:
                    track_id = source_tracks[link.source_proposal_id]
                    source_proposal_id = link.source_proposal_id
                    link_iou = link.mask_iou
                target_tracks[proposal_id] = track_id
                observations.append(
                    {
                        "proposal_id": proposal_id,
                        "track_id": track_id,
                        "source_proposal_id": source_proposal_id,
                        "link_iou": link_iou,
                    }
                )
            frame_id = proposal_frame_ids[source_index + 1]
            frame_payloads.append(
                {"frame_id": frame_id, "observations": observations}
            )
            link_count += len(links)
            print(
                f"frame {frame_id}: proposals={len(target_ids)} "
                f"linked={len(links)} new_tracks={len(target_ids) - len(links)}"
            )
            source_ids, source_masks, source_tracks = (
                target_ids,
                target_masks,
                target_tracks,
            )

    manifest = {
        "format": TRACKLET_FORMAT,
        "version": TRACKLET_VERSION,
        "source_proposals": str(args.proposals.resolve()),
        "keyframe_export": str(args.keyframes.resolve()),
        "model": model,
        "min_link_iou": args.min_link_iou,
        "max_seeds_per_batch": args.max_seeds_per_batch,
        "frame_count": len(frame_payloads),
        "track_count": next_track_index,
        "link_count": link_count,
        "frames": frame_payloads,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {link_count} SAM2 links across {next_track_index} tracks "
        f"to {manifest_path}"
    )


if __name__ == "__main__":
    main()
