#!/usr/bin/env python3
"""Segment every captured frame in a short rendered HM3D robot trajectory."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.experiments import (  # noqa: E402
    AdjacentMaskTracker,
    select_frame_window,
)
from fact3r.proposals.mask_filter import (  # noqa: E402
    MaskFilterConfig,
    filter_image_mask_proposals,
)
from fact3r.proposals.sam2_generator import (  # noqa: E402
    SAM2AutomaticMaskGenerator,
)
from fact3r.proposals.sam2_official_generator import (  # noqa: E402
    SAM2OfficialMaskGenerator,
)
from fact3r.visualization.association import (  # noqa: E402
    entity_colour,
    mask_boundary,
)


def _default_output(
    sequence: Path, first_frame: int, last_frame: int
) -> Path:
    return (
        REPOSITORY_ROOT
        / "logs"
        / "hm3d"
        / "one_second_sam2"
        / sequence.name
        / f"frames_{first_frame:06d}_{last_frame:06d}"
    )


def _rgb_uint8(values: np.ndarray) -> np.ndarray:
    rgb = np.asarray(values)
    if np.issubdtype(rgb.dtype, np.floating) and rgb.size:
        if float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _short_track(track_id: str) -> str:
    try:
        return f"T{int(track_id.rsplit('-', 1)[-1])}"
    except ValueError:
        return track_id[:10]


def _render_overlay(
    rgb: np.ndarray,
    proposals,
    tracked_frame,
    *,
    timestamp: float,
    alpha: float,
) -> Image.Image:
    canvas = _rgb_uint8(rgb).astype(np.float64)
    height, width = canvas.shape[:2]
    observations = {
        item.proposal_id: item for item in tracked_frame.observations
    }
    ordered = sorted(proposals, key=lambda item: item.area, reverse=True)
    for proposal in ordered:
        observation = observations[proposal.proposal_id]
        colour = entity_colour(observation.track_id).astype(np.float64)
        canvas[proposal.mask] = (
            (1.0 - alpha) * canvas[proposal.mask] + alpha * colour
        )
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    for proposal in ordered:
        observation = observations[proposal.proposal_id]
        boundary_colour = (
            np.asarray([40, 255, 80], dtype=np.uint8)
            if observation.source_proposal_id is not None
            else np.asarray([255, 55, 55], dtype=np.uint8)
        )
        canvas[mask_boundary(proposal.mask)] = boundary_colour

    header_height = 58
    panel = Image.new("RGB", (width, height + header_height), (15, 15, 15))
    panel.paste(Image.fromarray(canvas), (0, header_height))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (6, 5),
        f"HM3D dense SAM2 | frame {tracked_frame.frame_id} | "
        f"t={timestamp:.3f}s | masks={len(proposals)}",
        fill=(255, 255, 255),
    )
    draw.text(
        (6, 26),
        f"linked={tracked_frame.linked_count} | "
        f"new tracks={tracked_frame.new_track_count} | "
        "green=adjacent IoU link, red=new track",
        fill=(220, 220, 220),
    )
    for proposal in ordered:
        coordinates = np.argwhere(proposal.mask)
        if len(coordinates) == 0:
            continue
        row, column = np.median(coordinates, axis=0).astype(int)
        observation = observations[proposal.proposal_id]
        label = _short_track(observation.track_id)
        x = int(np.clip(column, 0, width - 1))
        y = int(
            np.clip(
                row + header_height,
                header_height,
                height + header_height - 1,
            )
        )
        box = draw.textbbox((x, y), label)
        draw.rectangle(box, fill=(0, 0, 0))
        draw.text((x, y), label, fill=(255, 255, 255))
    return panel


def _write_contact_sheet(
    paths: list[Path], output: Path, *, columns: int = 5
) -> None:
    tiles: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            tiles.append(source.convert("RGB").copy())
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), (30, 30, 30))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
    sheet.save(output)


def _write_gif(paths: list[Path], output: Path, duration_ms: int) -> None:
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            frames.append(source.convert("RGB").copy())
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--start-frame", type=int)
    start.add_argument("--start-second", type=float)
    parser.add_argument("--duration-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backend", choices=("official", "transformers"), default="official"
    )
    parser.add_argument("--model")
    parser.add_argument("--device", default="0")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=32)
    parser.add_argument("--pred-iou-threshold", type=float, default=0.88)
    parser.add_argument("--stability-score-threshold", type=float, default=0.95)
    parser.add_argument("--min-area-pixels", type=int, default=100)
    parser.add_argument("--min-area-fraction", type=float, default=0.001)
    parser.add_argument("--max-area-fraction", type=float, default=0.8)
    parser.add_argument("--erosion-pixels", type=int, default=1)
    parser.add_argument("--min-component-pixels", type=int, default=50)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.9)
    parser.add_argument("--min-link-iou", type=float, default=0.30)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    meta_path = args.sequence / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"HM3D metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fps = float(meta["fps"])
    total_frames = int(meta["num_frames"])
    frame_ids = select_frame_window(
        total_frames=total_frames,
        fps=fps,
        duration_seconds=args.duration_seconds,
        start_frame=args.start_frame,
        start_second=args.start_second,
    )
    output = args.output or _default_output(
        args.sequence, frame_ids[0], frame_ids[-1]
    )
    frame_output = output / "frames"
    frame_output.mkdir(parents=True, exist_ok=True)

    if args.backend == "official":
        model = args.model or "facebook/sam2-hiera-large"
        generator = SAM2OfficialMaskGenerator(
            model,
            device=args.device,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_threshold=args.pred_iou_threshold,
            stability_score_threshold=args.stability_score_threshold,
        )
    else:
        model = args.model or "facebook/sam2.1-hiera-large"
        generator = SAM2AutomaticMaskGenerator(
            model,
            device=args.device,
            points_per_batch=args.points_per_batch,
            pred_iou_threshold=args.pred_iou_threshold,
            stability_score_threshold=args.stability_score_threshold,
        )
    filter_config = MaskFilterConfig(
        min_score=args.pred_iou_threshold,
        min_area_pixels=args.min_area_pixels,
        min_area_fraction=args.min_area_fraction,
        max_area_fraction=args.max_area_fraction,
        erosion_pixels=args.erosion_pixels,
        min_component_pixels=args.min_component_pixels,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
    )
    tracker = AdjacentMaskTracker(args.min_link_iou)
    frame_entries: list[dict[str, object]] = []
    overlay_paths: list[Path] = []
    link_ious: list[float] = []
    track_lengths: Counter[str] = Counter()
    target_proposal_count = 0
    total_proposals = 0

    for window_index, frame_id in enumerate(frame_ids):
        image_path = args.sequence / f"{frame_id:06d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"HM3D RGB frame not found: {image_path}")
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        raw = list(generator.generate(rgb, frame_id=frame_id))
        proposals = filter_image_mask_proposals(
            raw,
            rgb.shape[:2],
            filter_config,
            frame_id=frame_id,
        )
        tracked = tracker.update(frame_id, proposals)
        observations = {
            item.proposal_id: item for item in tracked.observations
        }
        for observation in tracked.observations:
            track_lengths[observation.track_id] += 1
            if observation.link_iou is not None:
                link_ious.append(observation.link_iou)
        total_proposals += len(proposals)
        if window_index > 0:
            target_proposal_count += len(proposals)

        masks = (
            np.stack([proposal.mask for proposal in proposals], axis=0)
            if proposals
            else np.empty((0, *rgb.shape[:2]), dtype=bool)
        )
        evidence_name = f"frame_{frame_id:06d}_masks.npz"
        np.savez_compressed(
            frame_output / evidence_name,
            masks=masks,
            proposal_ids=np.asarray(
                [proposal.proposal_id for proposal in proposals], dtype=np.str_
            ),
            track_ids=np.asarray(
                [observations[item.proposal_id].track_id for item in proposals],
                dtype=np.str_,
            ),
            scores=np.asarray([proposal.score for proposal in proposals]),
            link_ious=np.asarray(
                [
                    np.nan
                    if observations[item.proposal_id].link_iou is None
                    else observations[item.proposal_id].link_iou
                    for item in proposals
                ],
                dtype=np.float64,
            ),
        )
        timestamp = frame_id / fps
        overlay = _render_overlay(
            rgb,
            proposals,
            tracked,
            timestamp=timestamp,
            alpha=args.overlay_alpha,
        )
        overlay_name = f"frame_{frame_id:06d}_overlay.png"
        overlay_path = frame_output / overlay_name
        overlay.save(overlay_path)
        overlay_paths.append(overlay_path)

        frame_entries.append(
            {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "source_image": str(image_path.resolve()),
                "raw_proposal_count": len(raw),
                "proposal_count": len(proposals),
                "linked_count": tracked.linked_count,
                "new_track_count": tracked.new_track_count,
                "mask_evidence": f"frames/{evidence_name}",
                "overlay": f"frames/{overlay_name}",
                "proposals": [
                    {
                        "proposal_id": proposal.proposal_id,
                        "score": proposal.score,
                        "area": proposal.area,
                        "bounding_box_xyxy": (
                            None
                            if proposal.bounding_box_xyxy is None
                            else proposal.bounding_box_xyxy.tolist()
                        ),
                        **asdict(observations[proposal.proposal_id]),
                    }
                    for proposal in proposals
                ],
            }
        )
        print(
            f"frame {frame_id}: raw={len(raw)} kept={len(proposals)} "
            f"linked={tracked.linked_count} new_tracks={tracked.new_track_count}"
        )

    contact_path = output / "contact_sheet.png"
    _write_contact_sheet(overlay_paths, contact_path)
    gif_path = output / "one_second_tracks.gif"
    if not args.no_gif:
        _write_gif(overlay_paths, gif_path, max(1, round(1000.0 / fps)))

    link_rate = (
        0.0
        if target_proposal_count == 0
        else len(link_ious) / target_proposal_count
    )
    manifest = {
        "format": "fact3r-hm3d-one-second-segmentation",
        "version": 1,
        "source_sequence": str(args.sequence.resolve()),
        "scene": args.sequence.name,
        "fps": fps,
        "requested_duration_seconds": args.duration_seconds,
        "captured_frame_count": len(frame_ids),
        "first_frame_id": frame_ids[0],
        "last_frame_id": frame_ids[-1],
        "backend": args.backend,
        "model": model,
        "filter_config": asdict(filter_config),
        "adjacent_link_iou_threshold": args.min_link_iou,
        "total_proposals": total_proposals,
        "mean_proposals_per_frame": total_proposals / len(frame_ids),
        "track_count": tracker.track_count,
        "multi_frame_track_count": sum(
            length > 1 for length in track_lengths.values()
        ),
        "adjacent_link_count": len(link_ious),
        "adjacent_link_rate": link_rate,
        "median_link_iou": (
            None if not link_ious else float(np.median(link_ious))
        ),
        "track_length_histogram": {
            str(length): count
            for length, count in sorted(Counter(track_lengths.values()).items())
        },
        "contact_sheet": contact_path.name,
        "gif": None if args.no_gif else gif_path.name,
        "frames": frame_entries,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Dense-window totals: frames={len(frame_ids)} "
        f"proposals={total_proposals} tracks={tracker.track_count} "
        f"adjacent_link_rate={link_rate:.1%} "
        f"median_link_iou={manifest['median_link_iou']}"
    )
    print(f"Wrote dense HM3D segmentation diagnostic to {manifest_path}")


if __name__ == "__main__":
    main()
