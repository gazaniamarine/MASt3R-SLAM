#!/usr/bin/env python3
"""Render RGB, entity overlays, a contact sheet, and an association GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.integrations.mast3r_slam import iter_exported_keyframes  # noqa: E402
from fact3r.proposals.storage import load_proposal_run_manifest  # noqa: E402
from fact3r.visualization.association import (  # noqa: E402
    display_frame_from_manifest,
    join_panels,
    render_association_panel,
    render_rgb_panel,
)


def _parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("mapping must have the form LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("mapping label and path cannot be empty")
    return label.strip(), Path(raw_path)


def _manifest_path(path: Path) -> Path:
    return path / "manifest.json" if path.is_dir() else path


def _load_mapping(
    label: str, path: Path
) -> tuple[str, dict[int, object]]:
    manifest_path = _manifest_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") not in {
        "fact3r-hungarian-baseline",
        "fact3r-balanced-sinkhorn",
        "fact3r-visibility-residual-transport",
    }:
        raise ValueError(f"unsupported mapping manifest {manifest_path}")
    return label, {
        int(entry["frame_id"]): display_frame_from_manifest(entry)
        for entry in payload["frames"]
    }


def _load_masks(
    proposal_directory: Path, run_entry: dict[str, object]
) -> dict[str, np.ndarray]:
    manifest_path = proposal_directory / str(run_entry["manifest"])
    frame = json.loads(manifest_path.read_text(encoding="utf-8"))
    masks: dict[str, np.ndarray] = {}
    for proposal in frame["proposals"]:
        with np.load(
            manifest_path.parent / proposal["file"], allow_pickle=False
        ) as payload:
            masks[str(proposal["proposal_id"])] = np.array(
                payload["mask"], dtype=bool, copy=True
            )
    return masks


def _resize_to_width(image: Image.Image, maximum_width: int) -> Image.Image:
    if image.width <= maximum_width:
        return image.copy()
    height = max(1, round(image.height * maximum_width / image.width))
    return image.resize(
        (maximum_width, height), Image.Resampling.LANCZOS
    )


def _write_contact_sheet(
    frame_paths: list[Path],
    output: Path,
    *,
    maximum_frames: int,
    columns: int,
    tile_width: int,
) -> None:
    count = min(maximum_frames, len(frame_paths))
    indices = np.linspace(0, len(frame_paths) - 1, count, dtype=np.int64)
    tiles = []
    for index in indices:
        with Image.open(frame_paths[int(index)]) as source:
            tiles.append(_resize_to_width(source.convert("RGB"), tile_width))
    tile_height = max(tile.height for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        (30, 30, 30),
    )
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(tile, (x, y))
    sheet.save(output)


def _write_gif(
    frame_paths: list[Path],
    output: Path,
    *,
    width: int,
    duration_ms: int,
) -> None:
    frames: list[Image.Image] = []
    for path in frame_paths:
        with Image.open(path) as source:
            frames.append(_resize_to_width(source.convert("RGB"), width))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=_parse_mapping,
        action="append",
        required=True,
        help="repeatable LABEL=PATH mapping manifest",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--contact-frames", type=int, default=12)
    parser.add_argument("--contact-columns", type=int, default=2)
    parser.add_argument("--contact-tile-width", type=int, default=1200)
    parser.add_argument("--gif-width", type=int, default=1200)
    parser.add_argument("--gif-duration-ms", type=int, default=350)
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.contact_frames <= 0 or args.contact_columns <= 0:
        raise ValueError("contact sheet dimensions must be positive")

    proposal_run = load_proposal_run_manifest(args.proposals)
    proposal_entries = {
        int(entry["frame_id"]): entry for entry in proposal_run["frames"]
    }
    keyframe_images = {
        keyframe.frame_id: np.array(keyframe.rgb, copy=True)
        for keyframe in iter_exported_keyframes(args.keyframes)
    }
    mappings = [_load_mapping(label, path) for label, path in args.mapping]
    frame_ids = [
        frame_id
        for frame_id in proposal_entries
        if frame_id in keyframe_images
        and all(frame_id in frames for _, frames in mappings)
    ][:: args.stride]
    if args.max_frames is not None:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise ValueError("no common keyframe, proposal, and mapping frames found")

    args.output.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    for frame_id in frame_ids:
        rgb = keyframe_images[frame_id]
        masks = _load_masks(args.proposals, proposal_entries[frame_id])
        panels = [render_rgb_panel(rgb, frame_id=frame_id)]
        panels.extend(
            render_association_panel(
                rgb,
                masks,
                mapping_frames[frame_id],
                title=label,
                alpha=args.alpha,
            )
            for label, mapping_frames in mappings
        )
        montage = join_panels(tuple(panels))
        frame_path = args.output / f"frame_{frame_id:06d}.png"
        montage.save(frame_path)
        frame_paths.append(frame_path)
        print(f"wrote {frame_path}")

    contact_path = args.output / "association_contact_sheet.png"
    _write_contact_sheet(
        frame_paths,
        contact_path,
        maximum_frames=args.contact_frames,
        columns=args.contact_columns,
        tile_width=args.contact_tile_width,
    )
    print(f"wrote {contact_path}")
    if not args.no_gif:
        gif_path = args.output / "association.gif"
        _write_gif(
            frame_paths,
            gif_path,
            width=args.gif_width,
            duration_ms=args.gif_duration_ms,
        )
        print(f"wrote {gif_path}")


if __name__ == "__main__":
    main()
