#!/usr/bin/env python3
"""Render every retained SAM2 proposal from one frame as a labelled contact sheet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.integrations.mast3r_slam import iter_exported_keyframes  # noqa: E402
from fact3r.visualization.association import mask_boundary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    if args.columns <= 0:
        raise ValueError("columns must be positive")
    keyframe = next(
        (
            item
            for item in iter_exported_keyframes(args.keyframes)
            if item.frame_id == args.frame_id
        ),
        None,
    )
    if keyframe is None:
        raise ValueError(f"frame {args.frame_id} is absent from the keyframes")
    run = json.loads(
        (args.proposals / "manifest.json").read_text(encoding="utf-8")
    )
    run_entry = next(
        (item for item in run["frames"] if int(item["frame_id"]) == args.frame_id),
        None,
    )
    if run_entry is None:
        raise ValueError(f"frame {args.frame_id} is absent from the proposals")
    frame_manifest_path = args.proposals / str(run_entry["manifest"])
    frame = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    rgb = np.asarray(keyframe.rgb, dtype=np.uint8)
    tile_width = 360
    header_height = 54
    tile_height = round(tile_width * rgb.shape[0] / rgb.shape[1]) + header_height
    tiles: list[Image.Image] = []
    for proposal_index, proposal in enumerate(frame["proposals"]):
        with np.load(
            frame_manifest_path.parent / str(proposal["file"]),
            allow_pickle=False,
        ) as payload:
            mask = np.asarray(payload["mask"], dtype=bool)
        canvas = rgb.astype(np.float32)
        green = np.asarray([20.0, 255.0, 70.0])
        canvas[mask] = 0.55 * canvas[mask] + 0.45 * green
        canvas[mask_boundary(mask)] = green
        image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        image.thumbnail((tile_width, tile_height - header_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height), (18, 18, 18))
        tile.paste(image, ((tile_width - image.width) // 2, header_height))
        draw = ImageDraw.Draw(tile)
        draw.text(
            (6, 5),
            f"#{proposal_index}  {proposal['proposal_id']}",
            fill=(255, 255, 255),
        )
        draw.text(
            (6, 27),
            f"score={float(proposal['score']):.3f}  area={proposal['mask_area']}",
            fill=(185, 230, 195),
        )
        tiles.append(tile)
    if not tiles:
        raise ValueError(f"frame {args.frame_id} has no retained proposals")
    rows = math.ceil(len(tiles) / args.columns)
    sheet = Image.new(
        "RGB", (args.columns * tile_width, rows * tile_height), (30, 30, 30)
    )
    for index, tile in enumerate(tiles):
        sheet.paste(
            tile,
            ((index % args.columns) * tile_width, (index // args.columns) * tile_height),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=95)
    print(f"Rendered {len(tiles)} individual masks to {args.output}")


if __name__ == "__main__":
    main()
