#!/usr/bin/env python3
"""Draw ground truth and the resolved answer on the rendered pointing map.

Companion to `score_goal_against_groundtruth.py`: turns "is the distance
number good" into something you can just look at instead of read. Opens the
map image `resolve_semantic_goal_vlm.py` already rendered (its `--request`
JSON's `rendered_map` field, which still has every shortlisted marker drawn
on it), adds a green cross at the ground-truth position from the same GOAT
episode file, a red ring around wherever the resolved answer actually landed,
and a line between them labelled with the distance.

    python3 fact3r-map/scripts/visualize_goal_against_groundtruth.py \
        --episodes datasets/goat/data/datasets/goat_bench/hm3d/v1/val_unseen/content/y9hTuugGdiq.json.gz \
        --category freezer \
        --request /tmp/goal_request_vlm_freezer.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.semantic_goal import world_xy_to_cell  # noqa: E402
from fact3r.vlm_nav.bev_pointing import _cell_to_pixel  # noqa: E402


def _ground_truth_yx(episodes_path: Path, category: str) -> tuple[float, float]:
    with gzip.open(episodes_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    key = next(
        (k for k in data["goals"] if k.rsplit("_", 1)[-1] == category), None
    )
    if key is None:
        raise SystemExit(
            f"no goal category {category!r} in {episodes_path}; have "
            + ", ".join(sorted(k.rsplit('_', 1)[-1] for k in data["goals"]))
        )
    x, _, z = data["goals"][key][0]["position"]
    return -float(z), float(x)


def _cross(draw: ImageDraw.ImageDraw, x: int, y: int, *, colour, radius: int = 9) -> None:
    draw.line((x - radius, y, x + radius, y), fill=colour, width=3)
    draw.line((x, y - radius, x, y + radius), fill=colour, width=3)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colour)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True, help="scene's goat *.json.gz")
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--request", type=Path, required=True,
        help="resolve_semantic_goal_vlm.py (or resolve_semantic_goal.py) output",
    )
    parser.add_argument("--output", type=Path, help="default: <rendered_map>.compare.png")
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    origin_xy = request["origin_xy"]
    resolution = float(request["resolution_metres"])
    height = int(request["grid_shape"][0])

    rendered_map = request.get("rendered_map")
    if rendered_map is None:
        raise SystemExit(
            f"{args.request} has no 'rendered_map' field -- this looks like "
            "resolve_semantic_goal.py output (Pipeline A), which has no "
            "annotated map image to draw on"
        )
    map_image_path = Path(rendered_map)

    gt_y, gt_x = _ground_truth_yx(args.episodes, args.category)
    gt_row, gt_col = world_xy_to_cell(gt_x, gt_y, origin_xy, resolution)
    gt_px, gt_py = _cell_to_pixel(float(gt_row), float(gt_col), height)

    winner_y, winner_x = request["winner"]["centroid_yx"]
    win_row, win_col = world_xy_to_cell(winner_x, winner_y, origin_xy, resolution)
    win_px, win_py = _cell_to_pixel(float(win_row), float(win_col), height)

    distance = math.hypot(winner_y - gt_y, winner_x - gt_x)

    image = Image.open(map_image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    _cross(draw, gt_px, gt_py, colour=(20, 200, 60))
    draw.ellipse((win_px - 14, win_py - 14, win_px + 14, win_py + 14), outline=(230, 30, 30), width=3)
    draw.line((win_px, win_py, gt_px, gt_py), fill=(230, 30, 30), width=2)
    mid_x, mid_y = (win_px + gt_px) // 2, (win_py + gt_py) // 2
    draw.text((mid_x + 6, mid_y - 6), f"{distance:.2f} m", fill=(230, 30, 30))

    verdict = "close" if distance < 2.0 else "far"
    caption = (
        f'"{args.category}": {distance:.2f} m ({verdict})  '
        "green cross = ground truth   red ring = resolved answer"
    )
    text_bbox = draw.textbbox((10, 10), caption)
    draw.rectangle(
        (text_bbox[0] - 4, text_bbox[1] - 3, text_bbox[2] + 4, text_bbox[3] + 3),
        fill=(255, 255, 255),
    )
    draw.text((10, 10), caption, fill=(0, 0, 0))

    output = args.output or map_image_path.with_suffix(".compare.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"ground truth (y, x) = ({gt_y:.2f}, {gt_x:.2f}) m")
    print(f"resolved     (y, x) = ({winner_y:.2f}, {winner_x:.2f}) m")
    print(f"distance: {distance:.2f} m ({verdict})")
    print(f"comparison image: {output}")


if __name__ == "__main__":
    main()
