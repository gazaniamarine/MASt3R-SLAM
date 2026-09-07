#!/usr/bin/env python3
"""Score a resolved goal against a GOAT episode's ground-truth object position.

Turns "does the pointed-at pixel look right" into a number instead of an eyeball
check on the rendered map. GOAT-Bench already carries the exact 3D position of
every named object in a scene (`datasets/goat/.../<scene>.json.gz`'s `goals`),
so a goal request's `winner.centroid_yx` -- the same schema field
`resolve_semantic_goal.py` and `resolve_semantic_goal_vlm.py` both write --
can be compared directly against it, in the map frame `execute_vlnce_return.py`
already uses (`habitat_to_map_xy`: map_x = x, map_y = -z).

    python3 fact3r-map/scripts/score_goal_against_groundtruth.py \
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


def ground_truth_yx(episodes_path: Path, category: str) -> tuple[float, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True, help="scene's goat *.json.gz")
    parser.add_argument("--category", required=True)
    parser.add_argument(
        "--request", type=Path, required=True,
        help="resolve_semantic_goal.py or resolve_semantic_goal_vlm.py output",
    )
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    centroid_yx = request["winner"]["centroid_yx"]
    gt_yx = ground_truth_yx(args.episodes, args.category)
    distance = math.hypot(centroid_yx[0] - gt_yx[0], centroid_yx[1] - gt_yx[1])
    print(f"ground truth (y, x) = ({gt_yx[0]:.2f}, {gt_yx[1]:.2f}) m")
    print(f"resolved     (y, x) = ({centroid_yx[0]:.2f}, {centroid_yx[1]:.2f}) m")
    print(f"distance to ground truth: {distance:.2f} m")


if __name__ == "__main__":
    main()
