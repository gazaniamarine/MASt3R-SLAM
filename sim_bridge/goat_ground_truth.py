#!/usr/bin/env python3
"""Look up GOAT-Bench's own annotated ground-truth object positions for one
scene + category, straight from the episode file -- no dependency on
anything Fact3R resolved. Used to check "did we actually find/reach the
real object" independent of whatever our own pipeline believes its goal is.

    python3 scripts/goat_ground_truth.py \\
        --episode datasets/goat/data/datasets/goat_bench/hm3d/v1/val_unseen/content/y9hTuugGdiq.json.gz \\
        --category pillow

GOAT scores an `object` goal as satisfied by reaching ANY instance of the
category (see fact3r-map's own goat-bench-fits-fact3r notes), so this prints
every instance's position -- a caller wanting one number should score
distance-to-nearest, not pick one instance arbitrarily.
"""

from __future__ import annotations

import argparse
import gzip
import json


def load_positions(episode_path: str, category: str) -> list[list[float]]:
    data = json.load(gzip.open(episode_path))
    goals = data["goals"]
    matches = [k for k in goals if k.rsplit("_", 1)[-1] == category or k.endswith(f"_{category}")]
    if not matches:
        available = sorted(k.rsplit("_", 1)[-1] for k in goals)
        raise SystemExit(f'category {category!r} not annotated in {episode_path}; available: {available}')
    key = matches[0]
    return [instance["position"] for instance in goals[key]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", required=True)
    p.add_argument("--category", required=True)
    args = p.parse_args()

    positions = load_positions(args.episode, args.category)
    print(f"{len(positions)} ground-truth instance(s) of {args.category!r}:")
    for pos in positions:
        print(f"  {pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}")


if __name__ == "__main__":
    main()
