#!/usr/bin/env python3
"""Chain R2R-CE episodes into long outbound tours with a named return target.

Runs without Matterport3D: chaining and landmark mining need only the episode
JSON. The emitted `tours.json` is the plain-JSON handoff to the habitat-sim
renderer, which runs under a different Python version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.experiments.vlnce import (  # noqa: E402
    build_tours,
    extract_stop_landmark,
    group_by_scene,
    load_split,
)

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2] / "datasets" / "vlnce" / "R2R_VLNCE_v1-3"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--num-legs", type=int, default=3)
    parser.add_argument(
        "--link-tolerance",
        type=float,
        default=2.0,
        help="Metres allowed between one leg's goal and the next leg's start.",
    )
    parser.add_argument("--max-per-scene", type=int, default=1)
    parser.add_argument("--scene", default=None, help="Restrict to one MP3D house.")
    parser.add_argument("--output", default=None, help="Where to write tours.json.")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Report chaining and landmark coverage without writing tours.",
    )
    args = parser.parse_args()

    episodes = load_split(args.dataset, args.split)
    if args.scene:
        episodes = [e for e in episodes if e.scene == args.scene]
        if not episodes:
            print(f"no episodes for scene {args.scene} in {args.split}", file=sys.stderr)
            return 1

    grouped = group_by_scene(episodes)
    landmarks = sum(1 for e in episodes if e.stop_landmark)
    print(f"{args.split}: {len(episodes)} episodes across {len(grouped)} scenes")
    print(
        f"  stop-landmark coverage: {landmarks}/{len(episodes)} "
        f"({100.0 * landmarks / len(episodes):.1f}%)"
    )

    tours = build_tours(
        episodes,
        num_legs=args.num_legs,
        link_tolerance=args.link_tolerance,
        max_per_scene=args.max_per_scene,
    )
    if not tours:
        print("no tour could be chained at this tolerance", file=sys.stderr)
        return 1

    lengths = [tour.outbound_length for tour in tours]
    print(
        f"  chained {len(tours)} tours of {args.num_legs} legs; "
        f"outbound length mean {sum(lengths) / len(lengths):.1f} m, "
        f"max {max(lengths):.1f} m"
    )
    for tour in tours:
        legs = " -> ".join(str(leg.episode.episode_id) for leg in tour.legs)
        print(
            f"    {tour.scene:14s} legs {legs:20s} "
            f"{tour.outbound_length:5.1f} m  return to {tour.return_query!r} "
            f"(leg {tour.return_leg_index})"
        )

    if args.stats_only:
        return 0

    output = Path(args.output) if args.output else Path("logs/vlnce") / args.split / "tours.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "fact3r-vlnce-tours",
        "version": 1,
        "split": args.split,
        "dataset": str(args.dataset),
        "num_legs": args.num_legs,
        "link_tolerance": args.link_tolerance,
        "link_metric": "euclidean",
        "tours": [tour.to_json() for tour in tours],
    }
    output.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {output}")
    print(
        "Links are straight-line here; the renderer re-checks each one against "
        "the navmesh and drops tours that cross a wall."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
