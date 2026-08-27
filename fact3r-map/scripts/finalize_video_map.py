#!/usr/bin/env python3
"""Validate pipeline artifacts and write a portable Fact3R map manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.map_bundle import create_video_map_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--tracklets", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    path = create_video_map_manifest(
        output=args.output,
        video=args.video,
        map_name=args.map_name,
        sequence_name=args.sequence_name,
        keyframes=args.keyframes,
        proposals=args.proposals,
        tracklets=args.tracklets,
        mapping=args.mapping,
        observations=args.observations,
        calibration=args.calibration,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"Finalized map with {payload['frame_count']} keyframes, "
        f"{payload['entity_count']} entities and "
        f"{payload['observation_count']} indexed observations "
        f"({payload['track_only_observation_count']} retained as 2D tracks) "
        f"at {path}"
    )


if __name__ == "__main__":
    main()
