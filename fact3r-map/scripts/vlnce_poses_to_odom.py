#!/usr/bin/env python3
"""Write the planar odom_*.csv that build_depth_semantic_bev.py expects.

Reads the habitat ground-truth poses a rendered tour carries and converts them
to the rover odometry format, so the metric depth + semantic BEV path can run on
a VLN-CE tour without a real wheel encoder.

Habitat poses are exact, so this is odometry without drift. That is a stronger
input than the rover ever gets; say so when comparing the two.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.experiments.habitat_odometry import (  # noqa: E402
    poses_to_odometry,
    read_tum_poses,
    write_odometry_csv,
)


def _default_source(path: Path) -> Path:
    """Accept either a render/frames directory or the pose file itself."""

    if path.is_dir():
        return path / "groundtruth.txt"
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poses",
        required=True,
        help="A rendered tour directory, a frames export, or a groundtruth.txt.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination CSV; defaults to odom_<name>.csv beside the poses.",
    )
    args = parser.parse_args()

    source = _default_source(Path(args.poses))
    if not source.is_file():
        print(f"pose file not found: {source}", file=sys.stderr)
        return 1

    rows = poses_to_odometry(read_tum_poses(source))

    if args.output:
        output = Path(args.output)
    else:
        # _find_odometry() globs for odom_*.csv, so the prefix matters.
        output = source.parent / f"odom_{source.parent.name}.csv"
    write_odometry_csv(rows, output)

    span = rows[-1].t - rows[0].t
    travelled = sum(
        ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5 for a, b in zip(rows[:-1], rows[1:])
    )
    print(
        f"wrote {len(rows)} odometry rows over {span:.1f} s "
        f"({travelled:.1f} m travelled) -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
