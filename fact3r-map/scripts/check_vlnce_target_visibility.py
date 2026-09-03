#!/usr/bin/env python3
"""Report which rendered tours actually showed their return target.

Run after `render_vlnce_tour.py --semantic`. A tour whose target was never
observed cannot produce a meaningful return result: a failure there says nothing
about the map. Drop those tours before mapping rather than explaining them away
afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.experiments.vlnce_visibility import (  # noqa: E402
    InstanceVisibility,
    score_target,
    summarise,
)


def _tour_directories(root: Path) -> list[Path]:
    if (root / "meta.json").is_file():
        return [root]
    return sorted(path for path in root.iterdir() if (path / "meta.json").is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--renders",
        required=True,
        help="A rendered tour directory, or the root holding several.",
    )
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--min-pixel-fraction", type=float, default=0.005)
    parser.add_argument("--output", default=None, help="Write the report as JSON.")
    args = parser.parse_args()

    root = Path(args.renders)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 1
    directories = _tour_directories(root)
    if not directories:
        print(f"no rendered tours under {root}", file=sys.stderr)
        return 1

    results = []
    report = []
    skipped = 0
    trivial = []
    for directory in directories:
        meta = json.loads((directory / "meta.json").read_text())
        optimal = float(meta.get("return_optimal_geodesic_m", float("nan")))
        success_distance = float(meta.get("success_distance_m", 3.0))
        if optimal == optimal and optimal < success_distance:
            # Standing still already succeeds, so nothing this tour reports
            # about return navigation means anything.
            trivial.append((directory.name, optimal))
            print(
                f"  [DROP] {directory.name}: return target only {optimal:.1f} m "
                f"away, inside the {success_distance:.1f} m success radius"
            )
            continue
        visibility_file = directory / "semantic_visibility.json"
        if not visibility_file.is_file():
            print(f"  {directory.name}: no semantic audit (re-render with --semantic)")
            skipped += 1
            continue
        payload = json.loads(visibility_file.read_text())
        instances = [
            InstanceVisibility.from_json(record) for record in payload["instances"]
        ]
        result = score_target(
            instances,
            meta["return_query"],
            meta["return_position"],
            radius=args.radius,
            min_frames=args.min_frames,
            min_pixel_fraction=args.min_pixel_fraction,
        )
        results.append(result)
        report.append({"tour": directory.name, **result.to_json()})
        flag = "OK " if result.usable else "DROP"
        print(f"  [{flag}] {directory.name}: {result.query!r} -> {result.verdict}")
        print(f"         {result.reason}")

    if not results:
        if trivial:
            print(
                "\nno usable tour: all %d were dropped as trivially successful. "
                "Re-render with a larger --min-return-distance." % len(trivial),
                file=sys.stderr,
            )
        else:
            print("no tour carried a semantic audit", file=sys.stderr)
        return 1

    summary = summarise(results)
    print(
        f"\n{summary['usable']}/{summary['tours']} tours have an observed return "
        f"target: {summary['by_verdict']}"
    )
    if skipped:
        print(f"({skipped} tour(s) skipped for lack of a semantic audit)")
    if trivial:
        print(
            f"({len(trivial)} tour(s) dropped as trivially successful: "
            + ", ".join(name for name, _ in trivial) + ")"
        )
    if summary["usable"] < summary["tours"]:
        print(
            "Drop the non-usable tours, or lower --min-frames / "
            "--min-pixel-fraction only with a reason."
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "format": "fact3r-vlnce-visibility-report",
                    "version": 1,
                    "radius": args.radius,
                    "min_frames": args.min_frames,
                    "min_pixel_fraction": args.min_pixel_fraction,
                    "summary": summary,
                    "trivial_returns": [
                        {"tour": name, "optimal_geodesic_m": distance}
                        for name, distance in trivial
                    ],
                    "tours": report,
                },
                indent=2,
            )
        )
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
