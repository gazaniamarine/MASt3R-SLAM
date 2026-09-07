#!/usr/bin/env python3
"""Overlay the resolved goal, navigability projection, and A* route onto
Pipeline B's own rendered marker map -- one picture showing what Qwen was
shown, which marker it picked, where the goal actually landed after the
navigability snap, and the route to it.

Builds on top of resolve_semantic_goal_vlm.py's own rendered_map PNG (every
SigLIP-shortlisted marker Qwen saw) rather than re-deriving it, since the
shortlist itself is not persisted anywhere except baked into that image's
pixels -- only the single winning candidate survives into the goal-request
JSON (see resolve_semantic_goal_vlm.py's `"candidates": [candidate]`).

    python3 fact3r-map/scripts/visualize_pipeline_result.py \\
        --run logs/rover/pipeline/mpl_20260826 --query "the countertop"

Runs against whichever of locate/goal/plan stages have already produced
output -- locate alone still draws the winning marker; add goal for the
navigable snap point; add plan for the A* route. `plans/map_dstt.npz` is
NOT namespaced per query (run_rover_pipeline_vlm.py always writes the same
filename), so this script cross-checks its `goal` array against this run's
own goal.json before trusting it belongs to the query being visualized, and
refuses to draw a mismatched route rather than silently showing the wrong
one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.semantic_goal import world_xy_to_cell  # noqa: E402
from fact3r.vlm_nav.bev_pointing import _cell_to_pixel  # noqa: E402


def _slug(value: str) -> str:
    """Same slug rule as run_rover_pipeline_vlm.py; duplicated rather than
    imported since that file is a standalone CLI script, not a module."""
    return "-".join(
        part for part in "".join(
            c.lower() if c.isalnum() else " " for c in value
        ).split() if part
    ) or "query"


def _draw_ring(draw: ImageDraw.ImageDraw, x: int, y: int, colour, *, radius: int = 18, width: int = 4) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=width)


def _draw_cross(draw: ImageDraw.ImageDraw, x: int, y: int, colour, *, size: int = 10, width: int = 3) -> None:
    draw.line((x - size, y, x + size, y), fill=colour, width=width)
    draw.line((x, y - size, x, y + size), fill=colour, width=width)


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, colour,
                 *, canvas_width: int, offset: int = 14) -> None:
    """Text anchored right of (x, y), flipped to the left if it would run
    off the canvas -- a label cut off at the edge is easy to miss entirely
    on a small map render."""
    width = draw.textbbox((0, 0), text)[2]
    tx = x + offset if x + offset + width <= canvas_width else x - offset - width
    draw.text((tx, y), text, fill=colour)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True,
                         help="run directory, e.g. logs/rover/pipeline/mpl_20260826")
    parser.add_argument("--query", required=True,
                         help="same query text run_rover_pipeline_vlm.py was given")
    parser.add_argument("--plan-index", type=int, default=0)
    parser.add_argument("--goal-match-tolerance-m", type=float, default=0.05,
                         help="max distance between goal.json's goal_yx and the plan "
                              "npz's own goal before the route is treated as stale")
    parser.add_argument("--scale", type=float, default=3.0,
                         help="upscale factor applied to the rendered map before "
                              "drawing overlays -- the base BEV render is often only "
                              "a couple hundred pixels across, too small to read "
                              "markers/text at native size")
    parser.add_argument("--output", type=Path,
                         help="default: <run>/locate/<slug>_pipeline_overview.png")
    args = parser.parse_args()

    slug = _slug(args.query)
    locate_dir = args.run / "locate"
    request_path = locate_dir / f"{slug}_goal_request.json"
    goal_path = locate_dir / f"{slug}_goal.json"
    plan_path = args.run / "plans" / "map_dstt.npz"
    output = args.output or locate_dir / f"{slug}_pipeline_overview.png"

    if not request_path.exists():
        raise SystemExit(f"no locate output for {args.query!r}: {request_path} does not exist")
    request = json.loads(request_path.read_text())
    origin_xy = request["origin_xy"]
    resolution = float(request["resolution_metres"])
    winner = request["winner"]

    rendered_map = Path(request["rendered_map"])
    if not rendered_map.is_file():
        raise SystemExit(f"rendered map referenced by the request no longer exists: {rendered_map}")
    image = Image.open(rendered_map).convert("RGB")
    orig_height = image.height
    scale = args.scale
    if scale != 1.0:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
        )
    draw = ImageDraw.Draw(image)

    def to_pixel(row: float, col: float) -> tuple[int, int]:
        x, y = _cell_to_pixel(row, col, orig_height)
        return round(x * scale), round(y * scale)

    def world_to_pixel(y: float, x: float) -> tuple[int, int]:
        row, col = world_xy_to_cell(x, y, origin_xy, resolution)
        return to_pixel(float(row), float(col))

    summary = {
        "query": args.query,
        "vlm": request["vlm"],
        "winner_group_id": winner["group_id"],
        "winner_score": winner.get("score"),
        "winner_cell_count": winner.get("cell_count"),
    }

    # Winning entity: a thick cyan ring around wherever Qwen's answer
    # resolved to (works whether it named a marker or pointed freeform --
    # resolve_pointing already did that resolution upstream, in
    # resolve_semantic_goal_vlm.py).
    wx, wy = to_pixel(*winner["centroid_cell_rc"])
    _draw_ring(draw, wx, wy, (0, 220, 220), radius=round(18 * scale), width=max(2, round(4 * scale)))
    _draw_label(draw, wx, wy - 8, f"WINNER: {winner['group_id']}", (0, 220, 220),
                canvas_width=image.width, offset=22)

    goal_yx = None
    if goal_path.exists():
        goal = json.loads(goal_path.read_text())
        summary.update({
            "goal_yx": goal.get("goal_yx"),
            "projection_distance_m": goal.get("projection_distance_m"),
            "distance_to_entity_m": goal.get("distance_to_entity_m"),
            "in_largest_component": goal.get("in_largest_component"),
            "goal_clearance_m": goal.get("goal_clearance_m"),
        })
        goal_yx = goal["goal_yx"]
        gpx, gpy = world_to_pixel(goal_yx[0], goal_yx[1])
        _draw_cross(draw, gpx, gpy, (0, 255, 0), size=round(12 * scale), width=max(2, round(4 * scale)))
        _draw_label(draw, gpx, gpy + 10, "GOAL (navigable)", (0, 255, 0),
                    canvas_width=image.width, offset=14)
        if goal.get("start_yx"):
            sy, sx = goal["start_yx"]
            spx, spy = world_to_pixel(sy, sx)
            _draw_cross(draw, spx, spy, (255, 165, 0), size=round(10 * scale), width=max(2, round(3 * scale)))
            _draw_label(draw, spx, spy - 22, "START", (255, 165, 0),
                        canvas_width=image.width, offset=12)
    else:
        print(f"[note] {goal_path} not found -- run the 'goal' stage to see the "
              f"navigability-snapped point")

    if plan_path.exists() and goal_yx is not None:
        # Position-matching alone is not proof this plan was actually computed
        # for THIS goal -- two different queries can legitimately resolve to
        # nearly the same point (e.g. both naming the same real object), and
        # a stale plan from an earlier, unrelated run would then pass a
        # position check while still predating this query's own goal.json
        # entirely. Caught exactly this way once already: plans/map_dstt.npz
        # left over from Pipeline A's original run coincidentally matched a
        # later Pipeline B goal to within 5cm. mtime is the real signal.
        stale_by_time = plan_path.stat().st_mtime < goal_path.stat().st_mtime
        with np.load(plan_path) as payload:
            plan_goal_yx = np.asarray(payload["goal"])[args.plan_index]
            mismatch = np.hypot(plan_goal_yx[0] - goal_yx[0], plan_goal_yx[1] - goal_yx[1])
            if stale_by_time or mismatch > args.goal_match_tolerance_m:
                print(
                    f"[warning] plans/map_dstt.npz predates {slug}_goal.json "
                    f"(written before this query was resolved)" if stale_by_time else
                    f"[warning] plans/map_dstt.npz's goal ({plan_goal_yx[0]:.2f}, "
                    f"{plan_goal_yx[1]:.2f}) is {mismatch:.2f}m from this query's own "
                    f"goal_yx ({goal_yx[0]:.2f}, {goal_yx[1]:.2f})"
                )
                print(
                    "  -> plans/map_dstt.npz isn't namespaced per query, so it likely "
                    f"belongs to a different query's 'plan' run. Re-run 'plan' for "
                    f"{args.query!r} to draw its real route; skipping the route "
                    f"overlay for now."
                )
            else:
                centerline_yx = np.asarray(payload["centerline"])[args.plan_index]
                est_collision_frac = float(np.asarray(payload["est_collision_frac"])[args.plan_index])
                points = [world_to_pixel(y, x) for y, x in centerline_yx]
                draw.line(points, fill=(255, 0, 255), width=max(2, round(3 * scale)))
                summary["route_waypoints"] = len(points)
                summary["est_collision_frac"] = est_collision_frac
    elif not plan_path.exists():
        print(f"[note] {plan_path} not found -- run the 'plan' stage to see the A* route")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"query:            {args.query!r}")
    print(f"Qwen picked:      marker={request['vlm']['marker']} reason={request['vlm']['reason']!r}")
    score = winner.get("score")
    print(f"winner:           {winner['group_id']} ({winner.get('cell_count')} cells"
          + (f", score={score:.4f})" if score is not None else ")"))
    if goal_yx is not None:
        print(f"goal (y, x):      ({goal_yx[0]:.2f}, {goal_yx[1]:.2f}) m")
    if "route_waypoints" in summary:
        print(f"route:            {summary['route_waypoints']} waypoints, "
              f"est_collision_frac={summary['est_collision_frac']:.3f}")
    print(f"\nvisualization:    {output}")
    print(f"summary:          {summary_path}")


if __name__ == "__main__":
    main()
