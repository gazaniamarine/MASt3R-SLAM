#!/usr/bin/env python3
"""Turn a located semantic entity into an endpoint the planner will accept.

Second half of stage 7. `resolve_semantic_goal.py` scored the query and handed
over a footprint and its weighted centroid; this decides where the robot may
actually stand, using the same `HM3DMap` clearance field and the same
`planner.navigable` / `largest_component` definitions the planner itself uses.
Reimplementing either would let the goal be legal by one definition and illegal
by the one that matters.

Two things make the centroid untrustworthy on its own:

* an entity's own footprint is occupied by construction, so its centre is never
  a place to drive to; and
* a U-shaped or split entity -- a desk seen from both ends, a wall-length
  bench -- puts its centroid in the wall between its parts, or in unobserved
  space, and the result still looks like a perfectly ordinary coordinate.

So the centroid is only ever a seed for a search over cells that are navigable
*and* in the requested component. When there is no such cell the script says
which of those two conditions failed and stops, rather than handing the planner
a point it will silently snap somewhere else.

Runs in the planner's environment: `HM3DMap` imports scipy, which the
segmentation environment does not have.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_PLANNER_ROOT = REPOSITORY_ROOT / "thirdparty" / "safediffuser"
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.semantic_goal import nearest_cell_in  # noqa: E402


def _load_planner(root: Path):
    if not (root / "diffuser" / "hm3d" / "map.py").exists():
        raise SystemExit(
            f"--planner-root {root} does not look like the SafeDiffuser_STT "
            "checkout (no diffuser/hm3d/map.py)"
        )
    sys.path.insert(0, str(root))
    from diffuser.hm3d import planner
    from diffuser.hm3d.map import HM3DMap

    return HM3DMap, planner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True,
                        help="output of resolve_semantic_goal.py")
    parser.add_argument("--grid", type=Path, required=True, help="the *.npy grid")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner-root", type=Path, default=DEFAULT_PLANNER_ROOT)
    # These have to match the settings the plan is later drawn with: clearance,
    # and therefore what counts as navigable, is a function of all three.
    parser.add_argument("--robot-radius", type=float, default=0.20)
    parser.add_argument("--unknown-slack", type=float, default=0.20)
    parser.add_argument("--no-exclude-exterior", dest="exclude_exterior",
                        action="store_false")
    parser.set_defaults(exclude_exterior=True)
    parser.add_argument(
        "--start", type=float, nargs=2, metavar=("Y", "X"),
        help="world (y, x) the robot starts from. The goal is then required to "
        "be reachable from it; without this the largest navigable component is "
        "used instead.",
    )
    parser.add_argument(
        "--start-from-track", action="store_true",
        help="use the last rover pose recorded by the fuse stage as the start",
    )
    parser.add_argument(
        "--max-projection", type=float, default=2.0,
        help="metres the goal may be moved off the entity before the match is "
        "treated as too far away to be about that entity at all",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=0,
        help="how many ranked entities to try before giving up (0: all of "
        "them). The ranker routinely returns several entities on exactly the "
        "same score -- one object that UOT never merged across frames -- and "
        "which of them is reachable is not something the ranker can see. Set 1 "
        "to demand that the top-ranked entity itself be reachable.",
    )
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    if request.get("format") != "fact3r-semantic-goal-request":
        raise SystemExit(f"not a goal request: {args.request}")
    candidates_in = request.get("candidates") or [request["winner"]]
    if args.max_candidates > 0:
        candidates_in = candidates_in[: args.max_candidates]

    HM3DMap, planner = _load_planner(args.planner_root)
    hm3d_map = HM3DMap.load(
        args.grid,
        robot_radius=args.robot_radius,
        unknown_slack=args.unknown_slack,
        exclude_exterior=args.exclude_exterior,
    )
    print(hm3d_map.summary())
    if (hm3d_map.n_rows, hm3d_map.n_cols) != tuple(request["grid_shape"]):
        raise SystemExit(
            f"grid {args.grid} is {hm3d_map.n_rows}x{hm3d_map.n_cols} but the "
            f"goal request was resolved on {request['grid_shape']} -- these are "
            "not the same map"
        )

    navigable = planner.navigable(hm3d_map)
    if not navigable.any():
        raise SystemExit(
            f"no cell on this grid has positive clearance at robot_radius "
            f"{args.robot_radius} m and unknown_slack {args.unknown_slack} m; "
            "there is nowhere to send the robot"
        )

    start_yx = None
    if args.start is not None:
        start_yx = [float(args.start[0]), float(args.start[1])]
    elif args.start_from_track:
        track = request.get("rover_track_yx")
        if not track:
            raise SystemExit(
                "--start-from-track was given but the goal request carries no "
                "rover track; the fuse stage writes it to <stem>.txt"
            )
        start_yx = [float(track[-1][0]), float(track[-1][1])]

    start_report = None
    if start_yx is None:
        component = planner.largest_component(navigable)
        component_source = "largest navigable component"
    else:
        rows, cols = hm3d_map.world_to_cell(np.asarray([start_yx]))
        start_row, start_col = int(rows[0]), int(cols[0])
        if not navigable[start_row, start_col]:
            # The rover's own last pose can sit just inside the clearance
            # margin -- it drove there with a body, not a point -- so it is
            # projected too, and by how far is reported rather than hidden.
            moved_row, moved_col, moved_cells = nearest_cell_in(
                navigable,
                start_row,
                start_col,
                prefer=hm3d_map.clearance,
                tolerance_cells=args.robot_radius / hm3d_map.res,
            )
            print(
                f"start ({start_yx[0]:.2f}, {start_yx[1]:.2f}) is not navigable; "
                f"projected {moved_cells * hm3d_map.res:.2f} m to the nearest "
                "cell that is"
            )
            start_row, start_col = moved_row, moved_col
        component = planner.component_containing(navigable, (start_row, start_col))
        component_source = f"component containing the start cell ({start_row}, {start_col})"
        if component is None:
            raise SystemExit("the start cell is not in any navigable component")
        projected = hm3d_map.cell_to_world(start_row, start_col)
        start_report = {
            "requested_yx": start_yx,
            "cell_rc": [start_row, start_col],
            "yx": [float(projected[0]), float(projected[1])],
        }
        start_yx = start_report["yx"]

    print(
        f"{component_source}: {int(component.sum())} cells "
        f"({component.sum() * hm3d_map.res ** 2:.1f} m2)"
    )

    rejected: list[dict[str, object]] = []
    chosen = None
    for candidate in candidates_in:
        label = str(candidate["group_id"])
        if int(candidate.get("cell_count", 0)) < 1:
            rejected.append({"group_id": label, "reason": "holds no BEV cell"})
            print(f"  reject {label}: holds no BEV cell, so it has no position")
            continue
        centroid_yx = candidate["centroid_yx"]
        centroid_row, centroid_col = (
            float(candidate["centroid_cell_rc"][0]),
            float(candidate["centroid_cell_rc"][1]),
        )
        inside = (
            0 <= int(centroid_row) < hm3d_map.n_rows
            and 0 <= int(centroid_col) < hm3d_map.n_cols
        )
        centroid_navigable = bool(
            inside and component[int(centroid_row), int(centroid_col)]
        )
        centroid_clearance = (
            float(hm3d_map.clearance[int(centroid_row), int(centroid_col)])
            if inside
            else float("nan")
        )
        print(
            f'  rank {candidate.get("rank", "?")} {label}: '
            f'{candidate["cell_count"]} cells, centroid (y, x) = '
            f"({centroid_yx[0]:.2f}, {centroid_yx[1]:.2f}) m, clearance "
            f"{centroid_clearance:+.3f} m, "
            f"{'navigable' if centroid_navigable else 'not navigable'}"
        )
        if centroid_navigable:
            goal_row, goal_col, moved_cells = int(centroid_row), int(centroid_col), 0.0
            reason = "weighted centroid was already navigable and in the component"
        else:
            try:
                goal_row, goal_col, moved_cells = nearest_cell_in(
                    component,
                    centroid_row,
                    centroid_col,
                    prefer=hm3d_map.clearance,
                    tolerance_cells=args.robot_radius / hm3d_map.res,
                )
            except ValueError as error:
                rejected.append({"group_id": label, "reason": str(error)})
                print(f"    reject: {error}")
                continue
            reason = (
                "weighted centroid was outside the navigable component "
                f"(clearance {centroid_clearance:+.3f} m); projected to the "
                "nearest cell inside it"
            )
        moved_metres = moved_cells * hm3d_map.res
        if moved_metres > args.max_projection:
            note = (
                f"nearest navigable cell is {moved_metres:.2f} m away, past "
                f"--max-projection {args.max_projection:.2f} m"
            )
            rejected.append({"group_id": label, "reason": note})
            print(f"    reject: {note}")
            continue
        chosen = {
            "candidate": candidate,
            "goal_rc": (goal_row, goal_col),
            "moved_metres": moved_metres,
            "reason": reason,
            "centroid_navigable": centroid_navigable,
            "centroid_clearance": centroid_clearance,
        }
        break

    if chosen is None:
        raise SystemExit(
            f'no ranked entity for "{request["query"]}" can be reached: '
            + "; ".join(
                f"{item['group_id']} ({item['reason']})" for item in rejected
            )
            + ". The match may be real but outside the observed free space -- "
            "check map_semantic.png, or raise --max-projection deliberately."
        )

    winner = chosen["candidate"]
    goal_row, goal_col = chosen["goal_rc"]
    moved_metres = float(chosen["moved_metres"])
    reason = str(chosen["reason"])
    centroid_yx = winner["centroid_yx"]
    centroid_navigable = bool(chosen["centroid_navigable"])
    centroid_clearance = float(chosen["centroid_clearance"])
    goal = hm3d_map.cell_to_world(goal_row, goal_col)
    goal_yx = [float(goal[0]), float(goal[1])]

    # How far the goal ended up from the thing that was asked for. Measured
    # against the footprint, not the centroid: for a split entity the centroid
    # is the one point guaranteed not to be on it.
    cells_yx = np.asarray(winner["cells_yx"], dtype=np.float64)
    distance_to_entity = float(
        np.linalg.norm(cells_yx - np.asarray(goal_yx), axis=1).min()
    )

    result = {
        "format": "fact3r-semantic-goal",
        "version": 1,
        "query": request["query"],
        "source_request": str(args.request.resolve()),
        "grid": str(args.grid.resolve()),
        "robot_radius": args.robot_radius,
        "unknown_slack": args.unknown_slack,
        "exclude_exterior": args.exclude_exterior,
        "group_id": winner["group_id"],
        "rank": winner.get("rank"),
        "rejected_candidates": rejected,
        "semantic_id": winner["semantic_id"],
        "score": winner["score"],
        "cell_count": winner["cell_count"],
        "best_frame_id": winner["best_frame_id"],
        "centroid_yx": centroid_yx,
        "centroid_navigable": centroid_navigable,
        "centroid_clearance_m": centroid_clearance,
        "goal_yx": goal_yx,
        "goal_cell_rc": [goal_row, goal_col],
        "goal_clearance_m": float(hm3d_map.clearance[goal_row, goal_col]),
        "projection_distance_m": moved_metres,
        "distance_to_entity_m": distance_to_entity,
        "projection_reason": reason,
        "component_source": component_source,
        "component_cells": int(component.sum()),
        "component_area_m2": float(component.sum() * hm3d_map.res ** 2),
        "in_largest_component": bool(
            planner.largest_component(navigable)[goal_row, goal_col]
        ),
        "start_yx": start_yx,
        "start": start_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"  {reason}")
    print(
        f"  goal (y, x) = ({goal_yx[0]:.3f}, {goal_yx[1]:.3f}) m  cell "
        f"({goal_row}, {goal_col})  clearance "
        f"{result['goal_clearance_m']:+.3f} m  moved {moved_metres:.2f} m  "
        f"{distance_to_entity:.2f} m from the entity"
    )
    if start_yx is not None:
        print(f"  start (y, x) = ({start_yx[0]:.3f}, {start_yx[1]:.3f}) m")
    print(f"goal: {args.output}")


if __name__ == "__main__":
    main()
