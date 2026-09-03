#!/usr/bin/env python3
"""Score a text query on a fused semantic BEV and locate the winning entity.

This is the half of the goal resolver that needs the SigLIP text encoder, so it
runs in the segmentation environment. It stops at a *candidate* point: the
confidence-weighted centroid of the entity's footprint, plus the footprint
itself. Deciding whether the robot may stand there is
`project_semantic_goal.py`'s job, because that needs `HM3DMap`'s clearance and
connected components, and `HM3DMap` needs scipy, which the SAM2 environment
does not have.

The split is a packaging constraint, not a design one -- together the two
scripts are stage 7, "locate".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    load_observation_index,
)
from fact3r.semantics.semantic_goal import (  # noqa: E402
    cell_centre_xy,
    group_cell_counts,
    weighted_centroid_cell,
)


def _query_module():
    """Reuse the query script's prompt ensemble and ranking, verbatim.

    Scoring the same map two different ways is how the goal and the picture the
    operator was shown drift apart, so this imports the ranking rather than
    restating it.
    """

    script = Path(__file__).resolve().parent / "query_semantic_bev.py"
    spec = importlib.util.spec_from_file_location("query_semantic_bev", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rover_track(stem: Path, floor: dict[str, np.ndarray]) -> list[list[float]] | None:
    """The camera trace from the fuse stage, in the grid's plane frame.

    `<stem>.txt` holds one TUM-style line per processed keyframe in the rover
    world (x, gravity-down, horizontal z); the grid is indexed in the floor
    plane. Projecting the trace through the same floor frame the grid was built
    with is what makes "start where the rover finished" expressible as a planner
    endpoint.
    """

    path = Path(f"{stem}.txt")
    if not path.exists():
        return None
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not points:
        return None
    relative = np.asarray(points, dtype=np.float64) - floor["origin"]
    plane_x = relative @ floor["u"]
    plane_y = relative @ floor["v"]
    # Planner order: (y, x).
    return np.stack([plane_y, plane_x], axis=1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True, help="stem or _semantic.json")
    parser.add_argument(
        "--query",
        required=False,
        default="",
        help="text goal; omit when giving --image",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="image goal, scored against the stored observation embeddings",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--top-k", type=int, default=5, help="candidates reported")
    parser.add_argument(
        "--observed-between",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="keep only entities seen in this keyframe window; use the arrival "
             "window of the leg whose instruction named the landmark",
    )
    parser.add_argument("--top-views", type=int, default=3)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--exact-query", action="store_true")
    parser.add_argument(
        "--tie-break",
        choices=("cells", "score-only"),
        default="cells",
        help="how to order entities whose scores are equal to 1e-6. Ties are "
        "not rare here: consecutive frames of one object that UOT never merged "
        "become separate entities with near-identical prototypes, and the "
        "arbitrary winner among them may hold a single cell. 'cells' prefers "
        "the one with the most map support, which is the one with a usable "
        "footprint; 'score-only' keeps the raw ranking order.",
    )
    parser.add_argument(
        "--include-unmapped",
        action="store_true",
        help="rank entities holding no BEV cell. They have no position, so the "
        "projection stage will reject the winner; for retrieval evaluation only.",
    )
    args = parser.parse_args()
    if args.image is None and not args.query:
        parser.error("give either --query or --image")
    if args.top_k <= 0 or args.top_views <= 0:
        raise ValueError("top-k and top-views must be positive")

    query_module = _query_module()
    map_path = query_module._map_manifest(args.map)
    map_manifest = json.loads(map_path.read_text(encoding="utf-8"))
    if map_manifest.get("format") != "fact3r-depth-semantic-bev":
        raise SystemExit(f"unsupported semantic BEV: {map_path}")

    grid_path = map_path.parent / str(map_manifest["grid_file"])
    with np.load(grid_path, allow_pickle=False) as payload:
        semantic_ids = np.array(payload["semantic_ids"], copy=True)
        semantic_confidence = np.array(payload["semantic_confidence"], copy=True)
        origin_xy = np.array(payload["origin_xy"], copy=True)
        resolution = float(payload["resolution"])
        floor = {
            "origin": np.array(payload["floor_origin"], copy=True),
            "u": np.array(payload["floor_u"], copy=True),
            "v": np.array(payload["floor_v"], copy=True),
        }

    index_path = Path(str(map_manifest["source_observation_index"]))
    _, index_manifest, embeddings = load_observation_index(index_path)
    started = perf_counter()
    if index_manifest.get("format") == "fact3r-qwen-embedding-observation-index":
        from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder

        print(f"loading {index_manifest['model']}...")
        encoder = Qwen3VLEmbeddingEncoder(
            str(index_manifest["model"]), device_map=args.device_map, dtype=args.dtype
        )
    else:
        print(f"loading {index_manifest['model']}...")
        encoder = Siglip2Encoder(str(index_manifest["model"]), device=args.device)
    load_seconds = perf_counter() - started

    if args.image is not None:
        # An image goal lands in the same embedding space as the stored mask
        # observations, so it is scored by plain cosine similarity rather than
        # through the text prompt ensemble.
        from PIL import Image as PILImage

        goal_image = PILImage.open(args.image).convert("RGB")
        vector = np.asarray(encoder.encode_images([goal_image]), dtype=np.float32)[0]
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise SystemExit(f"image goal produced a zero embedding: {args.image}")
        scores = np.asarray(embeddings, dtype=np.float32) @ (vector / norm)
        # An image goal has no prompt ensemble; the request records the image
        # it was matched against instead.
        prompts = []
        print(f"image goal: {args.image}")
    else:
        prompts = query_module._query_prompts(args.query, ensemble=not args.exact_query)
        print("query prompts: " + " | ".join(prompts))
        scores, _, _ = query_module._fuse_prompt_scores(
            embeddings,
            encoder.encode_text(prompts),
            agreement_prompts=1 if args.exact_query else 2,
        )

    group_metadata = {str(item["group_id"]): item for item in map_manifest["groups"]}
    cell_counts = group_cell_counts(semantic_ids, map_manifest["groups"])
    candidates = set(group_metadata)
    if not args.include_unmapped:
        candidates = {group for group in candidates if cell_counts.get(group, 0) > 0}
    dropped = len(group_metadata) - len(candidates)
    print(
        f"on-map filter: {len(candidates)} of {len(group_metadata)} entities hold "
        f"a BEV cell; {dropped} dropped"
    )
    if args.observed_between is not None:
        # "stop near the X" names an object at the end of that leg, so the only
        # honest candidates are entities the agent actually saw as it arrived
        # there. Without this a query like "doorway" ranks a doorway from a
        # different room first, and the score gap is far too small to fix it.
        first, last = args.observed_between
        seen_in_window = set()
        for observation in index_manifest["observations"]:
            entity = observation.get("entity_id")
            if entity is None:
                continue
            if first <= int(observation["frame_id"]) <= last:
                seen_in_window.add(str(entity))
        before = len(candidates)
        candidates = candidates & seen_in_window
        print(
            f"arrival-window filter [{first}, {last}]: {len(candidates)} of "
            f"{before} entities were observed there"
        )
        if not candidates:
            raise SystemExit(
                f"no mapped entity was observed in keyframes {first}-{last}; "
                "widen --observed-between or drop it"
            )
    if not candidates:
        raise SystemExit(
            "no entity in this map holds a BEV cell -- nothing can be located. "
            "Check the fuse stage's semantic_cell_count."
        )

    ranked = query_module._rank_groups(
        scores, list(index_manifest["observations"]), candidates,
        top_views=args.top_views,
    )
    if args.min_score is not None:
        ranked = [item for item in ranked if float(item["score"]) >= args.min_score]
    if not ranked:
        raise SystemExit(
            f'no mapped entity scored at or above --min-score for "{args.query}"'
        )
    if args.tie_break == "cells":
        # Stable, so entities that are genuinely separated by score keep their
        # order; only exact ties are re-ordered.
        ranked.sort(
            key=lambda item: (
                round(float(item["score"]), 6),
                cell_counts.get(str(item["group_id"]), 0),
            ),
            reverse=True,
        )
        tied = sum(
            1
            for item in ranked
            if round(float(item["score"]), 6) == round(float(ranked[0]["score"]), 6)
        )
        if tied > 1:
            print(
                f"tie-break: {tied} entities share the top score "
                f"{float(ranked[0]['score']):.4f}; taking the one with the most "
                f"BEV cells ({cell_counts.get(str(ranked[0]['group_id']), 0)})"
            )

    observations = list(index_manifest["observations"])
    reported = []
    for rank, group in enumerate(ranked[: args.top_k], start=1):
        group_id = str(group["group_id"])
        observation = observations[int(group["best_observation_index"])]
        semantic_id = int(group_metadata[group_id]["semantic_id"])
        rows, cols = np.nonzero(semantic_ids == semantic_id)
        if not len(rows):
            # Only reachable with --include-unmapped; such an entity has no
            # position at all, so it is carried without geometry and the
            # projection stage rejects it by name.
            geometry: dict[str, object] = {"cell_count": 0}
        else:
            weights = semantic_confidence[rows, cols]
            centroid_row, centroid_col = weighted_centroid_cell(rows, cols, weights)
            centroid_x, centroid_y = cell_centre_xy(
                centroid_row, centroid_col, origin_xy, resolution
            )
            cell_x, cell_y = cell_centre_xy(rows, cols, origin_xy, resolution)
            geometry = {
                "cell_count": int(len(rows)),
                # Fractional cell units, deliberately: the projection stage
                # needs the sub-cell position to break ties sensibly.
                "centroid_cell_rc": [centroid_row, centroid_col],
                # Planner order. Not validated here -- a U-shaped or split
                # entity puts its own centroid in a wall, and only the
                # clearance field knows that.
                "centroid_yx": [float(centroid_y), float(centroid_x)],
                "cells_rc": np.stack([rows, cols], axis=1).astype(int).tolist(),
                "cells_yx": np.stack([cell_y, cell_x], axis=1).tolist(),
                "cell_weights": weights.astype(float).tolist(),
            }
        reported.append(
            {
                **group_metadata[group_id],
                **group,
                **geometry,
                "rank": rank,
                "best_frame_id": int(observation["frame_id"]),
            }
        )
        print(
            f"rank {rank}: {group_id} score={float(group['score']):.4f} "
            f"cells={geometry['cell_count']} views={group['views']} "
            f"frame={int(observation['frame_id'])}"
        )

    winner = reported[0]
    if not winner["cell_count"]:
        raise SystemExit(
            f"winning entity {winner['group_id']} holds no BEV cell, so it has "
            "no position. This can only happen with --include-unmapped."
        )

    track = _rover_track(
        map_path.parent / map_path.name.replace("_semantic.json", ""), floor
    )
    request = {
        "format": "fact3r-semantic-goal-request",
        "version": 1,
        "query": args.query,
        "query_image": str(Path(args.image).resolve()) if args.image else None,
        "query_prompts": prompts,
        "source_map": str(map_path.resolve()),
        "grid_file": str(grid_path.resolve()),
        "origin_xy": [float(origin_xy[0]), float(origin_xy[1])],
        "resolution_metres": resolution,
        "grid_shape": list(semantic_ids.shape),
        "on_map_only": not args.include_unmapped,
        "tie_break": args.tie_break,
        "candidate_entities": len(candidates),
        "dropped_entities": dropped,
        "candidates": reported,
        "winner": reported[0],
        "rover_track_yx": track,
        "timing": {"model_load_seconds": load_seconds},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    centroid = winner["centroid_yx"]
    print(
        f"\nwinner {winner['group_id']} holds {winner['cell_count']} cells; "
        f"weighted centroid (y, x) = ({centroid[0]:.2f}, {centroid[1]:.2f}) m"
    )
    if track:
        print(
            f"rover track: {len(track)} poses, last (y, x) = "
            f"({track[-1][0]:.2f}, {track[-1][1]:.2f}) m"
        )
    print(f"goal request: {args.output}")


if __name__ == "__main__":
    main()
