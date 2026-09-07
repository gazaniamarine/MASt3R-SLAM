#!/usr/bin/env python3
"""Pipeline B: SigLIP shortlists candidates, Qwen points at the right one.

A real rover scan holds hundreds to thousands of mapped entities (14,555 on
the MPL scan this was first tested against). A rendered map cannot legibly
show all of them at once, and any static "biggest/smallest N" rule silently
drops the actual target long before Qwen ever sees it -- confirmed on that
same scan: the correct "3D printer" entity held 11 cells and ranked 316th by
size, well outside a 60-marker, biggest-first cut, so Qwen picked the largest
blob on the map (the floor) instead and rationalised it afterwards.

So Pipeline B is SigLIP-shortlist-then-Qwen-points, not SigLIP-free: the same
embedding ranking `resolve_semantic_goal.py` uses narrows thousands of
entities down to a handful (`--siglip-top-k`, small enough that the rendered
map stays legible -- a 3-marker synthetic image was read perfectly, a 60-one
was not), and only then does Qwen pick among them by pointing at the map,
rather than by a per-candidate photo verification, which is what
`query_semantic_bev_vlm_live.py`'s Qwen3VLVerifier already does. That
pointing step, not the shortlist, is what still makes this a different
pipeline from Pipeline A.

The output is deliberately the same `fact3r-semantic-goal-request` schema
`resolve_semantic_goal.py` writes, so `project_semantic_goal.py` --
navigability projection over `HM3DMap`'s clearance field -- and everything
after it (A* in `thirdparty/safediffuser`, the VLN-CE follower) run completely
unmodified against either pipeline's output. Only this stage differs; A* still
plans the whole route, exactly as it does for Pipeline A.

Runs in the segmentation environment, same as `resolve_semantic_goal.py` and
`query_semantic_bev_vlm_live.py` -- it needs transformers for Qwen3-VL, not
scipy:

    conda run --no-capture-output -n SAM2 python3 \
        fact3r-map/scripts/resolve_semantic_goal_vlm.py \
        --map logs/vlnce_runs/zsNo4HB9uLZ_t00/map \
        --query "the rug" \
        --output logs/vlnce_runs/zsNo4HB9uLZ_t00/goal_request_vlm.json
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
from fact3r.semantics.semantic_goal import group_cell_counts, world_xy_to_cell  # noqa: E402
from fact3r.vlm_nav.bev_pointing import (  # noqa: E402
    Qwen3VLPointer,
    render_pointing_bev,
    resolve_pointing,
)


def _query_module():
    """Reuse `query_semantic_bev.py`'s manifest-path resolution, verbatim.

    Same reasoning `resolve_semantic_goal.py` already gives for doing this:
    duplicating map-path handling is how the two goal resolvers would drift
    apart on what counts as a valid map.
    """

    script = Path(__file__).resolve().parent / "query_semantic_bev.py"
    spec = importlib.util.spec_from_file_location("query_semantic_bev", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True, help="stem or _semantic.json")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--robot-yx",
        type=float,
        nargs=2,
        metavar=("Y", "X"),
        help="mark the robot's current position on the rendered map, world metres",
    )
    parser.add_argument(
        "--rendered-map",
        type=Path,
        help="where to save the annotated map image Qwen is shown; "
             "default is --output with a .map.png suffix",
    )
    parser.add_argument(
        "--siglip-top-k",
        type=int,
        default=8,
        help="candidates the SigLIP shortlist hands to Qwen as markers -- keep "
             "this small, a dense marker map is what breaks the pointing step",
    )
    parser.add_argument("--siglip-device", default="0")
    parser.add_argument("--siglip-device-map", default="auto")
    parser.add_argument(
        "--siglip-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--top-views", type=int, default=3)
    parser.add_argument("--exact-query", action="store_true")
    parser.add_argument(
        "--min-score", type=float,
        help="drop shortlist candidates scoring below this before rendering",
    )
    parser.add_argument(
        "--vlm-model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="the checkpoint meant to also serve the NavDP integration later; "
             "keep this the 2B model rather than swapping in a bigger one",
    )
    parser.add_argument("--vlm-device-map", default="auto")
    parser.add_argument(
        "--vlm-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attention-implementation", choices=("eager", "sdpa", "flash_attention_2")
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--min-cells",
        type=int,
        default=1,
        help="entities holding fewer BEV cells than this get no marker -- low "
             "by default since --siglip-top-k already keeps the marker count "
             "small; a real target is often a handful of cells",
    )
    parser.add_argument(
        "--max-markers",
        type=int,
        default=60,
        help="cap on numbered markers so the rendered map stays legible; "
             "should rarely bind now that --siglip-top-k narrows the field first",
    )
    parser.add_argument(
        "--fallback-search-radius-cells",
        type=float,
        default=20.0,
        help="how far a freeform pixel answer may be from the nearest mapped "
             "entity before it is rejected outright",
    )
    args = parser.parse_args()
    if args.min_cells <= 0 or args.max_markers <= 0:
        raise ValueError("min-cells and max-markers must be positive")

    query_module = _query_module()
    map_path = query_module._map_manifest(args.map)
    map_manifest = json.loads(map_path.read_text(encoding="utf-8"))
    if map_manifest.get("format") != "fact3r-depth-semantic-bev":
        raise SystemExit(f"unsupported semantic BEV: {map_path}")

    grid_path = map_path.parent / str(map_manifest["grid_file"])
    with np.load(grid_path, allow_pickle=False) as payload:
        occupancy = np.array(payload["occupancy"], copy=True)
        semantic_ids = np.array(payload["semantic_ids"], copy=True)
        semantic_confidence = np.array(payload["semantic_confidence"], copy=True)
        origin_xy = np.array(payload["origin_xy"], copy=True)
        resolution = float(payload["resolution"])
    groups = list(map_manifest["groups"])

    # Stage 1: SigLIP narrows thousands of entities to a handful, exactly as
    # resolve_semantic_goal.py does. Pipeline B's own contribution starts
    # after this -- Qwen picks among the shortlist by pointing at the
    # rendered map, not by per-candidate photo verification.
    index_path = Path(str(map_manifest["source_observation_index"]))
    _, index_manifest, observation_embeddings = load_observation_index(index_path)
    started = perf_counter()
    if index_manifest.get("format") == "fact3r-qwen-embedding-observation-index":
        from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder

        print(f"loading {index_manifest['model']}...")
        encoder = Qwen3VLEmbeddingEncoder(
            str(index_manifest["model"]),
            device_map=args.siglip_device_map,
            dtype=args.siglip_dtype,
        )
    else:
        print(f"loading {index_manifest['model']}...")
        encoder = Siglip2Encoder(str(index_manifest["model"]), device=args.siglip_device)
    print(f"shortlist encoder ready: load={perf_counter() - started:.2f}s")

    cell_counts = group_cell_counts(semantic_ids, groups)
    on_map = {str(g["group_id"]) for g in groups if cell_counts.get(str(g["group_id"]), 0) > 0}
    print(f"on-map filter: {len(on_map)} of {len(groups)} entities hold a BEV cell")

    prompts = query_module._query_prompts(args.query, ensemble=not args.exact_query)
    print("query prompts: " + " | ".join(prompts))
    scores, _, _ = query_module._fuse_prompt_scores(
        observation_embeddings,
        encoder.encode_text(prompts),
        agreement_prompts=1 if args.exact_query else 2,
    )
    ranked = query_module._rank_groups(
        scores, list(index_manifest["observations"]), on_map, top_views=args.top_views
    )
    if args.min_score is not None:
        ranked = [item for item in ranked if float(item["score"]) >= args.min_score]
    if not ranked:
        raise SystemExit(f'no entity scored at or above --min-score for "{args.query}"')
    shortlist_ids = {str(item["group_id"]) for item in ranked[: args.siglip_top_k]}
    shortlisted_groups = [g for g in groups if str(g["group_id"]) in shortlist_ids]
    print(
        f"SigLIP shortlist: {len(shortlisted_groups)} candidates offered to Qwen "
        f"(top score {float(ranked[0]['score']):.4f})"
    )

    robot_cell = None
    if args.robot_yx is not None:
        goal_y, goal_x = args.robot_yx
        rows, cols = world_xy_to_cell(goal_x, goal_y, origin_xy, resolution)
        robot_cell = (float(rows), float(cols))

    rendered_map = args.rendered_map or args.output.with_suffix(".map.png")
    rendered_map, markers = render_pointing_bev(
        occupancy,
        semantic_ids,
        shortlisted_groups,
        origin_xy=origin_xy,
        resolution=resolution,
        confidence=semantic_confidence,
        robot_cell=robot_cell,
        min_cells=args.min_cells,
        max_markers=args.max_markers,
        output=rendered_map,
    )
    if not markers:
        raise SystemExit(
            "no shortlisted entity holds enough BEV cells to become a marker; "
            "lower --min-cells"
        )
    print(f"rendered {len(markers)} markers -> {rendered_map}")

    print(f"loading {args.vlm_model}...")
    pointer = Qwen3VLPointer(
        args.vlm_model,
        device_map=args.vlm_device_map,
        dtype=args.vlm_dtype,
        attention_implementation=args.attention_implementation,
        max_new_tokens=args.max_new_tokens,
    )
    pointer.load()
    print(f"Qwen ready: load={pointer.load_seconds:.2f}s")

    result = pointer.point(query=args.query, map_image=rendered_map, markers=markers)
    print(
        f"Qwen answer: marker={result.marker} xy=({result.x:.2f}, {result.y:.2f}) "
        f"reason={result.reason!r}"
    )

    candidate = resolve_pointing(
        result,
        markers,
        semantic_ids=semantic_ids,
        groups=shortlisted_groups,
        confidence=semantic_confidence,
        origin_xy=origin_xy,
        resolution=resolution,
        fallback_search_radius_cells=args.fallback_search_radius_cells,
    )
    # project_semantic_goal.py reads winner["score"] and winner["best_frame_id"]
    # with plain indexing; resolve_pointing()'s candidate shape doesn't carry
    # either, so attach them from the SigLIP shortlist ranking that already
    # scored this group (candidate["group_id"] is guaranteed to be in `ranked`
    # -- resolve_pointing can only resolve inside shortlisted_groups, which is
    # itself ranked[:siglip_top_k], and its freeform-pixel fallback raises
    # rather than returning a group outside that set).
    match = next(item for item in ranked if str(item["group_id"]) == candidate["group_id"])
    observation = index_manifest["observations"][int(match["best_observation_index"])]
    candidate["score"] = float(match["score"])
    candidate["best_frame_id"] = int(observation["frame_id"])
    candidate["rank"] = 1

    request = {
        "format": "fact3r-semantic-goal-request",
        "version": 1,
        "query": args.query,
        "source_map": str(map_path.resolve()),
        "grid_file": str(grid_path.resolve()),
        "origin_xy": [float(origin_xy[0]), float(origin_xy[1])],
        "resolution_metres": resolution,
        "grid_shape": list(semantic_ids.shape),
        "candidate_entities": len(markers),
        "candidates": [candidate],
        "winner": candidate,
        "rendered_map": str(rendered_map.resolve()),
        "vlm": {
            "model": pointer.model_name,
            "marker": result.marker,
            "x": result.x,
            "y": result.y,
            "reason": result.reason,
            "raw_output": result.raw_output,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

    centroid = candidate["centroid_yx"]
    print(
        f"\nwinner {candidate['group_id']} holds {candidate['cell_count']} cells; "
        f"weighted centroid (y, x) = ({centroid[0]:.2f}, {centroid[1]:.2f}) m"
    )
    print(f"goal request: {args.output}")


if __name__ == "__main__":
    main()
