#!/usr/bin/env python3
"""SigLIP shortlist -> Qwen3-VL photo verification -> memory recall -> a candidate.

This is "Pipeline C": the `locate` stage rebuilt so a VLM actually checks the
answer before it becomes the robot's destination, and a persistent per-map
memory lets a repeat or "go back to X" query skip straight to what was
already found instead of re-resolving from scratch.

    conda run --no-capture-output -n SAM2 python3 \\
        fact3r-map/scripts/resolve_semantic_goal_verified.py \\
        --map logs/rover/depth_semantic/map --query "the 3D printer" \\
        --output logs/rover/depth_semantic/goal_request_verified_3d_printer.json

Where this differs from `resolve_semantic_goal.py` (Pipeline A, pure SigLIP,
untouched by this script) and `resolve_semantic_goal_vlm.py` (Pipeline B,
Qwen picks by pointing at a rendered BEV blob with no label):

  1. SigLIP still produces the shortlist, with the identical ranking formula
     `resolve_semantic_goal.py` uses (`query_semantic_bev.py`'s
     `_query_prompts`/`_fuse_prompt_scores`/`_rank_groups`, imported
     verbatim, never restated).
  2. Before touching SigLIP or Qwen, a per-map memory file
     (`<stem>_goal_memory.jsonl`, next to the map manifest -- so it survives
     across `--run` directories and sessions) is checked for a matching or
     "go back to X"-style repeat of a past query; a hit skips straight to
     step 5.
  3. On a miss, the frozen shortlist is handed to `Qwen3VLVerifier`
     (`fact3r/semantics/vlm_verification.py`) against real photo/mask
     evidence of each candidate -- not a coloured, unlabelled BEV blob. Each
     candidate is judged independently yes/no/uncertain; only entities Qwen
     actually confirms become candidates here.
  4. A fresh, VLM-confirmed resolution is appended to the memory file.
  5. A single-marker confirmation image is rendered with
     `bev_pointing.render_pointing_bev` -- an operator-facing "here is what
     I'm going to" render of the already-decided winner, not a second blind
     guess (`Qwen3VLPointer.point`, Pipeline B's guessing step, is not used
     here at all).

The output is the same `fact3r-semantic-goal-request` schema
`resolve_semantic_goal.py` writes (plus a few additive fields --
`vlm`/`memory_hit`/`vlm_verified`/`confirmation_image` -- that
`project_semantic_goal.py` ignores), so everything downstream of `locate`
runs completely unmodified.

Fallback policy when nothing passes verification: stop loudly by default
(`SystemExit`), matching every other failure mode in this pipeline. Silent
fallback to an unverified guess would reintroduce the exact failure this
script exists to prevent (Qwen/SigLIP confidently picking the wrong thing).
Pass --allow-unverified-fallback to opt into an explicit, loudly-flagged
exception instead of stopping.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics import goal_memory  # noqa: E402
from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    load_observation_index,
)
from fact3r.semantics.semantic_goal import (  # noqa: E402
    cell_centre_xy,
    group_cell_counts,
    weighted_centroid_cell,
    world_xy_to_cell,
)
from fact3r.semantics.vlm_verification import (  # noqa: E402
    Qwen3VLVerifier,
    prepare_vlm_query,
    verify_prepared_query,
)
from fact3r.vlm_nav.bev_pointing import render_pointing_bev  # noqa: E402


def _query_module():
    """Reuse `query_semantic_bev.py`'s prompt ensemble and ranking, verbatim.

    Same reasoning `resolve_semantic_goal.py` already gives for doing this:
    scoring the same map two different ways is how the goal and the picture
    an operator was shown drift apart.
    """

    script = Path(__file__).resolve().parent / "query_semantic_bev.py"
    spec = importlib.util.spec_from_file_location("query_semantic_bev", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slug(value: str) -> str:
    return "-".join(
        part for part in "".join(
            character.lower() if character.isalnum() else " " for character in value
        ).split() if part
    ) or "query"


def _release_siglip() -> None:
    """Return SigLIP memory before the larger verifier is loaded."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _geometry(rows, cols, weights, origin_xy, resolution) -> dict[str, object]:
    centroid_row, centroid_col = weighted_centroid_cell(rows, cols, weights)
    centroid_x, centroid_y = cell_centre_xy(centroid_row, centroid_col, origin_xy, resolution)
    cell_x, cell_y = cell_centre_xy(rows, cols, origin_xy, resolution)
    return {
        "cell_count": int(len(rows)),
        "centroid_cell_rc": [centroid_row, centroid_col],
        "centroid_yx": [float(centroid_y), float(centroid_x)],
        "cells_rc": np.stack([rows, cols], axis=1).astype(int).tolist(),
        "cells_yx": np.stack([cell_y, cell_x], axis=1).tolist(),
        "cell_weights": weights.astype(float).tolist(),
    }


def _rover_track(stem: Path, floor: dict[str, np.ndarray]) -> list[list[float]] | None:
    """The camera trace from the fuse stage, in the grid's plane frame.

    Identical to `resolve_semantic_goal.py`'s own `_rover_track`; duplicated
    rather than imported since it is a private helper there, not part of a
    shared module.
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
    return np.stack([plane_y, plane_x], axis=1).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--map", type=Path, required=True, help="stem or _semantic.json")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0", help="SigLIP device")

    parser.add_argument("--top-views", type=int, default=3, help="retrieval ranking's per-group view aggregation")
    parser.add_argument("--exact-query", action="store_true")
    parser.add_argument("--tie-break", choices=("cells", "score-only"), default="cells")
    parser.add_argument(
        "--observed-between", type=int, nargs=2, metavar=("FIRST", "LAST"),
        help="keep only entities seen in this keyframe window",
    )
    parser.add_argument("--min-score", type=float, help="pre-shortlist retrieval score filter")

    parser.add_argument(
        "--siglip-top-k", type=int, default=8,
        help="size of the frozen shortlist handed to Qwen for verification",
    )

    parser.add_argument("--evidence-views", type=int, default=3, help="photo evidence images rendered per candidate")
    parser.add_argument("--min-entity-observations", type=int, default=2)
    parser.add_argument(
        "--min-siglip-score", type=float, default=0.10,
        help="documented only -- neutralized to -1.0 internally, since this "
             "script always forces its shortlist and cannot let this gate "
             "silently re-drop a candidate the ranking already included",
    )
    parser.add_argument(
        "--map-hard-negative-neighbors", type=int, default=3,
        help="documented only -- neutralized (weight 0) internally for the "
             "same reason as --min-siglip-score",
    )
    parser.add_argument("--max-entities", type=int, default=3)
    parser.add_argument("--min-vlm-confidence", type=float, default=0.75)
    parser.add_argument("--min-vlm-supporting-views", type=int, default=2)
    parser.add_argument(
        "--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct",
        help="needs ~18-20 GB free VRAM; pass Qwen/Qwen3-VL-2B-Instruct for a busy GPU",
    )
    parser.add_argument("--vlm-device-map", default="auto")
    parser.add_argument(
        "--vlm-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument(
        "--attention-implementation", choices=("eager", "sdpa", "flash_attention_2")
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="256 was found to silently truncate a batched (multi-candidate) "
             "listwise verification response mid-JSON, which the parser then "
             "treats as an unparseable 'uncertain' rejection -- a false "
             "negative, not a real model disagreement. 512 was confirmed "
             "sufficient for a 2-candidate batch; raise further if "
             "--vlm-batch-candidates increases",
    )
    parser.add_argument("--vlm-batch-candidates", type=int, default=2)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--force-reverify", action="store_true")

    parser.add_argument(
        "--memory-file", type=Path,
        help="default: <stem>_goal_memory.jsonl next to the map manifest",
    )
    parser.add_argument("--memory-min-similarity", type=float, default=0.90)
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument(
        "--force-reresolve", action="store_true",
        help="ignore a memory hit this run, but still refresh memory afterward",
    )

    parser.add_argument("--robot-yx", type=float, nargs=2, metavar=("Y", "X"))
    parser.add_argument(
        "--confirmation-image", type=Path,
        help="default: --output with a .confirmation.png suffix",
    )

    parser.add_argument(
        "--allow-unverified-fallback", action="store_true",
        help="on total VLM rejection, emit the top SigLIP-only candidate "
             "tagged vlm_verified=false with a loud warning, instead of "
             "stopping",
    )
    args = parser.parse_args()
    if args.siglip_top_k <= 0 or args.top_views <= 0 or args.evidence_views <= 0:
        raise ValueError("siglip-top-k, top-views, and evidence-views must be positive")

    query_module = _query_module()
    map_path = query_module._map_manifest(args.map)
    map_manifest = json.loads(map_path.read_text(encoding="utf-8"))
    if map_manifest.get("format") != "fact3r-depth-semantic-bev":
        raise SystemExit(f"unsupported semantic BEV: {map_path}")
    stem = map_path.parent / map_path.name.replace("_semantic.json", "")

    grid_path = map_path.parent / str(map_manifest["grid_file"])
    with np.load(grid_path, allow_pickle=False) as payload:
        occupancy = np.array(payload["occupancy"], copy=True)
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
    loaded_index = load_observation_index(index_path)
    _, index_manifest, observation_embeddings = loaded_index
    if index_manifest.get("format") != "fact3r-siglip-observation-index":
        raise SystemExit(
            "resolve_semantic_goal_verified.py requires a SigLIP observation "
            f"index; {index_path} is {index_manifest.get('format')!r}. Qwen "
            "verification is only proven against SigLIP-shaped indices."
        )

    group_metadata = {str(item["group_id"]): item for item in map_manifest["groups"]}
    cell_counts = group_cell_counts(semantic_ids, map_manifest["groups"])
    on_map_ids = {group for group in group_metadata if cell_counts.get(group, 0) > 0}
    print(f"on-map filter: {len(on_map_ids)} of {len(group_metadata)} entities hold a BEV cell")

    memory_path = args.memory_file or Path(f"{stem}_goal_memory.jsonl")

    print(f"loading {index_manifest['model']}...")
    encoder = Siglip2Encoder(str(index_manifest["model"]), device=args.device)
    siglip_load_seconds = encoder.load_seconds
    print(f"SigLIP ready: load={siglip_load_seconds:.2f}s")

    memory_records = [] if args.no_memory else goal_memory.load_memory(memory_path)
    memory_hit = None
    if memory_records and not args.force_reresolve:
        memory_hit = goal_memory.find_memory_hit(
            args.query, memory_records, encoder,
            min_similarity=args.memory_min_similarity,
        )

    accepted_entities: list[dict[str, object]] = []
    from_memory = False
    vlm_verified = True
    vlm_model_name = args.vlm_model
    query_prompts_used: list[str] = []

    if memory_hit is not None:
        surviving = [
            entity for entity in memory_hit["entities"]
            if cell_counts.get(str(entity["group_id"]), 0) > 0
        ]
        if surviving:
            print(
                f"memory hit: {len(surviving)} of {len(memory_hit['entities'])} "
                "remembered entities still hold a BEV cell"
            )
            accepted_entities = surviving
            from_memory = True
            vlm_model_name = str(memory_hit.get("vlm_model", args.vlm_model))
        else:
            print("memory hit's entities no longer hold a BEV cell on this grid; re-resolving")

    if not from_memory:
        query_prompts_used = query_module._query_prompts(args.query, ensemble=not args.exact_query)
        print("query prompts: " + " | ".join(query_prompts_used))
        scores, _, _ = query_module._fuse_prompt_scores(
            observation_embeddings,
            encoder.encode_text(query_prompts_used),
            agreement_prompts=1 if args.exact_query else 2,
        )
        candidates = set(on_map_ids)
        if args.observed_between is not None:
            first, last = args.observed_between
            seen_in_window = {
                str(observation["entity_id"])
                for observation in index_manifest["observations"]
                if observation.get("entity_id") is not None
                and first <= int(observation["frame_id"]) <= last
            }
            candidates &= seen_in_window
        if not candidates:
            raise SystemExit(f'no mapped entity is eligible for "{args.query}"')
        ranked = query_module._rank_groups(
            scores, list(index_manifest["observations"]), candidates,
            top_views=args.top_views,
        )
        if args.min_score is not None:
            ranked = [item for item in ranked if float(item["score"]) >= args.min_score]
        if not ranked:
            raise SystemExit(f'no mapped entity scored at or above --min-score for "{args.query}"')
        if args.tie_break == "cells":
            ranked.sort(
                key=lambda item: (
                    round(float(item["score"]), 6),
                    cell_counts.get(str(item["group_id"]), 0),
                ),
                reverse=True,
            )
        frozen_top_k = ranked[: args.siglip_top_k]
        frozen_candidate_ids = [str(item["group_id"]) for item in frozen_top_k]
        ranked_by_id = {str(item["group_id"]): item for item in frozen_top_k}
        print("frozen shortlist (Qwen may only accept/reject these):")
        for rank, item in enumerate(frozen_top_k, start=1):
            print(
                f"  #{rank} {item['group_id']} score={float(item['score']):.4f} "
                f"views={item['views']}"
            )

        output_dir = args.output.parent / f"{_slug(args.query)}_vlm_query"
        prepared = prepare_vlm_query(
            index=index_path,
            query=args.query,
            output=output_dir,
            encoder=encoder,
            max_candidates=len(frozen_candidate_ids),
            top_views=args.evidence_views,
            min_observations=args.min_entity_observations,
            min_siglip_score=-1.0,
            map_negative_weight=0.0,
            loaded_index=loaded_index,
            positive_prompts=query_prompts_used,
            forced_candidate_ids=frozen_candidate_ids,
            forced_observation_scores=scores,
        )
        if not prepared.candidates:
            raise SystemExit(
                "the frozen SigLIP shortlist contains no entity with enough "
                "observations for photo evidence; lower "
                "--min-entity-observations or raise --siglip-top-k"
            )

        del encoder
        _release_siglip()

        print(f"loading {args.vlm_model}...")
        verifier = Qwen3VLVerifier(
            args.vlm_model,
            device_map=args.vlm_device_map,
            dtype=args.vlm_dtype,
            attention_implementation=args.attention_implementation,
            max_new_tokens=args.max_new_tokens,
        )
        verifier.listwise_batch_size = args.vlm_batch_candidates
        result_path = verify_prepared_query(
            prepared,
            verifier=verifier,
            min_confidence=args.min_vlm_confidence,
            min_supporting_views=args.min_vlm_supporting_views,
            max_entities=args.max_entities,
            cache_directory=args.cache_directory,
            force_reverify=args.force_reverify,
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"Qwen ready: load={verifier.load_seconds:.2f}s")
        vlm_model_name = verifier.model_name

        if result["confident_match_found"]:
            for entity in result["entities"]:
                group_id = str(entity["candidate_id"])
                ranker = ranked_by_id.get(group_id, {})
                accepted_entities.append(
                    {
                        "group_id": group_id,
                        "score": float(entity["siglip_candidate_score"]),
                        "views": ranker.get("views"),
                        "best_observation_index": ranker.get("best_observation_index"),
                        "best_view_score": ranker.get("best_view_score"),
                        "best_frame_id": int(
                            (entity.get("best_revisit_view") or {}).get("frame_id")
                        ),
                        "vlm": entity["vlm"],
                    }
                )
                print(
                    f"accepted {group_id}: confidence="
                    f"{float(entity['vlm']['confidence']):.2f} "
                    f"reason={entity['vlm']['reason']!r}"
                )
        elif args.allow_unverified_fallback:
            vlm_verified = False
            top = frozen_top_k[0]
            group_id = str(top["group_id"])
            observation = index_manifest["observations"][int(top["best_observation_index"])]
            print(
                f'WARNING: no candidate passed Qwen verification for "{args.query}"; '
                f"falling back, UNVERIFIED, to top SigLIP candidate {group_id} "
                "(--allow-unverified-fallback)"
            )
            accepted_entities.append(
                {
                    "group_id": group_id,
                    "score": float(top["score"]),
                    "views": top.get("views"),
                    "best_observation_index": top.get("best_observation_index"),
                    "best_view_score": top.get("best_view_score"),
                    "best_frame_id": int(observation["frame_id"]),
                    "vlm": {
                        "decision": "not_verified",
                        "confidence": 0.0,
                        "reason": "--allow-unverified-fallback: no candidate passed Qwen verification",
                    },
                }
            )
        else:
            raise SystemExit(
                f'no candidate passed Qwen visual verification for "{args.query}"; '
                f"{result['checked_candidate_count']} checked, "
                f"{len(result['rejected_candidates'])} rejected. Pass "
                "--allow-unverified-fallback to accept the top SigLIP-only "
                "candidate instead of stopping."
            )

    reported: list[dict[str, object]] = []
    for entity in accepted_entities:
        group_id = str(entity["group_id"])
        semantic_id = int(group_metadata[group_id]["semantic_id"])
        rows, cols = np.nonzero(semantic_ids == semantic_id)
        if not len(rows):
            continue
        weights = semantic_confidence[rows, cols]
        geometry = _geometry(rows, cols, weights, origin_xy, resolution)
        candidate = {
            **group_metadata[group_id],
            "group_id": group_id,
            **geometry,
            "score": entity["score"],
            "best_frame_id": entity["best_frame_id"],
            "vlm": entity["vlm"],
        }
        for key in ("views", "best_observation_index", "best_view_score"):
            if entity.get(key) is not None:
                candidate[key] = entity[key]
        reported.append(candidate)
    if not reported:
        raise SystemExit(
            "every accepted entity holds 0 BEV cells on the current grid -- "
            "nothing can be navigated to. Re-run with --no-memory or "
            "--force-reresolve, or re-fuse the map."
        )
    for rank, candidate in enumerate(reported, start=1):
        candidate["rank"] = rank
    winner = reported[0]

    if not from_memory and vlm_verified and not args.no_memory:
        goal_memory.append_memory(
            memory_path,
            {
                "query_text": args.query,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "vlm_model": vlm_model_name,
                "entities": [
                    {
                        "group_id": candidate["group_id"],
                        "semantic_id": int(candidate["semantic_id"]),
                        "score": candidate["score"],
                        "best_frame_id": candidate["best_frame_id"],
                        "vlm": {
                            "decision": candidate["vlm"]["decision"],
                            "confidence": candidate["vlm"]["confidence"],
                            "reason": candidate["vlm"]["reason"],
                        },
                    }
                    for candidate in reported
                ],
            },
        )
        print(f"memory: appended resolution for {args.query!r} to {memory_path}")
    elif not from_memory and not vlm_verified:
        print("memory: not recording an unverified fallback resolution")

    robot_cell = None
    if args.robot_yx is not None:
        goal_y, goal_x = args.robot_yx
        rows, cols = world_xy_to_cell(goal_x, goal_y, origin_xy, resolution)
        robot_cell = (float(rows), float(cols))
    confirmation_image = args.confirmation_image or args.output.with_suffix(".confirmation.png")
    confirmation_image, _markers = render_pointing_bev(
        occupancy,
        semantic_ids,
        [group_metadata[winner["group_id"]]],
        origin_xy=origin_xy,
        resolution=resolution,
        confidence=semantic_confidence,
        robot_cell=robot_cell,
        min_cells=1,
        max_markers=1,
        output=confirmation_image,
    )

    track = _rover_track(stem, floor)
    request = {
        "format": "fact3r-semantic-goal-request",
        "version": 1,
        "query": args.query,
        "query_image": None,
        "query_prompts": query_prompts_used,
        "source_map": str(map_path.resolve()),
        "grid_file": str(grid_path.resolve()),
        "origin_xy": [float(origin_xy[0]), float(origin_xy[1])],
        "resolution_metres": resolution,
        "grid_shape": list(semantic_ids.shape),
        "on_map_only": True,
        "tie_break": args.tie_break,
        "candidate_entities": len(on_map_ids),
        "dropped_entities": len(group_metadata) - len(on_map_ids),
        "candidates": reported,
        "winner": winner,
        "rover_track_yx": track,
        "timing": {"model_load_seconds": siglip_load_seconds},
        "locate_pipeline": "siglip-qwen-verify-memory",
        "memory_hit": from_memory,
        "vlm_verified": vlm_verified,
        "vlm_model": vlm_model_name,
        "confirmation_image": str(confirmation_image.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

    centroid = winner["centroid_yx"]
    print(
        f"\nwinner {winner['group_id']} holds {winner['cell_count']} cells; "
        f"weighted centroid (y, x) = ({centroid[0]:.2f}, {centroid[1]:.2f}) m"
    )
    print(f"memory_hit={from_memory} vlm_verified={vlm_verified}")
    if track:
        print(
            f"rover track: {len(track)} poses, last (y, x) = "
            f"({track[-1][0]:.2f}, {track[-1][1]:.2f}) m"
        )
    print(f"confirmation image: {confirmation_image}")
    print(f"goal request: {args.output}")


if __name__ == "__main__":
    main()
