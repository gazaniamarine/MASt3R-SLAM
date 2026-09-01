#!/usr/bin/env python3
"""Resident SigLIP shortlist + Qwen3-VL verification for semantic BEV queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    load_observation_index,
)
from fact3r.semantics.vlm_verification import (  # noqa: E402
    Qwen3VLVerifier,
    prepare_vlm_query,
    verify_prepared_query,
)
from query_semantic_bev import (  # noqa: E402
    _fuse_prompt_scores,
    _query_prompts,
    _rank_groups,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower() or "query"


def _map_manifest(path: Path) -> Path:
    if path.suffix == ".json":
        return path
    candidate = Path(f"{path}_semantic.json")
    if not candidate.is_file():
        raise ValueError(f"cannot find semantic BEV manifest for {path}")
    return candidate


def _render_verified_bev(
    *,
    occupancy: np.ndarray,
    semantic_ids: np.ndarray,
    map_groups: dict[str, dict[str, object]],
    result: dict[str, object],
    output: Path,
) -> Path:
    height, width = occupancy.shape
    canvas = np.full((height, width, 3), 128, dtype=np.uint8)
    canvas[occupancy >= 0] = 245
    canvas[occupancy >= 65] = 35
    palette = [(25, 235, 90), (255, 165, 35), (45, 145, 255)]
    accepted = []
    for entity in result.get("entities", []):
        candidate_id = str(entity.get("candidate_id") or "")
        group = map_groups.get(candidate_id)
        if group is None:
            continue
        accepted.append((entity, group))
    for rank, (_, group) in enumerate(accepted):
        mask = semantic_ids == int(group["semantic_id"])
        colour = np.asarray(palette[rank % len(palette)], dtype=np.float32)
        canvas[mask] = (0.15 * canvas[mask] + 0.85 * colour).astype(np.uint8)

    map_image = Image.fromarray(canvas[::-1])
    legend_width = 430
    image = Image.new("RGB", (width + legend_width, height), (20, 20, 20))
    image.paste(map_image, (0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((width + 12, 10), f'VLM verified: "{result["query"]}"', fill="white")
    if not accepted:
        draw.text((width + 12, 38), "No candidate passed verification", fill=(245, 100, 100))
    for rank, (entity, _) in enumerate(accepted):
        y = 40 + rank * 58
        colour = palette[rank % len(palette)]
        vlm = entity["vlm"]
        draw.rectangle((width + 12, y, width + 28, y + 16), fill=colour)
        draw.text(
            (width + 36, y - 2),
            f"#{rank + 1} {entity['candidate_id']}",
            fill=(240, 240, 240),
        )
        draw.text(
            (width + 36, y + 17),
            f"{vlm['predicted_object']} | confidence={float(vlm['confidence']):.2f}",
            fill=(190, 230, 200),
        )
        frame = (entity.get("best_revisit_view") or {}).get("frame_id")
        draw.text((width + 36, y + 34), f"best observed frame={frame}", fill=(185, 185, 185))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _best_frame_image(output: Path, entity: dict[str, object]) -> Path | None:
    observations = list(entity.get("observations", []))
    if not observations:
        return None
    best_frame = (entity.get("best_revisit_view") or {}).get("frame_id")
    selected = next(
        (
            observation
            for observation in observations
            if observation.get("frame_id") == best_frame
        ),
        observations[0],
    )
    image = selected.get("image")
    return None if image is None else output / str(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True, help="BEV output stem or _semantic.json")
    parser.add_argument("--query", help="run once; omit for an interactive resident loop")
    parser.add_argument("--output", type=Path, help="root for verified query results")
    parser.add_argument("--siglip-device", default="0")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vlm-device-map", default="auto")
    parser.add_argument(
        "--vlm-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument(
        "--top-k",
        "--max-candidates",
        dest="max_candidates",
        type=int,
        default=5,
        help="freeze this many embedding results before VLM verification",
    )
    parser.add_argument("--evidence-views", type=int, default=2)
    parser.add_argument("--min-entity-observations", type=int, default=1)
    parser.add_argument("--min-vlm-confidence", type=float, default=0.65)
    parser.add_argument("--min-vlm-supporting-views", type=int, default=1)
    parser.add_argument("--max-entities", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--vlm-batch-candidates", type=int, default=2)
    parser.add_argument("--history-frames", type=int, default=1)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--force-reverify", action="store_true")
    args = parser.parse_args()
    for name in (
        "max_candidates",
        "evidence_views",
        "min_entity_observations",
        "min_vlm_supporting_views",
        "max_entities",
        "vlm_batch_candidates",
        "history_frames",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")

    map_path = _map_manifest(args.map)
    map_manifest = json.loads(map_path.read_text(encoding="utf-8"))
    if map_manifest.get("format") != "fact3r-depth-semantic-bev":
        raise ValueError(f"unsupported semantic BEV: {map_path}")
    index_path = Path(str(map_manifest["source_observation_index"]))
    loaded_index = load_observation_index(index_path)
    _, index_manifest, observation_embeddings = loaded_index
    if index_manifest.get("format") != "fact3r-siglip-observation-index":
        raise ValueError(
            "resident VLM verification currently requires a SigLIP observation index"
        )
    grid_path = map_path.parent / str(map_manifest["grid_file"])
    with np.load(grid_path, allow_pickle=False) as payload:
        occupancy = np.array(payload["occupancy"], copy=True)
        semantic_ids = np.array(payload["semantic_ids"], copy=True)
    map_groups = {
        str(item["group_id"]): dict(item) for item in map_manifest["groups"]
    }
    index_observations = list(index_manifest["observations"])
    output_root = args.output or map_path.parent / "vlm_live_queries"
    cache_directory = args.cache_directory or index_path.parent / "vlm_cache"

    print(f"Loading resident shortlist model {index_manifest['model']}...")
    encoder = Siglip2Encoder(
        str(index_manifest["model"]), device=args.siglip_device
    )
    encoder.encode_text(["an object"])
    print(f"SigLIP ready: load={encoder.load_seconds:.2f}s")
    print(f"Loading resident verifier {args.vlm_model}...")
    verifier = Qwen3VLVerifier(
        args.vlm_model,
        device_map=args.vlm_device_map,
        dtype=args.vlm_dtype,
        attention_implementation=args.attention_implementation,
        max_new_tokens=args.max_new_tokens,
    )
    verifier.listwise_batch_size = args.vlm_batch_candidates
    verifier.load()
    print(f"Qwen ready: load={verifier.load_seconds:.2f}s")
    keyframe_cache: dict[int, np.ndarray] = {}

    def run_query(query: str) -> None:
        started = perf_counter()
        output = output_root / _slug(query)
        query_prompts = _query_prompts(query)
        query_embeddings = encoder.encode_text(query_prompts)
        retrieval_scores, _, _ = _fuse_prompt_scores(
            observation_embeddings,
            query_embeddings,
            agreement_prompts=2,
        )
        retrieval_ranking = _rank_groups(
            retrieval_scores,
            index_observations,
            set(map_groups),
            top_views=3,
        )
        frozen_top_k = retrieval_ranking[: args.max_candidates]
        frozen_candidate_ids = [str(item["group_id"]) for item in frozen_top_k]
        print("Frozen embedding top-k (VLM may only accept/reject):")
        for rank, item in enumerate(frozen_top_k, start=1):
            observation = index_observations[int(item["best_observation_index"])]
            print(
                f"  #{rank} {item['group_id']} score={float(item['score']):.3f} "
                f"frame={observation['frame_id']}"
            )
        prepared = prepare_vlm_query(
            index=index_path,
            query=query,
            output=output,
            encoder=encoder,
            max_candidates=args.max_candidates,
            top_views=args.evidence_views,
            min_observations=args.min_entity_observations,
            min_siglip_score=-1.0,
            map_negative_weight=0.0,
            loaded_index=loaded_index,
            keyframe_cache=keyframe_cache,
            positive_prompts=query_prompts,
            forced_candidate_ids=frozen_candidate_ids,
            forced_observation_scores=retrieval_scores,
        )
        if not prepared.candidates:
            print("The frozen embedding top-k contains no eligible mapped entities.")
            return
        result_path = verify_prepared_query(
            prepared,
            verifier=verifier,
            min_confidence=args.min_vlm_confidence,
            min_supporting_views=args.min_vlm_supporting_views,
            max_entities=args.max_entities,
            cache_directory=cache_directory,
            force_reverify=args.force_reverify,
            max_observations_per_entity=args.history_frames,
            keyframe_cache=keyframe_cache,
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        bev_path = _render_verified_bev(
            occupancy=occupancy,
            semantic_ids=semantic_ids,
            map_groups=map_groups,
            result=result,
            output=output / "verified_semantic_bev.png",
        )
        if result["entities"]:
            for rank, entity in enumerate(result["entities"], start=1):
                frame_path = _best_frame_image(output, entity)
                print(
                    f"accepted #{rank}: {entity['candidate_id']} "
                    f"confidence={float(entity['vlm']['confidence']):.2f} "
                    f"frame={(entity.get('best_revisit_view') or {}).get('frame_id')}"
                )
                if frame_path is not None:
                    print(f"  observed frame: {frame_path}")
        else:
            print("No candidate passed Qwen visual verification.")
        timing = result["timing"]
        print(
            f"warm-query latency={perf_counter() - started:.3f}s "
            f"(shortlist={prepared.preparation_seconds:.3f}s, "
            f"VLM={float(timing['vlm_inference_seconds']):.3f}s, "
            f"cache_hits={timing['vlm_cache_hits']})"
        )
        print(f"verified frame gallery: {output / 'index.html'}")
        print(f"verified BEV:           {bev_path}")

    if args.query is not None:
        run_query(args.query.strip())
        return
    print("Ready. Enter object queries; Ctrl-D or 'quit' exits without unloading models.")
    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            print()
            break
        if query.casefold() in {"quit", "exit"}:
            break
        if query:
            run_query(query)


if __name__ == "__main__":
    main()
