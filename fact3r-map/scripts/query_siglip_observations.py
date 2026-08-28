#!/usr/bin/env python3
"""Retrieve persistent entities by text and render every stored observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    load_observation_index,
    query_observation_index,
)
from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder  # noqa: E402


def _default_output(index: Path, query: str) -> Path:
    safe_query = "".join(
        character if character.isalnum() else "-" for character in query.lower()
    ).strip("-")
    index_directory = index if index.is_dir() else index.parent
    return index_directory / "queries" / f"{safe_query or 'query'}-verified"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, mps, cuda, or CUDA index"
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-entities", type=int, default=3)
    parser.add_argument(
        "--positive-prompt",
        action="append",
        help="repeatable positive prompt; defaults to an ensemble built from --query",
    )
    parser.add_argument(
        "--confounder",
        action="append",
        help="repeatable visually confusable negative prompt",
    )
    parser.add_argument("--entity-top-views", type=int, default=3)
    parser.add_argument("--min-supporting-views", type=int, default=2)
    parser.add_argument("--min-view-margin", type=float, default=0.02)
    parser.add_argument(
        "--min-entity-margin",
        "--min-entity-score",
        dest="min_entity_margin",
        type=float,
        default=0.02,
    )
    parser.add_argument("--reference-mask-area", type=float, default=4096.0)
    parser.add_argument(
        "--no-map-hard-negatives",
        action="store_true",
        help="disable automatic negatives from visually nearest map entities",
    )
    parser.add_argument("--map-hard-negative-neighbors", type=int, default=3)
    parser.add_argument("--map-hard-negative-weight", type=float, default=1.0)
    parser.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="also rank pending/unassigned groups; disabled for navigation by default",
    )
    parser.add_argument(
        "--no-unanchored-tracks",
        action="store_true",
        help="exclude persistent 2D tracks that never acquired reliable 3D support",
    )
    parser.add_argument("--max-observations-per-entity", type=int)
    parser.add_argument("--gif-width", type=int, default=1000)
    parser.add_argument("--gif-duration-ms", type=int, default=400)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="rank only: one positive/negative prompt and no gallery or GIF",
    )
    args = parser.parse_args()

    _, manifest, _ = load_observation_index(args.index)
    if not manifest.get("semantic_query_capable", True):
        raise ValueError(
            "this index contains raw VLA appearance features only; build a "
            "Qwen3-VL-Embedding semantic index before running text queries"
        )
    output = args.output or _default_output(args.index, args.query)
    if manifest.get("format") == "fact3r-qwen-embedding-observation-index":
        print(f"Loading {manifest['model']} with device map {args.device_map}...")
        encoder = Qwen3VLEmbeddingEncoder(
            str(manifest["model"]),
            device_map=args.device_map,
            dtype=args.dtype,
        )
    else:
        print(f"Loading {manifest['model']} on {args.device}...")
        encoder = Siglip2Encoder(str(manifest["model"]), device=args.device)
    result_path = query_observation_index(
        index=args.index,
        query=args.query,
        output=output,
        encoder=encoder,
        max_entities=args.max_entities,
        top_views=args.entity_top_views,
        positive_prompts=(
            [args.query]
            if args.fast and args.positive_prompt is None
            else args.positive_prompt
        ),
        negative_prompts=(
            ["an unrelated object"]
            if args.fast and args.confounder is None
            else args.confounder
        ),
        confirmed_only=not args.include_unconfirmed,
        include_unanchored_tracks=not args.no_unanchored_tracks,
        min_supporting_views=args.min_supporting_views,
        min_view_margin=args.min_view_margin,
        min_entity_margin=args.min_entity_margin,
        reference_mask_area=args.reference_mask_area,
        automatic_map_negatives=not args.no_map_hard_negatives,
        map_negative_neighbors=args.map_hard_negative_neighbors,
        map_negative_weight=args.map_hard_negative_weight,
        max_observations_per_entity=args.max_observations_per_entity,
        gif_width=args.gif_width,
        gif_duration_ms=args.gif_duration_ms,
        render_outputs=not args.fast,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["confident_match_found"]:
        for entity in result["entities"]:
            print(
                f"rank {entity['rank']}: {entity['group_id']} "
                f"margin={entity['entity_margin']:.3f} "
                f"support={entity['supporting_view_count']}/"
                f"{entity['observation_count']} "
                f"best_frame={entity['best_revisit_view']['frame_id']} "
                f"frames={len(entity['observations'])}"
            )
    else:
        print(
            "No persistent 3D entity or tracked 2D observation passed the "
            "semantic confidence gates."
        )
    timing = result["timing"]
    print(
        f"Timing: load={timing['model_load_seconds']:.2f}s, "
        f"text={timing['text_encoding_seconds']:.2f}s, "
        f"rank={timing['ranking_seconds']:.2f}s, "
        f"render={timing['rendering_seconds']:.2f}s, "
        f"query-total={timing['total_query_seconds_excluding_model_load']:.2f}s"
    )
    if result["gallery"] is not None:
        print(f"Open gallery: {output / 'index.html'}")
    if result["gif"] is not None:
        print(f"GIF: {output / str(result['gif'])}")
    print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
