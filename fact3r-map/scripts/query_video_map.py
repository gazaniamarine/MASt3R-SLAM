#!/usr/bin/env python3
"""Query a finalized Fact3R video map by short name instead of artifact paths."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.map_bundle import load_video_map  # noqa: E402
from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    load_observation_index,
    query_observation_index,
)
from fact3r.semantics.vlm_verification import (  # noqa: E402
    Qwen3VLVerifier,
    prepare_vlm_query,
    verify_prepared_query,
)


def _slug(value: str) -> str:
    compact = "".join(
        character if character.isalnum() else "-" for character in value.lower()
    ).strip("-")
    return compact or "query"


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _collect_queries(values: list[str] | None) -> list[str]:
    queries = [value.strip() for value in (values or []) if value.strip()]
    if queries:
        return queries
    if not sys.stdin.isatty():
        raise ValueError("provide at least one --query")
    print("Enter queries one per line; submit an empty line to start retrieval.")
    while True:
        value = input("query> ").strip()
        if not value:
            break
        queries.append(value)
    if not queries:
        raise ValueError("at least one query is required")
    return queries


def _print_result(result_path: Path) -> None:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    print(f"\nQuery: {result['query']}")
    if result["confident_match_found"]:
        for entity in result["entities"]:
            confidence = entity.get("vlm", {}).get("confidence")
            suffix = "" if confidence is None else f" Qwen={confidence:.2f}"
            print(
                f"  {entity['entity_id']}: "
                f"{len(entity['observations'])} observations{suffix}"
            )
    else:
        print("  no confident match")
    print(f"  gallery: {result_path.parent / 'index.html'}")
    print(f"  results: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument(
        "--query",
        action="append",
        help="repeat for several queries; otherwise enter them interactively",
    )
    parser.add_argument("--mode", choices=("fast", "vlm"), default="fast")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", "--siglip-device", dest="device", default="auto")
    parser.add_argument("--max-entities", type=int, default=1)
    parser.add_argument("--confounder", action="append")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--evidence-views", type=int, default=2)
    parser.add_argument("--min-entity-observations", type=int, default=2)
    parser.add_argument("--min-siglip-score", type=float, default=0.10)
    parser.add_argument("--min-vlm-confidence", type=float, default=0.75)
    parser.add_argument("--min-vlm-supporting-views", type=int, default=2)
    parser.add_argument(
        "--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct"
    )
    parser.add_argument("--vlm-device-map", default="auto")
    parser.add_argument(
        "--vlm-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--force-reverify", action="store_true")
    args = parser.parse_args()

    _, video_map = load_video_map(args.map)
    observation_directory = Path(
        str(video_map["artifacts"]["observations"]["directory"])
    )
    _, observation_manifest, _ = load_observation_index(observation_directory)
    query_root = args.output or Path(str(video_map["query_directory"]))
    query_root.mkdir(parents=True, exist_ok=True)
    queries = _collect_queries(args.query)

    print(
        f"Map {video_map['map_name']}: {video_map['entity_count']} entities, "
        f"{video_map['observation_count']} observations"
    )
    print(f"Loading {observation_manifest['model']} on {args.device}...")
    encoder = Siglip2Encoder(
        str(observation_manifest["model"]), device=args.device
    )
    if args.mode == "fast":
        for query in queries:
            output = query_root / f"{_slug(query)}-fast"
            result_path = query_observation_index(
                index=observation_directory,
                query=query,
                output=output,
                encoder=encoder,
                max_entities=args.max_entities,
                negative_prompts=args.confounder,
            )
            _print_result(result_path)
        return

    prepared_queries = []
    for query in queries:
        output = query_root / f"{_slug(query)}-qwen-verified"
        prepared_queries.append(
            prepare_vlm_query(
                index=observation_directory,
                query=query,
                output=output,
                encoder=encoder,
                max_candidates=args.max_candidates,
                top_views=args.evidence_views,
                min_observations=args.min_entity_observations,
                min_siglip_score=args.min_siglip_score,
            )
        )
    del encoder
    _release_accelerator_memory()
    verifier = Qwen3VLVerifier(
        args.vlm_model,
        device_map=args.vlm_device_map,
        dtype=args.vlm_dtype,
        attention_implementation=args.attention_implementation,
        max_new_tokens=args.max_new_tokens,
    )
    for prepared in prepared_queries:
        result_path = verify_prepared_query(
            prepared,
            verifier=verifier,
            min_confidence=args.min_vlm_confidence,
            min_supporting_views=args.min_vlm_supporting_views,
            max_entities=args.max_entities,
            force_reverify=args.force_reverify,
        )
        _print_result(result_path)


if __name__ == "__main__":
    main()
