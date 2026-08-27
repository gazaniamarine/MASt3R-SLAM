#!/usr/bin/env python3
"""Retrieve with SigLIP, verify with Qwen3-VL, and render entity histories."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    load_observation_index,
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


def _default_output(index: Path, query: str) -> Path:
    directory = index if index.is_dir() else index.parent
    return directory / "queries" / f"{_slug(query)}-qwen-verified"


def _release_siglip() -> None:
    """Return SigLIP memory before the larger verifier is loaded."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Shortlist confirmed entities with SigLIP, verify their highlighted "
            "multi-view evidence with Qwen3-VL, and render every accepted view."
        )
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--siglip-device",
        "--device",
        dest="siglip_device",
        default="auto",
        help="auto, cpu, mps, cuda, or a CUDA index",
    )
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--evidence-views", type=int, default=3)
    parser.add_argument("--min-entity-observations", type=int, default=2)
    parser.add_argument("--map-hard-negative-neighbors", type=int, default=3)
    parser.add_argument("--map-hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--max-entities", type=int, default=3)
    parser.add_argument("--min-vlm-confidence", type=float, default=0.75)
    parser.add_argument("--min-vlm-supporting-views", type=int, default=2)
    parser.add_argument(
        "--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct"
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
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--force-reverify", action="store_true")
    parser.add_argument("--max-observations-per-entity", type=int)
    parser.add_argument("--gif-width", type=int, default=1000)
    parser.add_argument("--gif-duration-ms", type=int, default=400)
    args = parser.parse_args()

    _, manifest, _ = load_observation_index(args.index)
    output = args.output or _default_output(args.index, args.query)
    print(f"Loading shortlist model {manifest['model']} on {args.siglip_device}...")
    encoder = Siglip2Encoder(str(manifest["model"]), device=args.siglip_device)
    prepared = prepare_vlm_query(
        index=args.index,
        query=args.query,
        output=output,
        encoder=encoder,
        max_candidates=args.max_candidates,
        top_views=args.evidence_views,
        min_observations=args.min_entity_observations,
        map_negative_neighbors=args.map_hard_negative_neighbors,
        map_negative_weight=args.map_hard_negative_weight,
    )
    print(
        f"SigLIP shortlisted {len(prepared.candidates)} confirmed entities; "
        "releasing it before Qwen3-VL loads."
    )
    del encoder
    _release_siglip()

    verifier = Qwen3VLVerifier(
        args.vlm_model,
        device_map=args.vlm_device_map,
        dtype=args.vlm_dtype,
        attention_implementation=args.attention_implementation,
        max_new_tokens=args.max_new_tokens,
    )
    result_path = verify_prepared_query(
        prepared,
        verifier=verifier,
        min_confidence=args.min_vlm_confidence,
        min_supporting_views=args.min_vlm_supporting_views,
        max_entities=args.max_entities,
        cache_directory=args.cache_directory,
        force_reverify=args.force_reverify,
        max_observations_per_entity=args.max_observations_per_entity,
        gif_width=args.gif_width,
        gif_duration_ms=args.gif_duration_ms,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["confident_match_found"]:
        for entity in result["entities"]:
            candidate_id = entity.get("entity_id") or entity.get("track_id")
            print(
                f"accepted {candidate_id}: "
                f"Qwen confidence={entity['vlm']['confidence']:.2f}, "
                f"frames={len(entity['observations'])}"
            )
    else:
        print("No candidate passed Qwen3-VL verification.")
    if result["dynamic_confounders"]:
        print(
            "Dynamic confounders discovered: "
            + ", ".join(result["dynamic_confounders"])
        )
    timing = result["timing"]
    print(
        f"Qwen load={timing['vlm_model_load_seconds']:.2f}s, "
        f"inference={timing['vlm_inference_seconds']:.2f}s, "
        f"cache hits={timing['vlm_cache_hits']}"
    )
    print(f"Open gallery: {output / 'index.html'}")
    print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
