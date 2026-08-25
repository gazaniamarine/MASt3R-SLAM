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


def _default_output(index: Path, query: str) -> Path:
    safe_query = "".join(
        character if character.isalnum() else "-" for character in query.lower()
    ).strip("-")
    index_directory = index if index.is_dir() else index.parent
    return index_directory / "queries" / (safe_query or "query")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, mps, cuda, or CUDA index"
    )
    parser.add_argument("--max-entities", type=int, default=3)
    parser.add_argument("--min-entity-score", type=float)
    parser.add_argument("--entity-top-views", type=int, default=2)
    parser.add_argument("--max-observations-per-entity", type=int)
    parser.add_argument("--gif-width", type=int, default=1000)
    parser.add_argument("--gif-duration-ms", type=int, default=400)
    args = parser.parse_args()

    _, manifest, _ = load_observation_index(args.index)
    output = args.output or _default_output(args.index, args.query)
    print(f"Loading {manifest['model']} on {args.device}...")
    encoder = Siglip2Encoder(str(manifest["model"]), device=args.device)
    result_path = query_observation_index(
        index=args.index,
        query=args.query,
        output=output,
        encoder=encoder,
        max_entities=args.max_entities,
        min_entity_score=args.min_entity_score,
        top_views=args.entity_top_views,
        max_observations_per_entity=args.max_observations_per_entity,
        gif_width=args.gif_width,
        gif_duration_ms=args.gif_duration_ms,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    for entity in result["entities"]:
        print(
            f"rank {entity['rank']}: {entity['group_id']} "
            f"score={entity['entity_score']:.3f} "
            f"frames={len(entity['observations'])}"
        )
    print(
        f"Query took {result['timing']['total_query_seconds_excluding_model_load']:.2f}s "
        f"after model loading"
    )
    print(f"Open gallery: {output / 'index.html'}")
    print(f"GIF: {output / 'matches.gif'}")
    print(f"Results: {result_path}")


if __name__ == "__main__":
    main()
