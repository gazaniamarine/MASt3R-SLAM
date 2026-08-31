#!/usr/bin/env python3
"""Render an open-vocabulary text query on a fused depth-semantic BEV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    load_observation_index,
)


def _map_manifest(path: Path) -> Path:
    if path.suffix == ".json":
        return path
    candidate = Path(f"{path}_semantic.json")
    if candidate.exists():
        return candidate
    raise ValueError(f"cannot find semantic BEV manifest for {path}")


def _slug(value: str) -> str:
    return "-".join(part for part in "".join(
        character.lower() if character.isalnum() else " " for character in value
    ).split() if part) or "query"


def _normalise(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def _rank_groups(
    scores: np.ndarray,
    observations: list[dict[str, object]],
    allowed_groups: set[str],
    *,
    top_views: int,
) -> list[tuple[str, float, int]]:
    grouped: dict[str, list[float]] = {}
    for row, observation in enumerate(observations):
        group_id = str(observation["group_id"])
        if group_id in allowed_groups:
            grouped.setdefault(group_id, []).append(float(scores[row]))
    ranked = []
    for group_id, values in grouped.items():
        strongest = sorted(values, reverse=True)[:top_views]
        ranked.append((group_id, float(np.mean(strongest)), len(values)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def _render(
    occupancy: np.ndarray,
    semantic_ids: np.ndarray,
    selected: list[dict[str, object]],
    output: Path,
    query: str,
) -> None:
    height, width = occupancy.shape
    canvas = np.full((height, width, 3), 128, dtype=np.uint8)
    canvas[occupancy >= 0] = 245
    canvas[occupancy >= 65] = 35
    palette = [
        (30, 225, 90),
        (255, 165, 35),
        (45, 145, 255),
        (225, 65, 190),
        (245, 225, 45),
    ]
    for rank, match in enumerate(selected):
        mask = semantic_ids == int(match["semantic_id"])
        colour = np.asarray(palette[rank % len(palette)], dtype=np.float32)
        canvas[mask] = (0.18 * canvas[mask] + 0.82 * colour).astype(np.uint8)
    map_image = Image.fromarray(canvas[::-1])
    legend_width = 390
    image = Image.new("RGB", (width + legend_width, height), (22, 22, 22))
    image.paste(map_image, (0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((width + 12, 10), f'query: "{query}"', fill="white")
    for rank, match in enumerate(selected):
        y = 38 + rank * 42
        colour = palette[rank % len(palette)]
        draw.rectangle((width + 12, y, width + 27, y + 15), fill=colour)
        draw.text(
            (width + 35, y - 2),
            f"#{rank + 1} {match['group_id']}",
            fill=(240, 240, 240),
        )
        draw.text(
            (width + 35, y + 15),
            f"score={float(match['score']):.3f} views={match['views']}",
            fill=(185, 185, 185),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True, help="output stem or _semantic.json")
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-views", type=int, default=3)
    parser.add_argument("--min-score", type=float)
    args = parser.parse_args()
    if args.top_k <= 0 or args.top_views <= 0:
        raise ValueError("top-k and top-views must be positive")

    map_path = _map_manifest(args.map)
    map_manifest = json.loads(map_path.read_text(encoding="utf-8"))
    if map_manifest.get("format") != "fact3r-depth-semantic-bev":
        raise ValueError(f"unsupported semantic BEV: {map_path}")
    index_path = Path(str(map_manifest["source_observation_index"]))
    _, index_manifest, embeddings = load_observation_index(index_path)
    started = perf_counter()
    if index_manifest.get("format") == "fact3r-qwen-embedding-observation-index":
        from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder

        print(f"loading {index_manifest['model']}...")
        encoder = Qwen3VLEmbeddingEncoder(
            str(index_manifest["model"]),
            device_map=args.device_map,
            dtype=args.dtype,
        )
    else:
        print(f"loading {index_manifest['model']}...")
        encoder = Siglip2Encoder(str(index_manifest["model"]), device=args.device)
    load_seconds = perf_counter() - started
    text_started = perf_counter()
    query_embedding = _normalise(encoder.encode_text([args.query]))
    text_seconds = perf_counter() - text_started
    scores = _normalise(embeddings) @ query_embedding[0]
    group_metadata = {
        str(item["group_id"]): item for item in map_manifest["groups"]
    }
    ranked = _rank_groups(
        scores,
        list(index_manifest["observations"]),
        set(group_metadata),
        top_views=args.top_views,
    )
    if args.min_score is not None:
        ranked = [item for item in ranked if item[1] >= args.min_score]
    selected = []
    for group_id, score, views in ranked[: args.top_k]:
        selected.append(
            {
                **group_metadata[group_id],
                "score": score,
                "views": views,
            }
        )
    grid_path = map_path.parent / str(map_manifest["grid_file"])
    with np.load(grid_path, allow_pickle=False) as payload:
        occupancy = np.array(payload["occupancy"], copy=True)
        semantic_ids = np.array(payload["semantic_ids"], copy=True)
    output = args.output or map_path.parent / f"{_slug(args.query)}_semantic_bev.png"
    _render(occupancy, semantic_ids, selected, output, args.query)
    result_path = output.with_suffix(".json")
    result_path.write_text(
        json.dumps(
            {
                "format": "fact3r-depth-semantic-bev-query",
                "version": 1,
                "query": args.query,
                "source_map": str(map_path.resolve()),
                "matches": selected,
                "image": str(output.resolve()),
                "timing": {
                    "model_load_seconds": load_seconds,
                    "text_encoding_seconds": text_seconds,
                    "ranking_seconds": perf_counter() - text_started - text_seconds,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if selected:
        for rank, match in enumerate(selected, start=1):
            print(
                f"rank {rank}: {match['group_id']} "
                f"score={float(match['score']):.3f} views={match['views']}"
            )
    else:
        print("no mapped semantic entity passed the requested score threshold")
    print(f"semantic query map: {output}")
    print(f"results:            {result_path}")


if __name__ == "__main__":
    main()
