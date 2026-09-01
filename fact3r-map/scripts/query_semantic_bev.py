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
from fact3r.semantics.semantic_goal import group_cell_counts  # noqa: E402


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


def _query_prompts(query: str, *, ensemble: bool = True) -> list[str]:
    """Create generic visual-category fallbacks without object-specific rules."""

    cleaned = " ".join(query.strip().split())
    if not cleaned:
        raise ValueError("query must not be empty")
    if not ensemble:
        return [cleaned]
    words = [
        "".join(character for character in word if character.isalnum() or character == "-")
        for word in cleaned.lower().split()
    ]
    words = [word for word in words if word]
    without_article = words[1:] if words and words[0] in {"a", "an", "the"} else words
    base = " ".join(without_article) or cleaned
    head = without_article[-1] if without_article else cleaned
    article_for_base = "an" if base[:1].lower() in "aeiou" else "a"
    article_for_head = "an" if head[:1].lower() in "aeiou" else "a"
    prompts = [cleaned, base, f"{article_for_base} {base}", head, f"{article_for_head} {head}"]
    return list(dict.fromkeys(prompt for prompt in prompts if prompt))


def _fuse_prompt_scores(
    observation_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    *,
    agreement_prompts: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse prompt variants by their strongest agreement, not a brittle mean."""

    if agreement_prompts <= 0:
        raise ValueError("agreement_prompts must be positive")
    scores = _normalise(observation_embeddings) @ _normalise(text_embeddings).T
    count = min(agreement_prompts, scores.shape[1])
    strongest = np.partition(scores, scores.shape[1] - count, axis=1)[:, -count:]
    fused = np.mean(strongest, axis=1)
    winning_prompt = np.argmax(scores, axis=1)
    return (
        np.asarray(fused, dtype=np.float32),
        np.asarray(winning_prompt, dtype=np.int32),
        np.asarray(scores, dtype=np.float32),
    )


def _rank_groups(
    scores: np.ndarray,
    observations: list[dict[str, object]],
    allowed_groups: set[str],
    *,
    top_views: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[float, int]]] = {}
    for row, observation in enumerate(observations):
        group_id = str(observation["group_id"])
        if group_id in allowed_groups:
            grouped.setdefault(group_id, []).append((float(scores[row]), row))
    ranked = []
    for group_id, rows in grouped.items():
        strongest = sorted(rows, reverse=True)[:top_views]
        ranked.append(
            {
                "group_id": group_id,
                "score": float(np.mean([item[0] for item in strongest])),
                "views": len(rows),
                "best_observation_index": int(strongest[0][1]),
                "best_view_score": float(strongest[0][0]),
            }
        )
    return sorted(ranked, key=lambda item: float(item["score"]), reverse=True)


def _load_observation_image(
    observation: dict[str, object], index_manifest: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    frame_id = int(observation["frame_id"])
    keyframe_directory = Path(str(index_manifest["source_keyframes"]))
    keyframe_manifest = json.loads(
        (keyframe_directory / "manifest.json").read_text(encoding="utf-8")
    )
    entry = next(
        (
            item
            for item in keyframe_manifest["keyframes"]
            if int(item["frame_id"]) == frame_id
        ),
        None,
    )
    if entry is None:
        raise ValueError(f"source keyframe {frame_id} is missing")
    rgb_file = entry.get("rgb_file")
    if rgb_file is not None and (keyframe_directory / str(rgb_file)).exists():
        rgb = np.asarray(Image.open(keyframe_directory / str(rgb_file)).convert("RGB"))
    else:
        with np.load(keyframe_directory / str(entry["file"]), allow_pickle=False) as data:
            rgb = np.array(data["rgb"], dtype=np.uint8, copy=True)
    proposal_directory = Path(str(index_manifest["source_proposals"]))
    with np.load(
        proposal_directory / str(observation["mask_file"]), allow_pickle=False
    ) as data:
        mask = np.array(data["mask"], dtype=bool, copy=True)
    if mask.shape != rgb.shape[:2]:
        mask = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize(
                (rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST
            ),
            dtype=bool,
        )
    return rgb, mask


def _render_observed_frame(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    query: str,
    group_id: str,
    frame_id: int,
    score: float,
    rank: int,
    output: Path,
) -> None:
    canvas = np.asarray(rgb, dtype=np.float32).copy()
    colour = np.asarray([25.0, 235.0, 90.0], dtype=np.float32)
    canvas[mask] = 0.40 * canvas[mask] + 0.60 * colour
    boundary = mask & ~(
        np.roll(mask, 1, axis=0)
        & np.roll(mask, -1, axis=0)
        & np.roll(mask, 1, axis=1)
        & np.roll(mask, -1, axis=1)
    )
    canvas[boundary] = np.asarray([0.0, 255.0, 70.0])
    frame = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    banner_height = 58
    image = Image.new("RGB", (frame.width, frame.height + banner_height), (18, 18, 18))
    image.paste(frame, (0, banner_height))
    draw = ImageDraw.Draw(image)
    draw.text(
        (10, 8),
        f'#{rank} query "{query}" | frame {frame_id} | score {score:.3f}',
        fill="white",
    )
    draw.text((10, 30), group_id, fill=(80, 245, 125))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=94)


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
    parser.add_argument(
        "--exact-query",
        action="store_true",
        help="disable automatic phrase/head-noun prompt fusion",
    )
    parser.add_argument(
        "--include-unmapped",
        action="store_true",
        help="also rank entities that won no BEV cell. They have no position, "
        "so nothing can be navigated to them; kept for retrieval evaluation, "
        "where recall over the whole memory is the quantity of interest.",
    )
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
    query_prompts = _query_prompts(args.query, ensemble=not args.exact_query)
    query_embeddings = encoder.encode_text(query_prompts)
    text_seconds = perf_counter() - text_started
    scores, winning_prompt_rows, _ = _fuse_prompt_scores(
        embeddings,
        query_embeddings,
        agreement_prompts=1 if args.exact_query else 2,
    )
    print("query prompts: " + " | ".join(query_prompts))
    group_metadata = {
        str(item["group_id"]): item for item in map_manifest["groups"]
    }
    grid_path = map_path.parent / str(map_manifest["grid_file"])
    with np.load(grid_path, allow_pickle=False) as payload:
        occupancy = np.array(payload["occupancy"], copy=True)
        semantic_ids = np.array(payload["semantic_ids"], copy=True)
    cell_counts = group_cell_counts(semantic_ids, map_manifest["groups"])
    candidates = set(group_metadata)
    if not args.include_unmapped:
        candidates = {group for group in candidates if cell_counts.get(group, 0) > 0}
        print(
            f"on-map filter: {len(candidates)} of {len(group_metadata)} entities "
            f"hold a BEV cell; {len(group_metadata) - len(candidates)} dropped"
        )
    if not candidates:
        raise SystemExit(
            "no entity in this map holds a BEV cell -- there is nothing to "
            "rank. Re-run the fuse stage, or pass --include-unmapped to score "
            "the memory without positions."
        )
    ranked = _rank_groups(
        scores,
        list(index_manifest["observations"]),
        candidates,
        top_views=args.top_views,
    )
    if args.min_score is not None:
        ranked = [item for item in ranked if float(item["score"]) >= args.min_score]
    selected = []
    index_observations = list(index_manifest["observations"])
    for group in ranked[: args.top_k]:
        group_id = str(group["group_id"])
        observation = index_observations[int(group["best_observation_index"])]
        best_observation_index = int(group["best_observation_index"])
        selected.append(
            {
                **group_metadata[group_id],
                **group,
                "cell_count": int(cell_counts.get(group_id, 0)),
                "best_observation": {
                    "observation_index": int(group["best_observation_index"]),
                    "frame_id": int(observation["frame_id"]),
                    "proposal_id": str(observation["proposal_id"]),
                    "timestamp": observation.get("timestamp"),
                    "mask_file": str(observation["mask_file"]),
                    "winning_prompt": query_prompts[
                        int(winning_prompt_rows[best_observation_index])
                    ],
                },
            }
        )
    output = args.output or map_path.parent / f"{_slug(args.query)}_semantic_bev.png"
    _render(occupancy, semantic_ids, selected, output, args.query)
    frame_directory = output.parent / f"{output.stem}_observed_frames"
    for rank, match in enumerate(selected, start=1):
        observation = index_observations[int(match["best_observation_index"])]
        rgb, mask = _load_observation_image(observation, index_manifest)
        frame_path = frame_directory / (
            f"rank_{rank:02d}_frame_{int(observation['frame_id']):06d}.jpg"
        )
        _render_observed_frame(
            rgb,
            mask,
            query=args.query,
            group_id=str(match["group_id"]),
            frame_id=int(observation["frame_id"]),
            score=float(match["best_view_score"]),
            rank=rank,
            output=frame_path,
        )
        match["best_observation"]["image"] = str(frame_path.resolve())
    result_path = output.with_suffix(".json")
    result_path.write_text(
        json.dumps(
            {
                "format": "fact3r-depth-semantic-bev-query",
                "version": 1,
                "query": args.query,
                "query_prompts": query_prompts,
                "on_map_only": not args.include_unmapped,
                "candidate_entities": len(candidates),
                "dropped_entities": len(group_metadata) - len(candidates),
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
                f"score={float(match['score']):.3f} views={match['views']} "
                f"cells={match['cell_count']} "
                f"frame={match['best_observation']['frame_id']} "
                f"prompt={match['best_observation']['winning_prompt']!r}"
            )
            print(f"  observed frame: {match['best_observation']['image']}")
    else:
        print("no mapped semantic entity passed the requested score threshold")
    print(f"semantic query map: {output}")
    print(f"results:            {result_path}")


if __name__ == "__main__":
    main()
