#!/usr/bin/env python3
"""Resident, retrieval-only Qwen object-memory query loop."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path
import re
from time import perf_counter
import sys

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import load_observation_index  # noqa: E402
from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder  # noqa: E402
from fact3r.integrations.mast3r_slam import iter_exported_keyframes  # noqa: E402
from fact3r.visualization.association import mask_boundary  # noqa: E402


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(values / norms, dtype=np.float32)


def _entity_prototypes(
    embeddings: np.ndarray, observations: list[dict[str, object]]
) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, observation in enumerate(observations):
        group_id = observation.get("entity_id") or observation.get("track_id")
        if group_id is not None:
            grouped[str(group_id)].append(row)
    if not grouped:
        raise ValueError("semantic memory contains no persistent entities or tracks")
    group_ids = sorted(grouped)
    rows = [np.asarray(grouped[group_id], dtype=np.int64) for group_id in group_ids]
    prototypes = np.stack(
        [np.mean(embeddings[group_rows], axis=0) for group_rows in rows]
    )
    return _normalise_rows(prototypes), group_ids, rows


def _slug(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return compact[:80] or "query"


def _resize_to_width(image: Image.Image, maximum_width: int) -> Image.Image:
    if image.width <= maximum_width:
        return image.copy()
    height = max(1, round(image.height * maximum_width / image.width))
    return image.resize((maximum_width, height), Image.Resampling.LANCZOS)


def _render_results(
    query: str,
    results: list[dict[str, object]],
    *,
    manifest: dict[str, object],
    observations: list[dict[str, object]],
    output: Path,
    render_top_k: int,
) -> None:
    selected = results[: min(render_top_k, len(results))]
    if not selected:
        return
    needed_frames = {
        int(observations[int(result["observation_index"])]["frame_id"])
        for result in selected
    }
    keyframe_images = {
        keyframe.frame_id: np.array(keyframe.rgb, copy=True)
        for keyframe in iter_exported_keyframes(str(manifest["source_keyframes"]))
        if keyframe.frame_id in needed_frames
    }
    proposal_directory = Path(str(manifest["source_proposals"]))
    output.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[Path, dict[str, object]]] = []
    serialisable_results: list[dict[str, object]] = []
    for result in selected:
        observation = observations[int(result["observation_index"])]
        frame_id = int(observation["frame_id"])
        rgb = keyframe_images.get(frame_id)
        if rgb is None:
            raise ValueError(f"keyframe {frame_id} is missing from the source export")
        with np.load(
            proposal_directory / str(observation["mask_file"]),
            allow_pickle=False,
        ) as evidence:
            mask = np.array(evidence["mask"], dtype=bool, copy=True)
        if mask.shape != rgb.shape[:2]:
            raise ValueError("stored proposal mask and keyframe shape do not match")
        canvas = rgb.astype(np.float32)
        canvas[mask] = 0.55 * canvas[mask] + 0.45 * np.asarray(
            [40.0, 240.0, 100.0], dtype=np.float32
        )
        canvas[mask_boundary(mask)] = [40.0, 255.0, 80.0]
        header_height = 72
        image = Image.new(
            "RGB", (rgb.shape[1], rgb.shape[0] + header_height), (20, 20, 20)
        )
        image.paste(Image.fromarray(canvas.astype(np.uint8)), (0, header_height))
        draw = ImageDraw.Draw(image)
        draw.text(
            (8, 8),
            f'query "{query}" | rank {result["rank"]} | {result["group_id"]}',
            fill=(245, 245, 245),
        )
        draw.text(
            (8, 36),
            f'entity={float(result["score"]):.3f} | '
            f'best view={float(result["best_view"]):.3f} | '
            f'frame={frame_id} | views={result["views"]}',
            fill=(190, 230, 200),
        )
        filename = (
            f'rank_{int(result["rank"]):02d}_{_slug(str(result["group_id"]))}_'
            f"frame_{frame_id:06d}.jpg"
        )
        path = output / filename
        image.save(path, quality=92)
        public_result = {
            key: value for key, value in result.items() if key != "observation_index"
        }
        public_result.update(
            {
                "proposal_id": observation.get("proposal_id"),
                "timestamp": observation.get("timestamp"),
                "image": filename,
            }
        )
        serialisable_results.append(public_result)
        rendered.append((path, public_result))

    thumbs = []
    for path, _ in rendered:
        with Image.open(path) as source:
            thumbs.append(_resize_to_width(source.convert("RGB"), 480))
    columns = min(3, len(thumbs))
    tile_height = max(tile.height for tile in thumbs)
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 480, rows * tile_height), (25, 25, 25))
    for index, tile in enumerate(thumbs):
        sheet.paste(tile, ((index % columns) * 480, (index // columns) * tile_height))
    sheet.save(output / "contact_sheet.jpg", quality=90)

    cards = "\n".join(
        "<figure><img src=\"{}\"><figcaption>rank {} · {} · frame {} · "
        "entity {:.3f} · best view {:.3f}</figcaption></figure>".format(
            escape(path.name),
            result["rank"],
            escape(str(result["group_id"])),
            result["frame_id"],
            float(result["score"]),
            float(result["best_view"]),
        )
        for path, result in rendered
    )
    (output / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Qwen memory query</title>"
        "<style>body{background:#161616;color:#eee;font-family:sans-serif}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:12px}"
        "figure{margin:0;background:#222;padding:8px}img{width:100%;height:auto}"
        "figcaption{padding-top:6px}</style>"
        f"<h1>Query: {escape(query)}</h1><main>{cards}</main>",
        encoding="utf-8",
    )
    (output / "results.json").write_text(
        json.dumps(
            {
                "format": "fact3r-qwen-live-query-results",
                "version": 1,
                "query": query,
                "results": serialisable_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _print_query(
    query: str,
    *,
    encoder: Qwen3VLEmbeddingEncoder,
    embeddings: np.ndarray,
    observations: list[dict[str, object]],
    prototypes: np.ndarray,
    group_ids: list[str],
    group_rows: list[np.ndarray],
    top_k: int,
) -> list[dict[str, object]]:
    started = perf_counter()
    text_started = perf_counter()
    query_vector = encoder.encode_text([query])[0]
    text_seconds = perf_counter() - text_started
    rank_started = perf_counter()
    entity_scores = prototypes @ query_vector
    selected = np.argsort(-entity_scores)[: min(top_k, len(entity_scores))]
    results = []
    for rank, entity_index in enumerate(selected, start=1):
        rows = group_rows[int(entity_index)]
        observation_scores = embeddings[rows] @ query_vector
        best_local = int(np.argmax(observation_scores))
        best_row = int(rows[best_local])
        observation = observations[best_row]
        results.append(
            {
                "rank": rank,
                "group_id": group_ids[int(entity_index)],
                "score": float(entity_scores[int(entity_index)]),
                "best_view": float(observation_scores[best_local]),
                "frame_id": int(observation["frame_id"]),
                "views": len(rows),
                "observation_index": best_row,
            }
        )
    rank_seconds = perf_counter() - rank_started
    for result in results:
        print(
            f'rank {result["rank"]}: {result["group_id"]} '
            f'score={float(result["score"]):.3f} '
            f'best_view={float(result["best_view"]):.3f} '
            f'frame={result["frame_id"]} views={result["views"]}'
        )
    print(
        f"latency: text={text_seconds:.3f}s rank={rank_seconds:.3f}s "
        f"total={perf_counter() - started:.3f}s"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        help="gallery directory; defaults to INDEX/queries/QUERY",
    )
    parser.add_argument(
        "--render-top-k",
        type=int,
        default=5,
        help="save the strongest masked view for this many ranked entities",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.render_top_k <= 0:
        raise ValueError("render-top-k must be positive")

    manifest_path, manifest, embeddings = load_observation_index(args.index)
    if manifest.get("format") != "fact3r-qwen-embedding-observation-index":
        raise ValueError("live Qwen querying requires a Qwen embedding index")
    observations = list(manifest["observations"])
    prototypes, group_ids, group_rows = _entity_prototypes(
        embeddings, observations
    )
    print(f"Loading {manifest['model']} once...")
    encoder = Qwen3VLEmbeddingEncoder(
        str(manifest["model"]), device_map=args.device_map, dtype=args.dtype
    )
    warmup_started = perf_counter()
    encoder.encode_text(["an object"])
    print(
        f"Ready: {len(group_ids)} persistent groups, "
        f"load={encoder.load_seconds:.2f}s, warmup={perf_counter() - warmup_started:.2f}s"
    )

    def run_query(query: str, *, interactive: bool) -> None:
        results = _print_query(
            query,
            encoder=encoder,
            embeddings=embeddings,
            observations=observations,
            prototypes=prototypes,
            group_ids=group_ids,
            group_rows=group_rows,
            top_k=args.top_k,
        )
        if args.no_render:
            return
        if args.output is None:
            output = manifest_path.parent / "queries" / _slug(query)
        elif interactive:
            output = args.output / _slug(query)
        else:
            output = args.output
        render_started = perf_counter()
        _render_results(
            query,
            results,
            manifest=manifest,
            observations=observations,
            output=output,
            render_top_k=args.render_top_k,
        )
        print(f"gallery: {output / 'index.html'}")
        print(f"contact sheet: {output / 'contact_sheet.jpg'}")
        print(f"render latency: {perf_counter() - render_started:.3f}s")

    if args.query is not None:
        run_query(args.query, interactive=False)
        return

    print("Enter queries without restarting the model; Ctrl-D or 'quit' exits.")
    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            print()
            break
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        run_query(query, interactive=True)


if __name__ == "__main__":
    main()
