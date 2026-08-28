#!/usr/bin/env python3
"""Resident, retrieval-only Qwen object-memory query loop."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from time import perf_counter
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import load_observation_index  # noqa: E402
from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder  # noqa: E402


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
) -> None:
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
            (
                rank,
                group_ids[int(entity_index)],
                float(entity_scores[int(entity_index)]),
                float(observation_scores[best_local]),
                int(observation["frame_id"]),
                len(rows),
            )
        )
    rank_seconds = perf_counter() - rank_started
    for rank, group_id, score, best_score, frame_id, view_count in results:
        print(
            f"rank {rank}: {group_id} score={score:.3f} "
            f"best_view={best_score:.3f} frame={frame_id} views={view_count}"
        )
    print(
        f"latency: text={text_seconds:.3f}s rank={rank_seconds:.3f}s "
        f"total={perf_counter() - started:.3f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")

    _, manifest, embeddings = load_observation_index(args.index)
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

    if args.query is not None:
        _print_query(
            args.query,
            encoder=encoder,
            embeddings=embeddings,
            observations=observations,
            prototypes=prototypes,
            group_ids=group_ids,
            group_rows=group_rows,
            top_k=args.top_k,
        )
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
        _print_query(
            query,
            encoder=encoder,
            embeddings=embeddings,
            observations=observations,
            prototypes=prototypes,
            group_ids=group_ids,
            group_rows=group_rows,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
