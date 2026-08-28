#!/usr/bin/env python3
"""Resident functional-area retrieval over complete-frame Qwen memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    manifest_path = args.memory / "manifest.json" if args.memory.is_dir() else args.memory
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "fact3r-qwen-scene-memory":
        raise ValueError("unsupported Qwen scene memory")
    root = manifest_path.parent
    frame_embeddings = np.load(root / manifest["frame_embedding_file"])
    area_embeddings = np.load(root / manifest["area_embedding_file"])
    encoder = Qwen3VLEmbeddingEncoder(
        manifest["model"],
        device_map=args.device_map,
        dtype=args.dtype,
        image_instruction="Represent this complete scene and its functional area.",
        query_instruction=(
            "Retrieve complete scenes and functional areas relevant to the query."
        ),
    )
    encoder.encode_text(["a functional area"])
    print(
        f"Ready: {manifest['area_count']} areas; "
        f"navigation poses={manifest['navigation_pose_available']}"
    )

    def query_once(query: str) -> None:
        started = perf_counter()
        query_embedding = encoder.encode_text([query])[0]
        text_seconds = perf_counter() - started
        scores = area_embeddings @ query_embedding
        selected = np.argsort(-scores)[: min(args.top_k, len(scores))]
        for rank, area_index in enumerate(selected, start=1):
            area = manifest["areas"][int(area_index)]
            indices = np.asarray(area["frame_indices"], dtype=np.int64)
            view_scores = frame_embeddings[indices] @ query_embedding
            best_index = int(indices[int(np.argmax(view_scores))])
            best_frame = manifest["frames"][best_index]
            print(
                f"rank {rank}: {area['area_id']} score={float(scores[area_index]):.3f} "
                f"frames={area['start_frame_id']}-{area['end_frame_id']} "
                f"time={area['start_timestamp']}-{area['end_timestamp']} "
                f"best_frame={best_frame['frame_id']} "
                f"best_time={best_frame['timestamp']} "
                f"image={best_frame['image_file']}"
            )
            if rank == 1:
                if manifest["navigation_pose_available"]:
                    print(
                        "revisit_pose_world_from_camera="
                        + json.dumps(best_frame["camera_pose_world_from_camera"])
                    )
                else:
                    print(
                        "revisit pose unavailable: this real-video export contains "
                        "image-only identity poses"
                    )
        print(f"latency: {perf_counter() - started:.3f}s (text={text_seconds:.3f}s)")

    if args.query:
        query_once(args.query)
        return
    print("Ask functional-area questions; Ctrl-D or 'quit' exits.")
    while True:
        try:
            query = input("area-query> ").strip()
        except EOFError:
            print()
            break
        if query.lower() in {"quit", "exit"}:
            break
        if query:
            query_once(query)


if __name__ == "__main__":
    main()
