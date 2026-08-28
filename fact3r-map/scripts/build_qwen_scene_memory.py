#!/usr/bin/env python3
"""Encode complete frames into functional semantic areas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder  # noqa: E402
from fact3r.semantics.scene_memory import build_scene_memory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--area-seconds", type=float, default=5.0)
    parser.add_argument("--visual-split-similarity", type=float, default=0.65)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    encoder = Qwen3VLEmbeddingEncoder(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        image_instruction="Represent this complete scene and its functional area.",
        query_instruction=(
            "Retrieve complete scenes and functional areas relevant to the query."
        ),
    )
    manifest_path = build_scene_memory(
        keyframes=args.keyframes,
        output=args.output,
        encoder=encoder,
        batch_size=args.batch_size,
        area_seconds=args.area_seconds,
        visual_split_similarity=args.visual_split_similarity,
        max_frames=args.max_frames,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timing = manifest["timing"]
    print(
        f"Built {manifest['area_count']} areas from {manifest['frame_count']} frames "
        f"at {timing['frames_per_second']:.2f} FPS; "
        f"navigation poses={manifest['navigation_pose_available']}"
    )
    print(f"Wrote Qwen scene memory to {manifest_path}")


if __name__ == "__main__":
    main()
