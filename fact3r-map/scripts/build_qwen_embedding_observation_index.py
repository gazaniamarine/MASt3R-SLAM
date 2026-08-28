#!/usr/bin/env python3
"""Build a text-queryable mask index with Qwen3-VL-Embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import build_observation_index  # noqa: E402
from fact3r.semantics.qwen_embedding import Qwen3VLEmbeddingEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--tracklets", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=128 * 128)
    parser.add_argument("--max-pixels", type=int, default=224 * 224)
    parser.add_argument("--attention-implementation")
    parser.add_argument("--context-fraction", type=float, default=0.05)
    parser.add_argument("--outside-mask-alpha", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Loading semantic retriever {args.model} (no generation)...")
    encoder = Qwen3VLEmbeddingEncoder(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attention_implementation=args.attention_implementation,
    )
    manifest_path = build_observation_index(
        keyframes=args.keyframes,
        proposals=args.proposals,
        tracklets=args.tracklets,
        mapping=args.mapping,
        output=args.output,
        encoder=encoder,
        batch_size=args.batch_size,
        context_fraction=args.context_fraction,
        outside_mask_alpha=args.outside_mask_alpha,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timing = manifest["timing"]
    print(
        f"Encoded {manifest['observation_count']} masks in "
        f"{timing['image_encoding_seconds']:.2f}s "
        f"({timing['observations_per_encoding_second']:.1f} masks/s); "
        f"load={timing['model_load_seconds']:.2f}s"
    )
    print(f"Wrote text-queryable Qwen index to {manifest_path}")


if __name__ == "__main__":
    main()
