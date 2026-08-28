#!/usr/bin/env python3
"""Benchmark shared Qwen3-VL frame tokens as UOT appearance descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.qwen_visual_memory import (  # noqa: E402
    Qwen3VLFrameEncoder,
    build_qwen_visual_index,
    visual_link_diagnostics,
)
from fact3r.semantics.observation_index import load_observation_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Encode each complete frame once with Qwen3-VL, pool every SAM mask "
            "from its spatial tokens, and measure association quality and speed."
        )
    )
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--tracklets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--attention-implementation")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--compare-index",
        type=Path,
        help="optional existing SigLIP index for the same observations",
    )
    args = parser.parse_args()

    print(f"Loading visual backbone {args.model} (no text generation)...")
    encoder = Qwen3VLFrameEncoder(
        args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attention_implementation=args.attention_implementation,
    )
    manifest_path = build_qwen_visual_index(
        keyframes=args.keyframes,
        proposals=args.proposals,
        tracklets=args.tracklets,
        output=args.output,
        encoder=encoder,
        max_frames=args.max_frames,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timing = manifest["timing"]
    diagnostic = manifest["diagnostics"]
    print(
        f"Encoded {manifest['frame_count']} frames / "
        f"{manifest['observation_count']} masks in "
        f"{timing['frame_encoding_seconds']:.2f}s: "
        f"{timing['frames_per_encoding_second']:.3f} FPS, "
        f"{timing['observations_per_encoding_second']:.1f} masks/s; "
        f"load={timing['model_load_seconds']:.2f}s"
    )
    print(
        "Association diagnostic: "
        f"linked median={diagnostic['linked_cosine_median']}, "
        "strongest unrelated median="
        f"{diagnostic['strongest_unrelated_cosine_median']}, "
        f"margin={diagnostic['median_link_margin']}, "
        f"top1={diagnostic['previous_frame_top1_accuracy']}"
    )
    if args.compare_index is not None:
        _, comparison_manifest, comparison_embeddings = load_observation_index(
            args.compare_index
        )
        qwen_keys = {
            (int(item["frame_id"]), str(item["proposal_id"]))
            for item in manifest["observations"]
        }
        selected_rows = [
            int(item["index"])
            for item in comparison_manifest["observations"]
            if (int(item["frame_id"]), str(item["proposal_id"])) in qwen_keys
        ]
        selected_observations = [
            comparison_manifest["observations"][row] for row in selected_rows
        ]
        if len(selected_rows) != len(manifest["observations"]):
            raise ValueError(
                "comparison index does not contain the same frame/proposal rows"
            )
        comparison = visual_link_diagnostics(
            comparison_embeddings[selected_rows], selected_observations
        )
        print(
            f"Comparison ({comparison_manifest['model']}): "
            f"linked median={comparison['linked_cosine_median']}, "
            "strongest unrelated median="
            f"{comparison['strongest_unrelated_cosine_median']}, "
            f"margin={comparison['median_link_margin']}, "
            f"top1={comparison['previous_frame_top1_accuracy']}"
        )
    print(f"Wrote experimental VLA visual index to {manifest_path}")
    print(
        "This index can be passed to run_image_uot_mapping.py as "
        "--appearance-index, but it cannot yet answer text queries."
    )


if __name__ == "__main__":
    main()
