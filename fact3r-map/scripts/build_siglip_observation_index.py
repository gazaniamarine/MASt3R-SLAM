#!/usr/bin/env python3
"""Encode every saved SAM mask into a searchable SigLIP observation index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.observation_index import (  # noqa: E402
    Siglip2Encoder,
    build_observation_index,
)


def _default_output(proposals: Path, mapping: Path | None) -> Path:
    stage = (
        "fact3r_siglip_pre_uot"
        if mapping is None
        else "fact3r_siglip_observations"
    )
    return proposals.parent.parent / stage / proposals.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument(
        "--mapping",
        type=Path,
        help="optional Hungarian/Sinkhorn/UOT map used to attach persistent IDs",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model", default="google/siglip2-base-patch16-224"
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, mps, cuda, or CUDA index"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-fraction", type=float, default=0.15)
    parser.add_argument("--outside-mask-alpha", type=float, default=0.20)
    args = parser.parse_args()

    output = args.output or _default_output(args.proposals, args.mapping)
    print(f"Loading {args.model} on {args.device}...")
    encoder = Siglip2Encoder(args.model, device=args.device)
    manifest_path = build_observation_index(
        keyframes=args.keyframes,
        proposals=args.proposals,
        mapping=args.mapping,
        output=output,
        encoder=encoder,
        batch_size=args.batch_size,
        context_fraction=args.context_fraction,
        outside_mask_alpha=args.outside_mask_alpha,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    timing = manifest["timing"]
    print(
        f"Encoded {manifest['observation_count']} masks from "
        f"{manifest['frame_count']} frames in "
        f"{timing['image_encoding_seconds']:.2f}s "
        f"({timing['observations_per_encoding_second']:.1f} masks/s); "
        f"model load={timing['model_load_seconds']:.2f}s"
    )
    print(f"Wrote SigLIP observation index to {manifest_path}")


if __name__ == "__main__":
    main()
