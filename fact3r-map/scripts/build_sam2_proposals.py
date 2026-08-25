#!/usr/bin/env python3
"""Generate, filter, lift, and save SAM2 masks for exported MASt3R keyframes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.integrations.mast3r_slam import iter_exported_keyframes  # noqa: E402
from fact3r.proposals.mask_filter import MaskFilterConfig  # noqa: E402
from fact3r.proposals.proposal_pipeline import generate_lifted_proposals  # noqa: E402
from fact3r.proposals.sam2_generator import SAM2AutomaticMaskGenerator  # noqa: E402
from fact3r.proposals.sam2_official_generator import (  # noqa: E402
    SAM2OfficialMaskGenerator,
)
from fact3r.proposals.storage import save_frame_proposals  # noqa: E402


def _default_output(keyframe_directory: Path) -> Path:
    return keyframe_directory.parent.parent / "fact3r_sam2" / keyframe_directory.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backend",
        default="official",
        choices=("official", "transformers"),
        help="official: Meta's sam2 package; transformers: the HF mask-generation pipeline",
    )
    parser.add_argument(
        "--model",
        help="default: facebook/sam2-hiera-large (official) "
        "or facebook/sam2.1-hiera-large (transformers)",
    )
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--device", default="0", help="pipeline device, e.g. 0 or cpu")
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-threshold", type=float, default=0.88)
    parser.add_argument("--stability-score-threshold", type=float, default=0.95)
    parser.add_argument("--min-area-pixels", type=int, default=100)
    parser.add_argument("--min-area-fraction", type=float, default=0.001)
    parser.add_argument("--max-area-fraction", type=float, default=0.8)
    parser.add_argument("--erosion-pixels", type=int, default=1)
    parser.add_argument("--min-component-pixels", type=int, default=50)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.9)
    parser.add_argument("--min-geometry-confidence", type=float, default=0.0)
    parser.add_argument("--max-keyframes", type=int)
    args = parser.parse_args()

    output = args.output or _default_output(args.keyframes)
    output.mkdir(parents=True, exist_ok=True)
    if args.backend == "official":
        model = args.model or "facebook/sam2-hiera-large"
        generator = SAM2OfficialMaskGenerator(
            model,
            device=args.device,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_threshold=args.pred_iou_threshold,
            stability_score_threshold=args.stability_score_threshold,
        )
    else:
        model = args.model or "facebook/sam2.1-hiera-large"
        generator = SAM2AutomaticMaskGenerator(
            model,
            device=args.device,
            points_per_batch=args.points_per_batch,
            pred_iou_threshold=args.pred_iou_threshold,
            stability_score_threshold=args.stability_score_threshold,
        )
    filter_config = MaskFilterConfig(
        min_score=args.pred_iou_threshold,
        min_area_pixels=args.min_area_pixels,
        min_area_fraction=args.min_area_fraction,
        max_area_fraction=args.max_area_fraction,
        erosion_pixels=args.erosion_pixels,
        min_component_pixels=args.min_component_pixels,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        min_geometry_confidence=args.min_geometry_confidence,
    )

    summaries = []
    for keyframe_index, keyframe in enumerate(
        iter_exported_keyframes(args.keyframes)
    ):
        if args.max_keyframes is not None and keyframe_index >= args.max_keyframes:
            break
        generated = generate_lifted_proposals(keyframe, generator, filter_config)
        summary = save_frame_proposals(output, keyframe, generated)
        summaries.append(summary)
        print(
            f"frame {keyframe.frame_id}: kept {len(generated)} filtered SAM2 proposals"
        )

    run_manifest = {
        "format": "fact3r-sam2-proposals",
        "version": 1,
        "backend": args.backend,
        "model": model,
        "keyframe_export": str(args.keyframes.resolve()),
        "filter_config": {
            name: getattr(filter_config, name)
            for name in filter_config.__dataclass_fields__
        },
        "frame_count": len(summaries),
        "frames": [
            {
                "frame_id": summary["frame_id"],
                "proposal_count": summary["proposal_count"],
                "manifest": f"frame_{summary['frame_id']:06d}/manifest.json",
            }
            for summary in summaries
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote proposal map to {manifest_path}")


if __name__ == "__main__":
    main()

