#!/usr/bin/env python3
"""Caption the most prominent objects in every keyframe with BLIP.

For each exported keyframe: run SAM2 to get class-agnostic mask proposals,
rank them by score and size (bigger, higher-confidence masks first -- a
tiny corner speck should not outrank a large, clear object), keep the top
--top-k, crop each with GOAT's "bbox + padding" context (Chang et al. 2023
found padded crops beat both bare bounding boxes and full images for
region-to-text matching), and caption each padded crop with BLIP.

This is the description tier feeding both the entity-captioning pass (Stage
A: caption a persistent entity's evidence crops) and route-instruction
synthesis (caption the landmark crop at each leg of a return path) -- SigLIP
stays the retrieval/grouping mechanism, BLIP only ever describes crops it is
handed, and never decides which crops matter.

    conda run -n SAM2 python3 fact3r-map/scripts/caption_frame_objects_blip.py \\
        --keyframes logs/fact3r_real_uot/my_run/frames \\
        --output logs/fact3r_real_uot/my_run/blip_object_captions.json \\
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.integrations.mast3r_slam import iter_exported_keyframes  # noqa: E402
from fact3r.proposals.mask_generator import MaskProposal2D  # noqa: E402
from fact3r.proposals.sam2_official_generator import SAM2OfficialMaskGenerator  # noqa: E402
from fact3r.semantics.blip_captioner import BlipCaptioner  # noqa: E402


def rank_top_k(proposals: List[MaskProposal2D], k: int) -> List[MaskProposal2D]:
    """Largest, highest-score proposals first -- same heuristic as
    scripts/select_pixel_goal.py's candidate ranking, kept separate here
    since it's a few lines and the two scripts have no other shared need."""

    scored = []
    for proposal in proposals:
        if proposal.bounding_box_xyxy is None:
            continue
        size_bonus = 1.0 + math.log1p(max(proposal.area, 0))
        scored.append((proposal.score * size_bonus, proposal))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [proposal for _, proposal in scored[:k]]


def padded_crop(
    rgb: NDArray[np.uint8], bbox_xyxy: NDArray[np.floating], pad_frac: float,
) -> Tuple[NDArray[np.uint8], Tuple[float, float, float, float]]:
    """GOAT's "bbox + pad" context window: on average the best-performing
    crop choice for region-to-text matching (Chang et al. 2023, Table 2)."""

    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    pad_x, pad_y = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(float(width), x1 + pad_x)
    y1 = min(float(height), y1 + pad_y)
    crop = rgb[int(y0):int(y1), int(x0):int(x1)]
    return crop, (x0, y0, x1, y1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5,
                        help="most prominent objects to caption per frame")
    parser.add_argument("--sam2-model-id", default="facebook/sam2-hiera-large")
    parser.add_argument("--sam2-device", default="0")
    parser.add_argument("--points-per-side", type=int, default=24)
    parser.add_argument("--blip-model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--blip-device", default="auto")
    parser.add_argument("--bbox-padding-frac", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="caption only the first N keyframes (for a quick check)")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.bbox_padding_frac < 0:
        raise ValueError("bbox-padding-frac must not be negative")

    print(f"Loading SAM2 ({args.sam2_model_id})...")
    mask_generator = SAM2OfficialMaskGenerator(
        model_id=args.sam2_model_id, device=args.sam2_device,
        points_per_side=args.points_per_side,
    )
    print(f"Loading BLIP ({args.blip_model})...")
    captioner = BlipCaptioner(args.blip_model, device=args.blip_device)
    captioner.load()
    print(f"BLIP ready: load={captioner.load_seconds:.2f}s")

    frames_out = []
    started = perf_counter()
    frame_count = 0
    for keyframe in iter_exported_keyframes(args.keyframes):
        if args.max_frames is not None and frame_count >= args.max_frames:
            break
        frame_count += 1
        rgb = np.asarray(keyframe.rgb)
        proposals = mask_generator.generate(rgb, frame_id=keyframe.frame_id)
        top = rank_top_k(proposals, args.top_k)

        objects = []
        for proposal in top:
            crop, padded_bbox = padded_crop(rgb, proposal.bounding_box_xyxy, args.bbox_padding_frac)
            if crop.size == 0:
                continue
            caption = captioner.caption(crop)
            objects.append({
                "proposal_id": proposal.proposal_id,
                "score": float(proposal.score),
                "area": int(proposal.area),
                "bbox_xyxy": [float(value) for value in proposal.bounding_box_xyxy],
                "padded_bbox_xyxy": [float(value) for value in padded_bbox],
                "caption": caption,
            })
        frames_out.append({"frame_id": int(keyframe.frame_id), "objects": objects})
        print(f"  frame {keyframe.frame_id}: " + " | ".join(o["caption"] for o in objects))

    payload = {
        "format": "fact3r-blip-object-captions",
        "version": 1,
        "keyframes": str(args.keyframes),
        "sam2_model_id": args.sam2_model_id,
        "blip_model": args.blip_model,
        "top_k": args.top_k,
        "bbox_padding_frac": args.bbox_padding_frac,
        "frames": frames_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elapsed = perf_counter() - started
    print(f"Captioned {frame_count} frames in {elapsed:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
