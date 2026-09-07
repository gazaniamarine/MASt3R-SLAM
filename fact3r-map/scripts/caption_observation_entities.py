#!/usr/bin/env python3
"""Add SmolVLM captions to an EXISTING SigLIP observation index -- SAM2,
SigLIP, and UOT association are not touched, re-run, or re-derived here.

This reads an observation index already built by
fact3r.semantics.observation_index.build_observation_index() (SAM2 mask
proposals -> SigLIP embeddings -> UOT association assigns each observation
a persistent group_id/entity_id) and adds a fourth tier on top -- one
SmolVLM caption per persistent entity, using the crop that same pipeline
already localized (its own stored bounding_box_xyxy, on its own source
keyframe). Nothing about manifest.json, embeddings.npy, SAM2, SigLIP, or
UOT changes; this only ever writes a NEW sibling file.

Captions one representative view per entity (its highest-quality
observation: proposal_score weighted by mask size, same heuristic
live_caption_video.py's best_per_side already uses), not every observation
-- a real UOT-associated run can have tens of thousands of raw observations
for a few thousand persistent entities, and most entities only need one
good look to describe. --min-observations filters out single-frame noise
detections (this pipeline's own rank_semantic_entity_groups uses the same
kind of supporting-view-count gate, min_supporting_views, for the same
reason).

Representative-view selection also downweights elongated "sliver" boxes
(aspect ratio > 2, softly penalized rather than hard-excluded): visually
verified against real crops (fact3r-map/scripts/visualize_entity_captions.py)
that a first pass of this script -- picking the single highest score*size
view with no shape awareness -- was choosing thin wall-edge/doorframe/floor-
strip fragments for several entities, which SmolVLM then confidently
mis-captioned ("Door, pipe, wall." for a bare wall sliver, "Chair, computer,
desk." for a floor strip) rather than describing what was actually in the
crop. Since a persistent entity usually has many other observations, a
same-quality but better-shaped alternative view is picked in preference to
a sliver whenever one is available.

    conda run -n SAM2 python3 fact3r-map/scripts/caption_observation_entities.py \\
        --index logs/fact3r_real_uot/full_video_qwen_complete/siglip_observations \\
        --output logs/fact3r_real_uot/full_video_qwen_complete/siglip_observations/entity_captions.jsonl \\
        --min-observations 10 --max-entities 15
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.integrations.mast3r_slam import iter_exported_keyframes  # noqa: E402
from fact3r.semantics.observation_index import load_observation_index  # noqa: E402
from fact3r.semantics.smolvlm_captioner import SmolVLMCaptioner  # noqa: E402

OBJECT_LABEL_PROMPT = (
    "List the prominent objects in this image, 1 to 3 of them, separated by commas. "
    "Short labels only (1-3 words each), not full sentences. If nothing is clearly "
    "identifiable, say \"unclear\"."
)


def group_observations(observations: list) -> dict:
    groups: dict = {}
    for observation in observations:
        groups.setdefault(str(observation["group_id"]), []).append(observation)
    return groups


def _aspect_penalty(bbox_xyxy) -> float:
    """1.0 for a roughly square/normal box, falling off for elongated
    slivers (aspect ratio > 2) -- soft, not a hard cutoff, so a genuinely
    elongated real object (a shelf, a railing) still wins when it's the
    only view available."""

    if bbox_xyxy is None:
        return 1.0
    x0, y0, x1, y1 = bbox_xyxy
    width, height = max(1.0, x1 - x0), max(1.0, y1 - y0)
    aspect = max(width, height) / min(width, height)
    return 1.0 / max(1.0, aspect - 1.0)


def representative_observation(observations: list) -> dict:
    """Highest score*size*shape observation in a group -- same size_bonus
    heuristic as live_caption_video.py's best_per_side, applied here per
    persistent entity instead of per left/right side, plus an aspect-ratio
    penalty (see _aspect_penalty) that steers away from sliver crops when a
    better-shaped alternative view of the same entity exists."""

    def quality(observation):
        area = float(observation.get("mask_area", 0.0))
        size_bonus = 1.0 + math.log1p(max(area, 0.0))
        shape_bonus = _aspect_penalty(observation.get("bounding_box_xyxy"))
        return float(observation.get("proposal_score", 0.0)) * size_bonus * shape_bonus

    return max(observations, key=quality)


def padded_crop(rgb: np.ndarray, bbox_xyxy, pad_frac: float = 0.2) -> np.ndarray:
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    pad_x, pad_y = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(float(width), x1 + pad_x)
    y1 = min(float(height), y1 + pad_y)
    return rgb[int(y0):int(y1), int(x0):int(x1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="observation index dir or manifest.json path")
    parser.add_argument("--output", type=Path, required=True, help="entity_captions.jsonl to write")
    parser.add_argument("--min-observations", type=int, default=2,
                        help="skip entities seen in fewer frames than this (filters single-frame noise)")
    parser.add_argument("--max-entities", type=int, default=None,
                        help="caption only the top-N entities by observation count (omit for all)")
    parser.add_argument("--pad-frac", type=float, default=0.2)
    parser.add_argument("--smolvlm-model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--smolvlm-device", default="auto")
    args = parser.parse_args()

    if args.min_observations <= 0:
        raise ValueError("min-observations must be positive")

    manifest_path, manifest, _embeddings = load_observation_index(args.index)
    print(f"loaded {manifest_path}: {manifest['observation_count']} observations, "
          f"{manifest['frame_count']} frames")

    groups = group_observations(manifest["observations"])
    eligible = {gid: obs for gid, obs in groups.items() if len(obs) >= args.min_observations}
    print(f"{len(groups)} entities total, {len(eligible)} with >= {args.min_observations} observations")

    ranked = sorted(eligible.items(), key=lambda item: len(item[1]), reverse=True)
    if args.max_entities is not None:
        ranked = ranked[: args.max_entities]
    print(f"captioning {len(ranked)} entities")

    representatives = {gid: representative_observation(obs) for gid, obs in ranked}
    needed_frames = {int(observation["frame_id"]) for observation in representatives.values()}
    print(f"loading {len(needed_frames)} source keyframes from {manifest['source_keyframes']} ...")
    keyframe_images = {
        keyframe.frame_id: np.array(keyframe.rgb, copy=True)
        for keyframe in iter_exported_keyframes(manifest["source_keyframes"])
        if keyframe.frame_id in needed_frames
    }

    print(f"Loading SmolVLM ({args.smolvlm_model})...")
    captioner = SmolVLMCaptioner(args.smolvlm_model, device=args.smolvlm_device)
    captioner.load()
    print(f"SmolVLM ready: load={captioner.load_seconds:.2f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for group_id, observation in representatives.items():
            frame_id = int(observation["frame_id"])
            rgb = keyframe_images.get(frame_id)
            if rgb is None or observation.get("bounding_box_xyxy") is None:
                continue
            crop = padded_crop(rgb, observation["bounding_box_xyxy"], args.pad_frac)
            if crop.size == 0:
                continue
            caption_started = time.perf_counter()
            caption = captioner.caption(Image.fromarray(crop), prompt=OBJECT_LABEL_PROMPT)
            caption_s = time.perf_counter() - caption_started

            record = {
                "group_id": group_id,
                "entity_id": observation.get("entity_id"),
                "track_id": observation.get("track_id"),
                "observation_count": len(eligible[group_id]),
                "representative_frame_id": frame_id,
                "representative_proposal_id": observation.get("proposal_id"),
                "bbox_xyxy": observation.get("bounding_box_xyxy"),
                "caption": caption,
                "caption_seconds": caption_s,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            written += 1
            elapsed = time.perf_counter() - started
            print(f"[{elapsed:6.1f}s | {written}/{len(ranked)}] {group_id} "
                  f"(frame {frame_id}, {len(eligible[group_id])} views, {caption_s:.2f}s): {caption}")

    total_s = time.perf_counter() - started
    print(f"\n{written} entity captions in {total_s:.1f}s -> {args.output}")
    print(f"source index untouched: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
