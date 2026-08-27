#!/usr/bin/env python3
"""Compute adjacent-frame MASt3R reciprocal 2D matches without reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "thirdparty" / "mast3r"))
sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights",
        default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=8192)
    args = parser.parse_args()

    import torch
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from mast3r.fast_nn import fast_reciprocal_NNs
    from mast3r.model import AsymmetricMASt3R

    manifest = json.loads(
        (args.keyframes / "manifest.json").read_text(encoding="utf-8")
    )
    frames = manifest["keyframes"]
    if len(frames) < 2:
        raise ValueError("pair matching requires at least two sampled frames")
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = REPOSITORY_ROOT / weights
    print(f"Loading MASt3R matching weights from {weights}")
    model = AsymmetricMASt3R.from_pretrained(str(weights)).to(args.device)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    pairs: list[dict[str, object]] = []
    for pair_index, (source, target) in enumerate(zip(frames, frames[1:])):
        paths = [
            str(args.keyframes / source["rgb_file"]),
            str(args.keyframes / target["rgb_file"]),
        ]
        images = load_images(paths, size=512, verbose=False)
        with torch.inference_mode():
            result = inference(
                [tuple(images)], model, args.device, batch_size=1, verbose=False
            )
        descriptor_source = result["pred1"]["desc"].squeeze(0).detach()
        descriptor_target = result["pred2"]["desc"].squeeze(0).detach()
        source_xy, target_xy = fast_reciprocal_NNs(
            descriptor_source,
            descriptor_target,
            subsample_or_initxy1=args.subsample,
            device=args.device,
            dist="dot",
            block_size=args.block_size,
        )
        source_shape = tuple(int(value) for value in source["image_shape"])
        target_shape = tuple(int(value) for value in target["image_shape"])
        valid = (
            (source_xy[:, 0] >= 3)
            & (source_xy[:, 0] < source_shape[1] - 3)
            & (source_xy[:, 1] >= 3)
            & (source_xy[:, 1] < source_shape[0] - 3)
            & (target_xy[:, 0] >= 3)
            & (target_xy[:, 0] < target_shape[1] - 3)
            & (target_xy[:, 1] >= 3)
            & (target_xy[:, 1] < target_shape[0] - 3)
        )
        source_xy = np.asarray(source_xy[valid], dtype=np.int32)
        target_xy = np.asarray(target_xy[valid], dtype=np.int32)
        filename = f"pair_{pair_index:06d}.npz"
        np.savez_compressed(
            args.output / filename,
            source_xy=source_xy,
            target_xy=target_xy,
        )
        pairs.append(
            {
                "source_frame_id": int(source["frame_id"]),
                "target_frame_id": int(target["frame_id"]),
                "file": filename,
                "match_count": len(source_xy),
            }
        )
        print(
            f"pair {source['frame_id']}->{target['frame_id']}: "
            f"{len(source_xy)} reciprocal matches"
        )
    output_manifest = {
        "format": "fact3r-mast3r-pair-matches",
        "version": 1,
        "source_keyframes": str(args.keyframes.resolve()),
        "weights": str(weights),
        "pairs": pairs,
    }
    path = args.output / "manifest.json"
    path.write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote pairwise feature matches to {path}")


if __name__ == "__main__":
    main()
