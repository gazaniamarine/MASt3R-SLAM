#!/usr/bin/env python3
"""Draw each entity's box + SmolVLM caption onto its actual source keyframe --
the missing piece for eyeballing whether caption_observation_entities.py got
it right. That script never saved a crop or an image anywhere; this reads
its output (entity_captions.jsonl) back, re-derives the same padded crop
from the same source keyframe + bbox (nothing re-inferred, purely visual),
and saves one annotated PNG per entity.

    conda run -n SAM2 python3 fact3r-map/scripts/visualize_entity_captions.py \\
        --index logs/fact3r_real_uot/full_video_qwen_complete/siglip_observations \\
        --captions logs/fact3r_real_uot/full_video_qwen_complete/siglip_observations/entity_captions.jsonl \\
        --output-dir logs/fact3r_real_uot/full_video_qwen_complete/entity_caption_viz \\
        --max-entities 40

Also writes an index.html contact sheet (thumbnail + caption for every
entity in the batch) so a whole run can be scanned at a glance instead of
opening 600+ files one at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.integrations.mast3r_slam import iter_exported_keyframes  # noqa: E402
from fact3r.semantics.observation_index import load_observation_index  # noqa: E402

_BOX_COLOR = (66, 133, 244)


def padded_crop_bbox(bbox_xyxy, width: int, height: int, pad_frac: float = 0.2):
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    pad_x, pad_y = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    return (
        max(0.0, x0 - pad_x), max(0.0, y0 - pad_y),
        min(float(width), x1 + pad_x), min(float(height), y1 + pad_y),
    )


def annotate(rgb: np.ndarray, bbox_xyxy, caption: str, font, upscale: int = 2) -> Image.Image:
    image = Image.fromarray(rgb).resize(
        (rgb.shape[1] * upscale, rgb.shape[0] * upscale), Image.LANCZOS)
    width, _height = image.size
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = (v * upscale for v in bbox_xyxy)
    draw.rectangle([x0, y0, x1, y1], outline=_BOX_COLOR, width=3)

    chars_per_line = max(10, int(width / 8))
    lines = textwrap.wrap(caption, width=chars_per_line) or [caption]
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 4
    text_y = max(0, y0 - line_height * len(lines))
    for line in lines:
        text_w = draw.textlength(line, font=font)
        text_x = max(0, min(x0, width - text_w))
        box = draw.textbbox((text_x, text_y), line, font=font)
        draw.rectangle(box, fill=_BOX_COLOR)
        draw.text((text_x, text_y), line, fill=(255, 255, 255), font=font)
        text_y += line_height
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", required=True, help="observation index dir or manifest.json path")
    parser.add_argument("--captions", type=Path, required=True, help="entity_captions.jsonl from caption_observation_entities.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-entities", type=int, default=None,
                        help="visualize only the first N entities in the captions file (omit for all)")
    parser.add_argument("--pad-frac", type=float, default=0.2,
                        help="must match caption_observation_entities.py's --pad-frac for the drawn crop region to line up")
    args = parser.parse_args()

    _manifest_path, manifest, _embeddings = load_observation_index(args.index)
    source_keyframes = manifest["source_keyframes"]

    records = [json.loads(line) for line in args.captions.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_entities is not None:
        records = records[: args.max_entities]
    print(f"{len(records)} entities to visualize")

    needed_frames = {int(r["representative_frame_id"]) for r in records}
    print(f"loading {len(needed_frames)} source keyframes from {source_keyframes} ...")
    keyframe_images = {
        keyframe.frame_id: np.array(keyframe.rgb, copy=True)
        for keyframe in iter_exported_keyframes(source_keyframes)
        if keyframe.frame_id in needed_frames
    }

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    written = 0
    for index, record in enumerate(records):
        frame_id = int(record["representative_frame_id"])
        rgb = keyframe_images.get(frame_id)
        bbox = record.get("bbox_xyxy")
        if rgb is None or bbox is None:
            print(f"  {index}: skipped (no frame/bbox for {record['group_id']})")
            continue
        image = annotate(rgb, bbox, record["caption"], font)
        filename = f"{index:04d}_{record['group_id']}.png"
        image.save(args.output_dir / filename)
        cards.append((filename, record))
        written += 1
        print(f"  {index}: {filename} <- frame {frame_id}, caption: {record['caption']}")

    index_html = args.output_dir / "index.html"
    card_html = "\n".join(
        f'<figure><img src="{escape(name)}" loading="lazy">'
        f'<figcaption>{escape(record["group_id"])} '
        f'({record["observation_count"]} views): {escape(record["caption"])}</figcaption></figure>'
        for name, record in cards
    )
    index_html.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Entity caption check</title>"
        "<style>body{background:#161616;color:#eee;font-family:sans-serif}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}"
        "figure{margin:0;background:#222;padding:6px}img{width:100%;height:auto}"
        "figcaption{padding-top:4px;font-size:13px}</style>"
        f"<h1>{written} entity captions</h1><main>{card_html}</main>",
        encoding="utf-8",
    )
    print(f"\nwrote {written} annotated frames + {index_html} -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
