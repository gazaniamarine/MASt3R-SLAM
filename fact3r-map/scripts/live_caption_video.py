#!/usr/bin/env python3
"""Live SAM2+SigLIP+BLIP left/right observation logging over a real rover
video, streamed frame by frame -- no batch pre-pass, no pre-exported
keyframes.

Three tiers per crop, matching fact3r.semantics.observation_index's own
division of labour (that module's own docstring: "SigLIP-backed semantic
retrieval over persistent mask observations"), just applied live over a raw
video instead of that module's offline exported-keyframes + saved-proposal-
manifest pipeline (which needs a full MASt3R export and a separate SAM2
proposal-saving pass -- overkill for a single demo video with only an
odometry CSV):
  - SAM2   finds the crop (class-agnostic mask proposal, left/right split)
  - SigLIP encodes it into an embedding, saved to embeddings.npy (row order
    matches each object's "embedding_row" in live_landmarks.jsonl) -- the
    retrieval/similarity tier, so a later query can find "have I seen
    something like this crop before" without re-running any model
  - BLIP/SmolVLM captions it in words -- the human-readable description tier

Reads the video with cv2.VideoCapture (one frame at a time, like a live
camera), aligns each frame to the session's real wheel odometry (the video
and odometry clocks are NOT the same -- see [[mpl-session-clock-offset]]:
default --time-offset 26.80 is this specific 2026-08-26 MPL session's
measured lag, override for any other session), throttles by real distance
travelled (same 1m-by-default idea as memory_nav's outbound capture), and
runs SAM2 + SigLIP + the captioner inline on each throttled frame --
printing the left/right captions to the console the moment they're
produced, not after the whole video has been scanned.

    conda run -n SAM2 python3 fact3r-map/scripts/live_caption_video.py \\
        --video /home/nahar4/Gazania/MPL/manual_drive_20260826_180408.mp4 \\
        --odom /home/nahar4/Gazania/MPL/odom_home_session_20260826_180408.csv \\
        --time-offset 26.80 --capture-interval-m 1.0 \\
        --output-dir logs/mpl_live_captions/manual_drive_20260826_180408

This is a standalone demo/inspection tool for real footage -- it does not
share a file format with memory_nav's anchor/landmark store (there is no
"go back to X" query here, just "what did the rover see, and where, in this
recorded drive"), though the side-classification and captioning logic is
the same as caption_frame_objects_blip.py / memory_nav's caption_landmarks.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.proposals.sam2_official_generator import SAM2OfficialMaskGenerator  # noqa: E402
from fact3r.semantics.blip_captioner import BlipCaptioner  # noqa: E402
from fact3r.semantics.observation_index import Siglip2Encoder  # noqa: E402
from fact3r.semantics.smolvlm_captioner import SmolVLMCaptioner  # noqa: E402

_SIDE_COLORS = {"left": (66, 133, 244), "right": (251, 140, 0)}


def load_odometry(path: Path):
    """Same convention as fact3r-map/scripts/build_depth_semantic_bev.py's
    _load_odometry: timestamps rebased to start at 0 (odom-relative
    seconds), so a video-relative timestamp lines up after + time_offset."""

    t, x, y, theta = [], [], [], []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t.append(float(row["t"]))
            x.append(float(row["x"]))
            y.append(float(row["y"]))
            theta.append(float(row["theta"]))
    if len(t) < 2:
        raise ValueError(f"odometry needs at least two rows: {path}")
    t = np.asarray(t, dtype=np.float64)
    t -= t[0]
    return t, np.asarray(x), np.asarray(y), np.unwrap(np.asarray(theta))


def side_of(bbox_xyxy, image_width: int, dead_zone_frac: float = 0.15):
    x0, _, x1, _ = bbox_xyxy
    center_x = (x0 + x1) / 2.0
    image_center = image_width / 2.0
    dead_zone = image_width * dead_zone_frac
    if center_x < image_center - dead_zone:
        return "left"
    if center_x > image_center + dead_zone:
        return "right"
    return None


def best_per_side(proposals, image_width: int):
    best = {"left": None, "right": None}
    for proposal in proposals:
        if proposal.bounding_box_xyxy is None:
            continue
        side = side_of(proposal.bounding_box_xyxy, image_width)
        if side is None:
            continue
        size_bonus = 1.0 + math.log1p(max(int(np.asarray(proposal.mask).sum()), 0))
        score = proposal.score * size_bonus
        if best[side] is None or score > best[side][0]:
            best[side] = (score, proposal)
    return best


def padded_crop(rgb: np.ndarray, bbox_xyxy, pad_frac: float = 0.2) -> np.ndarray:
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    pad_x, pad_y = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(float(width), x1 + pad_x)
    y1 = min(float(height), y1 + pad_y)
    return rgb[int(y0):int(y1), int(x0):int(x1)]


def annotate(rgb: np.ndarray, objects: list, font) -> Image.Image:
    """Left's label goes above its box, right's goes below its box -- fixed
    anchor per side (not "whichever side has more room"), so the two labels
    never collide even when both boxes sit at the same height, as they
    often do near the top edge of a frame."""

    image = Image.fromarray(rgb)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    chars_per_line = max(10, int(width / 8))
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 4
    for obj in objects:
        color = _SIDE_COLORS.get(obj["side"], (200, 200, 200))
        x0, y0, x1, y1 = obj["bbox_xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f'{obj["side"]}: {obj["caption"]}'
        lines = textwrap.wrap(label, width=chars_per_line) or [label]
        if obj["side"] == "right":
            text_y = min(height - line_height * len(lines), y1 + 4)
        else:
            text_y = max(0, y0 - line_height * len(lines))
        for line in lines:
            text_w = draw.textlength(line, font=font)
            text_x = max(0, min(x0, width - text_w))
            box = draw.textbbox((text_x, text_y), line, font=font)
            draw.rectangle(box, fill=color)
            draw.text((text_x, text_y), line, fill=(255, 255, 255), font=font)
            text_y += line_height
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--odom", type=Path, required=True)
    parser.add_argument("--time-offset", type=float, required=True,
                        help="video-clock seconds to add before looking up odometry -- "
                             "measure with scripts/find_time_offset.py, never assume 0")
    parser.add_argument("--capture-interval-m", type=float, default=1.0,
                        help="minimum real distance travelled between inferences")
    parser.add_argument("--max-captures", type=int, default=None,
                        help="stop after this many inferred frames (omit for the whole video)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam2-model-id", default="facebook/sam2-hiera-large")
    parser.add_argument("--sam2-device", default="0")
    parser.add_argument("--points-per-side", type=int, default=24)
    parser.add_argument("--captioner", choices=("blip", "smolvlm"), default="blip",
                        help="which model captions each side's crop")
    parser.add_argument("--blip-model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--blip-device", default="auto")
    parser.add_argument("--smolvlm-model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--smolvlm-device", default="auto")
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--siglip-device", default="auto")
    args = parser.parse_args()

    odom_t, odom_x, odom_y, odom_theta = load_odometry(args.odom)

    print(f"Loading SAM2 ({args.sam2_model_id})...", flush=True)
    mask_generator = SAM2OfficialMaskGenerator(
        model_id=args.sam2_model_id, device=args.sam2_device, points_per_side=args.points_per_side)
    print(f"Loading SigLIP ({args.siglip_model})...", flush=True)
    siglip_encoder = Siglip2Encoder(args.siglip_model, device=args.siglip_device)
    print(f"SigLIP ready: load={siglip_encoder.load_seconds:.2f}s", flush=True)
    if args.captioner == "smolvlm":
        print(f"Loading SmolVLM ({args.smolvlm_model})...", flush=True)
        captioner = SmolVLMCaptioner(args.smolvlm_model, device=args.smolvlm_device)
    else:
        print(f"Loading BLIP ({args.blip_model})...", flush=True)
        captioner = BlipCaptioner(args.blip_model, device=args.blip_device)
    captioner.load()
    print(f"{args.captioner} ready: load={captioner.load_seconds:.2f}s", flush=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    landmarks_path = args.output_dir / "live_landmarks.jsonl"
    embeddings_path = args.output_dir / "embeddings.npy"

    last_capture_xy = None
    frame_index = 0
    written = 0
    embedding_rows: list = []
    started = time.perf_counter()
    print(f"streaming {args.video.name} @ {fps:.1f} fps, "
          f"throttled every {args.capture_interval_m}m of real travel ...", flush=True)

    with landmarks_path.open("w", encoding="utf-8") as landmarks_file:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            video_t = frame_index / fps
            odom_relative_t = video_t + args.time_offset
            frame_index += 1

            if odom_relative_t < odom_t[0] or odom_relative_t > odom_t[-1]:
                continue
            x = float(np.interp(odom_relative_t, odom_t, odom_x))
            y = float(np.interp(odom_relative_t, odom_t, odom_y))
            theta = float(np.interp(odom_relative_t, odom_t, odom_theta))

            here = np.array([x, y])
            if (last_capture_xy is not None
                    and np.linalg.norm(here - last_capture_xy) < args.capture_interval_m):
                continue
            last_capture_xy = here

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            infer_started = time.perf_counter()
            proposals = mask_generator.generate(rgb, frame_id=frame_index)
            sam2_s = time.perf_counter() - infer_started
            best = best_per_side(proposals, image_width=rgb.shape[1])

            objects = []
            siglip_s = 0.0
            caption_s = 0.0
            for side, entry in best.items():
                if entry is None:
                    continue
                score, proposal = entry
                crop = padded_crop(rgb, proposal.bounding_box_xyxy)
                if crop.size == 0:
                    continue
                crop_image = Image.fromarray(crop)

                siglip_started = time.perf_counter()
                embedding = siglip_encoder.encode_images([crop_image])[0]
                siglip_s += time.perf_counter() - siglip_started
                embedding_rows.append(embedding)
                embedding_row = len(embedding_rows) - 1

                caption_started = time.perf_counter()
                caption = captioner.caption(crop)
                caption_s += time.perf_counter() - caption_started

                objects.append({
                    "side": side, "caption": caption, "score": float(proposal.score),
                    "bbox_xyxy": [float(v) for v in proposal.bounding_box_xyxy],
                    "embedding_row": embedding_row,
                })
            infer_s = time.perf_counter() - infer_started

            record = {
                "frame_index": frame_index, "video_t": video_t, "odom_relative_t": odom_relative_t,
                "pose_xytheta": [x, y, theta], "objects": objects,
                "timing": {"sam2_seconds": sam2_s, "siglip_seconds": siglip_s,
                           "caption_seconds": caption_s, "total_seconds": infer_s},
            }
            landmarks_file.write(json.dumps(record) + "\n")
            landmarks_file.flush()

            annotated_path = args.output_dir / f"live_{written:04d}_f{frame_index:06d}.png"
            annotate(rgb, objects, font).save(annotated_path)

            summary = " | ".join(f'{o["side"]}={o["caption"]}' for o in objects) or "nothing on either side"
            elapsed = time.perf_counter() - started
            print(f"[{elapsed:6.1f}s wall | t={video_t:6.1f}s pos=({x:+5.2f},{y:+5.2f})] "
                  f"(sam2={sam2_s:.2f}s siglip={siglip_s:.2f}s caption={caption_s:.2f}s "
                  f"total={infer_s:.2f}s) {summary}", flush=True)
            written += 1
            if args.max_captures is not None and written >= args.max_captures:
                print(f"reached --max-captures={args.max_captures}, stopping", flush=True)
                break

    capture.release()
    if embedding_rows:
        np.save(embeddings_path, np.stack(embedding_rows).astype(np.float32))
    total_s = time.perf_counter() - started
    print(f"\n{written} frames captioned in {total_s:.1f}s -> {args.output_dir}")
    print(f"landmarks -> {landmarks_path}")
    print(f"embeddings ({len(embedding_rows)} rows, "
          f"dim={0 if not embedding_rows else embedding_rows[0].shape[0]}) -> {embeddings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
