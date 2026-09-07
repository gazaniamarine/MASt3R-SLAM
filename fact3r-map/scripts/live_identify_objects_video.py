#!/usr/bin/env python3
"""Live SmolVLM object identification over a real rover video -- no SAM2, no
bounding boxes, no left/right split, ONE SmolVLM call per throttled frame.

Started as two calls (a scene sentence + an objects list), then one combined
call asking for both -- SmolVLM-500M-Instruct turned out unable to do that
compound instruction reliably: it consistently skipped the scene sentence
and jumped straight to the objects list regardless of phrasing (three
prompt variants tried, all failed the same way, one degenerating into a
repeated-phrase loop). Rather than keep chasing a prompt that works around
a real small-model limitation, the scene-description half is dropped
entirely -- this asks only for the 2 to 4 most prominent inanimate objects,
which the model does reliably in a single call.

Same streaming/throttling shape as live_caption_video.py (cv2.VideoCapture
read loop, real-odometry distance throttle, live per-frame printing) and
the same clock-alignment convention (see [[mpl-session-clock-offset]]),
duplicated in full rather than imported -- separate CLI entry points, and
the shared logic is a dozen lines, cheaper to keep in step by hand than to
couple two otherwise-unrelated scripts.

    conda run -n SAM2 python3 fact3r-map/scripts/live_identify_objects_video.py \\
        --video /home/nahar4/Gazania/MPL/manual_drive_20260826_180408.mp4 \\
        --odom /home/nahar4/Gazania/MPL/odom_home_session_20260826_180408.csv \\
        --time-offset 26.80 --capture-interval-m 1.0 \\
        --output-dir logs/mpl_live_captions/objects
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fact3r.semantics.smolvlm_captioner import SmolVLMCaptioner  # noqa: E402

OBJECTS_PROMPT = (
    "List the 2 to 4 most prominent inanimate objects visible in this image, "
    "separated by commas. Do not include people or animals."
)

_NUMBERED_ITEM = re.compile(r"\b\d+\.\s*")


def load_odometry(path: Path):
    """Same convention as live_caption_video.py's load_odometry / fact3r-map's
    build_depth_semantic_bev.py: timestamps rebased to start at 0 (odom-
    relative seconds), so a video-relative timestamp lines up after +
    time_offset."""

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


def parse_objects(raw: str) -> list:
    """"a fan, a pillar, a whiteboard" -> ["a fan", "a pillar", "a whiteboard"].
    Also handles a numbered list ("1. fan 2. pillar 3. whiteboard") -- small
    models drift into that format about as often as commas, so split on
    whichever marker is actually present rather than assuming commas. Also
    strips a leading "Objects:" the model sometimes echoes back."""

    text = raw.strip()
    if text.lower().startswith("objects:"):
        text = text[len("objects:"):].strip()
    text = text.rstrip(".")
    if not text:
        return []
    if _NUMBERED_ITEM.search(text):
        parts = _NUMBERED_ITEM.split(text)
    else:
        parts = text.split(",")
    return [part.strip().strip(",").strip() for part in parts if part.strip(" ,.")]


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
    parser.add_argument("--smolvlm-model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--smolvlm-device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=40,
                        help="one short comma list -- 40 tokens is ample")
    args = parser.parse_args()

    odom_t, odom_x, odom_y, odom_theta = load_odometry(args.odom)

    print(f"Loading SmolVLM ({args.smolvlm_model})...", flush=True)
    captioner = SmolVLMCaptioner(
        args.smolvlm_model, device=args.smolvlm_device, max_new_tokens=args.max_new_tokens)
    captioner.load()
    print(f"SmolVLM ready: load={captioner.load_seconds:.2f}s", flush=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "objects.jsonl"

    last_capture_xy = None
    frame_index = 0
    written = 0
    started = time.perf_counter()
    print(f"streaming {args.video.name} @ {fps:.1f} fps, "
          f"throttled every {args.capture_interval_m}m of real travel ...", flush=True)

    with records_path.open("w", encoding="utf-8") as records_file:
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
            pil_image = Image.fromarray(rgb)

            infer_started = time.perf_counter()
            response = captioner.caption(pil_image, prompt=OBJECTS_PROMPT)
            objects = parse_objects(response)
            infer_s = time.perf_counter() - infer_started

            frame_path = args.output_dir / f"frame_{written:04d}_f{frame_index:06d}.png"
            pil_image.save(frame_path)

            record = {
                "frame_index": frame_index, "frame_path": str(frame_path),
                "video_t": video_t, "odom_relative_t": odom_relative_t,
                "pose_xytheta": [x, y, theta], "infer_seconds": infer_s,
                "objects": objects, "raw_response": response,
            }
            records_file.write(json.dumps(record) + "\n")
            records_file.flush()

            elapsed = time.perf_counter() - started
            print(f"[{elapsed:6.1f}s wall | t={video_t:6.1f}s pos=({x:+5.2f},{y:+5.2f})] "
                  f"({infer_s:.2f}s infer) objects: {', '.join(objects) or '(none)'}",
                  flush=True)
            written += 1
            if args.max_captures is not None and written >= args.max_captures:
                print(f"reached --max-captures={args.max_captures}, stopping", flush=True)
                break

    capture.release()
    total_s = time.perf_counter() - started
    print(f"\n{written} frames identified in {total_s:.1f}s -> {args.output_dir}")
    print(f"records -> {records_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
