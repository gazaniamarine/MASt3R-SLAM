#!/usr/bin/env python3
"""Live-capture a rover session (Zenoh camera + /rover/rpm) straight into the
.mp4 + odometry-CSV pair that run_rover_pipeline[_verified|_vlm].py already
expect via --video/--odom -- so those five-stage offline pipelines can run
against a session captured from the REAL rover instead of a separately
pre-recorded take + externally-measured --time-offset.

This is a RECORDER, not a driver: it never publishes cmd_vel. Drive the
rover with whatever already does that (teleop, or run_rover_multileg.py in
../qwen3vl-2b-navdp) while this runs alongside, passively, on the same Pi --
exactly like that project's own multileg_gui.py viewer. Camera + odometry
come off the SAME Zenoh contract nav_pipeline/zenoh_node.py already uses to
drive the rover, reused unchanged here (no new wire format, no camera/ESP32
changes).

Why --time-offset can be 0 here, unlike the ffmpeg-on-Pi recording path
(scripts/record_rover_run.sh) that scripts/find_time_offset.py exists to
correct: that path timestamps the video against the Pi's ffmpeg clock and
the odometry CSV against a separately-run session's clock, so the two
diverge by however long passed between starting each. Here ONE Python
process opens the video writer and starts the odometry logger a few
milliseconds apart in the same loop, both stamped off this process's own
time.time() -- there is no second clock to be offset from. Still verify
with find_time_offset.py if the two ever look unsynced (e.g. a very long
GC pause between the two starts).

Odometry CSV schema -- confirmed by reading, not assumed: fact3r-map's own
build_depth_semantic_bev.py::_load_odometry() reads columns "t","x","y",
"theta","v" by name via csv.DictReader, which is exactly what
OdometryLogger (nav_pipeline/odometry_logger.py, the same class
run_rover_multileg.py already uses) writes. No format translation needed --
this script just runs that same logger against the same rpm topic.

Usage (GPU machine, rover's --rover backend already up):
    python3 scripts/record_rover_live.py --pi-ip 172.18.186.125 \\
        --out datasets/rover/live_run1 --max-seconds 90 --query "a 3D printer"

Ctrl-C ends the take early and still writes a usable (shorter) video/CSV
pair -- same as --max-seconds elapsing.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SIBLING_NAV = REPO_ROOT.parent / "qwen3vl-2b-navdp"
if str(SIBLING_NAV) not in sys.path:
    sys.path.insert(0, str(SIBLING_NAV))

try:
    import zenoh
except ImportError:
    print("ERROR: zenoh not importable -- run this under the same conda env "
          "run_rover_multileg.py uses (navdp/internnav).")
    sys.exit(1)

try:
    from nav_pipeline.odometry_logger import OdometryLogger
    from nav_pipeline.zenoh_node import (
        CAMERA_COMPRESSED_KEYS, CAMERA_KEYS, RPM_KEYS,
        parse_compressed_image, parse_float32_multiarray, parse_image, serialize_string,
    )
except ImportError as e:
    print(f"ERROR: could not import nav_pipeline from {SIBLING_NAV} ({e}). "
          f"Expected the qwen3vl-2b-navdp checkout as a sibling of {REPO_ROOT}.")
    sys.exit(1)


class LiveCapture:
    def __init__(self, session: "zenoh.Session", odom: OdometryLogger):
        self.odom = odom
        self.lock = Lock()
        self.latest_rgb: "np.ndarray | None" = None
        self.rgb_id = 0            # bumped on every new frame, so the writer
        self._last_written_id = 0  # loop can tell "new" from "still the old one"
        self.rgb_t = 0.0
        self.rpm_samples = 0
        self._abort = False

        self.subs = (
            [session.declare_subscriber(k, self._on_image) for k in CAMERA_KEYS]
            + [session.declare_subscriber(k, self._on_image_compressed) for k in CAMERA_COMPRESSED_KEYS]
            + [session.declare_subscriber(k, self._on_rpm) for k in RPM_KEYS]
        )

    def _on_image(self, sample):
        img = parse_image(bytes(sample.payload))
        if img is not None and img.ndim == 3:
            with self.lock:
                self.latest_rgb, self.rgb_t, self.rgb_id = img, time.time(), self.rgb_id + 1

    def _on_image_compressed(self, sample):
        img = parse_compressed_image(bytes(sample.payload))
        if img is not None:
            with self.lock:
                self.latest_rgb, self.rgb_t, self.rgb_id = img, time.time(), self.rgb_id + 1

    def _on_rpm(self, sample):
        try:
            data = parse_float32_multiarray(bytes(sample.payload))
            if len(data) >= 2:
                imu_heading = data[2] if len(data) >= 3 else None
                imu_calib = data[3] if len(data) >= 4 else None
                lateral = data[4] if len(data) >= 5 else None
                self.odom.update(data[0], data[1], imu_heading_deg=imu_heading,
                                  imu_calib=imu_calib, lateral_m_s=lateral)
                self.rpm_samples += 1
        except Exception as e:
            print(f"[WARN] rpm parse failed: {e}")

    def request_abort(self, *_):
        self._abort = True


def main():
    ap = argparse.ArgumentParser(
        description="Live-capture the real rover (camera + odometry) for MASt3R-SLAM's "
                    "run_rover_pipeline*.py --video/--odom",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pi-ip", type=str, default=None, help="Pi IP for the Zenoh peer; omit for multicast")
    ap.add_argument("--out", type=str, required=True, help="output directory (created if missing)")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="hard cap on the take")
    ap.add_argument("--record-fps", type=float, default=10.0,
                    help="video write rate -- the frontend resamples to --sample-fps (2) anyway, "
                         "this just caps file size; the camera itself streams faster (~15Hz)")
    ap.add_argument("--input-wait-s", type=float, default=30.0,
                    help="max wait for the first camera frame + /rover/rpm sample before giving up")
    ap.add_argument("--imu-min-mag-calib", type=int, default=3)
    ap.add_argument("--query", type=str, default="a 3D printer",
                    help="only used to pre-fill the printed follow-up command, not recorded")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "rover.mp4"

    zcfg = zenoh.Config()
    if args.pi_ip:
        zcfg.insert_json5("connect/endpoints", f'["tcp/{args.pi_ip}:7447"]')
        print(f"[zenoh] connecting tcp/{args.pi_ip}:7447")
    else:
        print("[zenoh] multicast scouting")
    session = zenoh.open(zcfg)
    # Passive status feed for multileg_gui.py's "fact3r-map live recorder"
    # panel -- a viewer can show recording progress alongside the driving
    # run; this publisher works the same with zero viewers attached.
    status_pub = session.declare_publisher("mast3r_record/status")

    def _publish_record_status(recording: bool, t: float, frames: int, rpm_samples: int):
        try:
            status_pub.put(serialize_string(json.dumps(dict(
                recording=recording, t=round(t, 1), frames=frames,
                rpm_samples=rpm_samples, out=str(out_dir), max_seconds=args.max_seconds,
            ))))
        except Exception:
            pass

    odom = OdometryLogger(str(out_dir), imu_min_mag_calib=args.imu_min_mag_calib)
    odom.start_new_goal("live", reset_pose=True)   # fresh origin -- this IS the map's world frame
    odom_path = odom.path
    print(f"[odom] logging to {odom_path}")

    cap = LiveCapture(session, odom)
    signal.signal(signal.SIGINT, cap.request_abort)
    signal.signal(signal.SIGTERM, cap.request_abort)

    print(f"[inputs] waiting up to {args.input_wait_s:.0f}s for camera + /rover/rpm ...")
    t0 = time.time()
    while time.time() - t0 < args.input_wait_s:
        with cap.lock:
            have_cam = cap.latest_rgb is not None
        if have_cam and cap.rpm_samples > 0:
            break
        time.sleep(0.3)
    else:
        print(f"[fatal] no camera and/or /rover/rpm within {args.input_wait_s:.0f}s -- "
              f"is the Pi's --rover backend up? Aborting (no files written beyond the empty odom csv).")
        odom.close()
        session.close()
        sys.exit(4)
    print("[inputs] live -- recording starts now")

    writer = None
    frames_written = 0
    period = 1.0 / args.record_fps
    t_start = time.time()
    last_status = t_start
    try:
        while not cap._abort and time.time() - t_start < args.max_seconds:
            with cap.lock:
                rgb, rid = cap.latest_rgb, cap.rgb_id
            if rgb is not None and rid != cap._last_written_id:
                cap._last_written_id = rid
                bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                if writer is None:
                    h, w = bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_path), fourcc, args.record_fps, (w, h))
                    if not writer.isOpened():
                        print(f"[fatal] could not open video writer at {video_path}")
                        break
                    print(f"[video] writing {w}x{h} @ {args.record_fps}fps -> {video_path}")
                writer.write(bgr)
                frames_written += 1
            now = time.time()
            _publish_record_status(True, now - t_start, frames_written, cap.rpm_samples)
            if now - last_status > 10.0:
                print(f"[STATUS] t={now - t_start:5.1f}s frames={frames_written} "
                      f"rpm_samples={cap.rpm_samples} pose=({odom.x:+.2f},{odom.y:+.2f},"
                      f"{np.degrees(odom.theta):+.0f}deg)")
                last_status = now
            time.sleep(max(0.0, period - (time.time() - now)))
    finally:
        _publish_record_status(False, time.time() - t_start, frames_written, cap.rpm_samples)
        if writer is not None:
            writer.release()
        odom.close()
        session.close()

    dur = time.time() - t_start
    print("\n" + "=" * 64)
    if frames_written == 0:
        print("[fatal] recorded 0 frames -- camera stream stopped before anything was written.")
        sys.exit(5)
    print(f"Recorded {dur:.1f}s: {frames_written} frames -> {video_path}")
    print(f"                    {cap.rpm_samples} odom samples -> {odom_path}")
    print("=" * 64)
    print("\nNext (run_rover_pipeline_verified.py -- SigLIP + Qwen3-VL verification + memory):")
    print(f'  python3 scripts/run_rover_pipeline_verified.py \\\n'
          f'      --run logs/rover/pipeline/{out_dir.name} \\\n'
          f'      --video {video_path} --odom {odom_path} \\\n'
          f'      --time-offset 0.0 --query "{args.query}"')
    print("\n(--time-offset 0.0 because both files were timestamped by this one process's own "
          "clock -- verify with scripts/find_time_offset.py if the result looks off.)")


if __name__ == "__main__":
    main()
