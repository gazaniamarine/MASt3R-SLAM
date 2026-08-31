"""Run Depth-Anything-V2 over a video and write a colorized depth visualization.

Uses the HF transformers port of the Space at
https://huggingface.co/spaces/depth-anything/Depth-Anything-V2

Outputs a side-by-side (RGB | depth) mp4 by default, plus optional raw
inverse-depth arrays for downstream use.
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

CKPTS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out-dir", default="depth_out")
    p.add_argument("--encoder", default="large", choices=list(CKPTS))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument("--input-size", type=int, default=518)
    p.add_argument("--colormap", default="inferno", choices=["inferno", "spectral", "magma", "gray"])
    p.add_argument("--layout", default="sbs", choices=["sbs", "depth"])
    p.add_argument("--per-frame-norm", action="store_true",
                   help="normalize each frame independently (flickers, but max contrast)")
    p.add_argument("--norm-samples", type=int, default=120,
                   help="frames sampled to fix a global depth range")
    p.add_argument("--norm-pct", type=float, nargs=2, default=[2.0, 98.0],
                   metavar=("LO", "HI"), help="percentiles defining the depth range")
    p.add_argument("--tone", default="log", choices=["linear", "log", "gamma"],
                   help="curve across the range; log spreads far-field contrast")
    p.add_argument("--gamma", type=float, default=0.5, help="exponent when --tone gamma")
    p.add_argument("--save-npz", action="store_true", help="dump fp16 inverse depth per frame")
    p.add_argument("--save-png16", action="store_true", help="dump 16-bit PNG per frame")
    return p.parse_args()


def read_frames(cap, count, stride):
    """Pull up to `count` frames (already strided) or fewer at EOF."""
    frames = []
    while len(frames) < count:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
        for _ in range(stride - 1):
            if not cap.grab():
                break
    return frames


@torch.inference_mode()
def infer(model, processor, frames_bgr, device, dtype, hw):
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    inputs = processor(images=rgb, return_tensors="pt").to(device, dtype)
    out = model(**inputs)
    depths = processor.post_process_depth_estimation(
        out, target_sizes=[hw] * len(rgb)
    )
    return [d["predicted_depth"].float().cpu().numpy() for d in depths]


def colorize(inv_depth, lo, hi, cmap, tone="linear", gamma=0.5):
    if tone == "log":
        # Inverse depth is ~1/Z, so log() is uniform in log-distance: the
        # near floor stops eating the whole range.
        eps = 1e-3
        d = np.log(np.clip(inv_depth, eps, None))
        l, h = np.log(max(lo, eps)), np.log(max(hi, eps))
        x = np.clip((d - l) / max(h - l, 1e-6), 0.0, 1.0)
    else:
        x = np.clip((inv_depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        if tone == "gamma":
            x = x ** gamma
    u8 = (x * 255.0).astype(np.uint8)
    if cmap == "gray":
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    lut = {"inferno": cv2.COLORMAP_INFERNO,
           "magma": cv2.COLORMAP_MAGMA,
           "spectral": cv2.COLORMAP_TURBO}[cmap]
    return cv2.applyColorMap(u8, lut)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_out = (total + args.stride - 1) // args.stride
    if args.max_frames:
        n_out = min(n_out, args.max_frames)
    print(f"{args.video}: {W}x{H} @{src_fps:g}fps, {total} frames "
          f"-> {n_out} frames (stride {args.stride})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    ckpt = CKPTS[args.encoder]
    print(f"loading {ckpt} on {device} ({dtype})")
    processor = AutoImageProcessor.from_pretrained(ckpt)
    processor.size = {"height": args.input_size, "width": args.input_size}
    model = AutoModelForDepthEstimation.from_pretrained(ckpt).to(device, dtype).eval()

    # Pass 1: fix a global normalization range so the output does not flicker.
    lo = hi = None
    if not args.per_frame_norm:
        step = max(1, n_out // max(args.norm_samples, 1))
        vals = []
        for i in range(0, n_out, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * args.stride)
            ok, bgr = cap.read()
            if not ok:
                break
            d = infer(model, processor, [bgr], device, dtype, (H, W))[0]
            vals.append(np.percentile(d, args.norm_pct))
        vals = np.asarray(vals)
        lo, hi = float(np.median(vals[:, 0])), float(np.median(vals[:, 1]))
        print(f"global inverse-depth range: [{lo:.3f}, {hi:.3f}] from {len(vals)} samples")
        cap.release()
        cap = cv2.VideoCapture(args.video)

    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_w = W * 2 if args.layout == "sbs" else W
    out_fps = src_fps / args.stride
    vpath = os.path.join(args.out_dir, f"{stem}_depth_{args.encoder}.mp4")
    writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, H))

    npz_dir = os.path.join(args.out_dir, "raw")
    if args.save_npz or args.save_png16:
        os.makedirs(npz_dir, exist_ok=True)

    done = 0
    t0 = time.time()
    while done < n_out:
        batch = read_frames(cap, min(args.batch_size, n_out - done), args.stride)
        if not batch:
            break
        depths = infer(model, processor, batch, device, dtype, (H, W))
        for bgr, d in zip(batch, depths):
            if args.per_frame_norm:
                l, h = np.percentile(d, args.norm_pct)
            else:
                l, h = lo, hi
            vis = colorize(d, l, h, args.colormap, args.tone, args.gamma)
            writer.write(np.hstack([bgr, vis]) if args.layout == "sbs" else vis)
            if args.save_npz:
                np.save(os.path.join(npz_dir, f"{done:06d}.npy"), d.astype(np.float16))
            if args.save_png16:
                x = np.clip((d - l) / max(h - l, 1e-6), 0, 1)
                cv2.imwrite(os.path.join(npz_dir, f"{done:06d}.png"),
                            (x * 65535).astype(np.uint16))
            done += 1
        el = time.time() - t0
        print(f"\r{done}/{n_out}  {done/el:.1f} fps  eta {(n_out-done)/max(done/el,1e-6):.0f}s",
              end="", flush=True)

    writer.release()
    cap.release()
    print(f"\nwrote {vpath} ({done} frames, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
