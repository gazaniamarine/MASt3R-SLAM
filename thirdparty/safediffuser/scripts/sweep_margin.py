#!/usr/bin/env python3
"""How much real-world safety margin can the tube buy, and what does it cost?

    python3 scripts/sweep_margin.py --grids <a.npy> <b.npy>

`radius_margin` is the gap held between the tube wall and the nearest obstacle.
Raising it is the direct way to stop plans riding the wall -- with the cap in
place, a plan's worst clearance comes out at essentially the margin itself.

It is not free. The tube radius is capped at d(k) - margin, so as the margin
approaches the corridor's clearance the tube closes onto the centerline and the
trajectory has nowhere to go but along it. At that point the diffusion prior has
stopped contributing and the result is A* with extra steps. So the sweep reports
the deviation from the centerline next to the safety numbers: that column is
what says whether you are still running a diffusion planner.

Note this is a different knob from robot_radius. robot_radius shifts the
clearance field itself and so changes which cells A* may use at all -- push it
too far and the map disconnects. radius_margin only shrinks the tube, leaving
the centerline search untouched.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from diffuser.models.temporal import TemporalUnet          # noqa: E402
from diffuser.hm3d.diffusion import HM3DGaussianDiffusion  # noqa: E402
from diffuser.hm3d.map import HM3DMap                      # noqa: E402
from diffuser.hm3d import planner                          # noqa: E402

DEFAULT_CKPT = "logs/pretrained/maze2d-large-v1/diffusion/H384_T256/state_1920000.pt"
LAMBDA_EXACT = 1.0 / (2.0 * (1.0 - np.exp(-3.0)))


def build(ckpt, device):
    model = TemporalUnet(horizon=384, transition_dim=6, cond_dim=4, dim_mults=(1, 4, 8))
    d = HM3DGaussianDiffusion(model, horizon=384, observation_dim=4, action_dim=2,
                              n_timesteps=256, loss_type='l2', clip_denoised=True,
                              predict_epsilon=False, action_weight=1,
                              loss_weights=None, loss_discount=1)
    d.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['ema'])
    return d.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--margins", type=float, nargs="+",
                    default=[0.02, 0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--n-plans", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--robot-radius", type=float, default=0.20)
    ap.add_argument("--unknown-slack", type=float, default=0.50)
    ap.add_argument("--min-separation", type=float, default=4.0,
                    help="minimum start-goal distance, metres. Must match the "
                         "value plan_hm3d.py is run with, or the sweep sizes "
                         "the margin against a different problem set.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion = build(args.ckpt, device)
    diffusion.use_stt_guidance = True
    diffusion.lambda_stt = LAMBDA_EXACT

    for grid in args.grids:
        m = HM3DMap.load(grid, robot_radius=args.robot_radius,
                         unknown_slack=args.unknown_slack)
        nmin, nmax = m.norm_frame()

        # One fixed problem set per scene, reused at every margin, so the
        # columns differ only by the knob and not by which start/goal was drawn.
        rng = np.random.default_rng(args.seed)
        probs = []
        for _ in range(args.n_plans):
            s, g = planner.sample_endpoints(m, rng,
                                            min_separation=args.min_separation)
            line, _ = planner.centerline(m, s, g, 384)
            probs.append((s, g, line))

        print(f"\n{m.name}  (centerline clearance: min "
              f"{min(m.clearance_at(l).min() for _, _, l in probs):.3f} m)")
        print("%8s %10s %11s %11s %12s %12s" % (
            "margin", "coll%", "min_clear", "mean_clear", "dev_mean", "dev_max"))

        for margin in args.margins:
            C, MC, MEC, DEV, DEVX = [], [], [], [], []
            for s, g, line in probs:
                diffusion.bind_map(m, line, radius_min_real=0.25, eta=0.6,
                                   radius_margin=margin)
                sn = 2.0 * (s - nmin) / (nmax - nmin) - 1.0
                gn = 2.0 * (g - nmin) / (nmax - nmin) - 1.0
                cond = {
                    0: torch.tensor([[sn[0], sn[1], 0., 0.]], dtype=torch.float32, device=device),
                    383: torch.tensor([[gn[0], gn[1], 0., 0.]], dtype=torch.float32, device=device),
                }
                with torch.no_grad():
                    x = diffusion.conditional_sample(cond, horizon=384,
                                                     return_diffusion=False, verbose=False)
                t = (x[0, :, 2:4].cpu().numpy() + 1.0) * 0.5 * (nmax - nmin) + nmin
                cl = m.clearance_at(t)
                dev = np.linalg.norm(t - line, axis=1)
                C.append((cl <= 0).mean()); MC.append(cl.min()); MEC.append(cl.mean())
                DEV.append(dev.mean()); DEVX.append(dev.max())
            print("%8.2f %9.2f%% %11.3f %11.3f %12.3f %12.3f" % (
                margin, 100 * np.mean(C), np.mean(MC), np.mean(MEC),
                np.mean(DEV), np.mean(DEVX)))


if __name__ == "__main__":
    main()
