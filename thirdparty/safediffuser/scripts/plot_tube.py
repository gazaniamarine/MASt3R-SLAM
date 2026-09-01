#!/usr/bin/env python3
"""Draw the DSTT safety tube Gamma_j(k) over each scene's floorplan.

    python3 scripts/plot_tube.py --grids <a.npy> <b.npy> ...

The tube is the object the method actually reasons about, and it is the thing
that had to change to move off maze2d, so it is worth looking at directly.

Gamma_j(k) = {x : |x - c(k)| <= r_j(k)} is a union of disks along the
centerline, and that is how it is drawn here -- rasterised by stamping a disk
of radius r_j(k) at every horizon step -- rather than as a band between two
offset curves. The offset-curve shortcut is wrong exactly where the picture
matters: on a tight turn the outer offset opens a wedge the tube does not
contain, and the inner offset self-intersects.

Four diffusion steps are overlaid to show the prescribed-time contraction:
at j=N the tube is wide and the prior is free to shape the path inside it, and
by j=0 it has collapsed to the capped floor, pinning the trajectory into the
space the map says is actually free.

The radii come from the bound model's own compute_dstt_tube, not from a copy of
the formula, so what is plotted is what the sampler used.
"""
import argparse
import pathlib
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from diffuser.models.temporal import TemporalUnet          # noqa: E402
from diffuser.hm3d.diffusion import HM3DGaussianDiffusion  # noqa: E402
from diffuser.hm3d.map import HM3DMap                      # noqa: E402
from diffuser.hm3d import planner                          # noqa: E402
from diffuser.hm3d import plans_dir_for                     # noqa: E402
from diffuser.hm3d.tube import tube_mask                   # noqa: E402

DEFAULT_CKPT = "logs/pretrained/maze2d-large-v1/diffusion/H384_T256/state_1920000.pt"
LAMBDA_EXACT = 1.0 / (2.0 * (1.0 - np.exp(-3.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--seed", type=int, default=1)
    # phi_j = exp(-lambda_r * s) with s = (N-j)/N, so the tube does most of its
    # contracting in the first 40% of denoising: by j=150 phi is already 0.29.
    # Sampling j uniformly would show four nearly identical outlines, so these
    # are spaced through the contraction instead.
    ap.add_argument("--steps", type=int, nargs="+", default=[255, 225, 170, 0])
    ap.add_argument("--radius-margin", type=float, default=0.15)
    # Must match plan_hm3d.py. Left at the 0.50 default on a partially observed
    # scene the figure is drawn on a different navigable set than the plans it
    # is meant to illustrate: at 0.50 a centerline may sit 0.30 m inside
    # never-observed space, so the tube would be shown riding through the void.
    ap.add_argument("--unknown-slack", type=float, default=0.50)
    ap.add_argument("--min-separation", type=float, default=4.0)
    # See plan_hm3d.py: the border flood fill assumes an enclosed building and
    # over-claims the map on open, partially observed scenes.
    ap.add_argument("--no-exclude-exterior", dest="exclude_exterior",
                    action="store_false",
                    help="do not treat border-connected unknown as hard")
    ap.set_defaults(exclude_exterior=True)
    ap.add_argument("--out", default=None,
                    help="output png; defaults to safety_tubes.png in the run's plans/ folder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TemporalUnet(horizon=384, transition_dim=6, cond_dim=4, dim_mults=(1, 4, 8))
    diffusion = HM3DGaussianDiffusion(
        model, horizon=384, observation_dim=4, action_dim=2, n_timesteps=256,
        loss_type='l2', clip_denoised=True, predict_epsilon=False,
        action_weight=1, loss_weights=None, loss_discount=1)
    diffusion.load_state_dict(
        torch.load(args.ckpt, map_location='cpu', weights_only=False)['ema'])
    diffusion = diffusion.to(device).eval()
    diffusion.use_stt_guidance = True
    diffusion.lambda_stt = LAMBDA_EXACT

    n = len(args.grids)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 5.6 * nrow))
    axes = np.atleast_1d(axes).ravel()

    # The lightest shade used to be #cfe4f7 on a #eaf4fb exterior -- two pale
    # blues, so the widest tube was invisible against the outdoors, which is
    # exactly the region a wide tube is at risk of reaching into. The map keeps
    # the warm/neutral end of the palette and the tube gets the blues outright.
    shades = ["#b9d3ec", "#7fb0dd", "#3f83c4", "#12457e"]
    widths = [0.8, 1.0, 1.2, 1.6]
    profiles = []

    for ax, grid in zip(axes, args.grids):
        m = HM3DMap.load(grid, exclude_exterior=args.exclude_exterior,
                         unknown_slack=args.unknown_slack)
        nmin, nmax = m.norm_frame()
        rng = np.random.default_rng(args.seed)

        # Longest of a few candidates: a 1 m hop shows nothing about a tube.
        best = None
        for _ in range(8):
            s, g = planner.sample_endpoints(m, rng,
                                            min_separation=args.min_separation)
            line, _ = planner.centerline(m, s, g, 384)
            L = np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
            if best is None or L > best[0]:
                best = (L, s, g, line)
        _, s, g, line = best

        diffusion.bind_map(m, line, radius_min_real=0.25, eta=0.6,
                           radius_margin=args.radius_margin)
        sn = 2.0 * (s - nmin) / (nmax - nmin) - 1.0
        gn = 2.0 * (g - nmin) / (nmax - nmin) - 1.0
        cond = {
            0: torch.tensor([[sn[0], sn[1], 0., 0.]], dtype=torch.float32, device=device),
            383: torch.tensor([[gn[0], gn[1], 0., 0.]], dtype=torch.float32, device=device),
        }
        with torch.no_grad():
            x = diffusion.conditional_sample(cond, horizon=384,
                                             return_diffusion=False, verbose=False)
        traj = (x[0, :, 2:4].cpu().numpy() + 1.0) * 0.5 * (nmax - nmin) + nmin

        base = np.zeros(m.prob.shape)
        base[m.free] = 1.0
        base[m.undetermined] = 2.0
        base[m.occupied] = 3.0
        base[m.exterior] = 4.0
        ax.imshow(base, origin="lower", interpolation="nearest",
                  cmap=matplotlib.colors.ListedColormap(
                      ["#e0dcd6", "#ffffff", "#f0c674", "#111111", "#efe6da"]),
                  vmin=0, vmax=4)

        profile = {"name": m.name, "clearance": m.clearance_at(line), "radii": {}}
        # Widest first, so each contraction is drawn on top of the one it sits
        # inside and the nesting is visible rather than painted over.
        for shade, lw, j in zip(shades, widths, sorted(args.steps, reverse=True)):
            with torch.no_grad():
                c_real, r_real = diffusion.compute_dstt_tube(
                    cond[0][:, 0:2], cond[383][:, 0:2], 384, j)
            profile["radii"][j] = r_real[0, :, 0].cpu().numpy()
            mask = tube_mask(m, c_real[0].cpu().numpy(), r_real[0, :, 0].cpu().numpy())
            ax.contourf(mask.astype(float), levels=[0.5, 1.5],
                        colors=[shade], alpha=0.75)
            # A drawn outline as well as a fill: where two contractions differ
            # by less than a cell the fills are indistinguishable, and the
            # boundary is the part of the tube that carries the guarantee.
            ax.contour(mask.astype(float), levels=[0.5], colors=[shade],
                       linewidths=lw)

        def cells(p):
            return ((p[:, 1] - m.origin_x) / m.res - 0.5,
                    (p[:, 0] - m.origin_y) / m.res - 0.5)

        cx, cy = cells(line)
        tx, ty = cells(traj)
        ax.plot(cx, cy, "--", color="#555555", lw=1.2)
        ax.plot(tx, ty, "-", color="#c62828", lw=2.0)
        # Same shape vocabulary as plan_hm3d.py's figure: triangle = start,
        # cross = goal, so the two figures can be read side by side.
        ax.plot(cx[0], cy[0], "^", color="#2e7d32", ms=11, mec="k", mew=0.9,
                zorder=6)
        ax.plot(cx[-1], cy[-1], "X", color="#c62828", ms=12, mec="k", mew=0.9,
                zorder=6)

        # Crop to the tube, since the plan covers a fraction of the grid.
        pad = int(1.0 / m.res)
        ax.set_xlim(max(0, cx.min() - pad), min(m.n_cols, cx.max() + pad))
        ax.set_ylim(max(0, cy.min() - pad), min(m.n_rows, cy.max() + pad))
        ax.set_xticks([]); ax.set_yticks([])
        profiles.append(profile)
        rmin = float(r_real[0, :, 0].min())
        rmax = float(r_real[0, :, 0].max())
        ax.set_title(f"{m.name}\nfinal tube radius {rmin:.2f}-{rmax:.2f} m, "
                     f"path {np.linalg.norm(np.diff(traj,axis=0),axis=1).sum():.1f} m",
                     fontsize=10)

    for ax in axes[n:]:
        ax.axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=s, alpha=0.75) for s in shades]
    handles += [plt.Line2D([], [], color="#555555", ls="--"),
                plt.Line2D([], [], color="#c62828", lw=2),
                plt.Line2D([], [], ls="none", marker="^", color="#2e7d32",
                           ms=11, mec="k"),
                plt.Line2D([], [], ls="none", marker="X", color="#c62828",
                           ms=12, mec="k")]
    labels = [f"tube at j={j}" for j in sorted(args.steps, reverse=True)] + [
        "A* centerline", "sampled plan", "start", "goal"]
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("DSTT safety tube contracting over the reverse diffusion (j: 255 -> 0)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out = (pathlib.Path(args.out) if args.out
           else plans_dir_for(args.grids[0]) / "safety_tubes.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("wrote", out)

    # --- companion figure: the radius profile along the horizon ---------------
    # The overlay above shows where the tube is; this shows why it is that wide.
    # The cap d(k)-margin is drawn against the radii so the binding constraint is
    # visible: wherever a radius curve runs along the cap, it is the map -- not
    # the contraction schedule -- setting the tube width, which is precisely the
    # behaviour the r_min floor used to prevent.
    fig2, axes2 = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.4 * nrow),
                               sharex=True)
    axes2 = np.atleast_1d(axes2).ravel()
    for ax, pr in zip(axes2, profiles):
        k = np.arange(len(pr["clearance"]))
        ax.plot(k, pr["clearance"], color="#444444", lw=1.4, label="clearance d(k)")
        ax.plot(k, np.clip(pr["clearance"] - args.radius_margin, 0, None),
                color="#c62828", lw=1.2, ls=":", label=f"cap d(k)-{args.radius_margin}")
        for shade, j in zip(shades, sorted(args.steps, reverse=True)):
            ax.plot(k, pr["radii"][j], color=shade, lw=1.6, label=f"r_j, j={j}")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_title(pr["name"], fontsize=10)
        ax.set_ylabel("metres")
        ax.set_xlabel("horizon step k")
    for ax in axes2[len(profiles):]:
        ax.axis("off")
    h, l = axes2[0].get_legend_handles_labels()
    fig2.legend(h, l, loc="lower center", ncol=7, frameon=False,
                bbox_to_anchor=(0.5, -0.02))
    fig2.suptitle("Tube radius along the horizon: the map caps it, not the schedule",
                  fontsize=13)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.97])
    out2 = out.with_name(out.stem + "_profiles.png")
    fig2.savefig(out2, dpi=115, bbox_inches="tight")
    print("wrote", out2)


if __name__ == "__main__":
    main()
