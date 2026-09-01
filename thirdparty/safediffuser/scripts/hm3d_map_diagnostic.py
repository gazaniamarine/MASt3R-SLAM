#!/usr/bin/env python3
"""Show what the unknown_slack knob does to a grid's navigable set.

    python3 scripts/hm3d_map_diagnostic.py --grid <path to *.npy>

The HM3D grids are ~80% unknown, so the choice of how to treat unobserved space
decides the shape of the planning problem before any diffusion runs. This draws
the occupancy map, the clearance field, and the navigable mask at several slack
values, with the largest connected component highlighted -- that component is
what a start/goal pair can actually be drawn from.
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from diffuser.hm3d.map import HM3DMap  # noqa: E402
from diffuser.hm3d import plans_dir_for  # noqa: E402


def largest_component(nav):
    lab, n = label(nav, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(nav), 0
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (sizes.argmax() + 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--slacks", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--robot-radius", type=float, default=0.20)
    ap.add_argument("--out", default=None,
                    help="output png; defaults to the run's plans/ folder")
    args = ap.parse_args()

    base = HM3DMap.load(args.grid, robot_radius=args.robot_radius)
    print(base.summary())

    ncol = 2 + len(args.slacks)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.6))

    # Grid row 0 is the low-y edge; origin="lower" makes the plot read like a map.
    kw = dict(origin="lower", interpolation="nearest")

    cat = np.full(base.prob.shape, 0.0)
    cat[base.free] = 1.0
    cat[base.undetermined] = 2.0
    cat[base.occupied] = 3.0
    cat[base.exterior] = 4.0
    axes[0].imshow(cat, cmap=matplotlib.colors.ListedColormap(
        ["#dcdcdc", "#ffffff", "#f0c674", "#1a1a1a", "#b0d8f0"]), vmin=0, vmax=4, **kw)
    axes[0].set_title(f"occupancy -- blue=exterior {base.exterior.mean():.0%}\n"
                      f"grey=enclosed unknown, black=occupied")

    cl = np.ma.masked_invalid(base.clearance)
    im = axes[1].imshow(cl, cmap="viridis", vmin=-1.0, vmax=1.0, **kw)
    axes[1].set_title("clearance (m), slack=0.50")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    for ax, slack in zip(axes[2:], args.slacks):
        m = HM3DMap.load(args.grid, robot_radius=args.robot_radius,
                         unknown_slack=slack)
        nav = m.clearance > 0
        big, n = largest_component(nav)
        shown = np.zeros(nav.shape)
        shown[nav] = 1.0
        shown[big] = 2.0
        ax.imshow(shown, cmap=matplotlib.colors.ListedColormap(
            ["#1a1a1a", "#7a4fa3", "#5ce65c"]), vmin=0, vmax=2, **kw)
        rr, cc = np.nonzero(big)
        span = ((rr.max() - rr.min()) * m.res, (cc.max() - cc.min()) * m.res) if big.any() else (0, 0)
        ax.set_title(f"slack {slack:.2f} m -- nav {nav.mean():.0%}\n"
                     f"largest comp {big.sum() / max(nav.sum(), 1):.0%} of nav, "
                     f"{span[0]:.1f}x{span[1]:.1f} m")
        print(f"slack {slack:.2f}: nav {nav.mean():.1%}, {n} components, "
              f"largest spans {span[0]:.1f} x {span[1]:.1f} m")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(base.name, fontsize=11)
    fig.tight_layout()
    out = (pathlib.Path(args.out) if args.out
           else plans_dir_for(args.grid) / f"map_diagnostic_{base.name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
