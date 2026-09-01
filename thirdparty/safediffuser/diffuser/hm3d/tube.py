"""Rasterising Gamma_j(k) onto the grid, so the tube can be drawn.

Gamma_j(k) = {x : |x - c(k)| <= r_j(k)} is a union of disks along the
centerline, and that is how it is drawn -- by stamping a disk of radius r_j(k)
at every horizon step -- rather than as a band between two offset curves. The
offset-curve shortcut is wrong exactly where the picture matters: on a tight
turn the outer offset opens a wedge the tube does not contain, and the inner
offset self-intersects.

Lives here rather than in a script because both plot_tube.py (one plan, four
diffusion steps) and plan_hm3d.py (many plans, the endpoints of the
contraction) need the same rasterisation, and a second copy of it would be a
second chance for the picture to disagree with the sampler.
"""
import numpy as np
import torch


def disk_offsets(radius_cells):
    r = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = (yy ** 2 + xx ** 2) <= radius_cells ** 2
    return yy[keep], xx[keep]


def tube_mask(hm3d_map, centre_yx, radius_m):
    """Union of the per-step disks, as a boolean grid mask.

    Radii are rounded to 0.01 cell before the offsets are cached: consecutive
    horizon steps have near-identical radii, so without the cache this rebuilds
    the same stencil 384 times.
    """
    mask = np.zeros(hm3d_map.prob.shape, dtype=bool)
    rows, cols = hm3d_map.world_to_cell(centre_yx)
    cache = {}
    for r0, c0, rad in zip(rows, cols, np.asarray(radius_m) / hm3d_map.res):
        key = round(float(rad), 2)
        if key not in cache:
            # Never smaller than half a cell: a tube that has contracted to
            # zero is still a curve, and dropping it would draw nothing at all
            # where the map is tightest -- which is the interesting place.
            cache[key] = disk_offsets(max(key, 0.5))
        dy, dx = cache[key]
        rr, cc = r0 + dy, c0 + dx
        ok = ((rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1]))
        mask[rr[ok], cc[ok]] = True
    return mask


def radii_at(diffusion, start_norm, goal_norm, steps):
    """{j: r_j(k)} from the bound model's own compute_dstt_tube.

    Read from the model rather than recomputed from the formula, so a plot can
    never show a tube the sampler did not use.
    """
    out = {}
    for j in steps:
        with torch.no_grad():
            _, r = diffusion.compute_dstt_tube(start_norm, goal_norm,
                                               diffusion.horizon, j)
        out[j] = r[0, :, 0].cpu().numpy()
    return out


def radii_at_batch(diffusion, start_norm, goal_norm, steps):
    """{j: r_j(k) for every batch element}, shaped (B, H).

    The batched form of `radii_at`: with one centerline bound per batch element
    the radii differ per element too, so a single (H,) row can no longer stand
    for the batch. Same contract otherwise -- read from the bound model, never
    recomputed from the formula.
    """
    out = {}
    for j in steps:
        with torch.no_grad():
            _, r = diffusion.compute_dstt_tube(start_norm, goal_norm,
                                               diffusion.horizon, j)
        out[j] = r[:, :, 0].cpu().numpy()
    return out
