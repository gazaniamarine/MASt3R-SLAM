# SafeDiffuser + DSTT, vendored

Stage 5 of the rover pipeline (`scripts/run_rover_pipeline.py`) plans with this
code. It lives here rather than in a sibling checkout so that the pipeline is
one clone and one collaborator invite.

## Provenance

| | |
|---|---|
| Upstream | https://github.com/Weixy21/SafeDiffuser (branch `maze2d`) |
| Licence | MIT, Copyright (c) 2025 Wei Xiao — see `LICENSE`, unmodified |
| Vendored from | `~/Gazania/SafeDiffuser_STT` on 2026-09-01 |

The whole `diffuser` package is copied, not a trimmed subset. It is 684 KB, and
`diffuser/models/diffusion.py` imports `diffuser.utils` which reaches a further
13 modules, so a partial copy would break on the first lazy import rather than
at load time.

## What is local to this project, not upstream

Everything under `diffuser/hm3d/` and `scripts/plan_hm3d.py` is work done here —
the occupancy grid as a planning world, A* centerlines, the DSTT tube, and the
grid-planning entry point. Upstream has no equivalent.

| file | what it is |
|---|---|
| `diffuser/hm3d/map.py` | `HM3DMap`: the three-way free/occupied/unknown grid, the `min(sd_hard, sd_soft + slack) - robot_radius` clearance field, and the exterior flood fill |
| `diffuser/hm3d/planner.py` | A* over the clearance field, plus smoothing, resampling, repair, and `diverse_centerlines` for K distinct routes |
| `diffuser/hm3d/tube.py` | the DSTT tube radii the guidance projects onto |
| `diffuser/hm3d/diffusion.py` | `HM3DGaussianDiffusion`, which binds a map and tube to the pretrained maze2d prior |
| `diffuser/hm3d/roadmap.py` | the PRM alternative to A*. **Not used by the pipeline** — it is kept because `--route-source prm` is still `plan_hm3d.py`'s own default for older benchmark invocations |
| `scripts/plan_hm3d.py` | the entry point; `--route-source astar` is what the pipeline passes |

## Weights are not in git

`--ckpt` defaults to

    thirdparty/safediffuser/logs/pretrained/maze2d-large-v1/diffusion/H384_T256/state_1920000.pt

which is 29.6 MB and matched by the repo's `logs/` ignore rule, the same way the
2.75 GB MASt3R checkpoint under `checkpoints/` is kept out. Transfer both
alongside the clone.

## Updating it

This is a copy, so it does not track upstream. If `~/Gazania/SafeDiffuser_STT`
moves ahead, re-sync with:

    rsync -a --exclude='__pycache__' --exclude='*.pyc' \
        ~/Gazania/SafeDiffuser_STT/diffuser/ thirdparty/safediffuser/diffuser/

Or point the pipeline back at an external checkout without copying anything:

    python3 scripts/run_rover_pipeline.py … --planner-root ~/Gazania/SafeDiffuser_STT
