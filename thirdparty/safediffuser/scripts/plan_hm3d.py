#!/usr/bin/env python3
"""Plan with SafeDiffuser + DSTT on a MASt3R-SLAM occupancy grid.

    python3 scripts/plan_hm3d.py --grid <path/to/scene.npy> --n-plans 4

This is plan_maze2d.py with the d4rl environment removed. There is no simulator
to step and no reward to accumulate: the occupancy grid *is* the world, the
plan is judged directly against it, and a rollout would only re-run a PD
controller we have no dynamics model for here. What remains is the part under
test -- the pretrained maze2d prior, steered at sampling time by a tube built
from real geometry.

The prior is used zero-shot. It was trained on maze2d-large-v1 and has never
seen a house, so it contributes smooth, goal-reaching motion while the DSTT
guidance contributes the obstacles. Whether that trade holds is what the
printed clearance statistics are there to answer.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from diffuser.models.temporal import TemporalUnet          # noqa: E402
from diffuser.hm3d.diffusion import HM3DGaussianDiffusion  # noqa: E402
from diffuser.hm3d.map import HM3DMap                      # noqa: E402
from diffuser.hm3d import planner                          # noqa: E402
from diffuser.hm3d import plans_dir_for                     # noqa: E402
from diffuser.hm3d.tube import radii_at_batch, tube_mask     # noqa: E402

DEFAULT_CKPT = "logs/pretrained/maze2d-large-v1/diffusion/H384_T256/state_1920000.pt"


def load_model(ckpt, device):
    """Rebuild the trained architecture and load its EMA weights.

    The shapes are the ones recorded in the run's model/diffusion configs
    (transition_dim 6 = 2 action + 4 observation, dim_mults (1,4,8)); they are
    hardcoded rather than unpickled because the pickles reference the d4rl
    dataset class, which needs mujoco to import.
    """
    model = TemporalUnet(horizon=384, transition_dim=6, cond_dim=4,
                         dim_mults=(1, 4, 8))
    diffusion = HM3DGaussianDiffusion(
        model, horizon=384, observation_dim=4, action_dim=2, n_timesteps=256,
        loss_type='l2', clip_denoised=True, predict_epsilon=False,
        action_weight=1, loss_weights=None, loss_discount=1)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    diffusion.load_state_dict(state['ema'])
    print(f"loaded EMA weights from step {state['step']:,}")
    return diffusion.to(device).eval()


def to_norm(yx, norm_mins, norm_maxs):
    return 2.0 * (np.asarray(yx) - norm_mins) / (norm_maxs - norm_mins) - 1.0


def to_real(yx_norm, norm_mins, norm_maxs):
    return (np.asarray(yx_norm) + 1.0) * 0.5 * (norm_maxs - norm_mins) + norm_mins


def evaluate(traj_yx, hm3d_map, goal_yx):
    """Judge a plan against the grid it was planned on."""
    clearance = hm3d_map.clearance_at(traj_yx)
    step = np.linalg.norm(np.diff(traj_yx, axis=0), axis=1)
    return {
        "collision_frac": float((clearance <= 0).mean()),
        "min_clearance_m": float(clearance.min()),
        "mean_clearance_m": float(clearance.mean()),
        "path_length_m": float(step.sum()),
        "max_step_m": float(step.max()),
        "goal_error_m": float(np.linalg.norm(traj_yx[-1] - goal_yx)),
    }


def _mean_pairwise(trajs):
    """Mean over pairs of the per-step mean distance between two trajectories."""
    if len(trajs) < 2:
        return 0.0
    d = [float(np.linalg.norm(trajs[a] - trajs[b], axis=1).mean())
         for a in range(len(trajs)) for b in range(a + 1, len(trajs))]
    return float(np.mean(d))


def report_diversity(plans):
    """Separate the diversity the planner supplied from the model's own.

    Two numbers, because they support different claims and conflating them is
    how a diversity result stops meaning anything:

    within_route_spread_m
        Samples drawn against the *same* tube from the *same* endpoints. Nothing
        but the diffusion sampling distinguishes them, so this is the generative
        number -- the one an optimiser or a regressor cannot produce at all,
        because both return exactly one trajectory per query.

    between_route_spread_m
        Spread across tubes. This one is the A* route search's, not the
        model's. It measures scene coverage, and it is reported so that it is
        never mistaken for the line above. With a single route there are no
        pairs to compare and it is necessarily zero, which the line says
        outright rather than leaving it to look like a failure.
    """
    out = {}
    for q in sorted({p["query"] for p in plans}):
        qp = [p for p in plans if p["query"] == q]
        routes = sorted({p["route"] for p in qp})
        within = [_mean_pairwise([p["traj"] for p in qp if p["route"] == r])
                  for r in routes]
        firsts = [next(p["traj"] for p in qp if p["route"] == r) for r in routes]
        out[q] = {
            "n_trajectories": len(qp),
            "n_routes": len(routes),
            "within_route_spread_m": float(np.mean(within)) if within else 0.0,
            "between_route_spread_m": _mean_pairwise(firsts),
            "collision_free_frac": float(
                np.mean([p["metrics"]["collision_frac"] == 0.0 for p in qp])),
        }
    print("\n" + "=" * 60)
    print("diversity")
    for q, d in out.items():
        print(f"  query {q}: {d['n_trajectories']} trajectories over "
              f"{d['n_routes']} routes")
        print(f"    within-route spread  {d['within_route_spread_m']:.3f} m "
              f"<- diffusion (generative)")
        print(f"    between-route spread {d['between_route_spread_m']:.3f} m "
              f"<- A* (scene coverage)"
              + ("  [1 route: no pairs to compare]" if d['n_routes'] < 2 else ""))
        print(f"    collision-free       {d['collision_free_frac'] * 100:.0f}% "
              f"of trajectories")
    return out


def render(hm3d_map, plans, out_path, tube_steps=(255, 0), crop=True):
    """Trajectories and their tubes over the map, in cell coordinates.

    The tube is the object the guidance actually enforces, so it is drawn, not
    just the curve that came out of it. Two diffusion steps are shown per plan:
    j=N, the widest the tube ever is and the room the prior had to shape the
    path, and j=0, what it has contracted to by the time sampling ends -- the
    trajectory has to be inside that one. Drawing all 256 would be a smear;
    drawing only j=0 would hide the fact that the tube moves.

    Filled at low alpha rather than outlined only, because with several plans
    the outlines cross and it stops being clear which ribbon belongs to which
    trajectory. The j=0 outline is drawn solid on top so the binding boundary
    stays legible where the fills overlap.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    cat = np.zeros(hm3d_map.prob.shape)
    cat[hm3d_map.free] = 1.0
    cat[hm3d_map.undetermined] = 2.0
    cat[hm3d_map.occupied] = 3.0
    cat[hm3d_map.exterior] = 4.0
    ax.imshow(cat, origin="lower", interpolation="nearest",
              cmap=matplotlib.colors.ListedColormap(
                  ["#d9d9d9", "#ffffff", "#f0c674", "#1a1a1a", "#efe6da"]),
              vmin=0, vmax=4)

    def to_cells(yx):
        return ((yx[:, 1] - hm3d_map.origin_x) / hm3d_map.res - 0.5,
                (yx[:, 0] - hm3d_map.origin_y) / hm3d_map.res - 0.5)

    j_wide, j_final = max(tube_steps), min(tube_steps)
    for i, p in enumerate(plans):
        color = f"C{p.get('route', i)}"
        if not p.get("draw_tube", True):
            continue
        for j, alpha, lw, ls in [(j_wide, 0.13, 0.6, ":"),
                                 (j_final, 0.30, 1.2, "-")]:
            if j not in p.get("radii", {}):
                continue
            mask = tube_mask(hm3d_map, p["centerline"], p["radii"][j]).astype(float)
            ax.contourf(mask, levels=[0.5, 1.5], colors=[color], alpha=alpha)
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=lw,
                       linestyles=ls)

    for i, p in enumerate(plans):
        cx, cy = to_cells(p["centerline"])
        tx, ty = to_cells(p["traj"])
        r = p.get("route", i)
        first = p.get("draw_tube", True)
        ax.plot(cx, cy, "--", color="#555555", lw=0.9,
                label="route centerline" if i == 0 else None)
        ax.plot(tx, ty, "-", lw=1.6, alpha=0.85, color=f"C{r}",
                label=(f"route {r} (min clear "
                       f"{p['metrics']['min_clearance_m']:.2f} m)") if first else None)
        # Distinct shapes, not just distinct colours: with eight plans the
        # colours repeat visually once tubes are drawn over them, and a circle
        # at both ends leaves no way to tell which end the plan started from.
        # The index label is what pairs a start with its own goal.
        if not first:
            continue
        ax.plot(cx[0], cy[0], "^", color=f"C{r}", ms=9, mec="k", mew=0.8, zorder=6)
        ax.plot(cx[-1], cy[-1], "X", color=f"C{r}", ms=10, mec="k", mew=0.8, zorder=6)
        for cxx, cyy in [(cx[0], cy[0]), (cx[-1], cy[-1])]:
            ax.annotate(str(r), (cxx, cyy), textcoords="offset points",
                        xytext=(7, 5), fontsize=7, fontweight="bold",
                        color="k", zorder=7,
                        path_effects=[pe.withStroke(linewidth=2, foreground="w")])

    if crop:
        # The plans cover a fraction of the grid, and at full extent a 0.2 m
        # tube is a couple of pixels wide. Crop to what was planned in, plus a
        # metre of surrounding map for context.
        allpts = np.concatenate([p["traj"] for p in plans]
                                + [p["centerline"] for p in plans])
        ax_x, ax_y = to_cells(allpts)
        pad = int(1.0 / hm3d_map.res)
        ax.set_xlim(max(0, ax_x.min() - pad), min(hm3d_map.n_cols, ax_x.max() + pad))
        ax.set_ylim(max(0, ax_y.min() - pad), min(hm3d_map.n_rows, ax_y.max() + pad))

    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], ls="none", marker="^", color="#444444",
                           ms=9, mec="k"),
                plt.Line2D([], [], ls="none", marker="X", color="#444444",
                           ms=10, mec="k"),
                plt.Rectangle((0, 0), 1, 1, fc="#777777", alpha=0.13),
                plt.Rectangle((0, 0), 1, 1, fc="#777777", alpha=0.30)]
    labels += ["start (numbered)", "goal (numbered)",
               f"tube $\\Gamma_{{j={j_wide}}}$ (widest)",
               f"tube $\\Gamma_{{j={j_final}}}$ (final)"]
    ax.set_xticks([])
    ax.set_yticks([])
    # Outside the axes: the crop is tight by design, and an inset legend covers
    # the part of the map the crop was made to show.
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=7.5, frameon=False)
    ax.set_title(f"{hm3d_map.name} -- SafeDiffuser+DSTT, zero-shot maze2d prior")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print("wrote", out_path)


def _check_endpoint(hm3d_map, nav, args, name, yx):
    """One endpoint, in world (y, x), against the set A* will actually search."""
    r, c = hm3d_map.world_to_cell(np.asarray([yx]))
    r, c = int(r[0]), int(c[0])
    if not nav[r, c]:
        raise SystemExit(
            f"--{name} ({yx[0]:.2f}, {yx[1]:.2f}) is not navigable: cell "
            f"({r}, {c}) has clearance {hm3d_map.clearance[r, c]:+.3f} m at "
            f"robot_radius {args.robot_radius} m and unknown_slack "
            f"{args.unknown_slack} m. Endpoints are world (y, x) metres -- note "
            "the order; the grid yaml's origin is [x, y, 0]."
        )
    return np.asarray(yx, dtype=float), (r, c)


def validate_explicit_endpoints(hm3d_map, args):
    """Reject an unusable endpoint before the checkpoint is loaded.

    Called ahead of `load_model` on purpose: a mistyped goal is most often an
    axis swap, and waiting until the sampler has 1.9 M steps of
    weights in memory to say so wastes a minute to deliver a one-line message.
    Draws no random numbers, so a run with no explicit endpoints is untouched.
    """
    if args.start is None and args.goal is None:
        return
    nav = planner.navigable(hm3d_map)
    if args.start is not None and args.goal is not None:
        _, start_rc = _check_endpoint(hm3d_map, nav, args, "start", args.start)
        _, goal_rc = _check_endpoint(hm3d_map, nav, args, "goal", args.goal)
        comp = planner.component_containing(nav, start_rc)
        if comp is None or not comp[goal_rc]:
            raise SystemExit(
                f"--start ({args.start[0]:.2f}, {args.start[1]:.2f}) and --goal "
                f"({args.goal[0]:.2f}, {args.goal[1]:.2f}) are both navigable "
                "but lie in different components, so no path between them "
                "exists. Widening --unknown-slack joins components by licensing "
                "travel through unobserved space; do that deliberately."
            )
        return
    name = "start" if args.start is not None else "goal"
    _check_endpoint(hm3d_map, nav, args, name,
                    args.start if args.start is not None else args.goal)


def resolve_endpoints(hm3d_map, rng, args):
    """Settle one query's start and goal, honouring whatever the caller fixed.

    With neither flag this is exactly `sample_endpoints` as before -- same call,
    same rng draws -- because that is what every existing runbook invocation
    expects and a benchmark whose problems moved is not the same benchmark.

    With one or both given, `validate_explicit_endpoints` has already checked
    them against the navigable set, so what is left here is drawing the other
    end when only one was supplied.
    """
    if args.start is None and args.goal is None:
        return planner.sample_endpoints(
            hm3d_map, rng, min_separation=args.min_separation)
    if args.start is not None and args.goal is not None:
        return (np.asarray(args.start, dtype=float),
                np.asarray(args.goal, dtype=float))
    fixed_is_start = args.start is not None
    fixed_yx = np.asarray(args.start if fixed_is_start else args.goal, dtype=float)
    partner = planner.sample_partner(
        hm3d_map, rng, fixed_yx, min_separation=args.min_separation)
    return (fixed_yx, partner) if fixed_is_start else (partner, fixed_yx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="path to a *.npy grid")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--n-plans", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unknown-slack", type=float, default=0.50)
    ap.add_argument("--robot-radius", type=float, default=0.20)
    # The border flood fill assumes a BUILDING: walls stop it, so only the
    # outdoors is marked hard. That precondition fails on a partially observed
    # open hall, where no closed wall loop exists and the fill pours through
    # every unobserved gap until it meets the ribbon the camera actually saw.
    # On the rover pilot grid it claimed 86.6% of the map as exterior and left
    # 0.2 m2 navigable. Turning it off returns that space to the SOFT set, so
    # unknown_slack still governs how far a plan may push into it rather than
    # it becoming free.
    ap.add_argument("--no-exclude-exterior", dest="exclude_exterior",
                    action="store_false",
                    help="do not treat border-connected unknown as a hard "
                         "obstacle. Use on open/partially-observed scenes that "
                         "are not enclosed by reconstructed walls.")
    ap.set_defaults(exclude_exterior=True)
    ap.add_argument("--radius-min", type=float, default=0.25,
                    help="tube floor r_min, metres")
    ap.add_argument("--eta", type=float, default=0.6)
    # Chosen from scripts/sweep_margin.py: the worst-case clearance a plan ends
    # up with is essentially this value, and past ~0.20 m the tube closes onto
    # the centerline and the diffusion prior stops contributing anything.
    ap.add_argument("--radius-margin", type=float, default=0.15,
                    help="gap held between the tube wall and the nearest obstacle")
    # p_sample applies pos <- pos - (lambda * 2 * phi_gain) * (pos - projection),
    # so lambda*2*phi_gain is a relaxation factor onto the tube boundary: below 1
    # it under-relaxes and leaves the trajectory outside the tube, at 1 it lands
    # exactly on it, above 1 it reflects past it and the sampler diverges (at
    # lambda 1.5, 99% of the horizon ends up inside walls). phi_gain peaks at
    # 1-exp(-mu) on the final step, so the value that projects exactly is
    # 1/(2*(1-exp(-mu))) -- 0.526 for the mu=3.0 hardcoded in p_sample. This is
    # a derived constant, not a tuned one; the paper's 0.5 under-relaxes by 5%,
    # which is precisely the residual collision rate it leaves behind.
    ap.add_argument("--lambda-stt", type=float, default=1.0 / (2.0 * (1.0 - np.exp(-3.0))),
                    help="guidance gain; stability requires lambda < 1/(1-exp(-mu))")
    ap.add_argument("--min-separation", type=float, default=4.0)
    # Endpoints in world (y, x) metres -- the order HM3DMap.cell_to_world
    # returns, NOT the (x, y) of the grid yaml's origin. Absent, endpoints are
    # sampled as before, which is what makes this a map-quality benchmark;
    # supplied, it is a navigator.
    ap.add_argument("--start", type=float, nargs=2, metavar=("Y", "X"),
                    help="explicit start in world (y, x) metres")
    ap.add_argument("--goal", type=float, nargs=2, metavar=("Y", "X"),
                    help="explicit goal in world (y, x) metres, e.g. from "
                         "fact3r-map/scripts/project_semantic_goal.py")
    ap.add_argument("--n-routes", type=int, default=4,
                    help="distinct centerlines per start/goal query (K)")
    ap.add_argument("--samples-per-route", type=int, default=4,
                    help="diffusion samples drawn per centerline (S). This is "
                         "the generative axis: same tube, same endpoints, S "
                         "different trajectories.")
    ap.add_argument("--route-min-deviation", type=float, default=0.40,
                    help="mean separation, metres, below which two routes are "
                         "treated as the same route")
    # The contraction floor. At 0 the tube closes to r_min and every sample of
    # a route collapses onto the same curve, which is the behaviour that made
    # the batch redundant; raising it leaves the sampler corridor width to
    # differ in. Safe at any value -- see compute_dstt_tube.
    ap.add_argument("--spread", type=float, default=0.6,
                    help="fraction of the spatial radius the tube keeps at j=0")
    ap.add_argument("--no-guidance", action="store_true",
                    help="ablate DSTT: sample the bare prior")
    ap.add_argument("--out", default=None,
                    help="output dir; defaults to the grid's run plans/ folder")
    ap.add_argument("--tube-steps", type=int, nargs=2, default=[255, 0],
                    metavar=("J_WIDE", "J_FINAL"),
                    help="the two diffusion steps whose tubes are drawn")
    ap.add_argument("--full-extent", action="store_true",
                    help="draw the whole grid instead of cropping to the plans")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hm3d_map = HM3DMap.load(args.grid, robot_radius=args.robot_radius,
                            unknown_slack=args.unknown_slack,
                            exclude_exterior=args.exclude_exterior)
    print(hm3d_map.summary())
    norm_mins, norm_maxs = hm3d_map.norm_frame()
    print(f"  norm box side {norm_maxs[0] - norm_mins[0]:.2f} m "
          f"({(norm_maxs[0] - norm_mins[0]) / 2:.3f} m per normalized unit)")

    # Before the checkpoint: an unreachable endpoint is a one-line message and
    # there is no reason to spend a model load to deliver it.
    validate_explicit_endpoints(hm3d_map, args)

    diffusion = load_model(args.ckpt, device)
    diffusion.use_stt_guidance = not args.no_guidance
    diffusion.lambda_stt = args.lambda_stt

    rng = np.random.default_rng(args.seed)

    if args.start is not None and args.goal is not None and args.n_plans > 1:
        print(f"note: both endpoints are fixed, so all {args.n_plans} queries "
              "are the same problem; the spread they show is the sampler's")

    plans = []
    for i in range(args.n_plans):
        start_yx, goal_yx = resolve_endpoints(hm3d_map, rng, args)

        # K genuinely different routes between the SAME endpoints, each
        # repeated S times so the sampler also gets to differ within a route.
        # The batch is K*S trajectories from one denoising pass.
        lines, raws = planner.diverse_centerlines(
            hm3d_map, start_yx, goal_yx, diffusion.horizon,
            k=args.n_routes, min_deviation=args.route_min_deviation)
        K = len(lines)
        S = args.samples_per_route
        batch_lines = np.repeat(lines, S, axis=0)          # (K*S, H, 2)
        route_of = np.repeat(np.arange(K), S)
        B = K * S

        diffusion.bind_map(hm3d_map, batch_lines, radius_min_real=args.radius_min,
                           eta=args.eta, radius_margin=args.radius_margin,
                           spread=args.spread)

        s_n = to_norm(start_yx, norm_mins, norm_maxs)
        g_n = to_norm(goal_yx, norm_mins, norm_maxs)
        # Velocity dims are left at 0 in normalized space: the maze2d velocity
        # range is symmetric, so 0 is "at rest" and it keeps the endpoints from
        # implying a direction the map knows nothing about.
        #
        # Every row of the batch carries the same endpoints -- the query is one
        # start/goal pair. What differs between rows is the bound tube and the
        # sampling noise, which is precisely the point: any spread in the output
        # is produced by the routes and the prior, not by asking B questions.
        start_row = torch.tensor([s_n[0], s_n[1], 0.0, 0.0], dtype=torch.float32,
                                 device=device)
        goal_row = torch.tensor([g_n[0], g_n[1], 0.0, 0.0], dtype=torch.float32,
                                device=device)
        cond = {
            0: start_row.repeat(B, 1),
            diffusion.horizon - 1: goal_row.repeat(B, 1),
        }

        # Pulled from the bound model before sampling, so the tubes that get
        # plotted are the same objects the guidance projected onto -- not a
        # second evaluation of the formula that could drift from them.
        radii = radii_at_batch(diffusion, cond[0][:, 0:2],
                               cond[diffusion.horizon - 1][:, 0:2], args.tube_steps)

        t0 = time.time()
        with torch.no_grad():
            x = diffusion.conditional_sample(cond, horizon=diffusion.horizon,
                                             return_diffusion=False, verbose=False)
        elapsed = time.time() - t0
        print(f"\nquery {i}: start ({start_yx[0]:.2f}, {start_yx[1]:.2f}) -> "
              f"goal ({goal_yx[0]:.2f}, {goal_yx[1]:.2f})  "
              f"{K} routes x {S} samples = {B} trajectories in {elapsed:.1f}s")

        j_final = min(args.tube_steps)
        for b in range(B):
            traj_yx = to_real(x[b, :, 2:4].cpu().numpy(), norm_mins, norm_maxs)
            line = batch_lines[b]
            metrics = evaluate(traj_yx, hm3d_map, goal_yx)
            metrics["query"] = i
            metrics["route"] = int(route_of[b])
            metrics["sample_time_s"] = round(elapsed / B, 2)
            metrics["centerline_length_m"] = float(
                np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
            metrics["astar_clearance_min_m"] = float(hm3d_map.clearance_at(line).min())
            metrics["centerline_dev_mean_m"] = float(
                np.linalg.norm(traj_yx - line, axis=1).mean())
            # Whether the trajectory actually ended up inside the set the
            # guidance was enforcing. Nothing structurally guarantees it --
            # p_sample applies a soft additive nudge, not a hard projection --
            # so it is measured.
            #
            # The tolerance is not cosmetic. stt_guidance is zero strictly
            # inside the tube and, at the derived lambda, an exact projection
            # outside it, so its fixed point for any point the prior pushes out
            # is the boundary itself. A bare `dev > r` test then just reports
            # which side of the boundary float32 rounding landed on -- it reads
            # ~60% while the worst actual overshoot is a few microns.
            # `max_tube_excess_m` is the number that carries the information;
            # the fraction is kept for the tail.
            dev = np.linalg.norm(traj_yx - line, axis=1)
            excess = dev - radii[j_final][b]
            metrics["max_tube_excess_m"] = float(excess.max())
            metrics["outside_final_tube_frac"] = float((excess > 1e-4).mean())
            metrics["final_tube_radius_mean_m"] = float(radii[j_final][b].mean())

            plans.append({"traj": traj_yx, "centerline": line,
                          "raw": raws[int(route_of[b])],
                          "start": start_yx, "goal": goal_yx, "metrics": metrics,
                          "radii": {j: radii[j][b] for j in args.tube_steps},
                          "route": int(route_of[b]), "query": i,
                          # One tube per route, not per sample: the S samples of
                          # a route share a tube, and drawing it S times just
                          # darkens it.
                          "draw_tube": b % S == 0})
            print(f"    route {route_of[b]} sample {b % S}: "
                  f"collision {metrics['collision_frac']:.3f}  "
                  f"min clear {metrics['min_clearance_m']:.2f} m  "
                  f"goal err {metrics['goal_error_m']:.2f} m  "
                  f"len {metrics['path_length_m']:.1f} m")

    diversity = report_diversity(plans)

    out = pathlib.Path(args.out) if args.out else plans_dir_for(args.grid)
    tag = "prior" if args.no_guidance else "dstt"
    render(hm3d_map, plans, out / f"{hm3d_map.name}_{tag}.png",
           tube_steps=args.tube_steps, crop=not args.full_extent)
    summary = {
        "grid": str(args.grid), "guidance": not args.no_guidance,
        "start": list(args.start) if args.start else None,
        "goal": list(args.goal) if args.goal else None,
        "endpoints": ("explicit" if args.start is not None and args.goal is not None
                      else "sampled" if args.start is None and args.goal is None
                      else "half-explicit"),
        "unknown_slack": args.unknown_slack, "robot_radius": args.robot_radius,
        "radius_min": args.radius_min, "eta": args.eta,
        "lambda_stt": args.lambda_stt,
        "radius_margin": args.radius_margin,
        "navigable_area_m2": float(
            planner.largest_component(planner.navigable(hm3d_map)).sum() * hm3d_map.res ** 2),
        "route_source": "astar",
        "n_routes": args.n_routes,
        "samples_per_route": args.samples_per_route,
        "spread": args.spread,
        "diversity": {str(k): v for k, v in diversity.items()},
        "plans": [p["metrics"] for p in plans],
    }
    (out / f"{hm3d_map.name}_{tag}.json").write_text(json.dumps(summary, indent=2))

    # The trajectories themselves, for scoring against habitat's navmesh. That
    # has to happen in a different interpreter -- habitat_sim and torch live in
    # different conda envs here -- so the handoff is a file, in the grid's own
    # plane coordinates, which is what grids.json's Sim(3) expects as input.
    np.savez(out / f"{hm3d_map.name}_{tag}.npz",
             traj=np.stack([p["traj"] for p in plans]),
             centerline=np.stack([p["centerline"] for p in plans]),
             start=np.stack([p["start"] for p in plans]),
             goal=np.stack([p["goal"] for p in plans]),
             stem=hm3d_map.name,
             route=np.array([p["route"] for p in plans]),
             query=np.array([p["query"] for p in plans]),
             tube_steps=np.array(args.tube_steps),
             tube_radius=np.stack([np.stack([p["radii"][j] for j in args.tube_steps])
                                   for p in plans]),
             est_collision_frac=np.array([p["metrics"]["collision_frac"] for p in plans]))

    agg = {k: float(np.mean([p["metrics"][k] for p in plans]))
           for k in ["collision_frac", "min_clearance_m", "path_length_m",
                     "goal_error_m", "outside_final_tube_frac", "max_tube_excess_m"]}
    print("\n" + "=" * 60)
    print(f"mean over {len(plans)} trajectories ({tag}):")
    for k, v in agg.items():
        print(f"  {k:20s} {v:.3f}")


if __name__ == "__main__":
    main()
