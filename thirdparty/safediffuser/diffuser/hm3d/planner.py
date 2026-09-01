"""A centerline for the DSTT tube that a floorplan can actually support.

`compute_dstt_tube` builds its nominal centerline as a straight start->goal
line and bends it with bump functions around whichever obstacle it hits. That
is sound for the maze2d demo, where the obstacles are two convex circles in an
open arena: a local sideways push always gets you around them.

A house is not that. 00807's navigable region is a room joined to a 16 m
corridor; the straight line between two rooms leaves free space almost
immediately and stays out, and no amount of local deformation recovers it,
because getting there means going *through a doorway* that may be nowhere near
the straight line. The deformation is a local fix to a global problem.

So the centerline comes from a search over the grid instead. What DSTT needs
from it is unchanged -- a path c(k) with a clearance profile d(k) -- and every
downstream piece (the r = eta*d ceiling, the PTZF contraction, the projection
guidance) is untouched. Only the source of c(k) moves from a straight line to
A*.

The search does not minimise length. A shortest path hugs corners, which is
exactly where clearance goes to zero and the tube collapses to r_min; steering
it toward the middle of a corridor costs a little distance and buys a radius
the guidance can work inside.
"""
import heapq

import numpy as np
from scipy.ndimage import label

# 8-connected: a 4-connected path through a diagonal doorway staircases, and
# the resampled centerline then carries a sawtooth the tube has to absorb.
_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1), (0, 1),
               (1, -1), (1, 0), (1, 1)]


def navigable(hm3d_map):
    """Cells the robot centre may occupy: clearance already nets off its radius."""
    return hm3d_map.clearance > 0


def largest_component(mask):
    lab, n = label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (sizes.argmax() + 1)


def component_containing(mask, rc):
    """The connected component of `mask` holding cell `rc`, or None if it is not
    in one.

    Same 8-connectivity as `largest_component`, and for the same reason A* is
    8-connected: a goal reachable only through a diagonal pinch is reachable.
    """
    lab, n = label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return None
    target = int(lab[int(rc[0]), int(rc[1])])
    return None if target == 0 else lab == target


def sample_partner(hm3d_map, rng, fixed_yx, min_separation=4.0, tries=500):
    """One navigable point in the same component as an endpoint already fixed.

    `sample_endpoints` invents both ends; this is the half of it that applies
    once a caller has supplied a goal (or a start) of its own and wants the
    other end drawn from somewhere a path to it actually exists. Same
    same-component requirement, same best-effort fallback when the component is
    too small to hold `min_separation`.
    """
    nav = navigable(hm3d_map)
    r, c = hm3d_map.world_to_cell(fixed_yx)
    comp = component_containing(nav, (int(r[0]), int(c[0])))
    if comp is None:
        raise ValueError("the fixed endpoint is not in any navigable component")
    rows, cols = np.nonzero(comp)
    fixed = np.asarray(fixed_yx, dtype=float)
    best = None
    for _ in range(tries):
        i = rng.integers(0, len(rows))
        candidate = hm3d_map.cell_to_world(rows[i], cols[i])
        d = np.linalg.norm(candidate - fixed)
        if d >= min_separation:
            return candidate
        if best is None or d > best[0]:
            best = (d, candidate)
    return best[1]


def astar(hm3d_map, start_rc, goal_rc, clearance_ref=0.50, clearance_weight=4.0,
          extra_penalty=None):
    """Least-cost cell path from start to goal, biased away from walls.

    Step cost is the geometric length scaled by a penalty that rises as
    clearance falls below `clearance_ref`, so the path prefers the middle of a
    corridor but will still squeeze through a tight doorway when that is the
    only way. The heuristic stays the plain Euclidean distance: the penalty is
    >= 1 everywhere, so straight-line distance never overestimates and A*
    remains admissible.
    """
    nav = navigable(hm3d_map)
    n_rows, n_cols = nav.shape
    if not nav[start_rc] or not nav[goal_rc]:
        raise ValueError("start or goal is not navigable")

    clearance = hm3d_map.clearance
    penalty = 1.0 + clearance_weight * np.clip(
        (clearance_ref - clearance) / clearance_ref, 0.0, 1.0)
    # `extra_penalty` is how diverse_centerlines pushes successive searches off
    # the routes it already has. It is only ever >= 0, so `penalty` stays >= 1
    # and the Euclidean heuristic below remains admissible.
    if extra_penalty is not None:
        penalty = penalty + extra_penalty

    res = hm3d_map.res
    goal_arr = np.array(goal_rc, dtype=float)

    def h(rc):
        return np.hypot(*(np.array(rc, dtype=float) - goal_arr)) * res

    open_heap = [(h(start_rc), 0.0, start_rc)]
    came_from = {}
    g_score = {start_rc: 0.0}
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal_rc:
            break
        closed.add(cur)
        r, c = cur
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols) or not nav[nr, nc]:
                continue
            step = res * (1.41421356 if dr and dc else 1.0)
            # Average the endpoint penalties so the cost of entering a cell does
            # not depend on which side it is entered from.
            ng = g + step * 0.5 * (penalty[r, c] + penalty[nr, nc])
            if ng < g_score.get((nr, nc), np.inf):
                g_score[(nr, nc)] = ng
                came_from[(nr, nc)] = cur
                heapq.heappush(open_heap, (ng + h((nr, nc)), ng, (nr, nc)))
    else:
        raise ValueError("no path between start and goal")

    path = [goal_rc]
    while path[-1] != start_rc:
        path.append(came_from[path[-1]])
    path.reverse()
    rows = np.array([p[0] for p in path])
    cols = np.array([p[1] for p in path])
    return hm3d_map.cell_to_world(rows, cols)


def smooth(path_yx, hm3d_map, iterations=60, step=0.25):
    """Pull the polyline taut, but never into a cell the robot cannot occupy.

    A* returns a path on the cell lattice, so it is faceted at 45-degree steps.
    Averaging each point with its neighbours removes the facets; the guard
    rejects any move that lands somewhere with less clearance than the robot
    needs, so smoothing can round a corner but cannot cut through one.
    """
    p = np.array(path_yx, dtype=float)
    if len(p) < 3:
        return p
    for _ in range(iterations):
        mid = 0.5 * (p[:-2] + p[2:])
        cand = p.copy()
        cand[1:-1] = p[1:-1] + step * (mid - p[1:-1])
        ok = hm3d_map.clearance_at(cand[1:-1]) > 0
        p[1:-1] = np.where(ok[:, None], cand[1:-1], p[1:-1])
    return p


def resample(path_yx, n):
    """Resample to exactly `n` points, uniform in arc length.

    The tube is indexed by horizon step k, and the guidance projects the k-th
    trajectory point onto the k-th centerline point. Uniform spacing is what
    makes that correspondence a sensible one -- with the raw A* spacing, the
    diagonal steps are 41% longer than the axis-aligned ones and the pairing
    drifts.
    """
    p = np.asarray(path_yx, dtype=float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(target, s, p[:, 0]),
                     np.interp(target, s, p[:, 1])], axis=1)


def repair(path_yx, hm3d_map, target=0.05, iterations=40, rate=0.6):
    """Walk any sub-target point back up the clearance gradient.

    A* only ever visits cells with positive clearance, but resampling draws
    straight segments between the points it returns, and a segment between two
    clear points can still clip the corner of a wall. On 00806 that put the
    centerline's worst clearance at -0.002 m, and because the tube is capped at
    the clearance, no margin setting could recover it -- the tube collapsed onto
    a centerline that was itself in collision, which is exactly the residual
    0.16% of horizon steps that survived every margin in the sweep.

    Endpoints are pinned: they are the start and goal the caller asked for, and
    moving them would silently answer a different query.
    """
    p = np.array(path_yx, dtype=float)
    for _ in range(iterations):
        d = hm3d_map.clearance_at(p)
        bad = d < target
        bad[0] = bad[-1] = False
        if not bad.any():
            break
        g = hm3d_map.clearance_grad_at(p[bad])
        n = np.linalg.norm(g, axis=1, keepdims=True)
        # Where the field is flat there is no direction to move; leaving the
        # point put is better than dividing by ~0 and flinging it somewhere.
        g = np.where(n > 1e-6, g / np.maximum(n, 1e-6), 0.0)
        p[bad] += rate * (target - d[bad])[:, None] * g
    return p


def centerline(hm3d_map, start_yx, goal_yx, horizon, target_clearance=0.05, **kw):
    """Full pipeline: A* -> smooth -> resample -> repair, at the diffusion horizon.

    Repair runs last, after resampling, because resampling is what introduces
    the violations it fixes.
    """
    r0, c0 = hm3d_map.world_to_cell(start_yx)
    r1, c1 = hm3d_map.world_to_cell(goal_yx)
    raw = astar(hm3d_map, (int(r0[0]), int(c0[0])), (int(r1[0]), int(c1[0])), **kw)
    line = resample(smooth(raw, hm3d_map), horizon)
    return repair(line, hm3d_map, target=target_clearance), raw


def sample_endpoints(hm3d_map, rng, min_separation=4.0, tries=500):
    """Two navigable points in the same component, at least `min_separation` apart.

    Same component because a start and goal on opposite sides of a wall have no
    path at all, and the failure would otherwise surface deep inside A*.
    """
    comp = largest_component(navigable(hm3d_map))
    rows, cols = np.nonzero(comp)
    if len(rows) < 2:
        raise ValueError("navigable component too small")
    best = None
    for _ in range(tries):
        i, j = rng.integers(0, len(rows), size=2)
        a = hm3d_map.cell_to_world(rows[i], cols[i])
        b = hm3d_map.cell_to_world(rows[j], cols[j])
        d = np.linalg.norm(a - b)
        if d >= min_separation:
            return a, b
        if best is None or d > best[0]:
            best = (d, a, b)
    return best[1], best[2]


def _disk(radius_cells):
    r = int(max(1, round(radius_cells)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = (yy ** 2 + xx ** 2) <= r ** 2
    return yy[keep], xx[keep]


def _deviation(a, b):
    """Mean distance between two centerlines already resampled to the same H."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1).mean())


def diverse_centerlines(hm3d_map, start_yx, goal_yx, horizon, k=4,
                        detour_radius=0.60, detour_weight=3.0,
                        min_deviation=0.40, target_clearance=0.05,
                        max_tries=None, **kw):
    """`k` centerlines between the same endpoints, each a different route.

    A* is optimal, so it returns the same path every time. Re-running it after
    stamping a cost penalty along the route it just found makes the next search
    prefer a genuinely different way through -- typically a different doorway,
    because that is where the alternatives in a house actually are. The penalty
    is additive on the step cost rather than a hard block: a corridor that is
    the *only* way through stays usable, it just gets expensive, so this
    degrades to fewer distinct routes rather than to no path at all.

    Routes closer than `min_deviation` metres (mean deviation) to one already
    kept are discarded, and the penalty is stamped anyway so the next attempt is
    pushed further. Returns `(lines, raws)` with `lines` of shape (K, H, 2),
    K <= k -- a scene with only one way between two rooms genuinely has one
    route, and inventing more would mean inventing detours nobody asked for.
    """
    r0, c0 = hm3d_map.world_to_cell(start_yx)
    r1, c1 = hm3d_map.world_to_cell(goal_yx)
    start_rc, goal_rc = (int(r0[0]), int(c0[0])), (int(r1[0]), int(c1[0]))

    extra = np.zeros_like(hm3d_map.clearance, dtype=float)
    dy, dx = _disk(detour_radius / hm3d_map.res)
    n_rows, n_cols = extra.shape

    lines, raws = [], []
    tries = max_tries if max_tries is not None else 3 * k
    for _ in range(tries):
        if len(lines) >= k:
            break
        try:
            raw = astar(hm3d_map, start_rc, goal_rc, extra_penalty=extra, **kw)
        except ValueError:
            break
        line = repair(resample(smooth(raw, hm3d_map), horizon), hm3d_map,
                      target=target_clearance)
        if all(_deviation(line, prev) >= min_deviation for prev in lines):
            lines.append(line)
            raws.append(raw)

        # Stamped whether or not the route was kept: a rejected near-duplicate
        # means the last penalty was too weak to move the search, so the only
        # way to make progress is to make that corridor cost more still.
        rows, cols = hm3d_map.world_to_cell(raw)
        rr = (rows[:, None] + dy[None, :]).ravel()
        cc = (cols[:, None] + dx[None, :]).ravel()
        ok = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
        extra[rr[ok], cc[ok]] += detour_weight

    if not lines:
        raise ValueError("no route between start and goal")
    return np.stack(lines), raws
