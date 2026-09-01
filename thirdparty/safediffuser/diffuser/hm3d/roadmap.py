"""A probabilistic roadmap over the navigable grid, and diverse routes on it.

`planner.astar` answers one query optimally, and optimally is the problem: it
returns the same route every time, so a batch of B trajectories bound to it is
B copies of one trajectory. Re-running A* under a penalty does produce distinct
routes, but it pays a full grid sweep per route and each successive penalty
pushes the next route into a worse corridor, so the k-th route is often a
detour nobody would take.

A roadmap separates the two costs. Sampling and connecting nodes is done once
per scene -- it depends only on the map, not on the query -- and every route
after that is a Dijkstra over a few hundred nodes rather than a million cells.
Distinct routes come from re-querying that small graph under a node penalty,
which is the same trick as the A* version but ~1000x cheaper, and cheap enough
that near-duplicates can be discarded and retried rather than accepted.

What this does NOT do is make the routes the diffusion model's idea. It is a
classical planner and it supplies the homotopy classes. The generative claim
rests on the samples drawn *within* each bound tube, not on the count of routes
coming out of here.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .planner import largest_component, navigable, repair, resample, smooth


class Roadmap:
    """Nodes in free space plus collision-checked straight edges between them."""

    def __init__(self, nodes, src, dst, length, penalty):
        self.nodes = nodes            # (N, 2) world (y, x)
        self.src = src
        self.dst = dst
        self.length = length          # metres
        self.penalty = penalty        # clearance cost multiplier, >= 1
        self.tree = cKDTree(nodes)
        self.n = len(nodes)


def _edge_is_clear(hm3d_map, a, b, step_m=None):
    """Straight segments a->b that stay in free space, checked at map resolution.

    Sampling coarser than one cell would step over a wall one cell thick, which
    is exactly the wall a doorway is made of.
    """
    step = step_m or hm3d_map.res
    d = np.linalg.norm(b - a, axis=1)
    n = int(max(2, np.ceil(d.max() / step) + 1))
    t = np.linspace(0.0, 1.0, n).reshape(1, n, 1)
    pts = a[:, None, :] + t * (b - a)[:, None, :]
    clear = hm3d_map.clearance_at(pts.reshape(-1, 2)).reshape(len(a), n)
    return clear.min(axis=1) > 0.0, clear.mean(axis=1)


def build(hm3d_map, n_nodes=4000, k_neighbours=16, rng=None,
          clearance_ref=0.50, clearance_weight=4.0, max_edge_m=3.0):
    """Sample `n_nodes` navigable points and connect each to its k nearest.

    Nodes are drawn with probability proportional to clearance, so they land in
    the middle of rooms and corridors rather than against the walls. That is the
    roadmap's version of the clearance penalty in `planner.astar`: a route made
    of mid-corridor waypoints starts with room around it, which is what keeps
    the tube from collapsing to r_min later.
    """
    rng = rng or np.random.default_rng(0)
    comp = largest_component(navigable(hm3d_map))
    rows, cols = np.nonzero(comp)
    if len(rows) < 2:
        raise ValueError("navigable component too small")

    w = hm3d_map.clearance[rows, cols].astype(float)
    w = np.clip(w, 1e-3, None) ** 1.5
    idx = rng.choice(len(rows), size=min(n_nodes, len(rows)),
                     replace=False, p=w / w.sum())
    nodes = hm3d_map.cell_to_world(rows[idx], cols[idx])

    tree = cKDTree(nodes)
    k = min(k_neighbours + 1, len(nodes))
    _, nbr = tree.query(nodes, k=k)
    src = np.repeat(np.arange(len(nodes)), k - 1)
    dst = nbr[:, 1:].ravel()
    keep = src < dst                                   # each undirected edge once
    src, dst = src[keep], dst[keep]

    seg = np.linalg.norm(nodes[dst] - nodes[src], axis=1)
    within = seg <= max_edge_m                         # long edges are slow to check
    src, dst, seg = src[within], dst[within], seg[within]

    ok, mean_clear = _edge_is_clear(hm3d_map, nodes[src], nodes[dst])
    src, dst, seg, mean_clear = src[ok], dst[ok], seg[ok], mean_clear[ok]

    penalty = 1.0 + clearance_weight * np.clip(
        (clearance_ref - mean_clear) / clearance_ref, 0.0, 1.0)
    return Roadmap(nodes, src, dst, seg, penalty)


def _attach(roadmap, hm3d_map, point, n_try=20):
    """Index of a roadmap node reachable from `point` by a clear straight edge."""
    n_try = min(n_try, roadmap.n)
    _, cand = roadmap.tree.query(point.reshape(1, 2), k=n_try)
    cand = np.atleast_1d(cand.ravel())
    a = np.repeat(point.reshape(1, 2), len(cand), axis=0)
    ok, _ = _edge_is_clear(hm3d_map, a, roadmap.nodes[cand])
    if not ok.any():
        raise ValueError("point does not connect to the roadmap")
    return int(cand[ok.argmax()])


def _shortest(roadmap, extra, start_i, goal_i):
    """Dijkstra under an additive per-node detour penalty. Returns node indices."""
    w = roadmap.length * roadmap.penalty + 0.5 * (extra[roadmap.src] + extra[roadmap.dst])
    n = roadmap.n
    g = csr_matrix((np.concatenate([w, w]),
                    (np.concatenate([roadmap.src, roadmap.dst]),
                     np.concatenate([roadmap.dst, roadmap.src]))), shape=(n, n))
    dist, pred = dijkstra(g, indices=start_i, return_predecessors=True)
    if not np.isfinite(dist[goal_i]):
        return None
    path, cur = [goal_i], goal_i
    while cur != start_i:
        cur = int(pred[cur])
        if cur < 0:
            return None
        path.append(cur)
    return np.array(path[::-1])


def routes(roadmap, hm3d_map, start_yx, goal_yx, horizon, k=4,
           detour_radius=0.60, detour_weight=20.0, min_deviation=0.40,
           target_clearance=0.05, max_tries=None):
    """`k` distinct centerlines between the same endpoints, shape (K, H, 2).

    Same contract as `planner.diverse_centerlines`, so the two are drop-in
    alternatives: K <= k, because a scene with one way between two rooms
    genuinely has one route and inventing more would mean inventing detours.
    """
    start_yx = np.asarray(start_yx, dtype=float).reshape(2)
    goal_yx = np.asarray(goal_yx, dtype=float).reshape(2)
    start_i = _attach(roadmap, hm3d_map, start_yx)
    goal_i = _attach(roadmap, hm3d_map, goal_yx)

    extra = np.zeros(roadmap.n)
    lines, raws = [], []
    tries = max_tries if max_tries is not None else 3 * k
    for _ in range(tries):
        if len(lines) >= k:
            break
        path_i = _shortest(roadmap, extra, start_i, goal_i)
        if path_i is None:
            break
        # Endpoints are pinned to what the caller asked for; the roadmap nodes
        # are only the waypoints in between.
        raw = np.vstack([start_yx, roadmap.nodes[path_i], goal_yx])
        line = repair(resample(smooth(raw, hm3d_map), horizon), hm3d_map,
                      target=target_clearance)
        if all(np.linalg.norm(line - prev, axis=1).mean() >= min_deviation
               for prev in lines):
            lines.append(line)
            raws.append(raw)

        # Stamped whether or not the route was kept: a rejected near-duplicate
        # means the last penalty was too weak to move the search off it.
        near = roadmap.tree.query_ball_point(roadmap.nodes[path_i], r=detour_radius)
        for group in near:
            extra[group] += detour_weight

    if not lines:
        raise ValueError("no route between start and goal")
    return np.stack(lines), raws
