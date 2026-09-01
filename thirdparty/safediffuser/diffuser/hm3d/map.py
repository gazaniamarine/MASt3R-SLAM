"""The HM3D occupancy grid as the planner's world.

This replaces the two hardcoded circles in `compute_dstt_tube` with a real
floorplan recovered by MASt3R-SLAM. The grid arrives from
MASt3R-SLAM/scripts/hm3d_occupancy.py as a ROS map_server pair plus the raw
int8 array:

    prob[r, c]  -1        unknown, never observed
                0..100    occupancy probability
    row 0 is the LOW-y edge, col 0 the LOW-x edge
    origin (from the .yaml) is the bottom-left CORNER in plane coordinates

so cell (r, c) has its centre at
    y = origin_y + (r + 0.5) * res,  x = origin_x + (c + 0.5) * res

Three-way, not two-way
----------------------
ROS defines only `p < free_thresh` as free and `p >= occupied_thresh` as
occupied; everything between is undetermined. On these reconstructions the
undetermined part is most of the map -- 00800 level0 is 80% unknown -- so how
it is treated decides what the planner does far more than any guidance gain.

Treating unknown as free lets plans run confidently through space that was
never observed. Treating it as a hard obstacle walls off a 20%-known map into
disconnected islands. So unknown is *soft*: an obstacle standing `unknown_slack`
metres further away than it really is.

    clearance = min(sd_known, sd_soft + unknown_slack) - robot_radius

`sd_*` are signed distances (positive outside, negative inside), so the penalty
deepens as a path pushes further into unobserved space rather than switching on
at its edge. Just past the frontier costs almost nothing; deep in the void the
clearance goes negative and the tube closes. `unknown_slack = 0` recovers hard
unknown-is-occupied, `unknown_slack = inf` recovers unknown-is-free.

The undetermined band (free_thresh..occupied_thresh) joins unknown in the soft
set for the same reason: ROS calls it "not known to be free", and on this data
it is mostly thin evidence at surface edges.

Inside vs outside
-----------------
Not all unknown space is the same kind of unknown. The grid is a rectangle
drawn around the reconstruction, so the building is ringed by unobserved cells
that are not gaps in our knowledge of the interior -- they are the outdoors.
Softening them lets a plan leave through a wall and travel around the outside
of the house, which is what actually happens on 00800: at slack 0.75 m the
largest navigable component is that halo, spanning the full grid, while the
interior stays a single room.

So unknown cells are split by a flood fill from the grid border through soft
cells: whatever the border reaches is exterior and becomes a hard obstacle, and
only unknown enclosed by observed geometry stays soft. Observed-free cells stop
the fill as well as occupied ones -- the camera stood in them, so they are
interior by construction.
"""
import pathlib

import numpy as np
from scipy.ndimage import distance_transform_edt, label

UNKNOWN = -1


def _signed_distance(mask, res):
    """Metres to the nearest True cell: positive outside it, negative inside.

    distance_transform_edt alone is one-sided -- it returns 0 everywhere inside
    the region, which would make every point inside a wall look equally bad and
    give the guidance no gradient to push a trajectory back out. Differencing
    the two transforms restores that gradient.
    """
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    if mask.all():
        return np.full(mask.shape, -np.inf, dtype=np.float32)
    outside = distance_transform_edt(~mask)
    inside = distance_transform_edt(mask)
    return ((outside - inside) * res).astype(np.float32)


class HM3DMap:

    def __init__(self, prob, res, origin_xy, *, robot_radius=0.20,
                 unknown_slack=0.50, occupied_thresh=0.65, free_thresh=0.25,
                 exclude_exterior=True, name=""):
        self.prob = np.asarray(prob)
        self.res = float(res)
        self.origin_x, self.origin_y = float(origin_xy[0]), float(origin_xy[1])
        self.robot_radius = float(robot_radius)
        self.unknown_slack = float(unknown_slack)
        self.name = name
        self.n_rows, self.n_cols = self.prob.shape

        p = self.prob
        self.unknown = p == UNKNOWN
        self.free = (p >= 0) & (p < free_thresh * 100)
        self.occupied = p >= occupied_thresh * 100
        # "Not known to be free and not known to be occupied" -- treated like
        # unknown, see the module docstring.
        self.undetermined = (p >= free_thresh * 100) & (p < occupied_thresh * 100)
        self.soft = self.unknown | self.undetermined

        self.exterior = (self._flood_from_border(self.soft) if exclude_exterior
                         else np.zeros_like(self.soft))
        self.soft = self.soft & ~self.exterior
        # The outdoors is as impassable as a wall, so it joins the hard set.
        self.blocked = self.occupied | self.exterior

        sd_known = _signed_distance(self.blocked, self.res)
        sd_soft = _signed_distance(self.soft, self.res)
        slack = self.unknown_slack
        self.clearance = np.minimum(
            sd_known,
            sd_soft + (np.inf if np.isinf(slack) else slack),
        ) - self.robot_radius

    # ---------------------------------------------------------------- io ----
    @classmethod
    def load(cls, npy_path, **kwargs):
        """Load a grid written by hm3d_occupancy.py, reading its .yaml sidecar."""
        npy_path = pathlib.Path(npy_path)
        prob = np.load(npy_path)
        yaml_path = npy_path.with_suffix(".yaml")
        meta = cls._read_yaml(yaml_path)
        return cls(prob, meta["resolution"], meta["origin"][:2],
                   occupied_thresh=meta.get("occupied_thresh", 0.65),
                   free_thresh=meta.get("free_thresh", 0.25),
                   name=npy_path.stem, **kwargs)

    @staticmethod
    def _flood_from_border(soft):
        """Soft cells connected to the grid border -- i.e. outside the building.

        4-connectivity, so the fill cannot squeeze through a diagonal pinhole in
        a wall; an 8-connected fill leaks into the interior wherever two wall
        cells only touch at a corner, which on a 20%-known map is everywhere.
        """
        lab, n = label(soft, structure=np.array([[0, 1, 0],
                                                 [1, 1, 1],
                                                 [0, 1, 0]]))
        if n == 0:
            return np.zeros_like(soft)
        border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
        touching = np.unique(border[border > 0])
        return np.isin(lab, touching)

    @staticmethod
    def _read_yaml(path):
        """Minimal reader for the five keys write_pgm_yaml emits.

        Deliberately not PyYAML: this keeps the planning path free of a
        dependency for a file whose format we control and whose every value is
        a float or a flat list of floats.
        """
        meta = {}
        for line in pathlib.Path(path).read_text().splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            val = val.strip()
            if val.startswith("["):
                meta[key.strip()] = [float(v) for v in val.strip("[]").split(",")]
            else:
                try:
                    meta[key.strip()] = float(val)
                except ValueError:
                    meta[key.strip()] = val
        return meta

    # ----------------------------------------------------------- frames ----
    def cell_to_world(self, rows, cols):
        """Cell indices -> (y, x) metres at the cell centre."""
        y = self.origin_y + (np.asarray(rows) + 0.5) * self.res
        x = self.origin_x + (np.asarray(cols) + 0.5) * self.res
        return np.stack([y, x], axis=-1)

    def world_to_cell(self, yx):
        """(y, x) metres -> integer (row, col), clipped to the grid."""
        yx = np.atleast_2d(yx)
        r = np.floor((yx[:, 0] - self.origin_y) / self.res).astype(int)
        c = np.floor((yx[:, 1] - self.origin_x) / self.res).astype(int)
        return np.clip(r, 0, self.n_rows - 1), np.clip(c, 0, self.n_cols - 1)

    @property
    def extent_m(self):
        """(y_extent, x_extent) of the whole grid, in metres."""
        return self.n_rows * self.res, self.n_cols * self.res

    def free_bounds(self):
        """(y_min, y_max, x_min, x_max) metres of the known-free bounding box."""
        rows, cols = np.nonzero(self.free)
        lo = self.cell_to_world(rows.min(), cols.min())
        hi = self.cell_to_world(rows.max(), cols.max())
        return lo[0], hi[0], lo[1], hi[1]

    def norm_frame(self, margin=0.05):
        """The square (y, x) box in metres that the prior's [-1, 1] maps onto.

        Square on purpose. GaussianDiffusion normalizes y and x independently,
        so a rectangular box would scale them by different factors and the
        prior's motion -- trained on a maze whose corridors are the same width
        in both axes -- would come out stretched. One shared metres-per-unit
        keeps a diagonal a diagonal, at the cost of some wasted [-1, 1] range on
        the shorter axis.

        Returns (norm_mins, norm_maxs) as (y, x) pairs, which is exactly what
        GaussianDiffusion.norm_mins/norm_maxs expect -- so "real" coordinates
        inside the DSTT code become metres in the grid's plane frame, and every
        length there (r_min, eta*clearance) is directly physical.
        """
        y0, y1, x0, x1 = self.free_bounds()
        cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
        side = max(y1 - y0, x1 - x0) * (1.0 + margin)
        half = 0.5 * side
        return (np.array([cy - half, cx - half], dtype=np.float64),
                np.array([cy + half, cx + half], dtype=np.float64))

    # -------------------------------------------------------- clearance ----
    def clearance_at(self, yx):
        """Bilinearly sampled clearance, in metres, at (y, x) world points.

        Bilinear rather than nearest so the field a trajectory feels is
        continuous: the tube radius is read straight off this, and a piecewise
        constant radius would make the guidance jump between neighbouring cells.
        """
        yx = np.atleast_2d(np.asarray(yx, dtype=np.float64))
        rf = (yx[:, 0] - self.origin_y) / self.res - 0.5
        cf = (yx[:, 1] - self.origin_x) / self.res - 0.5
        r0 = np.clip(np.floor(rf).astype(int), 0, self.n_rows - 1)
        c0 = np.clip(np.floor(cf).astype(int), 0, self.n_cols - 1)
        r1 = np.clip(r0 + 1, 0, self.n_rows - 1)
        c1 = np.clip(c0 + 1, 0, self.n_cols - 1)
        dr = np.clip(rf - r0, 0.0, 1.0)
        dc = np.clip(cf - c0, 0.0, 1.0)
        f = self.clearance
        return (f[r0, c0] * (1 - dr) * (1 - dc) + f[r1, c0] * dr * (1 - dc)
                + f[r0, c1] * (1 - dr) * dc + f[r1, c1] * dr * dc)

    def clearance_grad_at(self, yx):
        """Bilinearly sampled gradient of the clearance field, per metre.

        Points the way out of an obstacle, which is what the centerline repair
        needs: a point that has drifted into a wall can be walked back up this
        until it is clear again.
        """
        if not hasattr(self, '_grad'):
            gy, gx = np.gradient(self.clearance, self.res)
            self._grad = (gy.astype(np.float32), gx.astype(np.float32))
        yx = np.atleast_2d(np.asarray(yx, dtype=np.float64))
        rf = (yx[:, 0] - self.origin_y) / self.res - 0.5
        cf = (yx[:, 1] - self.origin_x) / self.res - 0.5
        r0 = np.clip(np.floor(rf).astype(int), 0, self.n_rows - 1)
        c0 = np.clip(np.floor(cf).astype(int), 0, self.n_cols - 1)
        r1 = np.clip(r0 + 1, 0, self.n_rows - 1)
        c1 = np.clip(c0 + 1, 0, self.n_cols - 1)
        dr = np.clip(rf - r0, 0.0, 1.0)[:, None]
        dc = np.clip(cf - c0, 0.0, 1.0)[:, None]
        g = np.stack(self._grad, axis=-1)
        return (g[r0, c0] * (1 - dr) * (1 - dc) + g[r1, c0] * dr * (1 - dc)
                + g[r0, c1] * (1 - dr) * dc + g[r1, c1] * dr * dc)

    def is_free(self, yx):
        r, c = self.world_to_cell(yx)
        return self.free[r, c]

    def summary(self):
        n = self.prob.size
        y0, y1, x0, x1 = self.free_bounds()
        return (
            f"{self.name}: {self.n_rows}x{self.n_cols} @ {self.res} m "
            f"({self.n_rows * self.res:.1f} x {self.n_cols * self.res:.1f} m)\n"
            f"  free {self.free.sum() / n:6.1%}   occupied {self.occupied.sum() / n:6.1%}   "
            f"unknown {self.unknown.sum() / n:6.1%}   undetermined {self.undetermined.sum() / n:6.1%}\n"
            f"  exterior {self.exterior.sum() / n:6.1%} (hard)   enclosed-soft {self.soft.sum() / n:6.1%}\n"
            f"  free bbox  y [{y0:.2f}, {y1:.2f}]  x [{x0:.2f}, {x1:.2f}] m\n"
            f"  clearance  max {self.clearance.max():.2f} m  "
            f"(robot_radius {self.robot_radius} m, unknown_slack {self.unknown_slack} m)"
        )
