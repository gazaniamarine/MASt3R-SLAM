"""GaussianDiffusion with the DSTT tube built from an occupancy grid.

Only `compute_dstt_tube` changes. Everything the paper's guidance does with the
tube -- the projection field in `stt_guidance`, the PTZF gain schedule and the
correction applied in `p_sample`, the endpoint handling in `p_sample_loop` --
is inherited untouched, so this is a swap of the obstacle model rather than a
reimplementation of the method. Subclassing rather than editing keeps the
maze2d results reproducible from the same file.

Two things differ from the maze2d tube:

Centerline
    Fixed, and computed once per plan by A* over the grid (see planner.py),
    instead of a straight line deformed by bump fields. It does not depend on
    the diffusion step, so it is set before sampling and simply broadcast here.

Radius
    The clearance d(k) is read from the map's distance field at the centerline
    points rather than from distances to two circles, and it is then *capped by
    that clearance* -- see below. The contraction is verbatim DSTT:
    r_j = r_min + exp(-lambda_r*s)*(r_spatial - r_min).

The r_min floor has to become a cap
-----------------------------------
The maze2d tube sets r_spatial = max(r_min, eta*d), a floor that stops the tube
from vanishing. Theorem 2's guarantee runs the other way -- r <= eta*d with
eta < 1 is what keeps the tube inside free space -- so wherever the floor binds,
the guarantee is void and the tube claims space that is inside a wall.

In an arena holding two circles the floor never binds, because d is metres.
In a house it binds constantly: on 00807 the centerline's clearance runs
0.05-0.40 m against an r_min of 0.25 m, so the final tube overhangs free space
on 9-21% of the horizon, and the sampled trajectory duly collides at very
nearly that rate. The guidance was never wrong; it was being handed an unsafe
tube and tracking it faithfully.

So the radius is additionally capped at d(k) - margin. Where clearance is
generous the cap is slack and behaviour is identical to the paper's; where the
corridor pinches, the tube closes onto the centerline instead of into the wall,
and the guidance pulls the trajectory to the one place that is actually free.

Because `norm_mins`/`norm_maxs` are set from the map's square metric box, the
"real" coordinates this class works in are metres in the grid's plane frame --
so r_min and eta*d are physical lengths, not maze cells.
"""
import numpy as np
import torch

from diffuser.models.diffusion import GaussianDiffusion


class HM3DGaussianDiffusion(GaussianDiffusion):

    def bind_map(self, hm3d_map, centerline_yx, radius_min_real=0.25, eta=0.6,
                 lambda_r=3.0, radius_margin=0.02, spread=0.0):
        """Attach the grid and the centerline(s) for the plan about to be sampled.

        `centerline_yx` is (H, 2) in metres, or (B, H, 2) to give each element of
        the sampled batch its own tube. The per-element form is what turns one
        query into many trajectories: B routes are bound at once and a single
        batched `conditional_sample` returns one trajectory per route, each
        certified against its own tube. It must already be resampled to the
        diffusion horizon -- the guidance pairs horizon step k with centerline
        point k, so the two lengths have to agree.
        """
        c = np.asarray(centerline_yx, dtype=np.float32)
        if c.ndim == 2:
            c = c[None]
        if c.ndim != 3 or c.shape[2] != 2:
            raise ValueError(f"centerline must be (H,2) or (B,H,2), got {c.shape}")
        if c.shape[1] != self.horizon:
            raise ValueError(
                f"centerline has {c.shape[1]} points but horizon is {self.horizon}")
        device = self.betas.device
        self.hm3d_map = hm3d_map
        self._centerline_real = torch.as_tensor(c, device=device)          # (B,H,2)
        # Clearance is a property of the (fixed) centerlines, so it is sampled
        # once here rather than at every one of the 256 denoising steps.
        clearance = np.stack([hm3d_map.clearance_at(ci) for ci in c]).astype(np.float32)
        self._clearance_real = torch.as_tensor(clearance, device=device)[..., None]
        self.spread = float(spread)
        self.radius_min_real = float(radius_min_real)
        self.eta = float(eta)
        self.lambda_r = float(lambda_r)
        self.radius_margin = float(radius_margin)

        norm_mins, norm_maxs = hm3d_map.norm_frame()
        self.norm_mins = torch.as_tensor(norm_mins, dtype=torch.float32, device=device)
        self.norm_maxs = torch.as_tensor(norm_maxs, dtype=torch.float32, device=device)
        return self

    def compute_dstt_tube(self, start_norm, goal_norm, horizon, j, **kwargs):
        """Gamma_j(k) around the A* centerline. Returns metres, as the base class does."""
        if not hasattr(self, '_centerline_real'):
            raise RuntimeError("call bind_map() before sampling")
        B = start_norm.shape[0]
        # One bound centerline is broadcast to the whole batch; B of them are
        # used as-is, one per element.
        c_real = self._centerline_real
        d = self._clearance_real
        if c_real.shape[0] != B:
            if c_real.shape[0] != 1:
                raise ValueError(
                    f"bound {c_real.shape[0]} centerlines but batch is {B}")
            c_real = c_real.expand(B, -1, -1)
            d = d.expand(B, -1, -1)

        # r_spatial(k) = max(r_min, eta*d(k)) -- the paper's floor.
        r_spatial = torch.maximum(
            torch.full((B, horizon, 1), self.radius_min_real, device=c_real.device),
            self.eta * d,
        )

        # ...then capped so the tube can never reach past the free space it sits
        # in. Clearance is already net of the robot radius and goes negative
        # inside obstacles, so the clamp at 0 is what stops a pinched centerline
        # from producing a negative radius (which would invert the projection in
        # stt_guidance and push the trajectory away from the only free space).
        r_cap = torch.clamp(d - self.radius_margin, min=0.0)
        r_spatial = torch.minimum(r_spatial, r_cap)

        # Prescribed-time contraction: wide at j=N, collapsing at j=0. The floor
        # it contracts toward is capped too, for the same reason.
        r_min = torch.minimum(
            torch.full_like(r_cap, self.radius_min_real), r_cap)

        # How far the contraction is allowed to close. At spread=0 the tube
        # contracts to r_min, which on a house is 3-25 cm -- narrow enough that
        # every sample in the batch is projected onto essentially the same
        # curve, and the batch returns one trajectory B times over. `spread`
        # raises that floor toward the spatial ceiling, leaving the sampler the
        # width of the corridor to differ in.
        #
        # This costs no safety at any setting: r_spatial is already capped at
        # d - margin, so a floor between r_min and r_spatial is bounded by the
        # same cap, and Theorem 2's r <= eta*d still holds.
        r_floor = torch.maximum(r_min, self.spread * r_spatial)
        s = (self.n_timesteps - j) / self.n_timesteps
        phi_j = float(np.exp(-self.lambda_r * s))
        return c_real, r_floor + phi_j * (r_spatial - r_floor)
