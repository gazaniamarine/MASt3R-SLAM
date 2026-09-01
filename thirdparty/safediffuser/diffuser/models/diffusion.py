import numpy as np
import torch
from torch import nn
import pdb
from torch.autograd import Variable
try:
    from qpth.qp import QPFunction, QPSolvers
except ImportError:
    from diffuser.utils.optional_deps import MissingDependency
    QPFunction = QPSolvers = MissingDependency(
        'qpth', 'the QP-based SafeDiffuser ablations (RoS/ReS/TVS diffuser)')
import einops

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.norm_mins = 0
        self.norm_maxs = 0
        self.safe1 = 0
        self.safe2 = 0

        # STT guidance flags (for ablation / testing)
        self.use_stt_guidance = True   # additive score-nudge toward the DSTT tube
        self.lambda_stt = 0.50         # guidance strength

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):

        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):

        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        if self.use_stt_guidance and hasattr(self, 'start_norm'):
            # we already have gamma_L, gamma_U from compute_dstt_tube (but not here)
            # we'll compute guidance in p_sample instead.
            pass

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()   #only for sampling
    def Shield(self, x0, xp10):  #Truncate method

        x = x0.clone()
        xp1 = xp10.clone()

        xp1 = xp1.squeeze(0)

        nBatch = xp1.shape[0]

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

        for k in range(nBatch):
            if b[k, 0] < 0: 
                theta = torch.atan2((xp1[k,2:3] - off_y)/yr, (xp1[k,3:4] - off_x)/xr)
                xp1[k,2] = yr*torch.sin(theta) + off_y
                xp1[k,3] = xr*torch.cos(theta) + off_x

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

         #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((xp1[:,2:3] - off_y)/yr)**4 + ((xp1[:,3:4] - off_x)/xr)**4 - 1

        self.safe1 = torch.min(b[:,0])
        self.safe2 = torch.min(b2[:,0])

        xp1 = xp1.unsqueeze(0)
        return xp1
    
    @torch.no_grad()   #only for sampling
    def GD(self, x0, xp10):    #classifier guidance or potential-based method

        x = x0.clone()
        xp1 = xp10.clone()

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        b = ((xp1[:,2:3] - off_y)/yr)**2 + ((xp1[:,3:4] - off_x)/xr)**2 - 1

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((xp1[:,2:3] - off_y)/yr)**4 + ((xp1[:,3:4] - off_x)/xr)**4 - 1

        for k in range(nBatch):
            if b[k, 0] < 0.1:  # 0, 0.2
                u1 = 0.2/(2*((xp1[k,2:3] - off_y)/yr)/yr)
                u2 = 0.2/(2*((xp1[k,3:4] - off_x)/xr)/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001  #note no 0.1/0.01 for GD, but has for potential
                xp1[k,3] = xp1[k,3] + u2*0.001
            elif b2[k, 0] < 0.1:  # 0, 0.2
                u1 = 0.2/(4*((xp1[k,2:3] - off_y)/yr)**3/yr)
                u2 = 0.2/(4*((xp1[k,3:4] - off_x)/xr)**3/xr)
                xp1[k,2] = xp1[k,2] + u1*0.001
                xp1[k,3] = xp1[k,3] + u2*0.001
            # else:
            #     x[k,2] = xp1[k,2]
            #     x[k,3] = xp1[k,3]

        self.safe1 = torch.min(b[:,0])
        self.safe2 = torch.min(b2[:,0])

        xp1 = xp1.unsqueeze(0)
        return xp1
    

    def _real_to_norm(self, pos_real):
        """
        Convert REAL maze (y, x) coordinates to the model's normalized
        [-1, 1] coordinates, using the SAME self.norm_mins / self.norm_maxs
        convention as Shield() / GD() / invariance(). pos_real: (..., 2)
        with last dim = (y, x).
        """
        range_y = self.norm_maxs[0] - self.norm_mins[0]
        range_x = self.norm_maxs[1] - self.norm_mins[1]
        y_norm = 2.0 * (pos_real[..., 0:1] - self.norm_mins[0]) / range_y - 1.0
        x_norm = 2.0 * (pos_real[..., 1:2] - self.norm_mins[1]) / range_x - 1.0
        return torch.cat([y_norm, x_norm], dim=-1)

    def _norm_to_real(self, pos_norm):
        """Inverse of _real_to_norm."""
        range_y = self.norm_maxs[0] - self.norm_mins[0]
        range_x = self.norm_maxs[1] - self.norm_mins[1]
        y_real = (pos_norm[..., 0:1] + 1.0) * 0.5 * range_y + self.norm_mins[0]
        x_real = (pos_norm[..., 1:2] + 1.0) * 0.5 * range_x + self.norm_mins[1]
        return torch.cat([y_real, x_real], dim=-1)

    def compute_dstt_tube(self, start_norm, goal_norm, horizon, j,
                           radius_min_real=0.35, eta=0.6, lambda_r=3.0,
                           pad=10, margin_real=0.10,
                           amplitude_cap_real=3.0, verbose=False):
        """
        Synthesizes the Diffusion Spatiotemporal Tube (DSTT) Gamma_j(k).

        The tube is diffusion-step-varying (Version 1 / DSTT-v1):
          - Nominal centerline c0(k)  [straight line, Sec I.B]
          - Obstacle clearance d(k)   [Sec I.B]
          - Spatial radius ceiling: r_spatial(k) = max(r_min, eta*d(k))  [Theorem 2]
          - Prescribed-time contraction [UniConFlow PTZF, Sec I.E upgrade]:
              phi_j = exp(-lambda_r * s),  s = (N-j)/N  in [0,1]
              s=0 at j=N (start of denoising) -> phi=1 -> wide tube
              s=1 at j=0 (end of denoising)   -> phi~0 -> tube collapses to r_min
              r_j(k) = r_min + phi_j * (r_spatial(k) - r_min)
          - Bump-supported circumvent field beta(k) = A*psi(k)*n(k)  [Sec I.D]
            (amplitude A computed from the current r_j, so deformation is
             naturally stronger when the tube is wide and weaker near convergence)

        All computation is in REAL maze coordinates; caller receives real-space
        centerline and radius (no normalization done here).

        Returns: centerline_real (B, H, 2), radius_real (B, H, 1)
        """
        device = self.betas.device
        B = start_norm.shape[0]
        H = horizon

        # ---- obstacles: single source of truth, REAL coordinates.      ----
        # ---- Same centers Shield()/GD()/invariance() use (incl. -0.5). ----
        if not hasattr(self, '_dstt_obstacle_centers_real'):
            self._dstt_obstacle_centers_real = torch.tensor([
                [5.0 - 0.5, 5.8 - 0.5],   # obstacle 1 (y, x)
                [2.0 - 0.5, 5.3 - 0.5],   # obstacle 2 (y, x)
            ], device=device)
        self._dstt_obstacle_radius_real = 1.0  # both treated as radius-1 circles for guidance

        # ---- nominal centerline c0(k), REAL coordinates [Sec I.B] ----
        start_real = self._norm_to_real(start_norm)   # (B, 2)
        goal_real = self._norm_to_real(goal_norm)      # (B, 2)
        t_phys = torch.linspace(0, 1, H, device=device).view(1, H, 1)
        c_real = start_real.unsqueeze(1) + t_phys * (goal_real.unsqueeze(1) - start_real.unsqueeze(1))  # (B,H,2)

        # ---- obstacle clearance d(k), REAL units [Sec I.B] ----
        centers = self._dstt_obstacle_centers_real.view(1, 1, -1, 2)      # (1,1,2,2)
        pts = c_real.unsqueeze(2)                                          # (B,H,1,2)
        dist_to_centers = torch.norm(pts - centers, dim=-1)                # (B,H,2)
        signed_dist_per_obs = dist_to_centers - self._dstt_obstacle_radius_real   # can be negative (inside obstacle)
        clearance, nearest_obs_idx = signed_dist_per_obs.clamp(min=0.0).min(dim=-1, keepdim=True)  # (B,H,1)

        # ---- spatial ceiling: r_spatial(k) = max(r_min, eta*d(k))  [Theorem 2] ----
        # ---- eta < 1 keeps the tube inside free space without a wall cap.    ----
        r_spatial = torch.maximum(
            torch.full((B, H, 1), radius_min_real, device=device),
            eta * clearance,
        )

        # ---- prescribed-time contraction (PTZF-inspired)                    ----
        # ---- s = denoising progress in [0,1]: 0 at j=N (start), 1 at j=0   ----
        # ---- phi_j = exp(-lambda_r * s): 1 at start, ~0 at end              ----
        # ---- r_j(k) = r_min + phi_j * (r_spatial(k) - r_min)               ----
        s = (self.n_timesteps - j) / self.n_timesteps
        phi_j = float(torch.exp(torch.tensor(-lambda_r * s)).item())
        r_desired_real = radius_min_real + phi_j * (r_spatial - radius_min_real)

        # ---- collision region (REAL units on both sides) ----
        collision_mask = clearance < r_desired_real   # (B, H, 1)

        # ---- deformation, REAL units [Sec I.D] ----
        beta_real = torch.zeros_like(c_real)

        if collision_mask.any():
            for b in range(B):
                mask_b = collision_mask[b, :, 0]
                # label connected collision components along the horizon
                labeled = torch.zeros(H, dtype=torch.long, device=device)
                comp_id = 0
                in_comp = False
                for i in range(H):
                    if mask_b[i]:
                        if not in_comp:
                            comp_id += 1
                            in_comp = True
                        labeled[i] = comp_id
                    else:
                        in_comp = False

                for comp in range(1, comp_id + 1):
                    idx = (labeled == comp).nonzero(as_tuple=True)[0]
                    if len(idx) == 0:
                        continue
                    k_a, k_b = int(idx.min()), int(idx.max())
                    k_c = (k_a + k_b) / 2.0
                    # extended support window [kL, kR] with kL < ka < kb < kR
                    # [Sec I.C], wide enough to fully contain the collision
                    # interval regardless of how long it is
                    k_start = max(0, k_a - pad)
                    k_end = min(H - 1, k_b + pad)
                    window = slice(k_start, k_end + 1)

                    # compact-support bump psi(k) [Sec I.D, Eq. for psi]
                    k_vals = torch.arange(k_start, k_end + 1, device=device).float().view(1, -1, 1)
                    half_width = (k_end - k_start + 1) / 2.0
                    z = (k_vals - k_c) / half_width
                    psi = torch.exp(-1.0 / (1.0 - z**2 + 1e-8))
                    psi = torch.where(torch.abs(z) < 1, psi, torch.zeros_like(psi))
                    psi = psi / (psi.max() + 1e-8)

                    # nearest obstacle for this component -> avoidance
                    # direction n(k) [Sec I.D]
                    comp_obs_idx = int(nearest_obs_idx[b, idx[0], 0])
                    obs_center = self._dstt_obstacle_centers_real[comp_obs_idx].view(1, 1, 2)

                    dir_vec = c_real[b:b+1, window, :] - obs_center
                    dist_to_this_obs = torch.norm(dir_vec, dim=-1, keepdim=True)
                    n = dir_vec / (dist_to_this_obs + 1e-8)
                    # NOT clamped at 0: when the centerline is deep inside the
                    # obstacle this is strongly negative, which is exactly
                    # what tells the amplitude it must push past the
                    # obstacle's full radius, not just past r_desired.
                    signed_dist_this_obs = dist_to_this_obs - self._dstt_obstacle_radius_real

                    # amplitude A, chosen per Theorem 1 so the deformed tube
                    # clears the obstacle: A* = max_k (r(k) - signed_dist(k))
                    # / psi(k) + margin. Computed ONLY within this
                    # component's own window so an unrelated collision
                    # elsewhere on the horizon can't inflate it; psi-near-
                    # zero tail points excluded to avoid dividing by a
                    # near-zero bump value.
                    overlap = (r_desired_real[b:b+1, window, :] - signed_dist_this_obs).clamp(min=0.0)
                    valid = psi > 0.20

                    if valid.any():
                        A_star = (overlap[valid] / psi[valid]).max()
                    else:
                        A_star = torch.tensor(0.0, device=device)

                    A_star = A_star + margin_real
                    # loose sanity ceiling only -- see docstring point (1)
                    A_star = torch.clamp(A_star, min=0.0, max=amplitude_cap_real)

                    beta_comp = A_star * psi * n
                    beta_real[b:b+1, window, :] += beta_comp

        # ---- deformed centerline c~(k) = c0(k) + beta(k) [Sec I.D] ----
        centerline_real = c_real + beta_real

        if verbose and j % 50 == 0:
            print(f"Step {j}: clearance(real) min/max = {clearance.min():.3f}/{clearance.max():.3f}, "
                  f"radius(real) = {r_desired_real.min():.3f}/{r_desired_real.max():.3f}, "
                  f"collisions = {collision_mask.sum().item()}, beta(real) norm = {beta_real.norm():.3f}")

        return centerline_real, r_desired_real


    def stt_potential(self, pos, centerline, radius):
        """
        Phi = dist(pos, Gamma)^2. Coordinate-agnostic: feed it real-space or
        normalized-space tensors as long as pos/centerline/radius are all in
        the SAME units (p_sample now calls this in real coordinates).
        pos: (B, H, 2)
        """
        error = pos - centerline           # (B, H, 2)
        norm_error = torch.norm(error, dim=-1, keepdim=True)  # (B, H, 1)
        scale = torch.clamp(radius / (norm_error + 1e-8), max=1.0)  # (B, H, 1)
        proj = centerline + scale * error
        diff = pos - proj
        return torch.sum(diff**2, dim=-1, keepdim=True).sum()  # scalar

    def stt_guidance(self, pos, centerline, radius):
        """
        g_j(tau) = -grad_tau Phi_j(tau) = -2*(pos - Pi_Gamma(pos))  [Theorem 4]
        Coordinate-agnostic; see stt_potential note above.
        """
        error = pos - centerline
        norm_error = torch.norm(error, dim=-1, keepdim=True) + 1e-8
        scale = torch.clamp(radius / norm_error, max=1.0)
        proj = centerline + scale * error
        grad_phi = 2 * (pos - proj)
        return -grad_phi

    def plot_dstt_geometry(self, start_norm, goal_norm, j,final_real=None, save_path=None):
        """
        Debug helper -- no diffusion sampling required. Plots the nominal
        centerline, the deformed DSTT centerline, the tube boundaries, and
        both obstacles in REAL maze coordinates for one (start, goal) pair
        at a given diffusion step j.

        Call this BEFORE debugging the full sampling loop: if the green
        deformed line doesn't visibly bend around the red obstacles here,
        the guidance can't work downstream no matter what lambda_stt is.

        Example:
            fig = model.plot_dstt_geometry(start_norm[:1], goal_norm[:1], j=500)
            fig.savefig('tube_check.png')
        """
        import matplotlib.pyplot as plt

        centerline_real, radius_real = self.compute_dstt_tube(start_norm[:1], goal_norm[:1], self.horizon, j)
        start_real = self._norm_to_real(start_norm[:1])
        goal_real = self._norm_to_real(goal_norm[:1])
        t_phys = torch.linspace(0, 1, self.horizon, device=centerline_real.device).view(1, self.horizon, 1)
        nominal_real = start_real.unsqueeze(1) + t_phys * (goal_real.unsqueeze(1) - start_real.unsqueeze(1))

        nominal = nominal_real[0].cpu().numpy()
        deformed = centerline_real[0].cpu().numpy()
        radius = radius_real[0, :, 0].cpu().numpy()

        tangent = np.gradient(deformed, axis=0)
        tangent_norm = tangent / (np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8)
        normal = np.stack([-tangent_norm[:, 1], tangent_norm[:, 0]], axis=1)
        upper = deformed + normal * radius[:, None]
        lower = deformed - normal * radius[:, None]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.plot(nominal[:, 1], nominal[:, 0], 'b--', label='nominal centerline')
        ax.plot(deformed[:, 1], deformed[:, 0], 'g-', linewidth=2, label='deformed centerline')
        ax.plot(upper[:, 1], upper[:, 0], color='orange', linewidth=1, label='tube boundary')
        ax.plot(lower[:, 1], lower[:, 0], color='orange', linewidth=1)
        if final_real is not None:
            ax.plot(
                final_real[0,:,1].cpu().numpy(),
                final_real[0,:,0].cpu().numpy(),
                'k',
                linewidth=3,
                label='final trajectory'
            )
            
        for center in self._dstt_obstacle_centers_real.cpu().numpy():
            circ = plt.Circle((center[1], center[0]), self._dstt_obstacle_radius_real,
                               color='red', alpha=0.3, label='obstacle')
            ax.add_patch(circ)

        ax.set_xlabel('x (real)')
        ax.set_ylabel('y (real)')
        ax.set_title(f'DSTT geometry @ diffusion step j={j}')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set_aspect('equal')
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig


    @torch.no_grad()   #only for sampling
    def invariance_umaze(self, x, xp1):   #  RoS-diffuser for umaze

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1.52/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1.52/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(2.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

       #CBF
        b = 1 - ((x[:,2:3] - off_y)/yr)**4 - ((x[:,3:4] - off_x)/xr)**4
        Lfb = 0
        Lgbu1 = -4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = -4*((x[:,3:4] - off_x)/xr)**3/xr

        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b

        self.safe1 = torch.min(b[:,0])

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1.2/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*0.6/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(2-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0])

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out
        rt = rt.unsqueeze(0)
        return rt
    
    @torch.no_grad()   #only for sampling
    def invariance_umaze_relax(self, x, xp1, t):  #  ReS-diffuser for umaze

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1.52/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1.52/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(2.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = 1 - ((x[:,2:3] - off_y)/yr)**4 - ((x[:,3:4] - off_x)/xr)**4
        Lfb = 0
        Lgbu1 = -4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = -4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe1 = torch.min(b[:,0])

        if t >= 10:
            sign = 100   #relax
        else:
            sign = 0   #non-relax

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        G = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1.2/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*0.6/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(2-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0])

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx0, rx1], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        q0 = torch.zeros_like(q).to(G.device)
        q = torch.cat([q, q0], dim = 1)
        Q = Variable(torch.eye(4))
        Q = Q.unsqueeze(0).expand(nBatch, 4, 4).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        rt = rt.unsqueeze(0)
        return rt

    @torch.no_grad()   #only for sampling
    def invariance(self, x, xp1):    #  RoS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01  # robust term 09/25
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b

        self.safe1 = torch.min(b[:,0] + 0.01)  # robust term 09/25

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01 # robust term 09/25
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0]+ 0.01) # robust term 09/25

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out
        # print(out)
        rt = rt.unsqueeze(0)
        return rt

    @torch.no_grad()   #only for sampling
    def invariance_cf(self, x, xp1):  # closed form solution,  RoS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b0 = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01  # robust term 09/25
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h0 = Lfb + k*b0

        self.safe1 = torch.min(b0[:,0] + 0.01)  # robust term 09/25

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01 # robust term 09/25
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0]+ 0.01) # robust term 09/25

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1
        h1 = Lfb + k*b
        
        q = -ref[:,2:4].to(b.device)
        
        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out
        # print(out)
        rt = rt.unsqueeze(0)
        return rt
        



    @torch.no_grad()   #only for sampling
    def invariance_relax(self, x, xp1, t):  #  ReS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        if t >= 10:   # debug  10
            sign = 100   #relax
        else:
            sign = 0   #non-relax

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        G = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
        G = G.unsqueeze(1)
        k = 1
        h = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx0, rx1], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        q0 = torch.zeros_like(q).to(G.device)
        q = torch.cat([q, q0], dim = 1)
        Q = Variable(torch.eye(4))
        Q = Q.unsqueeze(0).expand(nBatch, 4, 4).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        rt = rt.unsqueeze(0)
        return rt
    
    @torch.no_grad()   #only for sampling
    def invariance_relax_cf(self, x, xp1, t):  # closed-form solution, ReS-diffuser for maze2d-large-v1

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        if t >= 10:   # debug  10
            sign = 100   #relax
        else:
            sign = 0   #non-relax

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        G0 = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0], dim = 1)
        k = 1
        h0 = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx0, rx1], dim = 1)
        k = 1
        h1 = Lfb + k*b
        
   
        q = -ref[:,2:4].to(G0.device)
        q0 = torch.zeros_like(q).to(G0.device)
        q = torch.cat([q, q0], dim = 1)

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()    
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        # print(out)
        rt = rt.unsqueeze(0)
        return rt
    

    @torch.no_grad()   #only for sampling
    def invariance_relax_narrow(self, x, xp1, t):  #  ReS-diffuser for maze2d-large-v1,  narrow passage case

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        if t >= 10:   # debug  10
            sign = 1   #relax
        else:
            sign = 0   #non-relax

        #normalize obstacle 1,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])

        off_x = 2*(5.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.4  # 0.01
        Lfb = 0
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        rx0 = torch.zeros_like(Lgbu1).to(b.device)
        rx1 = sign*torch.ones_like(Lgbu1).to(b.device)

        self.safe1 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2, rx1, rx0, rx0, rx0, rx0, rx0], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1
        h1 = Lfb + k*b

        ########################################### obs 2
        off_x = 2*(5.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b2 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.6 #0.01
        Lfb = 0
        Lgbu12 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu22 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b2[:,0] + 0.01)

        G2 = torch.cat([-Lgbu12, -Lgbu22, rx0, rx1, rx0, rx0, rx0, rx0], dim = 1)
        G2 = G2.unsqueeze(1)
        k = 1
        h2 = Lfb + k*b2

        ########################################### obs 3
        off_x = 2*(3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b3 = ((x[:,2:3] - off_y)/yr/0.5)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu13 = 4*((x[:,2:3] - off_y)/yr/0.5)**3/yr
        Lgbu23 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G3 = torch.cat([-Lgbu13, -Lgbu23, rx0, rx0, rx1, rx0, rx0, rx0], dim = 1)
        G3 = G3.unsqueeze(1)
        k = 1
        h3 = Lfb + k*b3

        ########################################### obs 4
        off_x = 2*(8.5-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(3.5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b4 = ((x[:,2:3] - off_y)/yr/1.8)**4 + ((x[:,3:4] - off_x)/xr/1.8)**4 - 1 - 0.01
        Lfb = 0
        Lgbu14 = 4*((x[:,2:3] - off_y)/yr/1.8)**3/yr
        Lgbu24 = 4*((x[:,3:4] - off_x)/xr/1.8)**3/xr

        G4 = torch.cat([-Lgbu14, -Lgbu24, rx0, rx0, rx0, rx1, rx0, rx0], dim = 1)
        G4 = G4.unsqueeze(1)
        k = 1
        h4 = Lfb + k*b4

        ########################################### obs 5
        off_x = 2*(7.6-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(7-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b5 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.4 #0.01
        Lfb = 0
        Lgbu15 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu25 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G5 = torch.cat([-Lgbu15, -Lgbu25, rx0, rx0, rx0, rx0, rx1, rx0], dim = 1)
        G5 = G5.unsqueeze(1)
        k = 1
        h5 = Lfb + k*b5

        ########################################### obs 6
        off_x = 2*(10-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(6.3-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b6 = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - 1 - 0.01
        Lfb = 0
        Lgbu16 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu26 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        G6 = torch.cat([-Lgbu16, -Lgbu26, rx0, rx0, rx0, rx0, rx0, rx1], dim = 1)
        G6 = G6.unsqueeze(1)
        k = 1
        h6 = Lfb + k*b6

        b0 = torch.cat([b, b2, b3, b4, b5, b6], dim = 1)
        idx = torch.argmin(b0, dim = 1).cpu().numpy()
        G0 = torch.cat([G1, G2, G3, G4, G5, G6], dim = 1)
        h0 = torch.cat([h1, h2, h3, h4, h5, h6], dim = 1)
        rows = len(G0[:,0,0])
        G = []
        h = []
        for i in range(rows):
            G.append(G0[i:i+1,idx[i]:idx[i]+1])
            h.append(h0[i:i+1,idx[i]:idx[i]+1])
        G = torch.cat(G, dim = 0)
        h = torch.cat(h, dim = 0)


        

        # G = torch.cat([G1, G2, G3, G4, G5, G6], dim = 1)
        # h = torch.cat([h1, h2, h3, h4, h5, h6], dim = 1)

        # G = torch.cat([G1, G2, G3, G5], dim = 1)
        # h = torch.cat([h1, h2, h3, h5], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        q0 = torch.zeros_like(q).to(G.device)
        q = torch.cat([q, q0, q0, q0], dim = 1)
        Q = Variable(torch.eye(8))
        Q = Q.unsqueeze(0).expand(nBatch, 8, 8).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out[:,0:2]
        rt = rt.unsqueeze(0)
        return rt


    @torch.no_grad()   #only for sampling
    def invariance_time(self, x, xp1, t):  #  TVS-diffuser for maze2d-large-v1
        t_bias = 5  #50

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - nn.Sigmoid()(t_bias - t) -0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        G = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G = G.unsqueeze(1)
        k = 1  #0.3
        h = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - nn.Sigmoid()(t_bias - t) - 0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        G1 = G1.unsqueeze(1)
        k = 1  #0.4
        h1 = Lfb + k*b

        G = torch.cat([G, G1], dim = 1)
        h = torch.cat([h, h1], dim = 1)
        
   
        q = -ref[:,2:4].to(G.device)
        Q = Variable(torch.eye(2))
        Q = Q.unsqueeze(0).expand(nBatch, 2, 2).to(G.device)
        
        e = Variable(torch.Tensor())
        out = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q, q, G, h, e, e)

        rt = xp1.clone()      
        rt[:,2:4] = x[:,2:4] + out
        rt = rt.unsqueeze(0)
        return rt

    @torch.no_grad()   #only for sampling
    def invariance_time_cf(self, x, xp1, t):  # closed-form solution, TVS-diffuser for maze2d-large-v1
        t_bias = 5  #50 

        x = x.squeeze(0)
        xp1 = xp1.squeeze(0)

        nBatch = x.shape[0]
        ref = xp1 - x

        #normalize obstacle 1, x-1, y-0  x = 1/12*np.cos(theta) + 5.5/12, y = 1/9*np.sin(theta) + 5/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.8-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(5-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**2 + ((x[:,3:4] - off_x)/xr)**2 - nn.Sigmoid()(t_bias - t) -0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 2*((x[:,2:3] - off_y)/yr)/yr
        Lgbu2 = 2*((x[:,3:4] - off_x)/xr)/xr

        self.safe1 = torch.min(b[:,0] + 0.01)

        G0 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1  #0.3
        h0 = Lfb + k*b

        #normalize obstacle 2,  x = 1/12*np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta)) + 5.3/12, y = 1/9*np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta)) + 2/9
        xr = 2*1/(self.norm_maxs[1] - self.norm_mins[1])
        yr = 2*1/(self.norm_maxs[0] - self.norm_mins[0])
        off_x = 2*(5.3-0.5 - self.norm_mins[1])/(self.norm_maxs[1] - self.norm_mins[1]) - 1
        off_y = 2*(2-0.5 - self.norm_mins[0])/(self.norm_maxs[0] - self.norm_mins[0]) - 1

        #CBF
        b = ((x[:,2:3] - off_y)/yr)**4 + ((x[:,3:4] - off_x)/xr)**4 - nn.Sigmoid()(t_bias - t) - 0.01
        Lfb = nn.Sigmoid()(t_bias - t)*(1 - nn.Sigmoid()(t_bias - t))
        Lgbu1 = 4*((x[:,2:3] - off_y)/yr)**3/yr
        Lgbu2 = 4*((x[:,3:4] - off_x)/xr)**3/xr

        self.safe2 = torch.min(b[:,0] + 0.01)

        G1 = torch.cat([-Lgbu1, -Lgbu2], dim = 1)
        k = 1  #0.4
        h1 = Lfb + k*b
        
   
        q = -ref[:,2:4].to(G0.device)

        y1_bar = 1*G0  # H or Q = identity matrix
        y2_bar = 1*G1
        u_bar = -1*q
        p1_bar = h0 - torch.sum(G0*u_bar,dim = 1).unsqueeze(1)
        p2_bar = h1 - torch.sum(G1*u_bar,dim = 1).unsqueeze(1)

        G = torch.cat([torch.sum(y1_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y1_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y1_bar,dim = 1).unsqueeze(1).unsqueeze(0), torch.sum(y2_bar*y2_bar,dim = 1).unsqueeze(1).unsqueeze(0)], dim = 0)
        #G = 1*[y1_bar*y1_bar', y1_bar*y2_bar'; y2_bar*y1_bar', y2_bar*y2_bar']
        w_p1_bar = torch.clamp(p1_bar, max=0)
        w_p2_bar = torch.clamp(p2_bar, max=0)

        # G 0-(1,1), 1-(1,2), 2-(2,1), 3-(2,2)
        lambda1 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, torch.zeros_like(p1_bar), torch.where(G[1]*w_p1_bar < G[0]*p2_bar, w_p1_bar/G[0], torch.clamp(G[3]*p1_bar - G[2]*p2_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))
        
        lambda2 = torch.where(G[2]*w_p2_bar < G[3]*p1_bar, w_p2_bar/G[3], torch.where(G[1]*w_p1_bar < G[0]*p2_bar, torch.zeros_like(p1_bar), torch.clamp(G[0]*p2_bar - G[1]*p1_bar, max=0)/(G[0]*G[3] - G[1]*G[2])))

        out = lambda1*y1_bar + lambda2*y2_bar + u_bar
        rt = xp1.clone()    
        rt[:,2:4] = x[:,2:4] + out
        # print(out)
        rt = rt.unsqueeze(0)
        return rt        

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        xp1 = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        j = t[0].item()
        have_endpoints = hasattr(self, 'start_norm') and hasattr(self, 'goal_norm')

        if self.use_stt_guidance and have_endpoints:
            centerline_real, radius_real = self.compute_dstt_tube(self.start_norm, self.goal_norm, self.horizon, j)

            pos_norm = xp1[:, :, 2:4]            # (B, H, 2), normalized (y, x)
            pos_real = self._norm_to_real(pos_norm)

            g_real = self.stt_guidance(pos_real, centerline_real, radius_real)   # (B, H, 2)

            # ---- diagnostic only; no effect on sampling ----------------------
            # Set `self._diag = []` before sampling to collect these.
            #
            # The theory assumes the prior usually lands INSIDE Gamma_j, where
            # g == 0 and no correction is applied. `saturation` measures how
            # often that assumption holds: it is the fraction of (batch, horizon)
            # entries where the guidance is actually firing. `dev_over_r` is the
            # mean |x - c| / r of the RAW sample, so >1 means the prior wanted to
            # be outside the tube and the correction had to drag it back.
            if getattr(self, '_diag', None) is not None:
                with torch.no_grad():
                    dev = torch.norm(pos_real - centerline_real, dim=-1)      # (B,H)
                    r = radius_real[..., 0]                                   # (B,H)
                    self._diag.append({
                        'j': j,
                        'saturation': float((dev > r).float().mean()),
                        'dev_over_r': float((dev / r.clamp(min=1e-6)).mean()),
                    })
            # -------------------------------------------------------------------

            range_y = self.norm_maxs[0] - self.norm_mins[0]
            range_x = self.norm_maxs[1] - self.norm_mins[1]
            scale_to_norm = torch.tensor(
                [2.0 / range_y, 2.0 / range_x], device=g_real.device, dtype=g_real.dtype
            ).view(1, 1, 2)
            g_norm = g_real * scale_to_norm

            # Prescribed-time guidance gain: lambda_j = lambda_stt * phi_gain(j)
            #
            # phi_gain grows from ~0 at j=N (start of denoising, high noise)
            # to 1 at j=0 (end of denoising, converged). This is the
            # "weak early / strong late" schedule from UniConFlow (PTZF):
            #   - Early steps: trajectory stays close to the learned prior.
            #   - Late steps: safety constraint dominates, enforcing Gamma_j.
            #
            # This is opposite to the old lambda*(j/N) schedule (which was
            # strongest at j=N -- the wrong direction), and stronger than a
            # constant gain because it explicitly leaves room for the diffusion
            # model to do its job during the high-noise phase.
            #
            # phi_gain = 1 - exp(-mu * s),  s = (N-j)/N in [0,1]
            #   s=0 (j=N, start): phi_gain ~ 0  -> correction off
            #   s=1 (j=0, end):   phi_gain ~ 1  -> full correction
            mu = 3.0
            s = (self.n_timesteps - j) / self.n_timesteps
            phi_gain = 1.0 - float(torch.exp(torch.tensor(-mu * s)).item())
            correction = self.lambda_stt * phi_gain * g_norm

            pos_corrected = pos_norm + correction
            xp1[:, :, 2:4] = pos_corrected

            # Diagnostic only (no effect on sampling): per-obstacle minimum
            # clearance of the CURRENT sample from each obstacle boundary,
            # in the same convention as Shield()/invariance() elsewhere in
            # this file (>= 0 means clear of that obstacle). This is what
            # was printing 0.0/0.0 before -- nothing was ever assigning to
            # self.safe1/self.safe2 once the old CBF path was removed.
            if hasattr(self, '_dstt_obstacle_centers_real'):
                pos_real_diag = self._norm_to_real(pos_corrected)
                d0 = torch.norm(pos_real_diag - self._dstt_obstacle_centers_real[0], dim=-1) - self._dstt_obstacle_radius_real
                d1 = torch.norm(pos_real_diag - self._dstt_obstacle_centers_real[1], dim=-1) - self._dstt_obstacle_radius_real
                self.safe1 = d0.min()
                self.safe2 = d1.min()

        x = xp1
        return x
    

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        # Extract start and goal from cond. cond[t] holds the OBSERVATION-
        # sized conditioning value (it gets placed into x[:, t, action_dim:]
        # by apply_conditioning), so its own first 2 entries are position --
        # cond[0][:, 0:2] is correct here (not [:, 2:4], which would be the
        # velocity components of the observation).
        self.start_norm = cond[0][:, 0:2]
        self.goal_norm = cond[self.horizon - 1][:, 0:2]

        if verbose and not hasattr(self, '_tube_debug_done'):
            for j_dbg, fname in [(255, 'dstt_255.png'), (150, 'dstt_150.png'), (50, 'dstt_50.png')]:
                self.plot_dstt_geometry(self.start_norm[:1], self.goal_norm[:1], j=j_dbg, save_path=fname)
            self._tube_debug_done = True

        self.safe1 = torch.tensor(0.0, device=device)
        self.safe2 = torch.tensor(0.0, device=device)

        if verbose:
            mode = 'soft DSTT guidance' if self.use_stt_guidance else 'none'
            print(f"Safety mode: {mode}")

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        safe1, safe2 = [], []
        for i in reversed(range(0, self.n_timesteps)):
            if i < 0:
                i = 0
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps)
            x = apply_conditioning(x, cond, self.action_dim)
            safe1.append(self.safe1.unsqueeze(0))
            safe2.append(self.safe2.unsqueeze(0))
            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        self.safe1 = torch.cat(safe1, dim=0)
        self.safe2 = torch.cat(safe2, dim=0)

        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x


    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, return_diffusion = True, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, return_diffusion= return_diffusion, *args, **kwargs)   ## debug

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)


    def _vb_terms_bpd(
        self, x_start, conditions, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # batch_size = x_start.shape(0)
        # conditions = self._format_conditions(conditions, batch_size)

        true_mean, _, true_log_variance_clipped = self.q_posterior(
            x_start=x_start, x_t=x_t, t=t
        )
        
        mean, _, log_variance = self.p_mean_variance(
             x_t, conditions, t)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, mean, log_variance
        )
        kl = mean_flat(kl) / np.log(2.0)

        # import pdb; pdb.set_trace()
        # decoder_nll = -discretized_gaussian_log_likelihood(
        #     x_start, means=mean, log_scales=0.5 * log_variance
        # )
        
        # assert decoder_nll.shape == x_start.shape
        # decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        # output = torch.where((t == 0), decoder_nll, kl)

        return kl


    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)