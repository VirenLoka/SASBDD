"""
The contract between the SA surrogate and DiffSBDD's diffusion process.

Everything that has to agree between stage 1 (surrogate training) and stage 2
(reward-guided finetuning) lives here, so the two cannot silently diverge:

  * the noise schedule, byte-compatible with DiffSBDD's PredefinedNoiseSchedule
  * the corruption applied to (coords, one-hot) at a given noise level
  * the box-integral categorical readout that turns a continuous h into
    calibrated log-probabilities
  * the Gumbel straight-through estimator

Why the box integral rather than `softmax(h)`:
DiffSBDD's h channel is not logits and has no softmax anywhere.  It is a
continuous Gaussian relaxation of the one-hot divided by norm_values[1] (= 4),
so a clean one-hot maps to {0, 0.25}.  Handing that to `gumbel_softmax` treats a
0.25-wide range as logits, Gumbel noise of scale ~1 swamps it, and the resulting
"discrete" sample is close to a uniform draw -- a gradient that means nothing.

The calibrated construction already exists in the codebase, in
`EnVariationalDiffusion.log_pxh_given_z0_without_constants`
(equivariant_diffusion/en_diffusion.py:185): un-normalise h, centre it on 1,
integrate N(h, sigma_cat) over the unit box [-0.5, +0.5], normalise across
classes with logsumexp.  `categorical_logprobs_from_h` below is that same
construction with sigma_t in place of sigma_0.

Why condition on sigma rather than on t:
r(t) = sigma_t / alpha_t spans 0.022 .. 44.7 for the released checkpoint -- more
than three orders of magnitude.  Conditioning on log10(r) instead of the integer
t keeps the surrogate valid whatever stage 2 feeds it (a rescaled z_t, or
Tweedie's x0-hat with its own residual scale) and across schedule changes.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Noise schedule
#
# Reimplemented rather than imported because equivariant_diffusion.en_diffusion
# pulls in torch_scatter at import time, which the surrogate does not otherwise
# need.  `verify_against_diffsbdd()` checks the two agree when that import works.
# Source: en_diffusion.py:1105-1190.
# -----------------------------------------------------------------------------

def clip_noise_schedule(alphas2: np.ndarray, clip_value: float = 0.001) -> np.ndarray:
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = alphas2[1:] / alphas2[:-1]
    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    return np.cumprod(alphas_step, axis=0)


def polynomial_schedule(timesteps: int, s: float = 1e-4, power: float = 3.0) -> np.ndarray:
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2
    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)
    precision = 1 - 2 * s
    return precision * alphas2 + s


def cosine_beta_schedule(timesteps: int, s: float = 0.008, raise_to_power: float = 1.0) -> np.ndarray:
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)
    return alphas_cumprod


class NoiseSchedule(nn.Module):
    """DiffSBDD's forward-process schedule, exposed in terms of r = sigma / alpha.

    Buffers are indexed by integer timestep in [0, T].
    """

    def __init__(
        self,
        noise_schedule: str = "polynomial_2",
        timesteps: int = 500,
        precision: float = 5.0e-4,
    ):
        super().__init__()
        self.timesteps = int(timesteps)
        self.noise_schedule = noise_schedule

        if noise_schedule == "cosine":
            alphas2 = cosine_beta_schedule(self.timesteps)
        elif "polynomial" in noise_schedule:
            parts = noise_schedule.split("_")
            if len(parts) != 2:
                raise ValueError(f"malformed polynomial schedule: {noise_schedule!r}")
            alphas2 = polynomial_schedule(self.timesteps, s=precision, power=float(parts[1]))
        else:
            raise ValueError(f"unknown noise schedule: {noise_schedule!r}")

        sigmas2 = 1.0 - alphas2
        gamma = -(np.log(alphas2) - np.log(sigmas2))  # = -log(SNR)

        gamma_t = torch.from_numpy(gamma).float()
        alpha_t = torch.sqrt(torch.sigmoid(-gamma_t))
        sigma_t = torch.sqrt(torch.sigmoid(gamma_t))
        r_t = sigma_t / alpha_t

        self.register_buffer("gamma_t", gamma_t)
        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("sigma_t", sigma_t)
        self.register_buffer("r_t", r_t)
        self.register_buffer("log10_r_t", torch.log10(r_t))

        # Symmetric for polynomial_2 (-1.65 .. +1.65); used to map log10(r)
        # into roughly [-1, 1] before it reaches the model.
        self.register_buffer("log10_r_min", self.log10_r_t.min().clone())
        self.register_buffer("log10_r_max", self.log10_r_t.max().clone())

    # -- lookups ------------------------------------------------------------
    # Every lookup moves its index onto the buffers' device and returns a result
    # on the caller's device.  Buffers follow the module onto the GPU, while
    # timesteps are often built on the CPU (logging, fixed eval buckets), and
    # indexing a CUDA tensor with a CPU index tensor is an error.
    def _lookup(self, table: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        idx = t.to(table.device).long()
        return table[idx].to(t.device)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self._lookup(self.alpha_t, t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self._lookup(self.sigma_t, t)

    def r(self, t: torch.Tensor) -> torch.Tensor:
        """Noise-to-signal ratio sigma_t / alpha_t, i.e. the per-coordinate
        standard deviation of the corruption once rescaled into x0-space."""
        return self._lookup(self.r_t, t)

    def t_from_r(self, r: torch.Tensor) -> torch.Tensor:
        """Nearest integer timestep for a given r.  r_t is strictly increasing."""
        table = self.r_t
        target = r.to(table.device).contiguous().clamp(min=float(table[0]))
        idx = torch.searchsorted(table, target)
        return idx.clamp(0, self.timesteps).to(r.device)

    def normalize_log_r(self, r: torch.Tensor) -> torch.Tensor:
        """Map r onto ~[-1, 1] for the model's conditioning input."""
        log_r = torch.log10(r.clamp_min(1e-8))
        mid = (0.5 * (self.log10_r_max + self.log10_r_min)).to(r.device)
        half = (0.5 * (self.log10_r_max - self.log10_r_min)).to(r.device)
        return (log_r - mid) / half.clamp_min(1e-8)

    # -- sanity -------------------------------------------------------------
    def verify_against_diffsbdd(self, precision: float = 5.0e-4) -> Optional[bool]:
        """Compare against DiffSBDD's own PredefinedNoiseSchedule.

        Returns True/False if the check ran, or None if en_diffusion could not
        be imported (it needs torch_scatter, which the surrogate does not).
        None means "not checked" -- do not read it as a mismatch.
        """
        try:
            from equivariant_diffusion.en_diffusion import PredefinedNoiseSchedule
        except Exception:
            return None
        ref = PredefinedNoiseSchedule(self.noise_schedule, self.timesteps, precision)
        return bool(torch.allclose(ref.gamma.detach(), self.gamma_t, atol=1e-5))


# -----------------------------------------------------------------------------
# Calibrated categorical readout  (en_diffusion.py:185 with sigma_t for sigma_0)
# -----------------------------------------------------------------------------

def standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def categorical_logprobs_from_h(
    h_normalized: torch.Tensor,
    sigma_normalized: torch.Tensor,
    h_scale: float = 4.0,
    epsilon: float = 1e-10,
    min_sigma: float = 1e-3,
) -> torch.Tensor:
    """Calibrated log p(atom type | h) for DiffSBDD's continuous h channel.

    Args:
        h_normalized:     [N, K] h in DiffSBDD's *normalized* space, i.e. a clean
                          one-hot appears as {0, 1/h_scale}.
        sigma_normalized: scalar, [N, 1] or [N] -- std of the noise on
                          `h_normalized`, in that same normalized space.
        h_scale:          norm_values[1] from the DiffSBDD config (4.0).

    Returns:
        [N, K] log-probabilities, normalized across the class dimension.
    """
    if sigma_normalized.dim() == 1:
        sigma_normalized = sigma_normalized.unsqueeze(-1)

    onehot_est = h_normalized * h_scale
    sigma_cat = (sigma_normalized * h_scale).clamp_min(min_sigma)

    # A clean entry is 1 for the true class and 0 otherwise; centring on 1 puts
    # the true class at 0 and every other class at -1.
    centered = onehot_est - 1.0

    log_p_unnormalized = torch.log(
        standard_normal_cdf((centered + 0.5) / sigma_cat)
        - standard_normal_cdf((centered - 0.5) / sigma_cat)
        + epsilon
    )
    return log_p_unnormalized - torch.logsumexp(log_p_unnormalized, dim=-1, keepdim=True)


# -----------------------------------------------------------------------------
# Gumbel straight-through
# -----------------------------------------------------------------------------

def gumbel_straight_through(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = True,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
) -> torch.Tensor:
    """Discrete one-hot forward, soft gradient backward.

    `deterministic=True` drops the Gumbel noise and takes the argmax -- use it
    for evaluation so reported metrics are not sampling-noisy.  The straight-
    through path is kept either way, so gradients still flow to `logits`.
    """
    if deterministic:
        y_soft = (logits / tau).softmax(dim=-1)
    else:
        if generator is None:
            gumbels = -torch.empty_like(logits).exponential_().log()
        else:
            u = torch.rand(logits.shape, generator=generator,
                           device=logits.device, dtype=logits.dtype)
            gumbels = -torch.log(-torch.log(u.clamp_min(1e-20)).clamp_min(1e-20))
        y_soft = ((logits + gumbels) / tau).softmax(dim=-1)

    if not hard:
        return y_soft

    index = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
    return (y_hard - y_soft).detach() + y_soft


# -----------------------------------------------------------------------------
# Corruption used for stage-1 training
# -----------------------------------------------------------------------------

def sample_noise_levels(
    num_graphs: int,
    schedule: NoiseSchedule,
    t_min: int = 0,
    t_max: int = 200,
    sampling: str = "log_uniform_sigma",
    clean_fraction: float = 0.0,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-graph integer timesteps.

    `log_uniform_sigma` samples uniformly in log(r) rather than in t, which
    spreads samples evenly across noise *magnitude* instead of piling them at
    the high-t end where both channels have already collapsed to the marginal.

    The `clean_fraction` share is set to t=0 (r = 0.022, i.e. 0.02 A of
    coordinate jitter -- clean for every practical purpose) rather than to a
    separate r=0 case, so the sigma conditioning stays well defined.
    """
    device = device or schedule.r_t.device
    t_min = max(0, int(t_min))
    t_max = min(int(t_max), schedule.timesteps)
    if t_max < t_min:
        raise ValueError(f"t_max ({t_max}) < t_min ({t_min})")

    def _rand(shape):
        if generator is None:
            return torch.rand(shape, device=device)
        return torch.rand(shape, generator=generator, device=device)

    if sampling == "uniform_t":
        u = _rand((num_graphs,))
        t = (t_min + u * (t_max - t_min)).round().long()
    elif sampling == "log_uniform_sigma":
        lo = schedule.log10_r_t[t_min]
        hi = schedule.log10_r_t[t_max]
        u = _rand((num_graphs,))
        target_r = torch.pow(10.0, lo + u * (hi - lo))
        t = schedule.t_from_r(target_r).clamp(t_min, t_max)
    else:
        raise ValueError(f"unknown sampling mode: {sampling!r}")

    if clean_fraction > 0:
        t = torch.where(_rand((num_graphs,)) < clean_fraction,
                        torch.zeros_like(t), t)
    return t


def corrupt(
    pos: torch.Tensor,
    one_hot: torch.Tensor,
    batch: torch.Tensor,
    r: torch.Tensor,
    *,
    h_scale: float = 4.0,
    x_scale: float = 1.0,
    tau: float = 1.0,
    hard: bool = True,
    corrupt_coords: bool = True,
    corrupt_types: bool = True,
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-noise a batch of molecules in x0-space at per-graph level r.

        x~ = x0       + r * eps_x
        h~ = onehot/h_scale + r * eps_h   ->  box integral  ->  Gumbel-ST

    The type path goes through exactly the same box-integral + Gumbel-ST that
    stage 2 will apply to the denoiser's output, so the surrogate is trained on
    the input distribution it will actually be asked to score.

    Args:
        pos:     [N, 3] coordinates, in Angstrom (DiffSBDD's norm_values[0] = 1).
        one_hot: [N, K] clean one-hot atom types.
        batch:   [N] graph index per node, DiffSBDD's `lig_mask` convention.
        r:       [G] noise-to-signal ratio per graph.

    Returns:
        (corrupted positions [N, 3], corrupted one-hot [N, K])
    """
    r_node = r[batch].unsqueeze(-1)  # [N, 1]

    if corrupt_coords:
        if generator is None:
            eps_x = torch.randn_like(pos)
        else:
            eps_x = torch.randn(pos.shape, generator=generator,
                                device=pos.device, dtype=pos.dtype)
        pos = pos + (r_node / x_scale) * eps_x

    if corrupt_types:
        h_norm = one_hot / h_scale
        if generator is None:
            eps_h = torch.randn_like(h_norm)
        else:
            eps_h = torch.randn(h_norm.shape, generator=generator,
                                device=h_norm.device, dtype=h_norm.dtype)
        h_norm = h_norm + r_node * eps_h
        logprobs = categorical_logprobs_from_h(h_norm, r_node, h_scale=h_scale)
        one_hot = gumbel_straight_through(logprobs, tau=tau, hard=hard,
                                          generator=generator,
                                          deterministic=deterministic)

    return pos, one_hot


# -----------------------------------------------------------------------------
# Stage-2 helpers  (not used during stage-1 training; kept here so the two
# stages share one definition)
# -----------------------------------------------------------------------------

def tweedie_x0(
    z_t: torch.Tensor,
    eps_hat: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """x0-hat = (z_t - sigma_t * eps_hat) / alpha_t.

    Mirrors `EnVariationalDiffusion.xh_given_zt_and_epsilon`
    (en_diffusion.py:471).  This is the only quantity in a single denoising pass
    that depends on the diffusion model's weights, and therefore the only route
    by which a surrogate reward reaches theta.
    """
    # DiffSBDD inflates these to [G, 1] via inflate_batch_array; accept a plain
    # [G] vector too so callers can pass schedule.alpha(t) directly.
    if alpha_t.dim() == 1:
        alpha_t = alpha_t.unsqueeze(-1)
    if sigma_t.dim() == 1:
        sigma_t = sigma_t.unsqueeze(-1)
    a = alpha_t[batch]
    s = sigma_t[batch]
    return z_t / a - eps_hat * s / a


def tweedie_residual_r(r: torch.Tensor, residual_scale: float = 0.3) -> torch.Tensor:
    """Effective noise level of a Tweedie estimate at forward level r.

    x0-hat is sharper than a rescaled z_t because the denoiser removes most of
    the error.  `residual_scale` is the fraction it fails to remove; 0.3 is a
    starting point, and it should be calibrated empirically against the
    pretrained checkpoint by measuring ||x0_hat - x0|| / r on held-out data
    before stage 2 relies on it.
    """
    return r * float(residual_scale)
