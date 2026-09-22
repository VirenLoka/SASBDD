"""
The frozen SA surrogate as a differentiable reward, plus the Tweedie reward pass.

Gradient flows:  eps_theta  ->  Tweedie x0_hat  ->  box-integral categorical
                            ->  Gumbel straight-through  ->  surrogate  ->  scalar

The surrogate's own weights never receive gradient; only the path *through* it
matters.  `xh_given_zt_and_epsilon` (en_diffusion.py:471) is the only quantity
in a single denoising pass that depends on theta, which is why Tweedie is not
optional here.

Channel balance
---------------
Measured on stage 1: ||g_types|| / ||g_coords|| is on the order of 100x-1000x.
That is structural -- atom types reach the surrogate through a plain Linear,
while coordinates only enter through RBF-expanded distances behind a smooth
cutoff envelope.  Left alone, the type channel swamps coordinate updates.
`_ChannelBalance` rescales the two blocks of d(score)/d(xh) before they
propagate back into the denoiser, which is the only place the split is still
visible (at parameter level the two are already mixed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from common.validate import validate_surrogate_against_ddpm
from sa_surrogate.diffusion_bridge import tweedie_residual_r
from sa_surrogate.inference import SASurrogate


class _ChannelBalance(torch.autograd.Function):
    """Identity forward; rescales the coordinate and type gradient blocks back."""

    @staticmethod
    def forward(ctx, xh, n_dims, coord_w, type_w, normalize):
        ctx.n_dims = int(n_dims)
        ctx.coord_w = float(coord_w)
        ctx.type_w = float(type_w)
        ctx.normalize = bool(normalize)
        return xh

    @staticmethod
    def backward(ctx, grad):
        n = ctx.n_dims
        g_x, g_h = grad[:, :n], grad[:, n:]
        if ctx.normalize:
            # Equalise the two blocks first, so the configured weights express a
            # ratio rather than fighting a 100x-1000x scale difference.
            g_x = g_x / g_x.norm().clamp_min(1e-12)
            g_h = g_h / g_h.norm().clamp_min(1e-12)
        return torch.cat([g_x * ctx.coord_w, g_h * ctx.type_w], dim=-1), \
            None, None, None, None


class FrozenSASurrogate(nn.Module):
    """Wraps the stage-1 surrogate: frozen, validated, channel-balanced.

    Registered as a submodule so device placement is handled for us and so each
    stage-2 checkpoint records which surrogate produced it.  Its parameters are
    excluded from the optimiser because `configure_optimizers` only ever sees
    `self.ddpm.parameters()`.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        residual_scale: float = 0.3,
        residual_scale_types: Optional[float] = None,
        tau: Optional[float] = None,
        hard: bool = True,
        deterministic: bool = False,
        channel_balance: str = "normalize",   # none | normalize | manual
        coord_weight: float = 1.0,
        type_weight: float = 1.0,
        n_dims: int = 3,
    ):
        super().__init__()
        self.surrogate = SASurrogate(checkpoint, device=device)
        self.model = self.surrogate.model          # registered -> moves with .to()
        self.model.requires_grad_(False).eval()
        # Registered too: it is an nn.Module holding the schedule buffers, and
        # `evaluate_sa` indexes them with CUDA timestep tensors.  `.to()` is
        # in-place for buffers, so `self.surrogate.schedule` sees the same move.
        self.schedule = self.surrogate.schedule

        self.residual_scale = float(residual_scale)
        self.residual_scale_types = (float(residual_scale_types)
                                     if residual_scale_types is not None else None)
        self.tau = tau
        self.hard = bool(hard)
        self.deterministic = bool(deterministic)
        self.channel_balance = str(channel_balance)
        self.coord_weight = float(coord_weight)
        self.type_weight = float(type_weight)
        self.n_dims = int(n_dims)
        self.checkpoint = str(checkpoint)

    # -- keep it frozen even if something calls .train() on the parent --------
    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @property
    def higher_is_better(self) -> bool:
        return self.surrogate.higher_is_better

    @property
    def atom_encoder(self) -> Dict[str, int]:
        return self.surrogate.atom_encoder

    def validate_against(self, ddpm_atom_encoder, ddpm_diffusion_params,
                         strict: bool = True):
        return validate_surrogate_against_ddpm(
            self.surrogate.cfg, self.atom_encoder,
            ddpm_atom_encoder, ddpm_diffusion_params, strict=strict)

    def effective_r(self, r_forward: torch.Tensor) -> torch.Tensor:
        return tweedie_residual_r(r_forward, self.residual_scale)

    def effective_r_types(self, r_forward: torch.Tensor) -> Optional[torch.Tensor]:
        if self.residual_scale_types is None:
            return None
        return tweedie_residual_r(r_forward, self.residual_scale_types)

    def score(self, xh_lig_hat: torch.Tensor, lig_mask: torch.Tensor,
              r_eff: torch.Tensor,
              r_eff_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[G] scores in the surrogate's target space.  Differentiable in `xh`."""
        xh = xh_lig_hat
        if self.channel_balance != "none":
            xh = _ChannelBalance.apply(
                xh, self.n_dims, self.coord_weight, self.type_weight,
                self.channel_balance == "normalize")
        return self.surrogate.score_from_xh(
            xh, lig_mask, r_eff, tau=self.tau, hard=self.hard,
            deterministic=self.deterministic, r_types=r_eff_types)

    def as_loss(self, score: torch.Tensor) -> torch.Tensor:
        """Signed so that minimising the returned value improves SA."""
        return -score if self.higher_is_better else score


# -----------------------------------------------------------------------------
# Tweedie reward pass
# -----------------------------------------------------------------------------

def reward_pass(
    ddpm: nn.Module,
    ligand: Dict[str, torch.Tensor],
    pocket: Dict[str, torch.Tensor],
    t_min: int,
    t_max: int,
    conditional: bool,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """One denoising pass at a restricted low timestep, returning Tweedie's x0_hat.

    Independent of the DDPM loss's own timestep draw: that one must span [0, T]
    to keep the model a valid diffusion model, while the reward only carries
    signal at low t (atom types are near-uniform past t ~ 75 of 500).

    `ligand` and `pocket` are mutated by `ddpm.normalize`, so pass freshly built
    dicts -- not the ones already handed to `forward()`.
    """
    device = ligand["x"].device
    batch_size = ligand["size"].size(0)

    ligand, pocket = ddpm.normalize(ligand, pocket)

    hi = min(int(t_max), int(ddpm.T))
    lo = max(0, min(int(t_min), hi))
    if generator is None:
        t_int = torch.randint(lo, hi + 1, size=(batch_size, 1), device=device).float()
    else:
        t_int = torch.randint(lo, hi + 1, size=(batch_size, 1),
                              generator=generator, device=device).float()
    t = t_int / ddpm.T

    gamma_t = ddpm.inflate_batch_array(ddpm.gamma(t), ligand["x"])
    alpha_t = ddpm.alpha(gamma_t, ligand["x"])
    sigma_t = ddpm.sigma(gamma_t, ligand["x"])

    xh0_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
    xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)

    if conditional:
        xh0_lig[:, :ddpm.n_dims], xh0_pocket[:, :ddpm.n_dims] = \
            ddpm.remove_mean_batch(xh0_lig[:, :ddpm.n_dims],
                                   xh0_pocket[:, :ddpm.n_dims],
                                   ligand["mask"], pocket["mask"])
        z_t_lig, xh_pocket, _eps = ddpm.noised_representation(
            xh0_lig, xh0_pocket, ligand["mask"], pocket["mask"], gamma_t)
        net_out_lig, _ = ddpm.dynamics(z_t_lig, xh_pocket, t,
                                       ligand["mask"], pocket["mask"])
        second_arg = xh_pocket
    else:
        z_t_lig, z_t_pocket, _e1, _e2 = ddpm.noised_representation(
            xh0_lig, xh0_pocket, ligand["mask"], pocket["mask"], gamma_t)
        net_out_lig, _ = ddpm.dynamics(z_t_lig, z_t_pocket, t,
                                       ligand["mask"], pocket["mask"])
        second_arg = z_t_pocket

    xh_lig_hat = ddpm.xh_given_zt_and_epsilon(z_t_lig, net_out_lig, gamma_t,
                                              ligand["mask"])

    return {
        "xh_lig_hat": xh_lig_hat,
        "net_out_lig": net_out_lig,
        "z_t_lig": z_t_lig,
        "second_arg": second_arg,
        "t": t,
        "t_int": t_int.squeeze(-1),
        "gamma_t": gamma_t,
        "r_forward": (sigma_t / alpha_t).squeeze(-1),
    }


def anchor_loss(
    ref_ddpm: nn.Module,
    net_out_lig: torch.Tensor,
    z_t_lig: torch.Tensor,
    second_arg: torch.Tensor,
    t: torch.Tensor,
    lig_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    num_lig_atoms: torch.Tensor,
    n_dims: int,
    atom_nf: int,
) -> torch.Tensor:
    """||eps_theta - eps_frozen||^2 at the reward pass's own (z_t, t).

    Evaluated exactly where the reward perturbs theta -- which is where drift
    originates -- and it reuses the z_t already computed, so it costs one frozen
    forward rather than a second noising.  The DDPM loss on real data covers the
    rest of the schedule.
    """
    with torch.no_grad():
        ref_out, _ = ref_ddpm.dynamics(z_t_lig, second_arg, t, lig_mask, pocket_mask)
    sq = (net_out_lig - ref_out.detach()) ** 2
    per_graph = torch.zeros(num_lig_atoms.size(0), device=sq.device,
                            dtype=sq.dtype).index_add_(0, lig_mask, sq.sum(-1))
    return per_graph / ((n_dims + atom_nf) * num_lig_atoms.clamp(min=1))
