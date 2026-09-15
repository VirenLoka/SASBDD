"""
SA-guided finetuning of DiffSBDD.

Subclasses `LigandPocketDDPM` rather than editing it, so upstream stays intact.
Nothing in the parent's `forward()` is overridden -- the reward is a separate
denoising pass at its own timestep.

Per optimiser step:

  1. DDPM loss on real data, t ~ U[0, T]      (parent forward, unchanged)
  2. Reward pass at t ~ U[t_min, t_max]       (low t, where the type channel
     -> Tweedie x0_hat -> frozen surrogate       still carries signal)
  3. Anchor ||eps_theta - eps_frozen||^2 at the reward pass's own z_t
  4. Separate backward for (1) and (2+3), budget the aux gradient against the
     DDPM gradient, sum, clip, step.

The split backward is what makes the gradient budget and the clipping fix
possible, so the module uses manual optimisation.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

import utils
from lightning_modules import LigandPocketDDPM
from stage2.reward import FrozenSASurrogate, anchor_loss, reward_pass


def _global_norm(grads: Sequence[Optional[torch.Tensor]],
                 device: torch.device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for g in grads:
        if g is not None:
            total = total + g.detach().pow(2).sum()
    return total.sqrt()


class SAGuidedDDPM(LigandPocketDDPM):

    def __init__(self, *, surrogate: Dict[str, Any], reward: Dict[str, Any],
                 anchor: Dict[str, Any], **parent_kwargs):
        super().__init__(**parent_kwargs)

        # The split backward in training_step needs control of the optimiser.
        self.automatic_optimization = False

        self.reward_cfg = dict(reward)
        self.anchor_cfg = dict(anchor)
        self.surrogate_cfg = dict(surrogate)

        self.sa_surrogate = FrozenSASurrogate(
            checkpoint=surrogate["checkpoint"],
            device="cpu",                      # Lightning moves it with the module
            residual_scale=surrogate.get("residual_scale", 0.3),
            residual_scale_types=surrogate.get("residual_scale_types"),
            tau=surrogate.get("tau"),
            hard=surrogate.get("hard", True),
            deterministic=surrogate.get("deterministic", False),
            channel_balance=reward.get("channel_balance", "normalize"),
            coord_weight=reward.get("coord_weight", 1.0),
            type_weight=reward.get("type_weight", 1.0),
            n_dims=self.x_dims,
        )
        self.sa_surrogate.validate_against(
            self.lig_type_encoder, parent_kwargs["diffusion_params"],
            strict=surrogate.get("strict_validation", True))

        # Frozen reference for the KL-style anchor.  Deep-copied here so it is a
        # registered submodule and therefore saved and restored with the run;
        # the entry point loads the pretrained weights into BOTH this and
        # `self.ddpm`, so on a fresh finetune the reference really is the
        # pretrained model rather than a random init.
        if self.anchor_cfg.get("enabled", True):
            self.ref_ddpm = copy.deepcopy(self.ddpm)
            self.ref_ddpm.requires_grad_(False)
            self.ref_ddpm.eval()
        else:
            self.ref_ddpm = None

        self.conditional = (self.mode != "joint")
        self.grad_clip_ceiling = self.reward_cfg.get("grad_clip_ceiling")
        self._step_count = 0
        self._last_diag: Dict[str, float] = {}

    # -- keep frozen parts frozen through .train() ---------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "ref_ddpm", None) is not None:
            self.ref_ddpm.eval()
        if getattr(self, "sa_surrogate", None) is not None:
            self.sa_surrogate.eval()
        return self

    # -- reward weight -------------------------------------------------------
    def reward_weight_now(self) -> float:
        w = float(self.reward_cfg.get("weight", 0.0))
        warmup = int(self.reward_cfg.get("warmup_steps", 0))
        delay = int(self.reward_cfg.get("start_step", 0))
        if self._step_count < delay:
            return 0.0
        if warmup > 0:
            ramp = min(1.0, (self._step_count - delay) / warmup)
            w = w * max(0.0, ramp)
        return w

    # -- the aux pass --------------------------------------------------------
    def reward_and_anchor(self, data) -> Dict[str, torch.Tensor]:
        """Fresh ligand/pocket dicts: `ddpm.normalize` rebinds them, so reusing
        the ones already given to forward() would double-normalise."""
        ligand, pocket = self.get_ligand_and_pocket(data)
        lig_mask, pocket_mask = ligand["mask"], pocket["mask"]
        num_lig = ligand["size"]

        out = reward_pass(
            self.ddpm, ligand, pocket,
            t_min=int(self.reward_cfg.get("t_min", 0)),
            t_max=int(self.reward_cfg.get("t_max", 75)),
            conditional=self.conditional,
        )

        r_eff = self.sa_surrogate.effective_r(out["r_forward"])
        r_eff_types = self.sa_surrogate.effective_r_types(out["r_forward"])
        score = self.sa_surrogate.score(out["xh_lig_hat"], lig_mask, r_eff,
                                        r_eff_types)
        reward_term = self.sa_surrogate.as_loss(score).mean()

        terms: Dict[str, torch.Tensor] = {
            "sa_pred": score.detach().mean(),
            "reward_term": reward_term,
            "reward_t": out["t_int"].float().mean().detach(),
        }

        if self.ref_ddpm is not None:
            a = anchor_loss(
                self.ref_ddpm, out["net_out_lig"], out["z_t_lig"],
                out["second_arg"], out["t"], lig_mask, pocket_mask,
                num_lig, self.x_dims, self.ddpm.atom_nf).mean()
            terms["anchor"] = a
        return terms

    # -- optimisation --------------------------------------------------------
    def training_step(self, data, batch_idx, *args):
        opt = self.optimizers()
        params = [p for p in self.ddpm.parameters() if p.requires_grad]
        device = self.device

        # ---- 1. DDPM loss on real data (parent forward, untouched) ----------
        try:
            nll, info = self.forward(data)
        except RuntimeError as e:
            if self.trainer.num_devices < 2 and "out of memory" in str(e):
                print("WARNING: ran out of memory, skipping to the next batch")
                return None
            raise
        ddpm_loss = nll.mean(0)
        g_ddpm = torch.autograd.grad(ddpm_loss, params, allow_unused=True)

        # ---- 2. reward + anchor --------------------------------------------
        w = self.reward_weight_now()
        lam = float(self.anchor_cfg.get("weight", 0.0)) \
            if self.ref_ddpm is not None else 0.0
        g_aux: List[Optional[torch.Tensor]] = [None] * len(params)
        aux_terms: Dict[str, torch.Tensor] = {}

        if w > 0.0 or lam > 0.0:
            aux_terms = self.reward_and_anchor(data)
            aux = w * aux_terms["reward_term"]
            if lam > 0.0 and "anchor" in aux_terms:
                aux = aux + lam * aux_terms["anchor"]
            if aux.requires_grad:
                g_aux = list(torch.autograd.grad(aux, params, allow_unused=True))
            info["sa/reward_term"] = aux_terms["reward_term"].detach()
            info["sa/pred"] = aux_terms["sa_pred"]
            info["sa/reward_t"] = aux_terms["reward_t"]
            if "anchor" in aux_terms:
                info["sa/anchor"] = aux_terms["anchor"].detach()

        # ---- 3. budget the aux gradient against the DDPM gradient -----------
        n_ddpm = _global_norm(g_ddpm, device)
        n_aux = _global_norm(g_aux, device)
        rho = self.reward_cfg.get("grad_budget")
        if rho is not None and float(n_aux) > 0.0:
            scale = float(min(1.0, float(rho) * float(n_ddpm) / float(n_aux)))
        else:
            scale = 1.0

        opt.zero_grad(set_to_none=True)
        for p, gd, ga in zip(params, g_ddpm, g_aux):
            acc = None
            if gd is not None:
                acc = gd.clone()
            if ga is not None:
                acc = ga * scale if acc is None else acc.add_(ga, alpha=scale)
            p.grad = acc

        # ---- 4. clip, then step --------------------------------------------
        grad_norm = self._clip_gradients(params, n_ddpm)
        opt.step()
        self._step_count += 1

        info["loss"] = ddpm_loss.detach()
        info["sa/grad_ddpm"] = n_ddpm
        info["sa/grad_aux"] = n_aux
        # The single most useful number when stage 2 misbehaves.
        info["sa/grad_ratio"] = n_aux / n_ddpm.clamp_min(1e-12)
        info["sa/aux_scale"] = torch.tensor(scale, device=device)
        info["sa/weight"] = torch.tensor(w, device=device)
        info["sa/grad_total"] = grad_norm
        self._last_diag = {k: float(v) for k, v in info.items()
                           if torch.is_tensor(v) and v.numel() == 1}

        self.log_metrics(info, "train", batch_size=len(data["num_lig_atoms"]))
        return None

    def _clip_gradients(self, params, ddpm_norm: torch.Tensor) -> torch.Tensor:
        """Adaptive clipping, with the two fixes the reward term makes necessary.

        The upstream rule (lightning_modules.py:874) is
        `1.5*mean + 2*std` over a 50-step history, and it ratchets: under
        persistent clipping the queue fills with the threshold itself, std -> 0,
        so the threshold grows ~1.5x every 50 steps, unbounded.  Harmless for the
        well-behaved DDPM gradient, not harmless for a spiky Gumbel-ST reward.

        Two changes: a hard ceiling, and the queue is fed the DDPM-only norm so
        the reward term cannot inflate the threshold that protects the DDPM loss.
        """
        if not self.clip_grad:
            return _global_norm([p.grad for p in params], self.device)

        max_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
        if self.grad_clip_ceiling is not None:
            max_norm = min(max_norm, float(self.grad_clip_ceiling))

        total = torch.nn.utils.clip_grad_norm_(params, max_norm)
        self.gradnorm_queue.add(float(min(float(ddpm_norm), max_norm)))
        if float(total) > max_norm:
            print(f"Clipped gradient with value {float(total):.1f} "
                  f"while allowed {max_norm:.1f}")
        return total

    # -- reward-hacking detector --------------------------------------------
    @torch.no_grad()
    def evaluate_sa(self, n_samples: int, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Sample molecules, then compare the surrogate's score against RDKit's.

        This is the diagnostic that distinguishes real improvement from reward
        hacking: if `sa_pred` improves while `sa_true` does not, the model has
        learnt to fool the surrogate rather than to make easier molecules. A
        widening `sa_gap`, or falling validity, says the same thing.
        """
        from rdkit import Chem
        from analysis.molecule_builder import build_molecule
        from analysis.SA_Score.sascorer import calculateScore
        from sa_surrogate.dataset import to_target

        dataset = self.val_dataset
        if dataset is None or n_samples < 1:
            return {}
        batch_size = min(batch_size or self.eval_batch_size, n_samples)
        use_ob = (self.sa_surrogate.surrogate.target["label_builder"] != "edm")
        target_mode = self.sa_surrogate.surrogate.target["mode"]

        true_vals, pred_vals, n_built, n_valid = [], [], 0, 0

        for i in range(math.ceil(n_samples / batch_size)):
            n_batch = min(batch_size, n_samples - n_built)
            if n_batch < 1:
                break
            batch = dataset.collate_fn(
                [dataset[(i * batch_size + j) % len(dataset)] for j in range(n_batch)])
            _lig, pocket = self.get_ligand_and_pocket(batch)
            num_nodes_lig = self.ddpm.size_distribution.sample_conditional(
                n1=None, n2=pocket["size"])
            xh_lig, _xh_pocket, lig_mask, _ = self.ddpm.sample_given_pocket(
                pocket, num_nodes_lig)

            x = xh_lig[:, :self.x_dims].detach()
            atom_type = xh_lig[:, self.x_dims:].argmax(1).detach()

            # surrogate's own view of what it just produced, at the clean end
            one_hot = torch.zeros(len(atom_type), self.ddpm.atom_nf,
                                  device=x.device).scatter_(1, atom_type[:, None], 1.0)
            r_clean = self.sa_surrogate.surrogate.schedule.r(
                torch.zeros(len(num_nodes_lig), dtype=torch.long, device=x.device))
            pred = self.sa_surrogate.surrogate.score(one_hot, x, lig_mask.long(), r_clean)
            pred_vals.extend(pred.float().cpu().tolist())

            for xi, ti in zip(utils.batch_to_list(x.cpu(), lig_mask.cpu()),
                              utils.batch_to_list(atom_type.cpu(), lig_mask.cpu())):
                n_built += 1
                try:
                    mol = build_molecule(xi, ti, self.dataset_info,
                                         add_coords=True, use_openbabel=use_ob)
                    Chem.SanitizeMol(mol)
                    n_valid += 1
                    true_vals.append(to_target(float(calculateScore(mol)), target_mode))
                except Exception:
                    continue

        out: Dict[str, float] = {}
        if pred_vals:
            out["sa_pred"] = float(np.mean(pred_vals))
        if true_vals:
            out["sa_true"] = float(np.mean(true_vals))
            out["sa_true_std"] = float(np.std(true_vals))
        if pred_vals and true_vals:
            out["sa_gap"] = out["sa_pred"] - out["sa_true"]
        if n_built:
            out["sa_validity"] = n_valid / n_built
        return out

    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        if not self.trainer.is_global_zero:
            return
        every = int(self.reward_cfg.get("eval_sa_epochs", 0) or 0)
        n = int(self.reward_cfg.get("eval_sa_samples", 0) or 0)
        if every < 1 or n < 1 or (self.current_epoch + 1) % every != 0:
            return
        metrics = self.evaluate_sa(n)
        if metrics:
            print("[sa-eval] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            self.log_metrics(metrics, "val")
