"""
Measure the denoiser's residual error, so `surrogate.residual_scale` is a
measurement rather than a guess.

The surrogate is conditioned on the noise level of its input.  At stage 2 that
input is Tweedie's x0_hat, whose error is *smaller* than the forward r(t)
because the denoiser removes most of it.  `residual_scale` is the fraction it
fails to remove:

    r_eff(t) = residual_scale * r(t),   r(t) = sigma_t / alpha_t

Tell the surrogate the wrong level and it is being asked to score an input it
was never calibrated on -- no crash, just a quietly wrong reward.

The coordinate and type blocks are reported separately because there is no
reason for the denoiser to be equally good at both, and the two feed different
parts of the surrogate (sigma conditioning vs the categorical readout).  If they
differ materially, set `residual_scale` from the coordinate row and
`residual_scale_types` from the type row.

    python stage2/calibrate.py --override \
        base_checkpoint=/path/to/ckpt paths.processed_crossdock=/path/to/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.compat import ensure_dependencies  # noqa: E402


def calibrate(cfg, base_ckpt: Path, n_batches: int = 16,
              timesteps: Optional[Sequence[int]] = None, split: str = "val"):
    import torch
    from torch.utils.data import DataLoader

    from sa_surrogate.train import get_device
    from stage2.train import build_module

    # Outside a Trainer nothing moves the module for us, and DiffSBDD's EGNN
    # self-moves to `egnn_params.device` in its own __init__ (egnn_new.py:161)
    # -- so without an explicit .to() the EGNN sits on cuda while everything
    # else, including the batch, stays on the CPU.
    device = get_device(str(cfg.project.get("device", "auto")))
    module = build_module(cfg, base_ckpt, verbose=False).to(device).eval()
    print(f"[calibrate] device: {device}")
    ddpm = module.ddpm
    n_dims = module.x_dims
    conditional = module.conditional

    # Reuse the module's own setup so calibration measures whichever source
    # stage 2 will actually train on (npz or targetdiff_lmdb), rather than
    # duplicating the loader and silently diverging.
    if split == "test":
        module.setup("test")
        ds = module.test_dataset
    else:
        module.setup("fit")
        ds = module.val_dataset if split == "val" else module.train_dataset
    if ds is None or len(ds) == 0:
        raise SystemExit(f"the '{split}' split is empty; try --split train")
    dl = DataLoader(ds, batch_size=int(cfg.diffsbdd.get("batch_size", 8)),
                    shuffle=False, collate_fn=ds.collate_fn)

    timesteps = list(timesteps or [0, 5, 10, 25, 50, 75, 100, 150, 200])
    rows = []

    print(f"[calibrate] {split} split, {len(ds)} complexes, "
          f"{min(n_batches, len(dl))} batches per timestep")

    for t_fixed in timesteps:
        res_x, res_h, r_fwd_all = [], [], []
        for bi, data in enumerate(dl):
            if bi >= n_batches:
                break
            ligand, pocket = module.get_ligand_and_pocket(data)
            assert ligand["x"].device == device, (
                f"batch landed on {ligand['x'].device}, module on {device}")
            ligand, pocket = ddpm.normalize(ligand, pocket)
            B = ligand["size"].size(0)

            t_int = torch.full((B, 1), float(t_fixed), device=ligand["x"].device)
            t = t_int / ddpm.T
            gamma_t = ddpm.inflate_batch_array(ddpm.gamma(t), ligand["x"])
            alpha_t = ddpm.alpha(gamma_t, ligand["x"])
            sigma_t = ddpm.sigma(gamma_t, ligand["x"])

            xh0_lig = torch.cat([ligand["x"], ligand["one_hot"]], dim=1)
            xh0_pocket = torch.cat([pocket["x"], pocket["one_hot"]], dim=1)

            with torch.no_grad():
                if conditional:
                    xh0_lig[:, :n_dims], xh0_pocket[:, :n_dims] = \
                        ddpm.remove_mean_batch(xh0_lig[:, :n_dims],
                                               xh0_pocket[:, :n_dims],
                                               ligand["mask"], pocket["mask"])
                    z_t, second, _ = ddpm.noised_representation(
                        xh0_lig, xh0_pocket, ligand["mask"], pocket["mask"], gamma_t)
                    net_out, _ = ddpm.dynamics(z_t, second, t,
                                               ligand["mask"], pocket["mask"])
                else:
                    z_t, z_p, _, _ = ddpm.noised_representation(
                        xh0_lig, xh0_pocket, ligand["mask"], pocket["mask"], gamma_t)
                    net_out, _ = ddpm.dynamics(z_t, z_p, t,
                                               ligand["mask"], pocket["mask"])
                xh_hat = ddpm.xh_given_zt_and_epsilon(z_t, net_out, gamma_t,
                                                      ligand["mask"])

            err = xh_hat - xh0_lig
            res_x.append(err[:, :n_dims].pow(2).mean().sqrt().item())
            res_h.append(err[:, n_dims:].pow(2).mean().sqrt().item())
            r_fwd_all.append((sigma_t / alpha_t).mean().item())

        r_fwd = float(np.mean(r_fwd_all))
        rx, rh = float(np.mean(res_x)), float(np.mean(res_h))
        rows.append((t_fixed, r_fwd, rx, rh, rx / max(r_fwd, 1e-12), rh / max(r_fwd, 1e-12)))

    print()
    print(f"{'t':>5s} {'r(t)':>9s} {'rms(dx)':>9s} {'rms(dh)':>9s} "
          f"{'scale_x':>9s} {'scale_h':>9s}")
    for t_fixed, r_fwd, rx, rh, sx, sh in rows:
        print(f"{t_fixed:5d} {r_fwd:9.4f} {rx:9.4f} {rh:9.4f} {sx:9.4f} {sh:9.4f}")

    # Weight the recommendation by the band the reward actually uses.
    t_lo = int(cfg.reward.get("t_min", 0))
    t_hi = int(cfg.reward.get("t_max", 75))
    band = [(sx, sh) for t_fixed, _, _, _, sx, sh in rows if t_lo <= t_fixed <= t_hi]
    if band:
        mx = float(np.median([b[0] for b in band]))
        mh = float(np.median([b[1] for b in band]))
        print()
        print(f"Median over the reward band t in [{t_lo}, {t_hi}]:")
        print(f"  surrogate.residual_scale:       {mx:.3f}   (coordinates)")
        print(f"  surrogate.residual_scale_types: {mh:.3f}   (atom types)")
        if abs(mx - mh) > 0.15 * max(mx, mh, 1e-9):
            print("  -> they differ materially; set both rather than one.")
        else:
            print("  -> close enough that a single residual_scale is fine.")
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_dependencies()
    from common.config import add_config_args, load_config, resolve_path

    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, default="stage2_reward.yaml")
    p.add_argument("--base-checkpoint", default=None)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--timesteps", type=int, nargs="*", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.override)
    base = args.base_checkpoint or cfg.get("base_checkpoint") \
        or cfg.paths.get("diffsbdd_ckpt")
    if not base:
        raise SystemExit("pass --base-checkpoint or set base_checkpoint in the config")
    calibrate(cfg, Path(resolve_path(base)), args.n_batches, args.timesteps, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
