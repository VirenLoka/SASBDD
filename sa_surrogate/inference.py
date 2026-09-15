"""
Inference for the SA surrogate, and the entry point stage 2 will call.

Three ways in:

  SASurrogate.predict_smiles / predict_sdf   convenience, for checking the model
                                             against RDKit on real molecules.

  SASurrogate.score(one_hot, pos, batch, r)  raw tensors.  Differentiable, and
                                             respects the ambient grad mode.

  SASurrogate.score_from_xh(xh, batch, r)    THE STAGE-2 HOOK.  Takes DiffSBDD's
                                             concatenated [x, h] in normalised
                                             space -- exactly what
                                             `xh_given_zt_and_epsilon` returns as
                                             `xh_lig_hat` (en_diffusion.py:471,
                                             already surfaced by forward() and
                                             already consumed by the LJ auxiliary
                                             loss at lightning_modules.py:285) --
                                             applies the calibrated box-integral
                                             categorical readout and the Gumbel
                                             straight-through estimator, and
                                             returns a differentiable score per
                                             ligand.

Sign convention: the returned score is in the configured target space.
  target.mode = raw_sa     -> RDKit's 1..10 scale, LOWER is easier to synthesise,
                              so stage 2 should MINIMISE it.
  target.mode = pocket2mol -> (10 - sa) / 9, HIGHER is better, so MAXIMISE it.
`higher_is_better` on the instance states which, so stage 2 need not guess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from sa_surrogate.dataset import (SALMDBDataset, collate_fn, load_config,  # noqa: E402
                                  resolve_path, to_target)
from sa_surrogate.diffusion_bridge import (NoiseSchedule,  # noqa: E402
                                           categorical_logprobs_from_h,
                                           gumbel_straight_through)
from sa_surrogate.model import build_model  # noqa: E402
from sa_surrogate.train import evaluate, get_device  # noqa: E402


class SASurrogate:
    def __init__(self, checkpoint: str | Path, device: str = "auto",
                 use_ema: bool = True):
        self.device = get_device(device)
        ckpt = torch.load(str(resolve_path(checkpoint)), map_location="cpu",
                          weights_only=False)
        self.cfg = ckpt["config"]
        self.target = ckpt["target"]
        self.atom_encoder: Dict[str, int] = ckpt["atom_encoder"]
        self.num_classes = int(ckpt["num_classes"])

        self.model = build_model(self.cfg, self.num_classes)
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.model.to(self.device).eval()

        ccfg = self.cfg["corruption"]
        self.schedule = NoiseSchedule(ccfg["noise_schedule"], ccfg["timesteps"],
                                      ccfg["noise_precision"]).to(self.device)
        self.h_scale = float(ccfg["norm_values"][1])
        self.x_scale = float(ccfg["norm_values"][0])
        self.higher_is_better = (self.target["mode"] == "pocket2mol")

        print(f"[surrogate] {checkpoint} | target={self.target['mode']} "
              f"label={self.target['label_builder']} "
              f"higher_is_better={self.higher_is_better} "
              f"epoch={ckpt.get('epoch')} val_MAE={ckpt.get('select_mae'):.4f}")

    # -- core ---------------------------------------------------------------
    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        if not self.target["normalize"]:
            return y
        return y * self.target["std"] + self.target["mean"]

    def score(
        self,
        one_hot: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        r: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """[G] scores in the configured target space.

        Differentiable w.r.t. both `one_hot` and `pos`.  Does NOT wrap itself in
        no_grad -- stage 2 needs the graph.  Use `predict_*` for plain inference.
        """
        out = self.model(one_hot.to(self.device), pos.to(self.device),
                         batch.to(self.device),
                         None if r is None else r.to(self.device))
        pred = out["pred"]
        if self.cfg["model"]["head"] != "regression":
            centres = torch.linspace(0, 1, pred.size(-1), device=pred.device)
            pred = (pred.softmax(-1) * centres).sum(-1)
        return self.denormalize(pred)

    def score_from_xh(
        self,
        xh: torch.Tensor,
        batch: torch.Tensor,
        r: torch.Tensor,
        n_dims: int = 3,
        tau: Optional[float] = None,
        hard: bool = True,
        deterministic: bool = False,
        r_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Stage-2 hook.  `xh` is DiffSBDD's [N, n_dims + K] in normalised space.

        The h columns are NOT logits: they are a Gaussian relaxation of the
        one-hot divided by norm_values[1].  They are converted to calibrated
        log-probabilities with the same box integral the codebase uses at t=0
        (en_diffusion.py:185), then discretised with Gumbel straight-through.

        `r` is the noise level of `xh`.  For a Tweedie estimate that is the
        denoiser's *residual* level, not the forward r(t) -- see
        `diffusion_bridge.tweedie_residual_r`, and calibrate it against the
        pretrained checkpoint before relying on it (stage2/calibrate.py).

        `r_types` optionally supplies a different level for the categorical
        readout than for the model's sigma conditioning.  The denoiser's
        residual error is not necessarily the same size on the coordinate and
        type blocks, and calibration reports them separately.
        """
        if tau is None:
            tau = float(self.cfg["corruption"]["gumbel"]["tau"])

        xh = xh.to(self.device)
        batch = batch.to(self.device)
        r = r.to(self.device)

        pos = xh[:, :n_dims] * self.x_scale
        h_norm = xh[:, n_dims:]

        rt = r if r_types is None else r_types.to(self.device)
        logprobs = categorical_logprobs_from_h(h_norm, rt[batch].unsqueeze(-1),
                                               h_scale=self.h_scale)
        one_hot = gumbel_straight_through(logprobs, tau=tau, hard=hard,
                                          deterministic=deterministic)
        return self.score(one_hot, pos, batch, r)

    # -- convenience --------------------------------------------------------
    def _mols_to_batch(self, mols: Sequence[Chem.Mol]):
        pos_l, oh_l, b_l, keep = [], [], [], []
        for i, mol in enumerate(mols):
            if mol is None or mol.GetNumConformers() == 0:
                continue
            m = Chem.RemoveHs(Chem.Mol(mol))
            syms = [a.GetSymbol() for a in m.GetAtoms()]
            if any(s not in self.atom_encoder for s in syms):
                continue
            t = torch.tensor([self.atom_encoder[s] for s in syms], dtype=torch.long)
            oh = torch.zeros(len(t), self.num_classes).scatter_(1, t[:, None], 1.0)
            pos_l.append(torch.tensor(m.GetConformer().GetPositions(), dtype=torch.float32))
            oh_l.append(oh)
            b_l.append(torch.full((len(t),), len(keep), dtype=torch.long))
            keep.append(i)
        if not keep:
            return None, None, None, []
        return torch.cat(pos_l), torch.cat(oh_l), torch.cat(b_l), keep

    @torch.no_grad()
    def predict_mols(self, mols: Sequence[Chem.Mol]) -> List[Optional[float]]:
        pos, oh, b, keep = self._mols_to_batch(mols)
        out: List[Optional[float]] = [None] * len(mols)
        if not keep:
            return out
        scores = self.score(oh, pos, b).float().cpu().tolist()
        for i, s in zip(keep, scores):
            out[i] = float(s)
        return out

    @torch.no_grad()
    def predict_smiles(self, smiles: Sequence[str], seed: int = 0) -> List[Optional[float]]:
        """Embeds a 3D conformer per SMILES -- the model is geometry-based."""
        from rdkit.Chem import AllChem
        mols = []
        for smi in smiles:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                mols.append(None)
                continue
            m = Chem.AddHs(m)
            p = AllChem.ETKDGv3()
            p.randomSeed = seed
            mols.append(m if AllChem.EmbedMolecule(m, p) == 0 else None)
        return self.predict_mols(mols)

    @torch.no_grad()
    def predict_sdf(self, sdf_path: str | Path) -> List[Optional[float]]:
        return self.predict_mols(list(Chem.SDMolSupplier(str(sdf_path), sanitize=False)))


# -----------------------------------------------------------------------------
# reference SA, for comparing the surrogate against RDKit
# -----------------------------------------------------------------------------

def reference_sa(mols: Sequence[Chem.Mol], mode: str, builder: str = "sdf",
                 atom_encoder: Optional[dict] = None) -> List[Optional[float]]:
    from sa_surrogate.dataset import compute_labels
    from constants import dataset_params
    info = dataset_params["crossdock"]
    enc = atom_encoder or info["atom_encoder"]
    out: List[Optional[float]] = []
    for mol in mols:
        if mol is None:
            out.append(None)
            continue
        try:
            m = Chem.Mol(mol)
            Chem.SanitizeMol(m)
            m = Chem.RemoveHs(m)
            pos = np.asarray(m.GetConformer().GetPositions(), dtype=np.float32)
            types = np.asarray([enc[a.GetSymbol()] for a in m.GetAtoms()], dtype=np.uint8)
            sa = compute_labels(pos, types, m, info, (builder,))[builder]
            out.append(None if sa is None else to_target(sa, mode))
        except Exception:
            out.append(None)
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SA surrogate inference")
    p.add_argument("--config", default=str(_REPO_ROOT / "configs/stage1_surrogate.yaml"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smiles", nargs="+", help="score SMILES (3D-embedded first)")
    g.add_argument("--sdf", help="score every molecule in an SDF file")
    g.add_argument("--split", choices=["train", "val", "test"],
                   help="evaluate an LMDB split per noise level")
    p.add_argument("--compare-rdkit", action="store_true",
                   help="also report the true RDKit SA and the surrogate's error")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    ckpt = args.checkpoint or cfg["inference"]["checkpoint"]
    sur = SASurrogate(ckpt, device=args.device or cfg.get("device", "auto"))
    mode = sur.target["mode"]

    if args.split:
        ds = SALMDBDataset(lmdb_path=cfg["data"]["lmdb_path"], split=args.split,
                           label_builder=cfg["data"]["label_builder"],
                           target_mode=mode, normalize=bool(cfg["target"]["normalize"]))
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=int(cfg["inference"]["batch_size"]),
                        shuffle=False, collate_fn=collate_fn)
        buckets = [t for t in cfg["train"]["eval_t_buckets"]
                   if t <= cfg["corruption"]["t_max"]]
        res = evaluate(sur.model, dl, sur.schedule, sur.cfg, sur.device, ds, buckets)
        print(f"\n{args.split} split, {len(ds)} molecules, target={mode}")
        print(f"{'bucket':>8s} {'r':>7s} {'MAE':>7s} {'RMSE':>7s} "
              f"{'pearson':>8s} {'spearman':>9s} {'R2':>7s}")
        for k, m in res.items():
            print(f"{k:>8s} {m['r']:7.3f} {m['mae']:7.3f} {m['rmse']:7.3f} "
                  f"{m['pearson']:8.3f} {m['spearman']:9.3f} {m['r2']:7.3f}")
        return 0

    if args.smiles:
        names, preds = list(args.smiles), sur.predict_smiles(args.smiles)
        mols = []
        for smi in args.smiles:
            from rdkit.Chem import AllChem
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                m = Chem.AddHs(m)
                pp = AllChem.ETKDGv3()
                pp.randomSeed = 0
                m = m if AllChem.EmbedMolecule(m, pp) == 0 else None
            mols.append(m)
    else:
        mols = list(Chem.SDMolSupplier(str(args.sdf), sanitize=False))
        names = [m.GetProp("_Name") if (m is not None and m.HasProp("_Name")) else f"mol_{i}"
                 for i, m in enumerate(mols)]
        preds = sur.predict_mols(mols)

    refs = reference_sa(mols, mode, builder=sur.target["label_builder"],
                        atom_encoder=sur.atom_encoder) if args.compare_rdkit \
        else [None] * len(mols)

    hdr = f"{'molecule':<44s} {'predicted':>10s}"
    if args.compare_rdkit:
        hdr += f" {'rdkit':>8s} {'error':>8s}"
    print(hdr)
    errs = []
    for name, pr, rf in zip(names, preds, refs):
        line = f"{str(name)[:44]:<44s} " + (f"{pr:10.3f}" if pr is not None else f"{'--':>10s}")
        if args.compare_rdkit:
            line += f" {rf:8.3f}" if rf is not None else f" {'--':>8s}"
            if pr is not None and rf is not None:
                errs.append(abs(pr - rf))
                line += f" {pr - rf:8.3f}"
            else:
                line += f" {'--':>8s}"
        print(line)
    if errs:
        print(f"\nMAE vs RDKit over {len(errs)} molecules: {np.mean(errs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
