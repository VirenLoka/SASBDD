"""
Train the SA surrogate.

Each molecule is drawn clean from the LMDB and corrupted on the fly, so the same
molecule is seen at many noise levels over training.  The corruption path is the
one in diffusion_bridge.py -- the identical box-integral + Gumbel-ST that stage 2
will apply to the denoiser's output -- so the surrogate is trained on the input
distribution it will actually be asked to score.

Validation is reported per noise level, not just in aggregate.  The aggregate
number is close to meaningless here: at high sigma the Bayes-optimal prediction
is the marginal mean, so a model that has learnt nothing useful still scores
well on the high-noise share of the data.  The low-t rows are the ones that say
whether stage-2 guidance will carry signal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sa_surrogate.dataset import SALMDBDataset, collate_fn, load_config, resolve_path
from sa_surrogate.diffusion_bridge import NoiseSchedule, corrupt, sample_noise_levels
from sa_surrogate.model import build_model


# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    """Resolve the compute device.

    `auto` deliberately skips MPS.  PyG's TransformerConv aggregates through
    aten::scatter_reduce, which Apple's MPS backend does not implement, so MPS
    only runs at all with PYTORCH_ENABLE_MPS_FALLBACK=1 -- and the resulting
    CPU round-trips measured 11.2 s/epoch against 2.1 s/epoch on plain CPU for
    this model.  Selecting it automatically would be a 5x slowdown.  Ask for it
    explicitly (device: mps) if you want it anyway.
    """
    if name != "auto":
        dev = torch.device(name)
        if dev.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            print("[warn] MPS needs PYTORCH_ENABLE_MPS_FALLBACK=1 (PyG's scatter_reduce "
                  "is unimplemented there), and it must be set BEFORE torch is imported:\n"
                  "         PYTORCH_ENABLE_MPS_FALLBACK=1 python sa_surrogate/train.py ...\n"
                  "       Falling back to CPU, which is faster here regardless.")
            return torch.device("cpu")
        return dev
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EMA:
    """Exponential moving average of model weights, applied via swap/restore."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if k in self.shadow}
        model.load_state_dict({**model.state_dict(),
                               **{k: v.to(next(model.parameters()).dtype)
                                  for k, v in self.shadow.items()}}, strict=False)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self._backup:
            model.load_state_dict({**model.state_dict(), **self._backup}, strict=False)
            self._backup = {}


# -- metrics (torch-native, no scipy dependency) -------------------------------

def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = xc.norm() * yc.norm()
    return float((xc @ yc) / denom) if float(denom) > 0 else float("nan")


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Average ranks, so ties do not distort Spearman."""
    n = x.numel()
    order = torch.argsort(x)
    ranks = torch.empty(n, dtype=torch.float64, device=x.device)
    ranks[order] = torch.arange(n, dtype=torch.float64, device=x.device)
    xs = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    return _pearson(_rankdata(x.double()), _rankdata(y.double()))


def _r2(pred: torch.Tensor, true: torch.Tensor) -> float:
    if true.numel() < 2:
        return float("nan")
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if float(ss_tot) > 0 else float("nan")


def regression_metrics(pred: torch.Tensor, true: torch.Tensor) -> Dict[str, float]:
    err = pred - true
    return {
        "mae": float(err.abs().mean()),
        "rmse": float((err ** 2).mean().sqrt()),
        "pearson": _pearson(pred, true),
        "spearman": _spearman(pred, true),
        "r2": _r2(pred, true),
        "n": int(true.numel()),
    }


# -----------------------------------------------------------------------------
# train / eval steps
# -----------------------------------------------------------------------------

def _corrupt_batch(batch, schedule, ccfg, device, t=None, generator=None,
                   deterministic=False):
    """Returns (pos, one_hot, r).  `t=None` samples per the config."""
    pos = batch["pos"].to(device)
    one_hot = batch["one_hot"].to(device)
    bidx = batch["batch"].to(device)
    num_graphs = batch["num_nodes"].numel()

    if not ccfg.get("enabled", True):
        r = schedule.r(torch.zeros(num_graphs, dtype=torch.long, device=device))
        return pos, one_hot, r

    if t is None:
        t = sample_noise_levels(
            num_graphs, schedule,
            t_min=ccfg["t_min"], t_max=ccfg["t_max"],
            sampling=ccfg["sampling"], clean_fraction=ccfg.get("clean_fraction", 0.0),
            device=device, generator=generator,
        )
    else:
        t = torch.full((num_graphs,), int(t), dtype=torch.long, device=device)

    r = schedule.r(t)
    pos, one_hot = corrupt(
        pos, one_hot, bidx, r,
        h_scale=float(ccfg["norm_values"][1]), x_scale=float(ccfg["norm_values"][0]),
        tau=float(ccfg["gumbel"]["tau"]), hard=bool(ccfg["gumbel"]["hard"]),
        corrupt_coords=bool(ccfg.get("corrupt_coords", True)),
        corrupt_types=bool(ccfg.get("corrupt_types", True)),
        generator=generator, deterministic=deterministic,
    )
    return pos, one_hot, r


def compute_loss(out, y, cfg, bin_edges=None):
    tcfg = cfg["train"]
    if cfg["model"]["head"] == "regression":
        if tcfg["loss"] == "huber":
            loss = nn.functional.smooth_l1_loss(out["pred"], y, beta=float(tcfg["huber_beta"]))
        elif tcfg["loss"] == "mse":
            loss = nn.functional.mse_loss(out["pred"], y)
        else:
            raise ValueError(f"unknown loss: {tcfg['loss']!r}")
        if "aux_logits" in out and bin_edges is not None:
            tgt = torch.bucketize(y, bin_edges).clamp(0, out["aux_logits"].size(-1) - 1)
            loss = loss + float(cfg["model"]["aux_classification"]["weight"]) * \
                nn.functional.cross_entropy(out["aux_logits"], tgt)
    else:
        tgt = torch.bucketize(y, bin_edges).clamp(0, out["pred"].size(-1) - 1)
        loss = nn.functional.cross_entropy(out["pred"], tgt)
    return loss


@torch.no_grad()
def evaluate(model, loader, schedule, cfg, device, dataset, t_buckets) -> Dict[str, dict]:
    """Per-noise-level metrics, in raw SA units.

    Corruption uses a fixed generator and a deterministic Gumbel argmax so the
    numbers are comparable across epochs rather than resampling-noisy.
    """
    model.eval()
    results: Dict[str, dict] = {}
    for t in t_buckets:
        preds, trues = [], []
        gen = torch.Generator(device=device).manual_seed(1234 + int(t))
        for batch in tqdm(loader, desc=f"Eval t={t}", leave=False):
            pos, one_hot, r = _corrupt_batch(batch, schedule, cfg["corruption"],
                                             device, t=t, generator=gen,
                                             deterministic=True)
            out = model(one_hot, pos, batch["batch"].to(device), r)
            p = out["pred"]
            if cfg["model"]["head"] != "regression":
                # expected bin centre, so classification is comparable to regression
                centres = torch.linspace(0, 1, p.size(-1), device=device)
                p = (p.softmax(-1) * centres).sum(-1)
            preds.append(p.detach().float().cpu())
            trues.append(batch["y"].float())
        pred = dataset.denormalize(torch.cat(preds))
        true = dataset.denormalize(torch.cat(trues))
        results[f"t={t}"] = regression_metrics(pred, true)
        results[f"t={t}"]["r"] = float(schedule.r(torch.tensor(int(t))))
    return results


# -----------------------------------------------------------------------------
# main loop
# -----------------------------------------------------------------------------

def train(cfg: dict, args) -> Path:
    tcfg, ccfg = cfg["train"], cfg["corruption"]
    set_seed(int(cfg["seed"]))
    device = get_device(cfg.get("device", "auto"))
    print(f"[train] device={device}")

    common = dict(lmdb_path=cfg["data"]["lmdb_path"],
                  label_builder=cfg["data"]["label_builder"],
                  target_mode=cfg["target"]["mode"],
                  normalize=bool(cfg["target"]["normalize"]))
    train_ds = SALMDBDataset(split="train", **common)
    val_ds = SALMDBDataset(split="val", **common)
    print(f"[train] label={train_ds.label_builder} target={cfg['target']['mode']} "
          f"train={len(train_ds)} val={len(val_ds)} "
          f"(target mean={train_ds.target_mean:.3f} std={train_ds.target_std:.3f})")
    if len(val_ds) == 0:
        raise RuntimeError("empty validation split; increase data.val_fraction")

    dl_kw = dict(collate_fn=collate_fn, num_workers=int(tcfg["num_workers"]),
                 pin_memory=(device.type == "cuda"))
    train_dl = DataLoader(train_ds, batch_size=int(tcfg["batch_size"]),
                          shuffle=True, drop_last=False, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=int(tcfg["batch_size"]),
                        shuffle=False, **dl_kw)

    model = build_model(cfg, train_ds.num_classes).to(device)
    schedule = NoiseSchedule(ccfg["noise_schedule"], ccfg["timesteps"],
                             ccfg["noise_precision"]).to(device)
    print(f"[train] model params: {model.num_parameters():,}")

    bin_edges = torch.linspace(-2.5, 2.5, int(cfg["model"]["num_bins"]) - 1, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]))
    steps_per_epoch = max(1, len(train_dl))
    total_steps = steps_per_epoch * int(tcfg["epochs"])
    warmup = int(tcfg["warmup_steps"])

    def lr_at(step: int) -> float:
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        if tcfg["scheduler"] == "cosine":
            prog = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, prog))))
        return 1.0

    sched_lr = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    ema = EMA(model, float(tcfg["ema_decay"])) if float(tcfg["ema_decay"]) > 0 else None

    out_dir = resolve_path(tcfg["ckpt_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    t_buckets = [t for t in tcfg["eval_t_buckets"] if t <= ccfg["t_max"]]

    best = float("inf")
    best_path = out_dir / "best.pt"
    bad_epochs = 0
    step = 0

    for epoch in range(1, int(tcfg["epochs"]) + 1):
        model.train()
        running, nb, t0 = 0.0, 0, time.time()
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}", leave=False):
            pos, one_hot, r = _corrupt_batch(batch, schedule, ccfg, device)
            out = model(one_hot, pos, batch["batch"].to(device), r)
            loss = compute_loss(out, batch["y"].to(device), cfg, bin_edges)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   float(tcfg["grad_clip"]))
            opt.step()
            sched_lr.step()
            if ema:
                ema.update(model)

            running += loss.item()
            nb += 1
            step += 1
            if step % int(tcfg["log_every"]) == 0:
                print(f"  epoch {epoch} step {step} loss={running/max(nb,1):.4f} "
                      f"grad={float(gnorm):.2f} lr={sched_lr.get_last_lr()[0]:.2e}")

        train_loss = running / max(nb, 1)

        if epoch % int(tcfg["val_every"]) != 0:
            continue

        if ema:
            ema.apply_to(model)
        metrics = evaluate(model, val_dl, schedule, cfg, device, val_ds, t_buckets)
        if ema:
            ema.restore(model)

        # Selection uses the clean end of the schedule: that is where guidance
        # has to work, and the high-noise rows are near-constant by construction.
        sel_keys = [k for k in (f"t={t}" for t in t_buckets[:3]) if k in metrics]
        sel = float(np.mean([metrics[k]["mae"] for k in sel_keys]))

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} "
              f"low-noise val MAE={sel:.4f} ({time.time()-t0:.1f}s)")
        for k in metrics:
            m = metrics[k]
            print(f"    {k:>8s} (r={m['r']:6.3f})  MAE={m['mae']:.3f}  "
                  f"RMSE={m['rmse']:.3f}  rho={m['spearman']:.3f}  R2={m['r2']:.3f}")

        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "step": step,
                                "train_loss": train_loss, "select_mae": sel,
                                "val": metrics}) + "\n")

        if sel < best - 1e-5:
            best, bad_epochs = sel, 0
            payload = {
                "model": (ema.shadow if ema else model.state_dict()),
                "config": cfg,
                "epoch": epoch,
                "select_mae": sel,
                "metrics": metrics,
                "num_classes": train_ds.num_classes,
                "atom_encoder": train_ds.atom_encoder,
                "target": {"mode": cfg["target"]["mode"],
                           "normalize": bool(cfg["target"]["normalize"]),
                           "mean": train_ds.target_mean, "std": train_ds.target_std,
                           "label_builder": train_ds.label_builder},
            }
            torch.save(payload, best_path)
            print(f"    -> new best, saved to {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= int(tcfg["early_stop_patience"]):
                print(f"[train] early stop after {bad_epochs} epochs without improvement")
                break

    print(f"[train] best low-noise val MAE = {best:.4f} ({best_path})")
    return best_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train the SA surrogate")
    p.add_argument("--config", default=str(_REPO_ROOT / "configs/stage1_surrogate.yaml"))
    p.add_argument("--override", nargs="*", default=[],
                   help="dotted config overrides, e.g. train.epochs=5 model.hidden_dim=64")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.override)
    train(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
