"""
Stage-2 entry point: SA-guided finetuning of a pretrained DiffSBDD checkpoint.

    python stage2/train.py --config configs/stage2_reward.yaml \
        --override base_checkpoint=/path/to/crossdocked_fullatom_cond.ckpt \
                   paths.processed_crossdock=/path/to/processed_crossdock_noH_full

Architecture hyperparameters come from the checkpoint being finetuned, so they
cannot drift out of sync with the weights.  The `diffsbdd:` block in the config
overrides them where you want something different (learning rate, batch size,
evaluation cadence).

The pretrained weights are loaded into BOTH `ddpm` and the frozen `ref_ddpm`, so
the anchor really is anchored to the pretrained model rather than to a random
initialisation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.compat import ensure_dependencies  # noqa: E402  (must precede DiffSBDD)

_PARENT_KWARGS = (
    "outdir", "dataset", "datadir", "batch_size", "lr", "egnn_params",
    "diffusion_params", "num_workers", "augment_noise", "augment_rotation",
    "clip_grad", "eval_epochs", "eval_params", "visualize_sample_epoch",
    "visualize_chain_epoch", "auxiliary_loss", "loss_params", "mode",
    "node_histogram", "pocket_representation", "virtual_nodes",
)


def _to_plain(value: Any) -> Any:
    """argparse.Namespace / ConfigNode / dict -> plain nested dict."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__") and not isinstance(value, (dict, list, tuple)):
        return {k: _to_plain(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    return value


def build_logger(cfg, outdir: Path, run_name: str):
    import pytorch_lightning as pl

    kind = str(cfg.logging.get("logger", "csv")).lower()
    if kind in ("none", "false", ""):
        return False
    if kind == "csv":
        return pl.loggers.CSVLogger(save_dir=str(outdir), name=run_name)
    if kind == "tensorboard":
        return pl.loggers.TensorBoardLogger(save_dir=str(outdir), name=run_name)
    if kind == "wandb":
        wb = cfg.logging.wandb
        return pl.loggers.WandbLogger(
            save_dir=str(outdir), name=run_name, id=run_name,
            project=wb.get("project", "diffsbdd-sa"), group=wb.get("group"),
            entity=wb.get("entity"), mode=wb.get("mode", "online"))
    raise ValueError(f"unknown logging.logger: {kind!r}")


def build_module(cfg, base_ckpt_path: Path, verbose: bool = True):
    import torch
    from common.config import _wrap, deep_merge, resolve_path
    from stage2.lightning_module import SAGuidedDDPM

    ckpt = torch.load(str(base_ckpt_path), map_location="cpu", weights_only=False)
    if "hyper_parameters" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(f"{base_ckpt_path} is not a DiffSBDD Lightning checkpoint")

    hp = dict(ckpt["hyper_parameters"])
    node_histogram = hp.pop("node_histogram", None)          # large; merged back below
    hp_plain = {k: _to_plain(v) for k, v in hp.items()}

    overrides = _to_plain(cfg.get("diffsbdd", {})) or {}
    merged = deep_merge(hp_plain, {k: v for k, v in overrides.items() if v is not None})

    # datadir / surrogate checkpoint fall back to the shared paths block
    if not merged.get("datadir"):
        merged["datadir"] = cfg.paths.get("processed_crossdock")
    lmdb_source = str((_to_plain(cfg.get("data", {})) or {}).get(
        "source", "npz")).lower() == "targetdiff_lmdb"
    if not merged.get("datadir") and lmdb_source:
        merged["datadir"] = "."      # unused by the LMDB source
    if not merged.get("datadir"):
        raise ValueError(
            "no dataset directory: set paths.processed_crossdock (or "
            "diffsbdd.datadir) to the output of process_crossdock.py -- the raw "
            "crossdocked_pocket10 release is not enough, stage 2 needs the "
            "processed train/val/test.npz")
    merged["datadir"] = str(resolve_path(merged["datadir"]))

    if node_histogram is None:
        import numpy as np
        hist_file = Path(merged["datadir"], "size_distribution.npy")
        if not hist_file.exists():
            raise FileNotFoundError(f"{hist_file} not found and the checkpoint "
                                    f"carries no node_histogram")
        node_histogram = np.load(hist_file).tolist()
    merged["node_histogram"] = node_histogram

    outdir = Path(resolve_path(cfg.paths.get("logdir", "runs")), cfg.run_name)
    merged["outdir"] = str(outdir)

    surrogate = _to_plain(cfg.surrogate)
    if not surrogate.get("checkpoint"):
        surrogate["checkpoint"] = cfg.paths.get("surrogate_ckpt")
    if not surrogate.get("checkpoint"):
        raise ValueError("no surrogate checkpoint: set surrogate.checkpoint or "
                         "paths.surrogate_ckpt")
    surrogate["checkpoint"] = str(resolve_path(surrogate["checkpoint"]))

    parent_kwargs = {k: _wrap(merged[k]) if isinstance(merged.get(k), dict) else merged.get(k)
                     for k in _PARENT_KWARGS}

    data_cfg = _to_plain(cfg.get("data", {})) or {"source": "npz"}
    if not data_cfg.get("raw_dir"):
        data_cfg["raw_dir"] = cfg.paths.get("crossdocked_raw")

    if verbose:
        changed = [k for k in overrides
                   if k in hp_plain and _to_plain(overrides[k]) != hp_plain[k]]
        print(f"[stage2] base checkpoint : {base_ckpt_path}")
        print(f"[stage2] mode            : {parent_kwargs['mode']} / "
              f"{parent_kwargs['pocket_representation']}")
        print(f"[stage2] datadir         : {merged['datadir']}")
        print(f"[stage2] surrogate       : {surrogate['checkpoint']}")
        print(f"[stage2] overridden from config: {sorted(changed) or 'nothing'}")

    module = SAGuidedDDPM(
        surrogate=surrogate,
        reward=_to_plain(cfg.reward),
        anchor=_to_plain(cfg.anchor),
        data=data_cfg,
        **parent_kwargs,
    )

    # Pretrained weights -> both the trainable model and the frozen reference.
    sd = ckpt["state_dict"]
    missing, unexpected = module.load_state_dict(sd, strict=False)
    loaded_ddpm = len([k for k in sd if k.startswith("ddpm.")])
    if module.ref_ddpm is not None:
        ref_sd = {("ref_" + k): v for k, v in sd.items() if k.startswith("ddpm.")}
        module.load_state_dict(ref_sd, strict=False)
        module.ref_ddpm.requires_grad_(False).eval()
    if unexpected:
        raise RuntimeError(f"checkpoint has keys this model does not: {unexpected[:5]}")
    if verbose:
        print(f"[stage2] loaded {loaded_ddpm} tensors into ddpm"
              + (" and ref_ddpm" if module.ref_ddpm is not None else "")
              + f"; {len(missing)} new keys initialised (surrogate/reference)")
    return module


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_dependencies()

    from common.config import add_config_args, load_config, resolve_path, save_config

    p = argparse.ArgumentParser(description="SA-guided finetuning of DiffSBDD")
    add_config_args(p, default="stage2_reward.yaml")
    p.add_argument("--base-checkpoint", default=None,
                   help="pretrained DiffSBDD checkpoint (overrides the config)")
    p.add_argument("--resume", default=None, help="resume a stage-2 run")
    p.add_argument("--steps", type=int, default=None,
                   help="cap training batches per epoch (smoke tests)")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.override)

    base_ckpt = args.base_checkpoint or cfg.get("base_checkpoint") \
        or cfg.paths.get("diffsbdd_ckpt")
    if not base_ckpt:
        raise SystemExit(
            "no pretrained checkpoint: pass --base-checkpoint, or set "
            "base_checkpoint / paths.diffsbdd_ckpt in the config")
    base_ckpt = resolve_path(base_ckpt)
    if not Path(base_ckpt).exists():
        raise SystemExit(f"checkpoint not found: {base_ckpt}")

    import pytorch_lightning as pl

    pl.seed_everything(int(cfg.project.get("seed", 42)), workers=True)
    module = build_module(cfg, Path(base_ckpt))

    outdir = Path(resolve_path(cfg.paths.get("logdir", "runs")), cfg.run_name)
    outdir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, outdir / "config.resolved.yaml")

    tr = cfg.trainer
    callbacks = [pl.callbacks.ModelCheckpoint(
        dirpath=str(outdir / "checkpoints"),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val", save_top_k=int(tr.get("save_top_k", 3)),
        save_last=True, mode="min")]

    devices = tr.get("devices", 1)
    trainer = pl.Trainer(
        max_epochs=int(tr.get("max_epochs", 50)),
        logger=build_logger(cfg, outdir, cfg.run_name),
        callbacks=callbacks,
        enable_progress_bar=bool(tr.get("enable_progress_bar", True)),
        num_sanity_val_steps=int(tr.get("num_sanity_val_steps", 0)),
        accelerator=str(tr.get("accelerator", "auto")),
        devices=devices,
        strategy=("ddp" if isinstance(devices, int) and devices > 1 else None),
        accumulate_grad_batches=int(tr.get("accumulate_grad_batches", 1)),
        limit_train_batches=(args.steps if args.steps is not None
                             else tr.get("limit_train_batches", 1.0)),
        limit_val_batches=tr.get("limit_val_batches", 1.0),
        log_every_n_steps=int(cfg.logging.get("log_every_n_steps", 50)),
    )
    trainer.fit(model=module, ckpt_path=args.resume)
    print(f"[stage2] done; artefacts in {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
