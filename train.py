import argparse
from argparse import Namespace
from pathlib import Path
import warnings

import torch
import pytorch_lightning as pl
import numpy as np

from common.config import (add_config_args, load_config, merge_args_and_yaml,
                           resolve_path, save_config)
from lightning_modules import LigandPocketDDPM


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(f"Config parameter '{key}' (value: "
                          f"{config[key]}) will be overwritten with value "
                          f"{value} from the checkpoint.")
        config[key] = value
    return config


def build_logger(args, out_dir):
    """Logger is configurable now: wandb is no longer a hard requirement.

    Reads the `logging:` block if the config has one (configs/base.yaml), and
    otherwise falls back to the original wandb behaviour so existing configs
    keep working unchanged.
    """
    logging_cfg = getattr(args, 'logging', None)
    kind = 'wandb'
    if logging_cfg is not None:
        kind = str(getattr(logging_cfg, 'logger', 'wandb')).lower()

    if kind in ('none', 'false', ''):
        return False
    if kind == 'csv':
        return pl.loggers.CSVLogger(save_dir=args.logdir, name=args.run_name)
    if kind == 'tensorboard':
        return pl.loggers.TensorBoardLogger(save_dir=args.logdir,
                                            name=args.run_name)

    wandb_params = getattr(args, 'wandb_params', None)
    if logging_cfg is not None and hasattr(logging_cfg, 'wandb') and wandb_params is None:
        wandb_params = logging_cfg.wandb
    return pl.loggers.WandbLogger(
        save_dir=args.logdir,
        project=getattr(wandb_params, 'project', 'ligand-pocket-ddpm'),
        group=getattr(wandb_params, 'group', None),
        name=args.run_name,
        id=args.run_name,
        resume='must' if args.resume is not None else False,
        entity=getattr(wandb_params, 'entity', None),
        mode=getattr(wandb_params, 'mode', 'online'),
    )


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_config_args(p)
    p.add_argument('--resume', type=str, default=None)
    args = p.parse_args()
    if args.config is None:
        p.error('--config is required')

    # Project-wide loader: supports `defaults:` inheritance and dotted
    # --override, and still reads the original flat configs unchanged.
    config = load_config(args.config, args.override).to_dict()
    config.pop('diffsbdd', None)   # legacy configs are mirrored there; not a kwarg

    assert 'resume' not in config

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(
            ckpt_path, map_location=torch.device('cpu'))['hyper_parameters']

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)

    out_dir = Path(args.logdir, args.run_name)
    histogram_file = Path(resolve_path(args.datadir), 'size_distribution.npy')
    histogram = np.load(histogram_file).tolist()
    pl_module = LigandPocketDDPM(
        outdir=out_dir,
        dataset=args.dataset,
        datadir=args.datadir,
        batch_size=args.batch_size,
        lr=args.lr,
        egnn_params=args.egnn_params,
        diffusion_params=args.diffusion_params,
        num_workers=args.num_workers,
        augment_noise=args.augment_noise,
        augment_rotation=args.augment_rotation,
        clip_grad=args.clip_grad,
        eval_epochs=args.eval_epochs,
        eval_params=args.eval_params,
        visualize_sample_epoch=args.visualize_sample_epoch,
        visualize_chain_epoch=args.visualize_chain_epoch,
        auxiliary_loss=args.auxiliary_loss,
        loss_params=args.loss_params,
        mode=args.mode,
        node_histogram=histogram,
        pocket_representation=args.pocket_representation,
        virtual_nodes=args.virtual_nodes
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, Path(out_dir, 'config.resolved.yaml'))

    logger = build_logger(args, out_dir)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(out_dir, 'checkpoints'),
        filename="best-model-epoch={epoch:02d}",
        monitor="loss/val",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    gpus = getattr(args, 'gpus', 1)
    accelerator = getattr(args, 'accelerator', None) or (
        'gpu' if torch.cuda.is_available() else 'auto')

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=args.enable_progress_bar,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accelerator=accelerator, devices=gpus,
        strategy=('ddp' if gpus > 1 else None)
    )

    trainer.fit(model=pl_module, ckpt_path=ckpt_path)
