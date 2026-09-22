"""
Verify that stage 2 runs entirely on the GPU.

Why this exists: DiffSBDD's EGNN moves *itself* onto `egnn_params.device` inside
its own `__init__` (equivariant_diffusion/egnn_new.py:161), so a freshly built
module is legitimately split across devices -- EGNN on cuda, everything else on
the CPU -- until something moves the rest.  Under a Trainer that happens
automatically; in a standalone script it does not, which is what made
`calibrate.py` fail with

    Expected all tensors to be on the same device, but found at least two
    devices, cuda:0 and cpu!

This script reports the split before and after placement, then runs real
training steps through a Trainer so any remaining misplacement raises rather
than lurking.

    python stage2/device_check.py                     # uses configs/stage2_reward.yaml
    python stage2/device_check.py --no-train-steps    # audit only, no fit
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.compat import ensure_dependencies  # noqa: E402


def device_census(module) -> Dict[str, Counter]:
    """Where every parameter and buffer actually lives, grouped by top-level
    submodule so a split is obvious."""
    out: Dict[str, Counter] = {}
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        top = name.split(".")[0] or "<root>"
        out.setdefault(top, Counter())[str(tensor.device)] += 1
    return out


def print_census(module, title: str) -> None:
    census = device_census(module)
    print(f"  {title}")
    for top in sorted(census):
        counts = ", ".join(f"{d}:{n}" for d, n in sorted(census[top].items()))
        split = "  <-- SPLIT" if len(census[top]) > 1 else ""
        print(f"    {top:16s} {counts}{split}")


def misplaced(module, expected: str) -> List[Tuple[str, str]]:
    bad = []
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if str(tensor.device).split(":")[0] != expected:
            bad.append((name, str(tensor.device)))
    return bad


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_dependencies()

    from common.config import add_config_args, load_config, resolve_path
    from sa_surrogate.train import get_device
    from stage2.train import build_module

    p = argparse.ArgumentParser(description=__doc__)
    add_config_args(p, default="stage2_reward.yaml")
    p.add_argument("--base-checkpoint", default=None)
    p.add_argument("--steps", type=int, default=2, help="training steps to run")
    p.add_argument("--no-train-steps", action="store_true")
    args = p.parse_args(argv)

    import torch

    cfg = load_config(args.config, args.override)
    base = args.base_checkpoint or cfg.get("base_checkpoint") \
        or cfg.paths.get("diffsbdd_ckpt")
    if not base:
        raise SystemExit("pass --base-checkpoint or set base_checkpoint in the config")

    print("=== environment ===")
    print(f"  torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device 0: {torch.cuda.get_device_name(0)}")
    device = get_device(str(cfg.project.get("device", "auto")))
    expected = device.type
    print(f"  resolved device: {device}")

    print("\n=== module placement ===")
    module = build_module(cfg, Path(resolve_path(base)), verbose=False)
    print_census(module, "as built (EGNN self-moves in its own __init__):")
    module = module.to(device)
    print_census(module, f"after .to({device}):")

    bad = misplaced(module, expected)
    if bad:
        print(f"\n  {len(bad)} tensor(s) NOT on {expected}:")
        for name, dev in bad[:20]:
            print(f"    {name} -> {dev}")
        return 1
    print(f"\n  all parameters and buffers on {expected}")

    # The surrogate is reached through a plain attribute, so check it followed.
    sur = module.sa_surrogate
    print(f"  surrogate.device property  : {sur.surrogate.device}")
    print(f"  surrogate schedule buffers : {sur.schedule.r_t.device}")
    print(f"  frozen reference           : "
          f"{next(module.ref_ddpm.parameters()).device if module.ref_ddpm else 'disabled'}")

    print("\n=== reward path on a real batch ===")
    module.setup("fit")
    ds = module.train_dataset
    n = min(int(cfg.diffsbdd.get("batch_size", 8)), len(ds))
    batch = ds.collate_fn([ds[i] for i in range(n)])
    print(f"  batch built on {batch['lig_coords'].device} (DataLoader output is "
          f"always CPU; the module moves it)")

    terms = module.reward_and_anchor(batch)
    for k, v in terms.items():
        if torch.is_tensor(v):
            print(f"  {k:14s} {float(v):12.6f}   on {v.device}")
    assert terms["reward_term"].device.type == expected, "reward left the device"

    print("\n=== training steps through a Trainer ===")
    if args.no_train_steps:
        print("  skipped (--no-train-steps)")
    else:
        import pytorch_lightning as pl

        from common.compat import trainer_strategy
        acc = "gpu" if expected == "cuda" else expected
        trainer = pl.Trainer(
            max_steps=int(args.steps), accelerator=acc, devices=1,
            strategy=trainer_strategy(1), logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
            num_sanity_val_steps=0, limit_val_batches=0,
        )
        trainer.fit(module)
        print(f"  {args.steps} step(s) completed on {acc}")
        for k in ("sa/grad_ddpm", "sa/grad_aux", "sa/grad_ratio", "sa/pred"):
            if k in module._last_diag:
                print(f"    {k:16s} {module._last_diag[k]:.6g}")

    if torch.cuda.is_available() and expected == "cuda":
        print(f"\n  peak GPU memory: "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
