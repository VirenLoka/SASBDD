"""
Optional-dependency shims, installed only when the real package is absent.

DiffSBDD imports `torch_scatter` and `wandb` at module scope, so both must exist
before `lightning_modules` can even be imported.  Neither is actually needed for
stage-2 reward finetuning:

  torch_scatter  is used only for scatter_add / scatter_mean, both of which are
                 a few lines of native torch.  The real package requires
                 compilation against a specific torch build, so having a
                 fallback removes a brittle install step.

  wandb          is only reached through the chain-visualisation helpers.  With
                 `logging.logger` set to anything but `wandb`, nothing calls it.

`ensure_dependencies()` must run BEFORE importing lightning_modules.  Call it at
the top of any stage-2 entry point.  If the real package is installed it is left
strictly alone.
"""

from __future__ import annotations

import sys
import types
import warnings
from typing import Dict

import torch


def _expand_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    if index.dim() == 1 and src.dim() > 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(shape).expand_as(src)
    return index.to(torch.int64)


def _scatter_add(src, index, dim=0, out=None, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() else 0
    if out is None:
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, _expand_index(index, src, dim), src)


def _scatter_mean(src, index, dim=0, out=None, dim_size=None):
    summed = _scatter_add(src, index, dim, None, dim_size)
    counts = _scatter_add(torch.ones_like(src), index, dim, None, summed.shape[dim])
    result = summed / counts.clamp(min=1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _make_torch_scatter() -> types.ModuleType:
    mod = types.ModuleType("torch_scatter")
    mod.scatter_add = _scatter_add
    mod.scatter_mean = _scatter_mean
    mod.scatter_sum = _scatter_add
    mod.__doc__ = "Minimal native-torch stand-in installed by stage2.compat."
    mod.__SHIM__ = True
    return mod


class _WandbStub:
    """No-op stand-ins for the handful of wandb symbols DiffSBDD touches."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        return _WandbStub()


def _make_wandb() -> types.ModuleType:
    mod = types.ModuleType("wandb")

    def _log(*a, **k):
        return None

    mod.log = _log
    mod.Image = _WandbStub
    mod.Video = _WandbStub
    mod.Molecule = _WandbStub
    mod.init = _log
    mod.finish = _log
    mod.run = None
    mod.__doc__ = "No-op stand-in installed by stage2.compat."
    mod.__SHIM__ = True
    return mod


def ensure_dependencies(verbose: bool = True) -> Dict[str, str]:
    """Install shims for missing optional deps.  Returns {name: 'real'|'shim'}."""
    status: Dict[str, str] = {}

    for name, factory in (("torch_scatter", _make_torch_scatter),
                          ("wandb", _make_wandb)):
        if name in sys.modules:
            status[name] = "shim" if getattr(sys.modules[name], "__SHIM__", False) else "real"
            continue
        try:
            __import__(name)
            status[name] = "real"
        except ImportError:
            sys.modules[name] = factory()
            status[name] = "shim"

    if verbose:
        shimmed = [k for k, v in status.items() if v == "shim"]
        if shimmed:
            print(f"[compat] using built-in stand-ins for: {', '.join(shimmed)} "
                  f"(install the real packages for full functionality)")
    if status.get("wandb") == "shim":
        warnings.filterwarnings("ignore", message=".*wandb.*")
    return status


def verify_shim_against_real(atol: float = 1e-6) -> bool:
    """If the real torch_scatter is present, check the shim agrees with it."""
    try:
        import torch_scatter as real
    except ImportError:
        return True
    if getattr(real, "__SHIM__", False):
        return True
    src = torch.randn(64, 7)
    idx = torch.randint(0, 9, (64,))
    ok = torch.allclose(real.scatter_add(src, idx, dim=0, dim_size=9),
                        _scatter_add(src, idx, dim=0, dim_size=9), atol=atol)
    ok &= torch.allclose(real.scatter_mean(src, idx, dim=0, dim_size=9),
                         _scatter_mean(src, idx, dim=0, dim_size=9), atol=atol)
    return bool(ok)


# -----------------------------------------------------------------------------
# Library version compatibility
#
# The repo was written against pytorch-lightning 1.8 (what environment.yaml
# pins).  Lightning 2.0 removed several hooks and changed two signatures, and
# torch 2.6 flipped a `torch.load` default.  These helpers keep one codebase
# working on both rather than forcing an environment downgrade.
# -----------------------------------------------------------------------------

def _version_tuple(mod_name: str):
    try:
        mod = __import__(mod_name)
        raw = getattr(mod, "__version__", "0")
    except Exception:
        return None
    parts = []
    for chunk in str(raw).split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def pl_version():
    return _version_tuple("pytorch_lightning")


def pl_is_v2() -> bool:
    v = pl_version()
    return bool(v and v[0] >= 2)


def trainer_strategy(devices):
    """`strategy=None` is valid on PL 1.x and rejected on 2.x, which wants
    "auto".  Multi-device is "ddp" on both."""
    n = devices if isinstance(devices, int) else 1
    if n > 1:
        return "ddp"
    return "auto" if pl_is_v2() else None


def torch_load(path, **kwargs):
    """`torch.load` with weights_only=False where the argument exists.

    torch 2.6 flipped the default to True, which refuses to unpickle Lightning
    checkpoints and the CrossDocked split files.
    """
    import inspect

    import torch as _torch

    try:
        if "weights_only" in inspect.signature(_torch.load).parameters:
            kwargs.setdefault("weights_only", False)
    except (TypeError, ValueError):
        pass
    return _torch.load(path, **kwargs)


def three_to_one(resname: str) -> str:
    """Three-letter residue code to one letter.

    `Bio.PDB.Polypeptide.three_to_one` was removed in biopython 1.80 in favour
    of the `protein_letters_3to1` mapping.
    """
    try:
        from Bio.PDB.Polypeptide import three_to_one as _legacy
        return _legacy(resname)
    except (ImportError, AttributeError):
        pass
    from Bio.PDB.Polypeptide import protein_letters_3to1
    key = resname.strip()
    for candidate in (key.upper(), key.capitalize(), key):
        if candidate in protein_letters_3to1:
            return protein_letters_3to1[candidate]
    raise KeyError(resname)


KNOWN_ISSUES = [
    ("pytorch_lightning", (2, 0, 0), "*_epoch_end hooks removed; "
     "configure_gradient_clipping dropped optimizer_idx; strategy=None invalid"),
    ("torch", (2, 6, 0), "torch.load defaults to weights_only=True"),
    ("Bio", (1, 80, 0), "Bio.PDB.Polypeptide.three_to_one removed"),
    ("numpy", (2, 0, 0), "NumPy 2 ABI; rebuild compiled extensions if imports fail"),
]


def report() -> int:
    """Print the environment and flag known-incompatible versions."""
    import importlib

    print("environment")
    names = ["torch", "pytorch_lightning", "torch_geometric", "torch_scatter",
             "rdkit", "Bio", "numpy", "lmdb", "openbabel", "wandb", "yaml"]
    for name in names:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "present")
            note = " (shim)" if getattr(mod, "__SHIM__", False) else ""
            print(f"  {name:20s} {ver}{note}")
        except Exception:
            print(f"  {name:20s} MISSING")

    print("\nhandled version differences")
    any_flagged = False
    for name, threshold, what in KNOWN_ISSUES:
        v = _version_tuple(name)
        if v is None:
            continue
        if v >= threshold:
            any_flagged = True
            print(f"  {name} {'.'.join(map(str, v))} >= "
                  f"{'.'.join(map(str, threshold))}: {what}")
    if not any_flagged:
        print("  none apply to these versions")
    print(f"\n  pl_is_v2()          = {pl_is_v2()}")
    print(f"  trainer_strategy(1) = {trainer_strategy(1)!r}")
    print(f"  trainer_strategy(4) = {trainer_strategy(4)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
