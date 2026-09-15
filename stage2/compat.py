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
