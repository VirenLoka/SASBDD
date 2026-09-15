"""
Project-wide YAML configuration.

One loader for every entry point: stage-1 surrogate training, stage-2 reward
finetuning, and DiffSBDD's own train / test / generate / optimize / inpaint
scripts.

Three things it has to do that a bare `yaml.safe_load` does not:

1. **Inheritance.**  `defaults: [base.yaml]` at the top of a config pulls in and
   deep-merges another file, so shared constants (atom vocabulary, diffusion
   schedule, paths) live in exactly one place.  Later files win, and the current
   file wins over everything it includes.

2. **Namespace compatibility.**  `LigandPocketDDPM.__init__` takes
   `egnn_params` / `diffusion_params` / `eval_params` as `argparse.Namespace`
   and reaches into them with attribute access, `in`, and
   `egnn_params.__dict__.get(...)`.  `ConfigNode` subclasses Namespace and
   converts nested dicts recursively, so all three access styles work and the
   existing module is untouched.

3. **Dotted overrides.**  `--override reward.weight=0.05 diffsbdd.batch_size=8`
   from any entry point, parsed as YAML scalars so types survive.

Backwards compatibility: DiffSBDD's original flat configs (configs/*.yml) still
load.  If a config has no `diffsbdd:` block, the top level is treated as one.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

# Keys that LigandPocketDDPM and friends expect to find at the top level of the
# DiffSBDD block; used to detect a legacy flat config.
_LEGACY_MARKERS = ("egnn_params", "diffusion_params", "datadir", "dataset")


class ConfigNode(argparse.Namespace):
    """Nested config with attribute *and* mapping access.

    Subclasses argparse.Namespace so `__dict__`, `in` and attribute access all
    behave the way the existing DiffSBDD code already assumes.
    """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, _wrap(v))

    # -- mapping-ish ---------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, _wrap(value))

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> Iterable[str]:
        return self.__dict__.keys()

    def items(self) -> Iterable:
        return self.__dict__.items()

    def values(self) -> Iterable:
        return self.__dict__.values()

    def setdefault(self, key: str, default: Any) -> Any:
        if not hasattr(self, key):
            setattr(self, key, _wrap(default))
        return getattr(self, key)

    # -- conversion ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, ConfigNode) else v)
                for k, v in self.__dict__.items()}

    def copy(self) -> "ConfigNode":
        return _wrap(copy.deepcopy(self.to_dict()))

    def __repr__(self) -> str:
        return f"ConfigNode({self.to_dict()!r})"


def _wrap(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, dict):
        return ConfigNode(**value)
    if isinstance(value, (list, tuple)):
        return type(value)(_wrap(v) for v in value)
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(v) for v in value)
    return value


# -----------------------------------------------------------------------------
# loading
# -----------------------------------------------------------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive merge; `override` wins.  Neither input is mutated."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path(p: Union[str, Path, None]) -> Optional[Path]:
    """Config paths are relative to the repo root unless absolute."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _load_raw(path: Union[str, Path], _seen: Optional[set] = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        cand = CONFIG_DIR / path
        path = cand if cand.exists() else (REPO_ROOT / path)
    path = path.resolve()

    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular config include involving {path}")
    _seen = _seen | {path}

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")

    parents = raw.pop("defaults", []) or []
    if isinstance(parents, str):
        parents = [parents]

    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            local = path.parent / parent_path
            parent_path = local if local.exists() else parent_path
        merged = deep_merge(merged, _load_raw(parent_path, _seen))
    return deep_merge(merged, raw)


def apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply `a.b.c=value` strings.  Values are parsed as YAML scalars, so
    `3`, `1e-4`, `true`, `null` and `[1, 2]` all keep their types."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            value = yaml.safe_load(raw)
        except Exception:
            value = raw
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return out


def load_config(
    path: Union[str, Path],
    overrides: Optional[Sequence[str]] = None,
    normalize_legacy: bool = True,
) -> ConfigNode:
    """Load a config file, resolve `defaults:`, apply overrides, wrap it."""
    raw = _load_raw(path)
    raw = apply_overrides(raw, overrides or [])
    if normalize_legacy and "diffsbdd" not in raw and \
            any(k in raw for k in _LEGACY_MARKERS):
        # A legacy flat DiffSBDD config: move it under `diffsbdd:` so every
        # consumer sees one shape, but keep the top-level keys too so old code
        # reading them directly still works.
        raw = deep_merge(raw, {"diffsbdd": {k: v for k, v in raw.items()
                                            if k not in ("defaults",)}})
    return _wrap(raw)


def save_config(cfg: Union[ConfigNode, Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_unwrap(cfg), f, sort_keys=False, default_flow_style=False)


# -----------------------------------------------------------------------------
# entry-point helpers
# -----------------------------------------------------------------------------

def add_config_args(parser: argparse.ArgumentParser,
                    default: Optional[str] = None) -> argparse.ArgumentParser:
    """Attach `--config` / `--override` to any entry point."""
    parser.add_argument("--config", type=str, default=default,
                        help="YAML config; relative paths resolve against configs/")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="dotted config overrides, e.g. reward.weight=0.05")
    return parser


def config_from_args(args: argparse.Namespace,
                     section: Optional[str] = None) -> Optional[ConfigNode]:
    """Load the config named by `--config`, optionally returning one section."""
    if getattr(args, "config", None) is None:
        return None
    cfg = load_config(args.config, getattr(args, "override", []))
    return cfg if section is None else cfg.get(section)


def apply_config_defaults(args: argparse.Namespace, cfg: Optional[ConfigNode],
                          keys: Optional[Iterable[str]] = None,
                          parser: Optional[argparse.ArgumentParser] = None
                          ) -> argparse.Namespace:
    """Fill argparse values from a config section, without clobbering anything
    the user actually typed.

    This is what keeps the migration additive: every existing command line keeps
    working unchanged, and the config only supplies values the user left alone.

    Precedence, highest first:
      1. flags the user typed on the command line
      2. positional arguments the user supplied
      3. the config section
      4. the parser's own defaults
    """
    if cfg is None:
        return args

    explicit: set = set()
    positional: set = set()
    if parser is not None:
        import sys as _sys
        argv = set(_sys.argv[1:])
        for action in parser._actions:
            if not action.option_strings:
                positional.add(action.dest)
            elif any(opt in argv for opt in action.option_strings):
                explicit.add(action.dest)

    for key in (keys if keys is not None else list(cfg.keys())):
        if key in explicit:
            continue                       # user typed this flag
        value = cfg.get(key, None)
        if value is None:
            continue
        if key in positional and getattr(args, key, None) is not None:
            continue                       # user supplied this positional
        setattr(args, key, _unwrap(value))
    return args


def merge_args_and_yaml(args: argparse.Namespace,
                        config_dict: Dict[str, Any]) -> argparse.Namespace:
    """Backwards-compatible shim for DiffSBDD's original train.py helper."""
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        arg_dict[key] = _wrap(value) if isinstance(value, dict) else value
    return args
