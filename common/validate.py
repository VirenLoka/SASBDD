"""
Consistency checks between the frozen SA surrogate and the diffusion model.

The surrogate was calibrated against one specific forward process.  If stage 2
feeds it inputs from a diffusion model with a different vocabulary ordering or a
different noise schedule, nothing crashes -- the scores are simply wrong, and the
reward silently optimises the wrong thing.  These checks turn that into a loud
failure at startup.

Deliberately validated rather than shared-by-construction: stage 1 and stage 2
configs each declare their own constants, so a surrogate trained days earlier
with a different config is still caught.
"""

from __future__ import annotations

from typing import Any, Dict, List


class ConsistencyError(RuntimeError):
    pass


def _as_dict(node: Any) -> Dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        return node
    if hasattr(node, "to_dict"):
        return node.to_dict()
    if hasattr(node, "__dict__"):
        return dict(node.__dict__)
    return {}


def check_vocab(surrogate_encoder: Dict[str, int],
                ddpm_encoder: Dict[str, int]) -> List[str]:
    """Same symbols AND the same index for each -- column order is what makes a
    one-hot from the diffusion model mean the same thing to the surrogate."""
    problems: List[str] = []
    s, d = dict(surrogate_encoder), dict(ddpm_encoder)
    if set(s) != set(d):
        problems.append(
            f"atom vocabularies differ: surrogate has {sorted(set(s) - set(d))} "
            f"extra, diffusion model has {sorted(set(d) - set(s))} extra")
    for sym in sorted(set(s) & set(d)):
        if s[sym] != d[sym]:
            problems.append(
                f"atom '{sym}' is column {s[sym]} for the surrogate but "
                f"{d[sym]} for the diffusion model")
    return problems


def check_schedule(surrogate_corruption: Dict[str, Any],
                   ddpm_diffusion: Dict[str, Any]) -> List[str]:
    """The surrogate's sigma conditioning is only meaningful on the schedule it
    was trained against."""
    problems: List[str] = []
    pairs = [
        ("timesteps", "diffusion_steps"),
        ("noise_schedule", "diffusion_noise_schedule"),
        ("noise_precision", "diffusion_noise_precision"),
        ("norm_values", "normalize_factors"),
    ]
    for s_key, d_key in pairs:
        if s_key not in surrogate_corruption or d_key not in ddpm_diffusion:
            continue
        s_val, d_val = surrogate_corruption[s_key], ddpm_diffusion[d_key]
        if isinstance(s_val, (list, tuple)) or isinstance(d_val, (list, tuple)):
            same = [float(x) for x in s_val] == [float(x) for x in d_val]
        elif isinstance(s_val, str) or isinstance(d_val, str):
            same = str(s_val) == str(d_val)
        else:
            same = abs(float(s_val) - float(d_val)) < 1e-12
        if not same:
            problems.append(
                f"noise schedule mismatch: surrogate corruption.{s_key}={s_val!r} "
                f"but diffusion model {d_key}={d_val!r}")
    return problems


def validate_surrogate_against_ddpm(
    surrogate_cfg: Any,
    surrogate_atom_encoder: Dict[str, int],
    ddpm_atom_encoder: Dict[str, int],
    ddpm_diffusion_params: Any,
    strict: bool = True,
) -> List[str]:
    """Raises ConsistencyError on any mismatch when `strict`."""
    problems = check_vocab(surrogate_atom_encoder, ddpm_atom_encoder)
    problems += check_schedule(_as_dict(_as_dict(surrogate_cfg).get("corruption")),
                               _as_dict(ddpm_diffusion_params))
    if problems and strict:
        raise ConsistencyError(
            "the SA surrogate is not compatible with this diffusion model:\n  - "
            + "\n  - ".join(problems)
            + "\n\nThe surrogate would still produce numbers, they would just be "
              "meaningless. Retrain the surrogate against this schedule, or set "
              "surrogate.strict_validation=false to proceed anyway."
        )
    return problems
