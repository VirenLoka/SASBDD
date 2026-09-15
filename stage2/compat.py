"""Backwards-compatible re-export.

The shims moved to `common/compat.py` when the inference entry points started
needing them too -- they import `lightning_modules`, which imports `wandb` at
module scope.
"""

from common.compat import (ensure_dependencies, verify_shim_against_real)  # noqa: F401
