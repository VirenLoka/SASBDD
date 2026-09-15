"""
Compute baseline MAE benchmarks for the SA surrogate.

Prints simple, dumb baselines that any trained model should beat:

  1. Mean predictor       -- always predicts the training-set mean SA
  2. Median predictor     -- always predicts the training-set median SA
  3. Per-size-bin mean    -- mean SA for molecules of similar size (±2 atoms)
  4. Random predictor     -- uniform random value in [min_SA, max_SA]

These baselines are evaluated in raw SA space (the same units the training
loop reports), at every noise bucket, so you can directly compare them to the
per-t MAE columns in the training log.

Usage:
    python sa_surrogate/benchmark_baselines.py
    python sa_surrogate/benchmark_baselines.py --config configs/stage1_surrogate.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sa_surrogate.dataset import SALMDBDataset, collate_fn, load_config, resolve_path
from sa_surrogate.diffusion_bridge import NoiseSchedule


# -----------------------------------------------------------------------------
# Baseline predictors
# -----------------------------------------------------------------------------

class MeanPredictor:
    """Always predicts the training-set mean."""
    name = "Mean predictor"

    def __init__(self, targets: np.ndarray, sizes: np.ndarray):
        self.mean = float(targets.mean())

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.mean)


class MedianPredictor:
    """Always predicts the training-set median."""
    name = "Median predictor"

    def __init__(self, targets: np.ndarray, sizes: np.ndarray):
        self.median = float(np.median(targets))

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.median)


class PerSizeMeanPredictor:
    """Predicts the mean SA for molecules of similar atom count (±2 atoms)."""
    name = "Per-size-bin mean"

    def __init__(self, targets: np.ndarray, sizes: np.ndarray, window: int = 2):
        self.global_mean = float(targets.mean())
        self.bin_means: Dict[int, float] = {}
        for s in np.unique(sizes):
            mask = np.abs(sizes - s) <= window
            self.bin_means[int(s)] = float(targets[mask].mean()) if mask.any() else self.global_mean

    def predict_for_sizes(self, sizes: np.ndarray) -> np.ndarray:
        return np.array([self.bin_means.get(int(s), self.global_mean) for s in sizes])


class RandomPredictor:
    """Predicts uniform random values in [min_SA, max_SA]."""
    name = "Random predictor"

    def __init__(self, targets: np.ndarray, sizes: np.ndarray, seed: int = 42):
        self.lo = float(targets.min())
        self.hi = float(targets.max())
        self.rng = np.random.default_rng(seed)

    def predict(self, n: int) -> np.ndarray:
        return self.rng.uniform(self.lo, self.hi, size=n)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_baselines(
    train_ds: SALMDBDataset,
    val_ds: SALMDBDataset,
    t_buckets: List[int],
    schedule: NoiseSchedule,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all baselines on the validation set.

    Returns {baseline_name: {"overall_mae": ..., "t=0": ..., ...}}.
    """
    # Collect all training targets (denormalized to raw SA)
    print("Collecting training targets...")
    train_targets = []
    train_sizes = []
    for i in tqdm(range(len(train_ds)), desc="Loading train data"):
        item = train_ds[i]
        y_raw = train_ds.denormalize(item["y"]).item()
        train_targets.append(y_raw)
        train_sizes.append(item["num_nodes"])
    train_targets = np.array(train_targets)
    train_sizes = np.array(train_sizes)

    # Collect all val targets (denormalized to raw SA)
    print("Collecting validation targets...")
    val_targets = []
    val_sizes = []
    for i in tqdm(range(len(val_ds)), desc="Loading val data"):
        item = val_ds[i]
        y_raw = val_ds.denormalize(item["y"]).item()
        val_targets.append(y_raw)
        val_sizes.append(item["num_nodes"])
    val_targets = np.array(val_targets)
    val_sizes = np.array(val_sizes)

    # Build baselines
    baselines = [
        MeanPredictor(train_targets, train_sizes),
        MedianPredictor(train_targets, train_sizes),
        PerSizeMeanPredictor(train_targets, train_sizes),
        RandomPredictor(train_targets, train_sizes),
    ]

    results: Dict[str, Dict[str, float]] = {}
    n_val = len(val_targets)

    for bl in baselines:
        metrics: Dict[str, float] = {}

        if isinstance(bl, PerSizeMeanPredictor):
            preds = bl.predict_for_sizes(val_sizes)
        else:
            preds = bl.predict(n_val)

        # Overall MAE
        overall_mae = float(np.abs(preds - val_targets).mean())
        overall_rmse = float(np.sqrt(np.mean((preds - val_targets) ** 2)))
        metrics["overall_mae"] = overall_mae
        metrics["overall_rmse"] = overall_rmse

        # The SA surrogate evaluates at specific noise levels (t_buckets).
        # Baselines are noise-agnostic: their predictions don't change with t.
        # So the MAE is the same at every t -- but we report it per-bucket
        # for direct comparison with the training log.
        for t in t_buckets:
            r = float(schedule.r(torch.tensor(int(t))))
            metrics[f"t={t}"] = overall_mae
            metrics[f"t={t}_r"] = r

        results[bl.name] = metrics

    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Compute baseline MAE benchmarks for the SA surrogate")
    p.add_argument("--config",
                   default=str(_REPO_ROOT / "configs/stage1_surrogate.yaml"))
    args = p.parse_args(argv)

    cfg = load_config(args.config)

    common = dict(
        lmdb_path=cfg["data"]["lmdb_path"],
        label_builder=cfg["data"]["label_builder"],
        target_mode=cfg["target"]["mode"],
        normalize=bool(cfg["target"]["normalize"]),
    )
    train_ds = SALMDBDataset(split="train", **common)
    val_ds = SALMDBDataset(split="val", **common)

    print(f"[benchmark] train={len(train_ds)} val={len(val_ds)} "
          f"label={train_ds.label_builder} target={cfg['target']['mode']}")
    print(f"[benchmark] target mean={train_ds.target_mean:.3f} "
          f"std={train_ds.target_std:.3f}")

    ccfg = cfg["corruption"]
    schedule = NoiseSchedule(ccfg["noise_schedule"], ccfg["timesteps"],
                             ccfg["noise_precision"])
    t_buckets = [t for t in cfg["train"]["eval_t_buckets"]
                 if t <= ccfg["t_max"]]

    results = evaluate_baselines(train_ds, val_ds, t_buckets, schedule)

    # ---- Print results -------------------------------------------------------
    print("\n" + "=" * 80)
    print("BASELINE BENCHMARKS  (validation set, raw SA units)")
    print("=" * 80)

    for name, metrics in results.items():
        print(f"\n  {name}")
        print(f"    Overall MAE  = {metrics['overall_mae']:.4f}")
        print(f"    Overall RMSE = {metrics['overall_rmse']:.4f}")
        for t in t_buckets:
            r = metrics[f"t={t}_r"]
            mae = metrics[f"t={t}"]
            print(f"      t={t:>3d} (r={r:6.3f})  MAE={mae:.4f}")

    # ---- Summary table -------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"{'Baseline':<25s} {'Overall MAE':>12s} {'Overall RMSE':>13s} "
          f"{'Low-noise MAE':>14s}")
    print("-" * 80)

    for name, metrics in results.items():
        # Low-noise MAE: average of first 3 t-buckets (same selection as train.py)
        sel_keys = [f"t={t}" for t in t_buckets[:3]]
        low_noise_mae = float(np.mean([metrics[k] for k in sel_keys]))
        print(f"{name:<25s} {metrics['overall_mae']:>12.4f} "
              f"{metrics['overall_rmse']:>13.4f} {low_noise_mae:>14.4f}")

    print("-" * 80)
    print("\nYour trained model should beat ALL of these baselines.")
    print("Compare the 'low-noise val MAE' from training against the")
    print("'Low-noise MAE' column above.\n")

    # Show comparison hint based on training log
    mean_mae = results["Mean predictor"]["overall_mae"]
    print(f"  If your model's low-noise val MAE < {mean_mae:.4f},")
    print(f"  it has learned something beyond the trivial mean predictor.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
