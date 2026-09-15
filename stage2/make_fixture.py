"""
Synthetic `processed_crossdock` directory, for exercising stage 2 without the
50 GB release.

Writes train/val/test.npz plus size_distribution.npy in exactly the layout
`ProcessedLigandPocketDataset` expects.  Ligands are real 3D-embedded molecules
over DiffSBDD's 10-type vocabulary; **pockets are random points in a shell
around each ligand**.  This exercises shapes, masks, batching and the whole
gradient path -- it is not chemistry, and no number produced from it means
anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from constants import dataset_params  # noqa: E402


def _build_split(n_complexes: int, atom_nf: int, rng: np.random.Generator,
                 seed_offset: int) -> Dict[str, np.ndarray]:
    from sa_surrogate.dataset import _fixture_molecules, mol_to_record

    encoder = dataset_params["crossdock"]["atom_encoder"]
    lig_coords, lig_one_hot, lig_mask = [], [], []
    pock_coords, pock_one_hot, pock_mask = [], [], []
    names, receptors = [], []

    idx = 0
    for path, mol in _fixture_molecules(n_complexes * 3, seed=seed_offset):
        rec = mol_to_record(mol, encoder, min_atoms=4, max_atoms=106)
        if rec is None:
            continue

        x = rec["pos"].astype(np.float32)
        oh = np.zeros((len(rec["types"]), atom_nf), dtype=np.float32)
        oh[np.arange(len(rec["types"])), rec["types"].astype(int)] = 1.0

        # random pocket shell: plausible shapes, no chemistry
        n_pocket = int(rng.integers(48, 96))
        centre = x.mean(0)
        direction = rng.normal(size=(n_pocket, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        radius = rng.uniform(4.0, 9.0, size=(n_pocket, 1))
        p_x = (centre + direction * radius).astype(np.float32)
        p_types = rng.integers(0, 4, size=n_pocket)  # C/N/O/S
        p_oh = np.zeros((n_pocket, atom_nf), dtype=np.float32)
        p_oh[np.arange(n_pocket), p_types] = 1.0

        lig_coords.append(x)
        lig_one_hot.append(oh)
        lig_mask.append(np.full(len(x), idx, dtype=np.int64))
        pock_coords.append(p_x)
        pock_one_hot.append(p_oh)
        pock_mask.append(np.full(n_pocket, idx, dtype=np.int64))
        names.append(f"fixture_{idx:05d}")
        receptors.append(f"fixture_{idx:05d}.pdb")

        idx += 1
        if idx >= n_complexes:
            break

    if idx == 0:
        raise RuntimeError("fixture produced no usable complexes")

    return {
        "names": np.array(names),
        "receptors": np.array(receptors),
        "lig_coords": np.concatenate(lig_coords),
        "lig_one_hot": np.concatenate(lig_one_hot),
        "lig_mask": np.concatenate(lig_mask),
        "pocket_coords": np.concatenate(pock_coords),
        "pocket_one_hot": np.concatenate(pock_one_hot),
        "pocket_mask": np.concatenate(pock_mask),
    }


def make_fixture(outdir: Path, n_train: int = 48, n_val: int = 12,
                 n_test: int = 12, seed: int = 0,
                 hist_shape: Sequence[int] = (107, 1671)) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    atom_nf = len(dataset_params["crossdock"]["atom_encoder"])
    rng = np.random.default_rng(seed)

    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        data = _build_split(n, atom_nf, rng, seed_offset=seed + hash(name) % 1000)
        np.savez(outdir / f"{name}.npz", **data)
        print(f"[fixture] {name}.npz: {len(data['names'])} complexes, "
              f"{len(data['lig_mask'])} ligand atoms, "
              f"{len(data['pocket_mask'])} pocket atoms")

    # Joint (n_ligand, n_pocket) histogram, same shape as the released one.
    hist = np.zeros(tuple(hist_shape), dtype=np.float64)
    hist[8:40, 40:110] = 1.0
    np.save(outdir / "size_distribution.npy", hist)
    print(f"[fixture] size_distribution.npy: {hist.shape}")
    print("[fixture] NOTE pockets are random shells, not real structures -- "
          "this validates plumbing only.")
    return outdir


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default=str(_REPO_ROOT / "stage2/fixture_data"))
    p.add_argument("--n-train", type=int, default=48)
    p.add_argument("--n-val", type=int, default=12)
    p.add_argument("--n-test", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    make_fixture(Path(args.outdir), args.n_train, args.n_val, args.n_test, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
