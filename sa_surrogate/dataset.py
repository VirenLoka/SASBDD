"""
CrossDocked ligand -> SA-labelled LMDB, and the Dataset that reads it back.

Pockets are ignored entirely: the surrogate scores ligands, so only the ligand
SDFs of the CrossDocked2020 pocket10 release are touched.  That keeps the build
cheap and means the 50 GB of pocket PDBs never have to be read.

Labels
------
Three SA labels are computed and stored per molecule:

  sdf        SA of the original RDKit molecule, using its real bonds.
  openbabel  SA(build_molecule(coords, types, use_openbabel=True)) -- the repo's
             default reconstruction, and therefore the reward actually measured
             at stage 2.
  edm        SA(make_mol_edm(...)) -- distance-cutoff bond inference, no OpenBabel.

Measured on the repo's own example ligands, `openbabel` reproduces the true SA
exactly (5.724 and 2.298) while `edm` is off by 0.93 and 2.54.  `openbabel` is
the default target: it agrees with the real chemistry *and* with the downstream
reward definition.  All three are stored so the choice can be revisited without
rebuilding.

Deduplication
-------------
CrossDocked docks the same ligand into many related pockets, so the ~100k
(pocket, ligand) training pairs collapse to far fewer unique molecules.  Without
deduplication the same molecule appears in train and validation and the reported
R^2 is fiction.  Molecules are deduplicated by canonical SMILES, and any SMILES
occurring in both the official train and test splits is assigned to test only.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

try:  # OpenBabel narrates bond perception to stderr; quiet it (cf. optimize.py:12)
    from openbabel import openbabel as _ob
    _ob.obErrorLog.StopLogging()
except Exception:
    pass

from constants import dataset_params  # noqa: E402
from analysis.SA_Score.sascorer import calculateScore  # noqa: E402

META_KEY = b"__meta__"
LABEL_BUILDERS = ("sdf", "openbabel", "edm")


# -----------------------------------------------------------------------------
# config helpers
# -----------------------------------------------------------------------------

def load_config(path: str | Path, overrides=None):
    """Delegates to the project-wide loader (common/config.py) so stage 1 and
    stage 2 share one config system.  The returned ConfigNode supports both
    dict-style and attribute access, so existing cfg["a"]["b"] usage is
    unchanged."""
    from common.config import load_config as _load
    return _load(path, overrides or [])


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repo root unless already absolute."""
    p = Path(p)
    return p if p.is_absolute() else (_REPO_ROOT / p)


# -----------------------------------------------------------------------------
# labelling
# -----------------------------------------------------------------------------

def _safe_sa(mol: Optional[Chem.Mol]) -> Optional[float]:
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return float(calculateScore(mol))
    except Exception:
        return None


def compute_labels(
    pos: np.ndarray,
    types: np.ndarray,
    mol_noH: Chem.Mol,
    dataset_info: dict,
    builders: Sequence[str] = LABEL_BUILDERS,
) -> Dict[str, Optional[float]]:
    """SA under each reconstruction route.  Missing/failed routes map to None."""
    out: Dict[str, Optional[float]] = {}

    if "sdf" in builders:
        out["sdf"] = _safe_sa(Chem.Mol(mol_noH))

    if "openbabel" in builders or "edm" in builders:
        from analysis.molecule_builder import build_molecule  # lazy: pulls openbabel

        pos_t = torch.from_numpy(np.asarray(pos, dtype=np.float32))
        types_t = torch.from_numpy(np.asarray(types, dtype=np.int64))

        if "openbabel" in builders:
            try:
                out["openbabel"] = _safe_sa(
                    build_molecule(pos_t, types_t, dataset_info, use_openbabel=True)
                )
            except Exception:
                out["openbabel"] = None

        if "edm" in builders:
            try:
                out["edm"] = _safe_sa(
                    build_molecule(pos_t, types_t, dataset_info,
                                   add_coords=True, use_openbabel=False)
                )
            except Exception:
                out["edm"] = None

    return out


def to_target(sa: float, mode: str) -> float:
    """raw_sa keeps RDKit's 1..10 scale (lower = easier to make).
    pocket2mol applies (10 - sa) / 9 so higher = better, matching
    MoleculeProperties.calculate_sa (analysis/metrics.py:141)."""
    if mode == "raw_sa":
        return float(sa)
    if mode == "pocket2mol":
        return float((10.0 - sa) / 9.0)
    raise ValueError(f"unknown target mode: {mode!r}")


# -----------------------------------------------------------------------------
# molecule -> record
# -----------------------------------------------------------------------------

def mol_to_record(
    mol: Chem.Mol,
    atom_encoder: Dict[str, int],
    min_atoms: int,
    max_atoms: int,
) -> Optional[dict]:
    """Heavy-atom coords + type indices, or None if the molecule is unusable."""
    try:
        mol = Chem.Mol(mol)
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None

    if mol.GetNumConformers() == 0:
        return None

    n = mol.GetNumAtoms()
    if n < min_atoms or n > max_atoms:
        return None

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    if any(s not in atom_encoder for s in symbols):
        return None  # element outside DiffSBDD's ligand vocabulary

    pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    if not np.isfinite(pos).all():
        return None

    types = np.asarray([atom_encoder[s] for s in symbols], dtype=np.uint8)

    # Match analysis/metrics.py:rdmol_to_smiles so dedup keys line up with the
    # repo's own notion of molecular identity.  RemoveStereochemistry mutates in
    # place and returns None, hence the copy-then-mutate.
    try:
        flat = Chem.Mol(mol)
        Chem.RemoveStereochemistry(flat)
        smiles = Chem.MolToSmiles(flat)
    except Exception:
        smiles = Chem.MolToSmiles(mol)

    return {"pos": pos, "types": types, "n": n, "smiles": smiles, "mol": mol}


# -----------------------------------------------------------------------------
# CrossDocked iteration
# -----------------------------------------------------------------------------

def load_crossdocked_split(crossdocked_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    split_path = crossdocked_dir / "split_by_name.pt"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found. `data.crossdocked_dir` must point at the "
            f"directory containing crossdocked_pocket10/ and split_by_name.pt"
        )
    split = torch.load(split_path, map_location="cpu", weights_only=False)
    return {k: list(v) for k, v in split.items()}


def iter_split_ligands(
    crossdocked_dir: Path, pairs: Sequence[Tuple[str, str]]
) -> Iterator[Tuple[str, Chem.Mol]]:
    """Yield (relative ligand path, molecule) for each *distinct* ligand file."""
    structure_dir = crossdocked_dir / "crossdocked_pocket10"
    seen_files = set()
    for _pocket_fn, ligand_fn in pairs:
        if ligand_fn in seen_files:
            continue
        seen_files.add(ligand_fn)
        sdf_path = structure_dir / ligand_fn
        if not sdf_path.exists():
            continue
        try:
            mol = Chem.SDMolSupplier(str(sdf_path), sanitize=False)[0]
        except Exception:
            continue
        if mol is not None:
            yield ligand_fn, mol


# -----------------------------------------------------------------------------
# fixture (lets train.py run end-to-end without the CrossDocked release)
# -----------------------------------------------------------------------------

_FIXTURE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Clc1ccccc1C(=O)Nc1ccc(cc1)S(=O)(=O)N", "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "CC1=C(C(=O)Nc2ccccc2)S(=O)(=O)c2ccccc21", "Brc1ccc(cc1)C(=O)Nc1ncccn1",
    "CCOC(=O)c1ccc(cc1)NC(=O)c1ccccc1Cl", "Fc1ccc(cc1)C1CCN(CC1)CCc1ccccc1",
    "OC(=O)c1ccc(cc1)P(=O)(O)O", "Ic1ccc(cc1)C(=O)NCC1CCNCC1",
    "CSc1ccc(cc1)C(=O)Nc1ccc(F)cc1", "CC(C)(C)OC(=O)N1CCC(CC1)Nc1ncnc2[nH]ccc12",
    "O=C(Nc1ccccc1)c1cccc(c1)S(=O)(=O)N1CCOCC1", "CN(C)CCOc1ccc(cc1)C(c1ccccc1)c1ccccc1",
    "OCC1OC(O)C(O)C(O)C1O", "c1ccc2[nH]ccc2c1", "CC(N)Cc1ccccc1",
    "O=S(=O)(N)c1ccc(cc1)NC(=O)C", "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C", "CC(=O)Nc1ccc(O)cc1", "OC(=O)Cc1ccccc1",
    "CC1=CC(=O)c2ccccc2C1=O", "NC(Cc1ccc(O)cc1)C(=O)O", "CC(C)NCC(O)COc1ccccc1",
    "Clc1cccc(Cl)c1C(=O)N1CCNCC1", "COc1cccc(c1)C(=O)Nc1ccc(Br)cc1",
    "CC(C)c1nc(cs1)CN(C)C", "OC1CCN(CC1)C(=O)c1ccc(F)cc1",
]


def _fixture_molecules(n_target: int, seed: int = 0) -> Iterator[Tuple[str, Chem.Mol]]:
    """3D-embedded real molecules spanning the full 10-type vocabulary."""
    from rdkit.Chem import AllChem

    idx = 0
    rng = np.random.default_rng(seed)
    while idx < n_target:
        smi = _FIXTURE_SMILES[idx % len(_FIXTURE_SMILES)]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            idx += 1
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(rng.integers(1, 2**31 - 1))
        if AllChem.EmbedMolecule(mol, params) != 0:
            idx += 1
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        yield f"fixture/{idx:06d}.sdf", mol
        idx += 1


# -----------------------------------------------------------------------------
# LMDB build
# -----------------------------------------------------------------------------

def build_lmdb(cfg: dict, fixture: int = 0, overwrite: bool = False) -> Path:
    import lmdb
    from tqdm import tqdm

    dcfg = cfg["data"]
    atom_encoder = dict(cfg["vocab"]["atom_encoder"])
    dataset_info = dataset_params[cfg["vocab"]["name"]]
    if dict(dataset_info["atom_encoder"]) != atom_encoder:
        raise ValueError(
            "config vocab.atom_encoder does not match DiffSBDD's "
            f"dataset_params['{cfg['vocab']['name']}']['atom_encoder']; the "
            "surrogate's one-hot columns must line up with the diffusion model's"
        )

    lmdb_path = resolve_path(dcfg["lmdb_path"])
    if lmdb_path.exists() and not overwrite:
        raise FileExistsError(f"{lmdb_path} exists; pass --overwrite to rebuild")
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    builders = LABEL_BUILDERS if dcfg.get("store_all_labels", True) \
        else (dcfg["label_builder"],)
    min_atoms, max_atoms = int(dcfg["min_atoms"]), int(dcfg["max_atoms"])
    max_molecules = dcfg.get("max_molecules")

    # ---- gather source molecules, tagged with their official split ----------
    sources: Any = []  # (official_split, path, mol); a generator for the LMDB source
    source_kind = str(dcfg.get("source", "crossdocked_raw")).lower()

    if fixture == 0 and source_kind == "targetdiff_lmdb":
        # The TargetDiff release stores each ligand's real bond graph, so the
        # 'sdf' label is exact chemistry rather than reconstructed geometry --
        # and the surrogate trains on precisely the ligand distribution stage 2
        # will show it.
        from stage2.crossdocked_lmdb import iter_lmdb_ligands, make_splits

        src_lmdb = resolve_path(dcfg.get("source_lmdb"))
        src_split = dcfg.get("source_split")
        if src_lmdb is None:
            raise ValueError("data.source is 'targetdiff_lmdb' but "
                             "data.source_lmdb is not set")
        splits = (make_splits(resolve_path(src_split), val_size=0)
                  if src_split else {"train": None, "test": []})

        def _stream():
            # Streamed rather than materialised: the train split is ~100k
            # records, and holding that many RDKit molecules before dedup is
            # gigabytes for no reason.
            for split_name in ("train", "test"):
                idx = splits.get(split_name)
                if idx is not None and len(idx) == 0:
                    continue
                print(f"[build] reading {split_name} ligands from {src_lmdb.name}")
                for path, mol in iter_lmdb_ligands(
                        src_lmdb, idx, dcfg.get("field_map") or None,
                        dcfg.get("max_molecules")):
                    yield split_name, path, mol

        sources = _stream()
    elif fixture > 0:
        print(f"[build] fixture mode: {fixture} synthetic molecules")
        for i, (path, mol) in enumerate(_fixture_molecules(fixture)):
            sources.append(("test" if i % 10 == 0 else "train", path, mol))
    else:
        cd_dir = dcfg.get("crossdocked_dir")
        if not cd_dir:
            raise ValueError(
                "data.crossdocked_dir is null. Either set it to the CrossDocked2020 "
                "pocket10 directory, or build a smoke-test dataset with --fixture N"
            )
        cd_dir = resolve_path(cd_dir)
        split = load_crossdocked_split(cd_dir)
        for split_name in ("train", "test"):
            if split_name not in split:
                continue
            pairs = split[split_name]
            print(f"[build] {split_name}: {len(pairs)} pairs")
            for path, mol in tqdm(iter_split_ligands(cd_dir, pairs),
                                  desc=f"read {split_name}"):
                sources.append((split_name, path, mol))

    # ---- deduplicate by canonical SMILES; test wins over train --------------
    dedup = bool(dcfg.get("dedup_by_smiles", True))
    if fixture > 0 and dedup:
        # The fixture cycles a fixed SMILES list with fresh conformers, so
        # SMILES-dedup would collapse it to one row per SMILES.  Keep each
        # conformer instead; the smoke test needs rows, not unique chemistry.
        # NOTE this means fixture train/val/test share molecules -- it is a
        # smoke test for the plumbing, never a measure of generalisation.
        print("[build] fixture mode: keeping each conformer (dedup disabled)")
        dedup = False
    by_smiles: Dict[str, dict] = {}
    n_seen = n_rejected = 0
    reassigned_to_test = 0

    for official_split, path, mol in tqdm(sources, desc="label"):
        n_seen += 1
        rec = mol_to_record(mol, atom_encoder, min_atoms, max_atoms)
        if rec is None:
            n_rejected += 1
            continue

        key = rec["smiles"] if dedup else f"{path}"
        if key in by_smiles:
            # A molecule in both official splits belongs to test, so the test
            # set stays honest.
            if official_split == "test" and by_smiles[key]["split"] == "train":
                by_smiles[key]["split"] = "test"
                reassigned_to_test += 1
            continue

        labels = compute_labels(rec["pos"], rec["types"], rec["mol"],
                                dataset_info, builders)
        if labels.get(dcfg["label_builder"]) is None:
            n_rejected += 1
            continue

        by_smiles[key] = {
            "pos": rec["pos"], "types": rec["types"], "n": rec["n"],
            "smiles": rec["smiles"], "sa": labels,
            "source": path, "split": official_split,
        }
        if max_molecules and len(by_smiles) >= int(max_molecules):
            break

    records = list(by_smiles.values())
    print(f"[build] {n_seen} molecules read, {n_rejected} rejected, "
          f"{len(records)} unique kept, {reassigned_to_test} moved train->test")
    if not records:
        raise RuntimeError("no usable molecules; check crossdocked_dir and filters")

    # ---- train / val / test -------------------------------------------------
    rng = np.random.default_rng(int(dcfg.get("split_seed", 0)))
    train_idx = [i for i, r in enumerate(records) if r["split"] == "train"]
    test_idx = [i for i, r in enumerate(records) if r["split"] == "test"]
    rng.shuffle(train_idx)
    n_val = int(round(float(dcfg.get("val_fraction", 0.05)) * len(train_idx)))
    val_idx, train_idx = train_idx[:n_val], train_idx[n_val:]
    splits = {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}
    print("[build] split sizes: " +
          ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    # ---- label statistics, computed on train only ---------------------------
    label_stats: Dict[str, dict] = {}
    for b in builders:
        vals = [records[i]["sa"][b] for i in splits["train"]
                if records[i]["sa"].get(b) is not None]
        if vals:
            a = np.asarray(vals, dtype=np.float64)
            label_stats[b] = {"mean": float(a.mean()), "std": float(a.std() + 1e-8),
                              "min": float(a.min()), "max": float(a.max()),
                              "count": int(a.size)}
    for b, s in label_stats.items():
        print(f"[build] SA[{b}]: mean={s['mean']:.3f} std={s['std']:.3f} "
              f"range=[{s['min']:.2f}, {s['max']:.2f}] n={s['count']}")

    # ---- write --------------------------------------------------------------
    map_size = int(float(dcfg.get("map_size_gb", 16)) * (1024 ** 3))
    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=True,
                    readonly=False, meminit=False, map_async=True)
    write_batch = int(dcfg.get("write_batch", 2048))
    try:
        txn = env.begin(write=True)
        for i, rec in enumerate(tqdm(records, desc="write")):
            payload = {k: rec[k] for k in ("pos", "types", "n", "smiles", "sa", "source")}
            txn.put(f"{i:09d}".encode(), pickle.dumps(payload, protocol=4))
            if (i + 1) % write_batch == 0:
                txn.commit()
                txn = env.begin(write=True)
        meta = {
            "num_records": len(records),
            "vocab": atom_encoder,
            "vocab_name": cfg["vocab"]["name"],
            "num_classes": len(atom_encoder),
            "splits": splits,
            "label_stats": label_stats,
            "label_builders": list(builders),
            "default_label_builder": dcfg["label_builder"],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fixture": int(fixture),
        }
        txn.put(META_KEY, pickle.dumps(meta, protocol=4))
        txn.commit()
    finally:
        env.sync()
        env.close()

    print(f"[build] wrote {len(records)} records to {lmdb_path}")
    return lmdb_path


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class SALMDBDataset(Dataset):
    """Reads the SA LMDB.  Emits clean molecules; corruption happens in the
    training loop so the same molecule is seen at many noise levels."""

    def __init__(
        self,
        lmdb_path: str | Path,
        split: str = "train",
        label_builder: Optional[str] = None,
        target_mode: str = "raw_sa",
        normalize: bool = True,
    ):
        super().__init__()
        self.lmdb_path = str(resolve_path(lmdb_path))
        self.split = split
        self.target_mode = target_mode
        self.normalize = normalize
        self._env = None  # opened lazily, per worker process

        meta = self.read_meta(self.lmdb_path)
        self.meta = meta
        self.num_classes = int(meta["num_classes"])
        self.atom_encoder = meta["vocab"]
        self.label_builder = label_builder or meta["default_label_builder"]
        if self.label_builder not in meta["label_builders"]:
            raise ValueError(
                f"label_builder {self.label_builder!r} not stored in this LMDB "
                f"(available: {meta['label_builders']}); rebuild with store_all_labels"
            )
        if split not in meta["splits"]:
            raise ValueError(f"unknown split {split!r}; have {list(meta['splits'])}")
        self.indices = list(meta["splits"][split])

        # Target statistics are always the *train* statistics, transformed the
        # same way the targets are, so val/test are scored on the same ruler.
        stats = meta["label_stats"][self.label_builder]
        lo = to_target(stats["mean"] - stats["std"], target_mode)
        hi = to_target(stats["mean"] + stats["std"], target_mode)
        self.target_mean = float(to_target(stats["mean"], target_mode))
        self.target_std = float(abs(hi - lo) / 2.0) or 1.0

    # -- lmdb plumbing ------------------------------------------------------
    @staticmethod
    def read_meta(lmdb_path: str | Path) -> dict:
        import lmdb
        env = lmdb.open(str(resolve_path(lmdb_path)), subdir=True, readonly=True,
                        lock=False, readahead=False)
        try:
            with env.begin(write=False) as txn:
                raw = txn.get(META_KEY)
            if raw is None:
                raise RuntimeError(f"{lmdb_path} has no {META_KEY!r} record")
            return pickle.loads(raw)
        finally:
            env.close()

    def _ensure_env(self):
        if self._env is None:
            import lmdb
            import os
            key = (self.lmdb_path, os.getpid())
            global _LMDB_ENVS
            if "_LMDB_ENVS" not in globals():
                _LMDB_ENVS = {}
            if key not in _LMDB_ENVS:
                _LMDB_ENVS[key] = lmdb.open(self.lmdb_path, subdir=True, readonly=True,
                                            lock=False, readahead=False, max_readers=512)
            self._env = _LMDB_ENVS[key]
        return self._env

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        env = self._ensure_env()
        with env.begin(write=False) as txn:
            raw = txn.get(f"{self.indices[i]:09d}".encode())
        rec = pickle.loads(raw)

        types = torch.from_numpy(rec["types"].astype(np.int64))
        one_hot = torch.zeros(len(types), self.num_classes, dtype=torch.float32)
        one_hot.scatter_(1, types.unsqueeze(1), 1.0)

        y = to_target(rec["sa"][self.label_builder], self.target_mode)
        if self.normalize:
            y = (y - self.target_mean) / self.target_std

        return {
            "pos": torch.from_numpy(rec["pos"].astype(np.float32)),
            "one_hot": one_hot,
            "y": torch.tensor(y, dtype=torch.float32),
            "num_nodes": int(rec["n"]),
            "smiles": rec["smiles"],
        }

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.target_std + self.target_mean if self.normalize else y

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None  # lmdb envs are not picklable across workers
        return state


def collate_fn(batch: List[dict]) -> dict:
    """Concatenate into DiffSBDD's flat + batch-index layout (its `lig_mask`)."""
    return {
        "pos": torch.cat([b["pos"] for b in batch], dim=0),
        "one_hot": torch.cat([b["one_hot"] for b in batch], dim=0),
        "batch": torch.cat([torch.full((b["num_nodes"],), i, dtype=torch.long)
                            for i, b in enumerate(batch)], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "num_nodes": torch.tensor([b["num_nodes"] for b in batch], dtype=torch.long),
        "smiles": [b["smiles"] for b in batch],
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build the SA surrogate LMDB")
    p.add_argument("--config", default=str(_REPO_ROOT / "configs/stage1_surrogate.yaml"))
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="dotted config overrides, e.g. data.source=targetdiff_lmdb")
    p.add_argument("--fixture", type=int, default=0,
                   help="build N synthetic molecules instead of reading CrossDocked")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--inspect", action="store_true", help="print LMDB metadata and exit")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.override)

    if args.inspect:
        meta = SALMDBDataset.read_meta(cfg["data"]["lmdb_path"])
        for k, v in meta.items():
            if k == "splits":
                print("  splits: " + ", ".join(f"{s}={len(ix)}" for s, ix in v.items()))
            elif k == "label_stats":
                for b, s in v.items():
                    print(f"  SA[{b}]: mean={s['mean']:.3f} std={s['std']:.3f} "
                          f"range=[{s['min']:.2f}, {s['max']:.2f}] n={s['count']}")
            else:
                print(f"  {k}: {v}")
        return 0

    build_lmdb(cfg, fixture=args.fixture, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
