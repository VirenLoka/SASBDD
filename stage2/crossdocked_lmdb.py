"""
Loader for the TargetDiff / Pocket2Mol preprocessed CrossDocked release:

    crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
    crossdocked_pocket10_pose_split.pt

This is a different artefact from `process_crossdock.py`'s output.  It stores one
pickled record per index holding separate protein and ligand arrays, with
elements as atomic numbers, and it ships train/test index lists rather than
train/val/test .npz files.

Three things make reading it awkward, all handled here:

1. **The pickles reference classes that do not exist in this repo.**  TargetDiff
   stores `ProteinLigandData` (a torch_geometric Data subclass from its own
   `utils.data`).  `_PermissiveUnpickler` substitutes an attribute-bag for any
   class it cannot import, so the record loads without vendoring TargetDiff.
   Note DiffSBDD *also* has a top-level `utils` module, so the import would
   otherwise resolve to the wrong thing and fail confusingly.

2. **Field names vary between forks.**  Fields are resolved through aliases and
   can be overridden from config (`data.field_map`).  `inspect_lmdb()` prints
   what a real record actually contains.

3. **The pocket is 10 A; the pretrained checkpoint saw 8 A.**  `pocket_cutoff`
   re-filters to match, reproducing DiffSBDD's residue-level rule
   (process_crossdock.py:51-58): keep a whole residue if any of its atoms is
   within the cutoff of any ligand atom.

Encoding follows `process_crossdock.py` exactly, including its quirks: ligands
whose elements fall outside the vocabulary are dropped, hydrogens are dropped,
and unknown non-hydrogen pocket atoms get an all-zero row rather than a one-hot
(upstream's `np.eye(1, K, K)` is out of range and yields zeros).
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from constants import FLOAT_TYPE, dataset_params  # noqa: E402
from dataset import ProcessedLigandPocketDataset  # noqa: E402

# Atomic number -> symbol, covering DiffSBDD's vocabulary plus what shows up in
# protein structures.
ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 34: "Se", 35: "Br",
    53: "I", 80: "Hg",
}

FIELD_ALIASES: Dict[str, Sequence[str]] = {
    "protein_pos": ("protein_pos", "pocket_pos", "protein_coords", "pocket_coords"),
    "protein_element": ("protein_element", "pocket_element", "protein_atomic_numbers",
                        "protein_atom_type"),
    "protein_atom_name": ("protein_atom_name", "pocket_atom_name"),
    "protein_atom_to_aa_type": ("protein_atom_to_aa_type", "pocket_atom_to_aa_type"),
    "protein_res_id": ("protein_res_id", "protein_residue_index", "protein_atom_to_res",
                       "pocket_res_id"),
    "protein_filename": ("protein_filename", "pocket_filename", "protein_molecule_name"),
    "ligand_pos": ("ligand_pos", "lig_pos", "ligand_coords"),
    "ligand_element": ("ligand_element", "lig_element", "ligand_atomic_numbers"),
    "ligand_bond_index": ("ligand_bond_index", "lig_bond_index"),
    "ligand_bond_type": ("ligand_bond_type", "lig_bond_type"),
    "ligand_smiles": ("ligand_smiles", "smiles", "lig_smiles"),
    "ligand_filename": ("ligand_filename", "lig_filename"),
}


# -----------------------------------------------------------------------------
# permissive unpickling
# -----------------------------------------------------------------------------

class _AttrBag:
    """Stand-in for a class we cannot import.  Absorbs whatever state the
    pickle carries so the record's fields survive."""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def __getitem__(self, k):
        return self.__dict__[k]

    def __repr__(self):
        return f"_AttrBag({sorted(self.__dict__)})"


def _make_placeholder(module: str, name: str):
    return type(f"_Missing_{name}", (_AttrBag,), {"__module__": module})


class _PermissiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # DiffSBDD has its own top-level `utils`, so TargetDiff's
        # `utils.data.ProteinLigandData` would resolve to the wrong module.
        if module.startswith("utils") or "ProteinLigandData" in name:
            return _make_placeholder(module, name)
        try:
            return super().find_class(module, name)
        except Exception:
            return _make_placeholder(module, name)


def _loads(raw: bytes) -> Any:
    return _PermissiveUnpickler(io.BytesIO(bytes(raw))).load()


def record_to_dict(obj: Any) -> Dict[str, Any]:
    """Normalise whatever the pickle produced into a flat dict."""
    if isinstance(obj, dict):
        out = dict(obj)
    elif hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            out = dict(obj.to_dict())
        except Exception:
            out = dict(getattr(obj, "__dict__", {}))
    else:
        out = dict(getattr(obj, "__dict__", {}))

    # torch_geometric keeps fields inside a storage object
    for key in ("_store", "_global_store", "_mapping"):
        store = out.pop(key, None)
        if store is None:
            continue
        inner = store if isinstance(store, dict) else getattr(store, "__dict__", {})
        for k, v in dict(inner).items():
            if k.startswith("_"):
                continue
            out.setdefault(k, v)
    return {k: v for k, v in out.items() if not k.startswith("_")}


# -----------------------------------------------------------------------------
# field access
# -----------------------------------------------------------------------------

def resolve_field(rec: Dict[str, Any], canonical: str,
                  field_map: Optional[Dict[str, str]] = None) -> Optional[Any]:
    if field_map and canonical in field_map and field_map[canonical] in rec:
        return rec[field_map[canonical]]
    for alias in FIELD_ALIASES.get(canonical, (canonical,)):
        if alias in rec:
            return rec[alias]
    return None


def _to_numpy(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def elements_to_symbols(elements: Any) -> List[str]:
    """Atomic numbers or symbol strings -> symbol strings."""
    arr = _to_numpy(elements)
    if arr is None:
        return []
    flat = arr.reshape(-1)
    if flat.dtype.kind in ("U", "S", "O"):
        return [str(x).strip().capitalize() for x in flat]
    return [ATOMIC_NUMBER_TO_SYMBOL.get(int(x), f"Z{int(x)}") for x in flat]


# -----------------------------------------------------------------------------
# pocket cropping
# -----------------------------------------------------------------------------

def infer_residue_ids(atom_names: Optional[Sequence[str]],
                      aa_types: Optional[np.ndarray],
                      n_atoms: int) -> Optional[np.ndarray]:
    """Segment a flat atom list into residues.

    TargetDiff's `to_dict_atom()` does not store a residue index, so it is
    reconstructed: in PDB order a new standard residue begins at its backbone
    N.  Falls back to segmenting on amino-acid-type changes, and returns None
    if neither signal is available (caller then crops per atom).
    """
    if atom_names is not None and len(atom_names) == n_atoms:
        names = [str(a).strip().upper() for a in atom_names]
        starts = np.array([i for i, nm in enumerate(names) if nm == "N"])
        if len(starts) > 1 and starts[0] <= 1:
            res = np.zeros(n_atoms, dtype=np.int64)
            for r, s in enumerate(starts):
                res[s:] = r
            return res
    if aa_types is not None and len(aa_types) == n_atoms:
        aa = np.asarray(aa_types).reshape(-1)
        return np.concatenate([[0], np.cumsum(aa[1:] != aa[:-1])]).astype(np.int64)
    return None


def crop_pocket(pocket_xyz: np.ndarray, lig_xyz: np.ndarray, cutoff: float,
                res_ids: Optional[np.ndarray]) -> np.ndarray:
    """Boolean keep-mask.  Residue-level when residues are known, matching
    DiffSBDD (process_crossdock.py:51-58); atom-level otherwise."""
    d = np.linalg.norm(pocket_xyz[:, None, :] - lig_xyz[None, :, :], axis=-1).min(axis=1)
    close = d < cutoff
    if res_ids is None:
        return close
    keep_res = set(np.unique(res_ids[close]).tolist())
    return np.isin(res_ids, list(keep_res))


# -----------------------------------------------------------------------------
# dataset
# -----------------------------------------------------------------------------

class CrossDockedLMDBDataset(Dataset):
    """Drop-in replacement for `ProcessedLigandPocketDataset`.

    Emits the same per-item dict and reuses its `collate_fn`, so
    `LigandPocketDDPM.get_ligand_and_pocket` needs no changes.
    """

    collate_fn = staticmethod(ProcessedLigandPocketDataset.collate_fn)

    def __init__(
        self,
        lmdb_path: str | Path,
        indices: Sequence[int],
        dataset_name: str = "crossdock",
        pocket_cutoff: Optional[float] = 8.0,
        center: bool = True,
        field_map: Optional[Dict[str, str]] = None,
        max_ligand_atoms: Optional[int] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.lmdb_path = str(lmdb_path)
        self.pocket_cutoff = pocket_cutoff
        self.center = bool(center)
        self.field_map = dict(field_map or {})
        self.max_ligand_atoms = max_ligand_atoms
        self._env = None

        info = dataset_params[dataset_name]
        self.atom_encoder: Dict[str, int] = dict(info["atom_encoder"])
        self.num_classes = len(self.atom_encoder)

        keys = self._all_keys()
        self.keys = [keys[i] for i in indices if 0 <= i < len(keys)]
        if verbose:
            print(f"[crossdocked-lmdb] {len(self.keys)} records "
                  f"(pocket_cutoff={pocket_cutoff}, center={center})")
        self._rejects: Dict[str, int] = {}

    # -- lmdb plumbing ------------------------------------------------------
    def _open(self):
        import lmdb
        if self._env is None:
            path = self.lmdb_path
            self._env = lmdb.open(path, subdir=os.path.isdir(path), readonly=True,
                                  lock=False, readahead=False, meminit=False,
                                  max_readers=512)
        return self._env

    def _all_keys(self) -> List[bytes]:
        """Enumerate real keys once, so any keying scheme works (TargetDiff
        uses unpadded str(i); other forks zero-pad)."""
        env = self._open()
        with env.begin(write=False) as txn:
            keys = [k for k, _ in txn.cursor()]
        def sort_key(k: bytes):
            s = k.decode(errors="ignore")
            return (0, int(s)) if s.lstrip("-").isdigit() else (1, s)
        return sorted(keys, key=sort_key)

    def __len__(self) -> int:
        return len(self.keys)

    def raw_record(self, i: int) -> Dict[str, Any]:
        env = self._open()
        with env.begin(write=False) as txn:
            raw = txn.get(self.keys[i])
        if raw is None:
            raise KeyError(f"missing record for key {self.keys[i]!r}")
        return record_to_dict(_loads(raw))

    # -- conversion ---------------------------------------------------------
    def _one_hot(self, symbols: Sequence[str], allow_unknown: bool
                 ) -> Optional[np.ndarray]:
        """Ligands reject unknown elements (upstream drops the complex);
        pockets give them an all-zero row (upstream's out-of-range np.eye)."""
        oh = np.zeros((len(symbols), self.num_classes), dtype=np.float32)
        for i, s in enumerate(symbols):
            idx = self.atom_encoder.get(s.capitalize())
            if idx is not None:
                oh[i, idx] = 1.0
            elif not allow_unknown:
                return None
        return oh

    def build(self, i: int) -> Optional[Dict[str, Any]]:
        rec = self.raw_record(i)
        f = lambda name: resolve_field(rec, name, self.field_map)  # noqa: E731

        lig_xyz = _to_numpy(f("ligand_pos"))
        pock_xyz = _to_numpy(f("protein_pos"))
        if lig_xyz is None or pock_xyz is None:
            return self._reject("missing_pos")
        lig_xyz = lig_xyz.astype(np.float32).reshape(-1, 3)
        pock_xyz = pock_xyz.astype(np.float32).reshape(-1, 3)

        lig_sym = elements_to_symbols(f("ligand_element"))
        pock_sym = elements_to_symbols(f("protein_element"))
        if len(lig_sym) != len(lig_xyz) or len(pock_sym) != len(pock_xyz):
            return self._reject("length_mismatch")

        # hydrogens are dropped on both sides (DiffSBDD is heavy-atom only)
        lig_keep = np.array([s != "H" for s in lig_sym], dtype=bool)
        lig_xyz, lig_sym = lig_xyz[lig_keep], [s for s, k in zip(lig_sym, lig_keep) if k]
        pock_keep_h = np.array([s != "H" for s in pock_sym], dtype=bool)

        if len(lig_xyz) < 1:
            return self._reject("empty_ligand")
        if self.max_ligand_atoms and len(lig_xyz) > self.max_ligand_atoms:
            return self._reject("ligand_too_large")

        res_ids = infer_residue_ids(f("protein_atom_name"),
                                    _to_numpy(f("protein_atom_to_aa_type")),
                                    len(pock_sym))
        keep = pock_keep_h
        if self.pocket_cutoff is not None:
            keep = keep & crop_pocket(pock_xyz, lig_xyz, float(self.pocket_cutoff), res_ids)
        pock_xyz = pock_xyz[keep]
        pock_sym = [s for s, k in zip(pock_sym, keep) if k]
        if len(pock_xyz) < 1:
            return self._reject("empty_pocket")

        lig_oh = self._one_hot(lig_sym, allow_unknown=False)
        if lig_oh is None:
            return self._reject("ligand_element_out_of_vocab")
        pock_oh = self._one_hot(pock_sym, allow_unknown=True)

        if self.center:
            mean = (lig_xyz.sum(0) + pock_xyz.sum(0)) / (len(lig_xyz) + len(pock_xyz))
            lig_xyz = lig_xyz - mean
            pock_xyz = pock_xyz - mean

        name = f("ligand_filename") or f"record_{i}"
        receptor = f("protein_filename") or f"record_{i}"
        return {
            "lig_coords": torch.from_numpy(lig_xyz).to(FLOAT_TYPE),
            "lig_one_hot": torch.from_numpy(lig_oh).to(FLOAT_TYPE),
            "lig_mask": torch.zeros(len(lig_xyz)),
            "num_lig_atoms": torch.tensor(len(lig_xyz)),
            "pocket_coords": torch.from_numpy(pock_xyz).to(FLOAT_TYPE),
            "pocket_one_hot": torch.from_numpy(pock_oh).to(FLOAT_TYPE),
            "pocket_mask": torch.zeros(len(pock_xyz)),
            "num_pocket_nodes": torch.tensor(len(pock_xyz)),
            "names": str(name),
            "receptors": str(receptor),
        }

    def _reject(self, reason: str) -> None:
        self._rejects[reason] = self._rejects.get(reason, 0) + 1
        return None

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.build(i)
        if item is not None:
            return item
        # A rejected record must not stop training; fall forward to the next
        # usable one.  Counts are reported by `rejection_summary()`.
        for j in range(1, min(len(self), 64)):
            item = self.build((i + j) % len(self))
            if item is not None:
                return item
        raise RuntimeError(f"no usable record near index {i}: {self._rejects}")

    def rejection_summary(self) -> Dict[str, int]:
        return dict(self._rejects)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None          # lmdb envs do not cross process boundaries
        return state


# -----------------------------------------------------------------------------
# splits
# -----------------------------------------------------------------------------

def load_pose_split(split_path: str | Path) -> Dict[str, List[int]]:
    """Read crossdocked_pocket10_pose_split.pt.

    Accepts index lists (TargetDiff) or lists of (pocket, ligand) name pairs;
    pairs are converted to positions within the split's own ordering.
    """
    from common.compat import torch_load
    obj = torch_load(str(split_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"{split_path} does not contain a dict of splits")
    out: Dict[str, List[int]] = {}
    for name, value in obj.items():
        entries = list(value)
        if not entries:
            out[str(name)] = []          # e.g. the release's empty 'val' key
        elif isinstance(entries[0], (int, np.integer)) or torch.is_tensor(entries[0]):
            out[str(name)] = [int(x) for x in entries]
        else:
            out[str(name)] = list(range(len(entries)))
            print(f"[crossdocked-lmdb] split '{name}' holds name pairs, not "
                  f"indices; using positional order ({len(entries)} entries)")
    return out


def make_splits(split_path: str | Path, val_size: int = 300, seed: int = 0
                ) -> Dict[str, List[int]]:
    """The pose split has no validation set; carve one out of train.

    Carved by index and removed from train, so unlike the note in
    process_crossdock.py:290 there is no train/val overlap.
    """
    splits = load_pose_split(split_path)
    train = list(splits.get("train", []))
    # An empty 'val' key counts as absent -- the release ships one.
    if not splits.get("val") and val_size > 0 and train:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train))
        n_val = min(int(val_size), max(0, len(train) - 1))
        val_pos = set(perm[:n_val].tolist())
        splits["val"] = [train[i] for i in sorted(val_pos)]
        splits["train"] = [t for i, t in enumerate(train) if i not in val_pos]
        print(f"[crossdocked-lmdb] carved {len(splits['val'])} validation "
              f"records out of train (seed={seed}); no overlap")
    splits.setdefault("val", [])
    splits.setdefault("test", [])
    return splits


# -----------------------------------------------------------------------------
# inspection
# -----------------------------------------------------------------------------

def inspect_lmdb(lmdb_path: str | Path, split_path: Optional[str | Path] = None,
                 n: int = 2) -> None:
    import lmdb

    path = str(lmdb_path)
    env = lmdb.open(path, subdir=os.path.isdir(path), readonly=True,
                    lock=False, readahead=False)
    with env.begin(write=False) as txn:
        keys = [k for k, _ in txn.cursor()]
        print(f"records: {len(keys)}")
        print(f"stored as: {'directory' if os.path.isdir(path) else 'single file'}")
        print(f"first keys: {[k[:32] for k in keys[:5]]}")
        for i, key in enumerate(keys[:n]):
            rec = record_to_dict(_loads(txn.get(key)))
            print(f"\n--- record {key!r} : {len(rec)} fields ---")
            for k in sorted(rec):
                v = rec[k]
                arr = _to_numpy(v) if isinstance(v, (torch.Tensor, np.ndarray, list)) else None
                if arr is not None and arr.dtype != object:
                    print(f"  {k:32s} {str(arr.shape):16s} {arr.dtype}")
                else:
                    s = str(v)
                    print(f"  {k:32s} {type(v).__name__:16s} "
                          f"{s[:60]}{'...' if len(s) > 60 else ''}")
            print("  resolved ->")
            for canonical in FIELD_ALIASES:
                got = resolve_field(rec, canonical)
                mark = "ok " if got is not None else "MISSING"
                print(f"    {mark} {canonical}")
    env.close()

    if split_path:
        splits = load_pose_split(split_path)
        print("\nsplits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


def check_lmdb(lmdb_path: str | Path, split_path: Optional[str | Path] = None,
               n: int = 500, pocket_cutoff: Optional[float] = 8.0,
               dataset_name: str = "crossdock") -> None:
    """Run the real build path over `n` records and report what comes out.

    Worth running before a long job: it is the only way to confirm that residue
    inference and pocket cropping behave on your actual structures, and that the
    ligand vocabulary is covered.
    """
    from collections import Counter

    splits = make_splits(split_path, val_size=0) if split_path else None
    indices = splits["train"][:n] if splits else list(range(n))

    uncropped = CrossDockedLMDBDataset(lmdb_path, indices, dataset_name=dataset_name,
                                       pocket_cutoff=None, verbose=False)
    cropped = CrossDockedLMDBDataset(lmdb_path, indices, dataset_name=dataset_name,
                                     pocket_cutoff=pocket_cutoff, verbose=False)

    lig_n, pock_raw, pock_cut, res_n = [], [], [], []
    res_ok = 0
    lig_elems, pock_elems, bad_elems = Counter(), Counter(), Counter()
    ok = 0

    for i in range(len(cropped)):
        try:
            rec = cropped.raw_record(i)
        except Exception:
            continue
        lig_sym = elements_to_symbols(resolve_field(rec, "ligand_element"))
        pock_sym = elements_to_symbols(resolve_field(rec, "protein_element"))
        lig_elems.update(s for s in lig_sym if s != "H")
        pock_elems.update(s for s in pock_sym if s != "H")
        bad_elems.update(s for s in lig_sym
                         if s != "H" and s.capitalize() not in cropped.atom_encoder)

        res = infer_residue_ids(resolve_field(rec, "protein_atom_name"),
                                _to_numpy(resolve_field(rec, "protein_atom_to_aa_type")),
                                len(pock_sym))
        if res is not None:
            res_ok += 1
            res_n.append(len(np.unique(res)))

        a, b = uncropped.build(i), cropped.build(i)
        if a is not None:
            pock_raw.append(int(a["num_pocket_nodes"]))
        if b is not None:
            ok += 1
            lig_n.append(int(b["num_lig_atoms"]))
            pock_cut.append(int(b["num_pocket_nodes"]))

    def stat(name, xs, unit=""):
        if not xs:
            print(f"  {name:28s} (none)")
            return
        a = np.asarray(xs)
        print(f"  {name:28s} mean={a.mean():7.1f}  median={np.median(a):7.1f}  "
              f"min={a.min():5d}  max={a.max():5d}{unit}")

    print(f"\n=== checked {len(cropped)} records "
          f"({'train split' if splits else 'first n'}) ===")
    print(f"  usable: {ok}/{len(cropped)} ({ok / max(len(cropped), 1):.1%})")
    if cropped.rejection_summary():
        print(f"  rejected: {cropped.rejection_summary()}")
    stat("ligand heavy atoms", lig_n)
    stat("pocket atoms (no crop)", pock_raw)
    stat(f"pocket atoms ({pocket_cutoff} A crop)", pock_cut)
    if pock_raw and pock_cut:
        print(f"  {'crop keeps':28s} {np.mean(pock_cut) / np.mean(pock_raw):.1%} of pocket atoms")
    print(f"  {'residues inferred for':28s} {res_ok}/{len(cropped)} pockets")
    stat("residues per pocket", res_n)

    print(f"\n  ligand elements: {dict(lig_elems.most_common())}")
    missing = [k for k in cropped.atom_encoder if k not in lig_elems]
    if missing:
        print(f"  vocabulary columns never seen in ligands: {missing}")
    if bad_elems:
        print(f"  OUT-OF-VOCAB ligand elements (these complexes are dropped): "
              f"{dict(bad_elems.most_common())}")
    print(f"  pocket elements: {dict(pock_elems.most_common(8))}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Inspect a TargetDiff CrossDocked LMDB")
    p.add_argument("lmdb", help="path to ...processed_final.lmdb")
    p.add_argument("--split", default=None, help="crossdocked_pocket10_pose_split.pt")
    p.add_argument("-n", type=int, default=2, help="records to dump")
    p.add_argument("--check", type=int, default=0, metavar="N",
                   help="run the real build path over N records and report stats")
    p.add_argument("--pocket-cutoff", type=float, default=8.0)
    args = p.parse_args(argv)
    if args.check:
        check_lmdb(args.lmdb, args.split, args.check, args.pocket_cutoff)
    else:
        inspect_lmdb(args.lmdb, args.split, args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# ligand extraction (stage 1)
# -----------------------------------------------------------------------------

def ligand_mol_from_record(rec: Dict[str, Any],
                           field_map: Optional[Dict[str, str]] = None):
    """Exact RDKit molecule from a record's stored bond graph.

    The LMDB keeps `ligand_bond_index` / `ligand_bond_type`, so the molecule can
    be rebuilt with its real bonds and orders -- no distance-based or OpenBabel
    perception, and therefore an exact SA label.  Falls back to the stored SMILES
    (coordinates re-attached where possible), then to None.
    """
    from rdkit import Chem

    from constants import bond_dict

    f = lambda name: resolve_field(rec, name, field_map)  # noqa: E731

    pos = _to_numpy(f("ligand_pos"))
    symbols = elements_to_symbols(f("ligand_element"))
    bond_index = _to_numpy(f("ligand_bond_index"))
    bond_type = _to_numpy(f("ligand_bond_type"))

    if pos is not None and symbols and bond_index is not None and bond_type is not None:
        pos = pos.astype(np.float64).reshape(-1, 3)
        if len(symbols) == len(pos):
            bi = bond_index.reshape(2, -1) if bond_index.shape[0] == 2 else bond_index.T
            mol = Chem.RWMol()
            for s in symbols:
                mol.AddAtom(Chem.Atom(s))
            seen = set()
            for (a, b), t in zip(bi.T.tolist(), np.asarray(bond_type).reshape(-1).tolist()):
                a, b, t = int(a), int(b), int(t)
                if a == b or (min(a, b), max(a, b)) in seen:
                    continue          # the graph is stored both ways
                seen.add((min(a, b), max(a, b)))
                bt = bond_dict[t] if 0 < t < len(bond_dict) else Chem.rdchem.BondType.SINGLE
                mol.AddBond(a, b, bt)
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, tuple(pos[i]))
            mol.AddConformer(conf)
            out = mol.GetMol()
            try:
                Chem.SanitizeMol(out)
                return out
            except Exception:
                pass                  # fall through to SMILES

    # No ETKDG fallback: callers read coordinates off the returned molecule, so
    # re-embedding would silently swap the docked pose for an arbitrary
    # conformer.  Fall back to perceiving bonds from the real coordinates
    # instead, which keeps the geometry and is honest about the bonds.
    if pos is not None and symbols and len(symbols) == len(pos):
        from analysis.molecule_builder import build_molecule
        from constants import dataset_params
        encoder = dataset_params["crossdock"]["atom_encoder"]
        if all(s.capitalize() in encoder for s in symbols):
            try:
                types = torch.tensor([encoder[s.capitalize()] for s in symbols])
                mol = build_molecule(torch.tensor(pos, dtype=torch.float32), types,
                                     dataset_params["crossdock"],
                                     add_coords=True, use_openbabel=True)
                Chem.SanitizeMol(mol)
                return mol
            except Exception:
                pass
    return None


def iter_lmdb_ligands(lmdb_path: str | Path,
                      indices: Optional[Sequence[int]] = None,
                      field_map: Optional[Dict[str, str]] = None,
                      max_records: Optional[int] = None,
                      report: bool = True):
    """Yield (name, rdkit_mol) for stage-1 surrogate training.

    Records that cannot be turned into a sanitisable molecule are skipped, and
    the counts are reported at the end rather than swallowed -- a high skip rate
    means the bond fields are being misread, which would otherwise look like a
    small dataset rather than a bug.
    """
    ds = CrossDockedLMDBDataset(
        lmdb_path, indices if indices is not None else range(10 ** 9),
        pocket_cutoff=None, center=False, field_map=field_map, verbose=False)
    n = 0
    skipped = {"unreadable": 0, "unbuildable": 0}
    for i in range(len(ds)):
        try:
            rec = ds.raw_record(i)
        except Exception:
            skipped["unreadable"] += 1
            continue
        mol = ligand_mol_from_record(rec, field_map)
        if mol is None:
            skipped["unbuildable"] += 1
            continue
        name = resolve_field(rec, "ligand_filename", field_map) or f"record_{i}"
        yield str(name), mol
        n += 1
        if max_records and n >= max_records:
            break
    total = n + sum(skipped.values())
    if report and total:
        frac = sum(skipped.values()) / total
        print(f"[crossdocked-lmdb] ligands: {n} usable, "
              f"{skipped['unreadable']} unreadable, "
              f"{skipped['unbuildable']} unbuildable ({frac:.1%} skipped)")
        if frac > 0.25:
            print("[crossdocked-lmdb] WARNING >25% skipped -- check the bond "
                  "fields resolve correctly:\n"
                  f"    python stage2/crossdocked_lmdb.py {lmdb_path}")
