import argparse
from pathlib import Path

import torch
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()  # suppress OpenBabel messages

import utils
from common.config import add_config_args, apply_config_defaults, config_from_args
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from common.compat import ensure_dependencies as _ensure_deps
_ensure_deps(verbose=False)   # must precede lightning_modules (imports wandb)

from lightning_modules import LigandPocketDDPM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_config_args(parser, default='inference.yaml')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--pdbfile', type=str)
    parser.add_argument('--resi_list', type=str, nargs='+', default=None)
    parser.add_argument('--ref_ligand', type=str, default=None)
    parser.add_argument('--outfile', type=Path)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_nodes_lig', type=int, default=None)
    parser.add_argument('--all_frags', action='store_true')
    parser.add_argument('--sanitize', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--resamplings', type=int, default=10)
    parser.add_argument('--jump_length', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=None)
    args = parser.parse_args()
    args = apply_config_defaults(
        args, config_from_args(args, 'generate'), parser=parser)

    pdb_id = Path(args.pdbfile).stem

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.batch_size is None:
        args.batch_size = args.n_samples
    assert args.n_samples % args.batch_size == 0

    # Load model
    model = LigandPocketDDPM.load_from_checkpoint(
        args.checkpoint, map_location=device)
    model = model.to(device)

    if args.num_nodes_lig is not None:
        num_nodes_lig = torch.ones(args.n_samples, dtype=int) * \
                        args.num_nodes_lig
    else:
        num_nodes_lig = None

    molecules = []
    from tqdm import tqdm
    n_batches = args.n_samples // args.batch_size
    for i in tqdm(range(n_batches), desc="Generating ligands"):
        molecules_batch = model.generate_ligands(
            args.pdbfile, args.batch_size, args.resi_list, args.ref_ligand,
            num_nodes_lig, args.sanitize, largest_frag=not args.all_frags,
            relax_iter=(200 if args.relax else 0),
            resamplings=args.resamplings, jump_length=args.jump_length,
            timesteps=args.timesteps)
        molecules.extend(molecules_batch)

    # Make SDF files
    utils.write_sdf_file(args.outfile, molecules)
