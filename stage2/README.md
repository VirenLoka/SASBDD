# Stage 2 — SA-guided finetuning of DiffSBDD

Finetunes a pretrained DiffSBDD checkpoint so its samples score better under the
frozen stage-1 SA surrogate, without letting it drift into fooling the surrogate.

Implemented by **subclassing** `LigandPocketDDPM`, not by editing it. Upstream
files are untouched apart from the additive config migration.

---

## Per optimiser step

```
1.  DDPM loss on real data, t ~ U[0, T]                      (parent forward, unchanged)
2.  reward pass at t ~ U[t_min, t_max]                       (low t only)
      eps_theta -> Tweedie x0_hat -> box-integral -> Gumbel-ST -> frozen surrogate -> scalar
3.  anchor ||eps_theta - eps_frozen||^2 at the reward pass's own z_t
4.  separate backward for (1) and (2+3); budget the aux gradient against the
    DDPM gradient; sum; clip; step
```

The reward draws **its own timestep**, independent of the DDPM loss. The DDPM
term needs all of `[0, T]` to keep the model a valid diffusion model; the reward
only carries signal at low `t`, because atom types are near-uniform past
`t ≈ 75` of 500. Sharing one draw would spend ~85% of the batch on evaluations
with no signal.

Step 4 uses manual optimisation — the split backward is what makes both the
gradient budget and the clipping fix possible.

---

## Quick start

```bash
python stage2/train.py --config configs/stage2_reward.yaml \
  --override base_checkpoint=/path/to/crossdocked_fullatom_cond.ckpt \
             paths.processed_crossdock=/path/to/processed_crossdock_noH_full \
             paths.surrogate_ckpt=sa_surrogate/checkpoints/<run>/best.pt
```

Calibrate `residual_scale` first (see below):

```bash
python stage2/calibrate.py --override base_checkpoint=/path/to/ckpt paths.processed_crossdock=/path/to/processed
```

Smoke test with no data at all — synthetic complexes, validates the whole
gradient path:

```bash
python stage2/make_fixture.py && python stage2/train.py --steps 4 --override base_checkpoint=/path/to/ckpt paths.processed_crossdock=stage2/fixture_data logging.logger=none
```

### Data sources

Two are supported, selected by `data.source`:

| `data.source` | artefact |
|---|---|
| `npz` (default) | `process_crossdock.py` output — `{train,val,test}.npz` + `size_distribution.npy` |
| `targetdiff_lmdb` | `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` + `crossdocked_pocket10_pose_split.pt` |

```bash
python stage2/train.py --override   data.source=targetdiff_lmdb   data.lmdb_path=/path/to/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb   data.split_path=/path/to/crossdocked_pocket10_pose_split.pt   base_checkpoint=/path/to/crossdocked_fullatom_cond.ckpt
```

Check what a record actually contains before a long run:

```bash
python stage2/crossdocked_lmdb.py <lmdb> --split <pose_split.pt>
```

Four things the LMDB path handles that are easy to get wrong:

- **The pickles reference classes this repo does not have.** TargetDiff stores
  `ProteinLigandData` from its own `utils.data`; DiffSBDD has a *different*
  top-level `utils`, so a naive `pickle.loads` resolves to the wrong module and
  fails confusingly. A permissive unpickler substitutes an attribute-bag for any
  class it cannot import.
- **Keys are unpadded `str(i)`, so cursor order is lexicographic**: `0, 1, 10,
  11, …`. Taking keys in cursor order would silently map split index 2 to record
  `"10"`. Keys are sorted numerically.
- **Pockets are 10 Å; the pretrained checkpoint saw 8 Å.** `data.pocket_cutoff`
  (default 8.0) re-crops, residue-level where residues can be recovered from
  atom names — matching `process_crossdock.py:51-58` — and atom-level otherwise.
  Measured on fixture data: no crop → 175 atoms/pocket, 10 Å → 105, 8 Å → 89,
  6 Å → 62.
- **The pose split has no validation set.** One is carved out of train by index
  and removed from it (`data.val_size`, default 300), so unlike the leak
  acknowledged in `process_crossdock.py:290` there is no train/val overlap.

Encoding reproduces `process_crossdock.py` exactly, quirks included: hydrogens
dropped, ligands with out-of-vocabulary elements dropped, and unknown non-H
pocket atoms given an all-zero row (upstream's `np.eye(1, K, K)` is out of range
and yields zeros rather than a one-hot).

Docking during eval is off by default (`data.docking_eval`) — the parent's
`analyze_sample` docks whenever receptors are passed, which needs smina plus
receptor PDBs on disk, neither implied by having the LMDB.

**The `npz` source needs the processed files**, not the raw release:
`LigandPocketDDPM.setup()` reads `{train,val,test}.npz` and
`size_distribution.npy` from `datadir`, which is what `process_crossdock.py`
produces (`python process_crossdock.py <basedir> --no_H`).

---

## Files

| file | role |
|---|---|
| `train.py` | entry point; reads architecture from the base checkpoint |
| `lightning_module.py` | `SAGuidedDDPM` — loss assembly, gradient budget, clipping, reward-hacking detector |
| `reward.py` | frozen surrogate wrapper, Tweedie reward pass, anchor, channel balance |
| `calibrate.py` | measures `residual_scale` instead of guessing it |
| `compat.py` | native-torch stand-ins for `torch_scatter` / `wandb` when absent |
| `make_fixture.py` | synthetic `processed_crossdock` dir for smoke tests |
| `../configs/stage2_reward.yaml` | every knob |

---

## What was verified

Run against the released `crossdocked_fullatom_cond.ckpt` on fixture data:

- **Anchor is exactly `0.000e+00` at initialisation.** This is the sharpest
  available check that the frozen reference really received the *pretrained*
  weights rather than a random init — `train.py` loads the checkpoint into both
  `ddpm.*` and `ref_ddpm.*`. It becomes non-zero (`8.8e-03`) as soon as theta is
  perturbed.
- **Reward gradient reaches 111/115 denoiser parameters.** The four that do not
  are `dynamics.residue_decoder.{0,2}.{weight,bias}` — in `pocket_conditioning`
  mode the pocket is fixed conditioning, so nothing on that branch affects the
  ligand the surrogate scores. Expected, not a bug.
- **Frozen things stay frozen**: 0/134 surrogate and 0/116 reference parameters
  require grad, and `.train()` keeps both in eval while `ddpm` trains.
- **The gradient budget holds the ratio exactly at `rho`** (measured 0.5, 0.2,
  0.05 → effective 0.5000, 0.2000, 0.0500).
- Full `Trainer.fit` runs with checkpointing and validation.

### Why the budget is not optional

Measured on the pretrained model:

| quantity | value |
|---|---|
| `‖g_ddpm‖` | 0.263 |
| `‖g_reward‖` unweighted | 8.026 |
| ratio | **30.5×** |
| after `reward.weight = 0.01` | 0.305× |
| after `grad_budget = 0.2` | 0.200× |

`reward.weight` is a fixed scalar, so the ratio it produces drifts as both terms
change during training. The budget pins it at a known value instead. **Log
`sa/grad_ratio` every step** — it is the first number to look at when stage 2
misbehaves.

---

## Reward hacking

SA improves monotonically toward smaller, carbon-only, ring-free molecules, and
molecule size is drawn from a fixed prior rather than being under gradient
control. The accessible hack is "turn every heteroatom into carbon".

Three defences, all on by default:

1. **DDPM loss on real data** across the full schedule.
2. **Frozen-reference anchor** `λ‖eps_theta − eps_frozen‖²`, evaluated at the
   reward pass's own `z_t` — i.e. exactly where the reward perturbs theta, which
   is where drift originates. Reuses the already-computed `z_t`, so it costs one
   frozen forward rather than a second noising.
3. **Gradient budget** capping the reward at a fixed fraction of the DDPM
   gradient.

And a **detector**, `evaluate_sa()`, run every `reward.eval_sa_epochs`: it
samples molecules, scores them with the surrogate (`sa_pred`) *and* with RDKit
via the same OpenBabel reconstruction (`sa_true`), and reports `sa_gap` and
`sa_validity`.

> If `sa_pred` improves while `sa_true` does not — or `sa_gap` widens, or
> `sa_validity` falls — the model is fooling the surrogate, not making easier
> molecules. Stop and lower `reward.weight` or raise `anchor.weight`.

---

## Calibrate `residual_scale` before trusting a run

The surrogate is conditioned on the noise level of its input. At stage 2 that
input is Tweedie's `x0_hat`, whose error is *smaller* than the forward `r(t)`
because the denoiser removes most of it. `residual_scale` is the fraction it
fails to remove. Get it wrong and the surrogate is scoring an input it was never
calibrated on — no crash, just a quietly wrong reward.

`calibrate.py` measures it. On fixture data (random-shell pockets, so
**off-distribution for the denoiser — re-run on real data**) it reported:

```
surrogate.residual_scale:       0.692   (coordinates)
surrogate.residual_scale_types: 0.202   (atom types)
```

Both far from the 0.3 default, and differing 3.5× from each other — which is why
the two are separately configurable. `residual_scale` feeds the model's sigma
conditioning; `residual_scale_types` feeds the categorical box-integral readout.
Leave `residual_scale_types` null to use one value for both.

---

## Adaptive gradient clipping

Upstream `configure_gradient_clipping` (`lightning_modules.py:874`) uses
`1.5·mean + 2·std` over a 50-step history. **It ratchets**: under persistent
clipping the queue fills with the threshold itself, std → 0, mean → M, so the
threshold becomes 1.5·M, then 2.25·M — ×1.5 every 50 steps, unbounded. Harmless
for the well-behaved DDPM gradient; not harmless for a spiky Gumbel-ST reward.

`SAGuidedDDPM._clip_gradients` applies two fixes:

- a hard ceiling (`reward.grad_clip_ceiling`), and
- the queue is fed the **DDPM-only** norm, so the reward term cannot inflate the
  threshold that protects the DDPM loss.

Plus `reward.warmup_steps` ramps the weight in, so the gradient-norm
distribution does not shift abruptly underneath the queue.

`grad_clip_ceiling` needs calibrating: watch `sa/grad_total` for the first few
hundred steps and set it to roughly 3–5× typical.

---

## Channel balance

`‖g_types‖ / ‖g_coords‖` is structurally large — types reach the surrogate
through a plain `Linear`, coordinates only through RBF-expanded distances behind
a smooth cutoff envelope. Left alone the type channel swamps coordinate updates.

`reward.channel_balance: normalize` equalises the two blocks of
`d(score)/d(xh)` before they propagate into the denoiser, then applies
`coord_weight` / `type_weight` as a ratio. Set `none` to disable, `manual` to
apply the weights without equalising first.

This is also the knob for the `t_max` tradeoff: types decohere by `t ≈ 75` but
coordinates survive to `t ≈ 150–200`, so if you weight coordinates well above
types you can raise `reward.t_max`.

---

## Library versions

The repo was written against pytorch-lightning 1.8 (what `environment.yaml`
pinned). **1.8.x and 2.x both work now** — `common/compat.py` absorbs the
differences rather than forcing a downgrade. Verified end to end on PL 1.8.6 and
PL 2.4.0, including all 11 entry points.

```bash
python common/compat.py
```

prints your versions and which known differences are being handled. What it
covers:

| change | where it bit | handling |
|---|---|---|
| PL 2.0 removed `*_epoch_end` hooks | `lightning_modules.py:382` **and** `stage2/lightning_module.py` | renamed to `on_validation_epoch_end`, which exists in both 1.x and 2.x; the outputs argument was never used |
| PL 2.0 dropped `optimizer_idx` from `configure_gradient_clipping` | `lightning_modules.py:874` | signature widened to `(self, optimizer, *args, **kwargs)` |
| PL 2.0 rejects `strategy=None` | `train.py`, `stage2/train.py` | `compat.trainer_strategy()` returns `None` on 1.x, `"auto"` on 2.x |
| torch 2.6 flipped `torch.load` to `weights_only=True` | 7 call sites | `compat.torch_load()` sets it where the argument exists |
| biopython 1.80 removed `three_to_one` | `lightning_modules.py`, `process_crossdock.py`, `process_bindingmoad.py` | `compat.three_to_one()` falls back to `protein_letters_3to1` |

Note the first two are bugs in **upstream DiffSBDD**, not just in the stage-2
code — the base `LigandPocketDDPM` could not run on Lightning 2.x either.

## Known limitations

- **`residual_scale` defaults are guesses** until `calibrate.py` is run on real
  data. This is the most likely source of a silently wrong reward.
- **The anchor only covers the reward's timestep band.** Drift at high `t` is
  constrained by the DDPM loss alone. Covering the full range with the anchor
  would need a second frozen forward per step.
- **No off-manifold negatives in the surrogate.** Early in finetuning the model
  emits geometry the surrogate has never seen, where its predictions are
  extrapolation. Sigma conditioning mitigates this (it falls back to the
  marginal), but does not remove it.
- **`evaluate_sa` is slow**: it runs the full reverse process (~17 s for 4
  molecules on CPU). Keep `eval_sa_samples` modest, or raise
  `eval_sa_epochs`.
- **The fixture has random-shell pockets.** It validates shapes, masks,
  batching and the gradient path. No number produced from it means anything.
