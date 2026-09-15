# SA surrogate — stage 1 of SA-guided DiffSBDD

A σ-conditioned graph attention transformer that predicts a ligand's synthetic
accessibility from `(one-hot atom types, 3D coordinates)` at **every noise level**
of DiffSBDD's forward diffusion process.

Stage 2 will use it as a differentiable reward: run the denoiser, form `x̂₀` with
Tweedie's formula, discretise the type channel with a Gumbel straight-through
estimator, score, and backpropagate into the diffusion model's weights.

Nothing here modifies DiffSBDD. It is a self-contained subpackage that reuses the
repo's `constants.py`, `analysis/SA_Score/` and `analysis/molecule_builder.py`.

---

## Quick start

Smoke test with no data (builds ~600 synthetic 3D molecules spanning all 10 atom
types, then trains):

```bash
python sa_surrogate/dataset.py --fixture 600 --overwrite
```

```bash
python sa_surrogate/train.py --override device=cpu train.epochs=20
```

Real run — point `data.crossdocked_dir` in `configs/default.yaml` at the
CrossDocked2020 pocket10 directory (the one holding `crossdocked_pocket10/` and
`split_by_name.pt`), then:

```bash
python sa_surrogate/dataset.py --overwrite
```

```bash
python sa_surrogate/train.py
```

```bash
python sa_surrogate/inference.py --split test
```

Everything is driven by `configs/default.yaml`; any field can be overridden on
the command line with `--override key.subkey=value`.

**Apple silicon:** `device: auto` resolves to CUDA or CPU, never MPS. PyG's
`TransformerConv` aggregates through `aten::scatter_reduce`, which MPS does not
implement, so MPS runs only with `PYTORCH_ENABLE_MPS_FALLBACK=1` — and the
resulting CPU round-trips measured **11.2 s/epoch against 2.1 s/epoch on plain
CPU**. Asking for `device: mps` without the variable prints a warning and uses
CPU. CrossDocked-scale training belongs on a CUDA box regardless.

---

## Files

| file | role |
|---|---|
| `configs/default.yaml` | every knob; the only place constants are set |
| `diffusion_bridge.py` | **the stage-1 ↔ stage-2 contract**: noise schedule, corruption, box-integral categorical readout, Gumbel-ST, Tweedie helpers |
| `dataset.py` | CrossDocked ligands → SA-labelled LMDB; `SALMDBDataset`; `collate_fn` |
| `model.py` | `SAGraphTransformer` |
| `train.py` | training loop, EMA, per-noise-level validation |
| `inference.py` | `SASurrogate` — SMILES/SDF/split scoring **and the stage-2 hook** |
| `dataset/` | where the LMDB lands |

---

## Design decisions, and the measurements behind them

### 1. Trained on CrossDocked ligands, not QM9

QM9 is `{H,C,N,O,F}` with ≤9 heavy atoms. DiffSBDD's ligand vocabulary is 10
heavy-atom types including S/P/Cl/Br/I/B, with ligands of ~20–40 atoms. Training
on QM9 would leave 6 of 10 one-hot columns at their initialisation, and guidance
would then discover it can move the predicted score arbitrarily by pushing mass
onto the iodine or boron column. The vocabulary here is asserted equal to
`dataset_params['crossdock']['atom_encoder']` at build time and the build fails
loudly if it ever drifts.

### 2. SA labels come from OpenBabel reconstruction

At stage 2 the reward you actually measure is `SA(build_molecule(x, h))`, which
infers bonds from coordinates. So "the SA of this molecule" and "the SA of what
`build_molecule` reconstructs" are different quantities. Measured on the repo's
own examples:

| ligand | heavy atoms | true SDF bonds | OpenBabel rebuild | distance-cutoff (`edm`) |
|---|---|---|---|---|
| `5ndu_C_8V2` | 49 | 5.724 | **5.724** | 6.653 |
| `3rfm_B_CFF` (caffeine) | 14 | 2.298 | **2.298** | 4.842 |

OpenBabel is exact on both; the distance-cutoff path is off by 0.9 and 2.5 SA
units — larger than the spread you are trying to optimise over. Across 494
fixture molecules the `edm` route is biased **+1.6 SA units** on average
(3.56 vs 1.94). All three labels are stored in the LMDB so the choice can be
revisited without a rebuild; `openbabel` is the default and gets both the real
chemistry *and* the downstream reward definition right.

### 3. Conditioned on σ, not on the timestep

`r(t) = σ_t/α_t` spans **0.022 → 44.7** for the released checkpoint. The model is
conditioned on `log₁₀(r)` mapped to ~[-1,1], so it stays valid whether stage 2
feeds it a rescaled `z_t` or Tweedie's `x̂₀` with its own residual scale, and
across schedule changes. FiLM conditioning is zero-initialised (AdaLN-Zero
style), so it starts as the identity and learns to turn on.

### 4. Types decohere far earlier than geometry

This is the single most actionable finding, and it falls straight out of
`normalize_factors = [1, 4]`: dividing the one-hot by 4 gives the type signal an
amplitude of 0.25 against coordinate amplitudes of ~1.5 Å — roughly a 6× SNR
disadvantage. Probability that an atom keeps its true type under the calibrated
box integral, and the corresponding coordinate noise:

| t | r | p(correct type) | coord noise |
|---|---|---|---|
| 0 | 0.022 | 1.000 | 0.02 Å |
| 10 | 0.036 | 0.998 | 0.04 Å |
| 25 | 0.074 | 0.686 | 0.07 Å |
| 50 | 0.144 | 0.266 | 0.14 Å |
| 75 | 0.217 | 0.168 | 0.22 Å |
| 100 | 0.293 | 0.135 | 0.29 Å |
| 200 | 0.646 | 0.107 | 0.65 Å |

(uniform would be 0.100). **The type channel — exactly what Gumbel-ST is for —
carries signal only up to t ≈ 75, about 15% of the schedule.** Coordinates
survive to t ≈ 150–200. A trained model reproduces this: held-out Spearman runs
0.47 → 0.27 → 0.16 → 0.12 → ~0 across t = 25/50/75/100/150.

Two consequences for stage 2: concentrate guidance at low t, and weight the type
and coordinate gradient components **separately**, because their useful ranges
differ.

This also makes the "Tweedie is useless at moderate t" objection largely
self-solving. Trained to predict the *clean* molecule's SA from a corrupted
input, the Bayes-optimal output at high σ is just the marginal mean — so the
model learns a constant there and **its input gradient vanishes on its own**.
Guidance anneals itself instead of needing a hand-tuned cutoff.

### 5. Regression, not classification

SA is continuous, smooth, and cheap to label exactly. Discretising throws away
gradient magnitude, which is the entire signal stage 2 needs — a classifier says
"this bin vs that bin" where you need "which direction, how hard" — and puts a
staircase in the guidance gradient. A classification head and an auxiliary binned
head are both implemented (`model.head`, `model.aux_classification`) but off by
default.

### 6. Edges from distances, never from bonds

At stage 2 there is no bond graph, only coordinates. Edges are a cdist-based
radius graph mirroring `EGNNDynamics.get_edges` (`dynamics.py:163`), which also
avoids `torch_cluster` — absent from DiffSBDD's environment. A cosine envelope
drives each edge's contribution smoothly to zero at the cutoff; with a hard
cutoff, `d(score)/d(coords)` is discontinuous every time an atom crosses the
boundary.

Verified: E(3) invariance to 3.7e-08, reflection invariance exactly 0, and
gradients reach both `one_hot` and `pos`.

---

## The stage-2 contract

```python
from sa_surrogate.inference import SASurrogate
from sa_surrogate.diffusion_bridge import tweedie_residual_r

sur = SASurrogate("sa_surrogate/checkpoints/<run>/best.pt")

# xh_lig_hat is what EnVariationalDiffusion.forward already returns
# (en_diffusion.py:471, consumed by the LJ aux loss at lightning_modules.py:285)
score = sur.score_from_xh(xh_lig_hat, ligand["mask"], r_eff)   # [G], differentiable
loss  = loss + weight * score.mean()      # raw_sa: LOWER is better -> minimise
```

`sur.higher_is_better` states the sign, so stage 2 never has to guess.
`score_from_xh` applies the calibrated box integral and the Gumbel-ST internally
— the *same* functions used to corrupt training data, which is what keeps the
train and inference input distributions identical.

**Why the box integral rather than `softmax(ĥ₀)`.** DiffSBDD's h channel is not
logits and has no softmax anywhere. Handing `ĥ₀` to `gumbel_softmax` treats a
0.25-wide range as logits; Gumbel noise of scale ~1 swamps it and the "discrete"
sample is close to a uniform draw. You would still get a gradient — it just
would not mean anything. `categorical_logprobs_from_h` reuses the construction
already in the codebase at `en_diffusion.py:185`, with σ_t in place of σ₀.

### Calibrate `residual_scale` before relying on it

`tweedie_residual_r(r, residual_scale)` assumes the denoiser removes a fixed
fraction of the error. The 0.3 default is a placeholder. Measure
`‖x̂₀ − x₀‖ / r` on held-out data with the pretrained checkpoint and replace it,
otherwise the surrogate is told a noise level that does not match its input.

### Budget the two gradient channels separately

On an untrained fixture model, `‖g_types‖ / ‖g_coords‖ ≈ 300`. The absolute
number will differ once trained, but the asymmetry is structural: types enter
through a `Linear` directly, coordinates only through RBF-expanded distances
behind an envelope. Summing the raw reward gradient lets the type channel swamp
coordinate updates. Scale them independently and log the ratio.

---

## Adaptive gradient clipping at stage 2

`configure_gradient_clipping` (`lightning_modules.py:874`) sets
`max_grad_norm = 1.5·mean + 2·std` over a 50-step history. **It ratchets.** If
clipping is persistent the queue fills with `max_grad_norm` values, std → 0,
mean → M, so the threshold becomes 1.5·M; refill again → 2.25·M, unbounded —
×1.5 per 50 steps. This never fires today because the DDPM gradient is
well-behaved; a spiky Gumbel-ST reward gradient is exactly what would trigger it.

Recommended, in order:

1. **Split the backward and budget the reward gradient.** Take
   `g_ddpm = autograd.grad(ddpm_loss, params, retain_graph=True)` and
   `g_sa = autograd.grad(sa_loss, params)` separately, rescale
   `g_sa ← g_sa · min(1, ρ‖g_ddpm‖/‖g_sa‖)` with ρ ≈ 0.1–0.3, write
   `p.grad = g_ddpm + g_sa`. Costs one extra backward. **Log `‖g_sa‖/‖g_ddpm‖`
   every step** — it is the first number to look at when stage 2 misbehaves.
2. **Hard ceiling**: `max_grad_norm = min(1.5·mean + 2·std, cap)`. One line,
   kills the ratchet. Do this regardless.
3. **Reuse the repo's own ramp**: `WeightSchedule` (`lightning_modules.py:903`)
   already exists for the LJ term — same problem, already solved.
4. **Feed the queue `‖g_ddpm‖` only**, so the original stabilisation semantics
   survive unchanged.
5. **Fallback with no extra backward**: warm up with the SA weight at 0, let the
   queue converge, then freeze it.

---

## Known limitations

- **Unique-ligand count.** CrossDocked's ~100k training pairs collapse to far
  fewer unique molecules (the same ligand is docked into many pockets).
  Deduplication is by canonical SMILES, and any SMILES in both the official train
  and test splits is assigned to test only, so the test set stays honest — but
  the effective dataset is roughly an order of magnitude smaller than the pair
  count. Check the printed unique count after the build.
- **No off-manifold negatives.** The surrogate only ever sees real ligands and
  noised versions of them. Early in stage 2 the diffusion model emits geometry
  far off that manifold, where predictions are extrapolation. The σ-conditioning
  mitigates this (it learns to fall back to the marginal), but adding
  DDPM-generated molecules labelled with their true RDKit SA would be the real
  fix, and needs the pretrained checkpoint.
- **SA is a hackable reward.** It improves monotonically toward smaller,
  carbon-only, ring-free molecules, and molecule size is drawn from a fixed prior
  rather than being under gradient control. The accessible hack is "turn every
  heteroatom into carbon". Stage 2 needs a KL anchor to a frozen reference model,
  a small ramped weight, and joint monitoring of validity/QED/Vina.
- The fixture is plumbing-only: it cycles a fixed SMILES list with fresh
  conformers, so its train/val/test share molecules. Never read a generalisation
  number off it.
