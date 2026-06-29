# Decaf Prediction and Evaluation

Guide for running Decaf predictions and evaluation metrics on the PoseBusters benchmark using the boltz codebase.

## Setup

### Prerequisites

- Conda environment `boltz` with boltz installed in editable mode
- Decaf checkpoint at `/tmp/decaf_ckpt.ckpt`
- Boltz1 checkpoint at `~/.boltz/boltz1_conf.ckpt`
- PoseBusters benchmark inputs at `/path/to/queries/` (YAML files)
- Reference structures at `/path/to/ref/` (CIF files)
- OpenStructure (`ost`) installed in the boltz env

### Activate environment

```bash
conda activate boltz
```

## Running Predictions

### Single GPU

```bash
python -m boltz.main predict \
    /path/to/queries \
    --checkpoint /tmp/decaf_ckpt.ckpt \
    --model boltz1 \
    --sampling_steps 10 \
    --diffusion_samples 5 \
    --recycling_steps 3 \
    --accelerator gpu \
    --out_dir /path/to/output \
    --no_kernels
```

### Multi-GPU parallel (recommended)

The `run_parallel_predict.sh` script splits inputs across GPUs:

```bash
PYTHON=/path/to/python \
CHECKPOINT=/path/to/checkpoint \
bash scripts/run_parallel_predict.sh \
    <input_dir> <output_dir> <num_gpus> <sampling_steps> <diffusion_samples> [--use_potentials]
```

#### Decaf SDE (10 steps, 5 samples)

```bash
PYTHON=$(which python) \
CHECKPOINT=/tmp/decaf_ckpt.ckpt \
bash scripts/run_parallel_predict.sh \
    /path/to/queries \
    /path/to/output_decaf_sde \
    4 10 5
```

#### Decaf SDE with FK steering

```bash
PYTHON=$(which python) \
CHECKPOINT=/tmp/decaf_ckpt.ckpt \
bash scripts/run_parallel_predict.sh \
    /path/to/queries \
    /path/to/output_decaf_fk \
    4 10 5 --use_potentials
```

#### Decaf with MC-GRAD (reward-aligned)

```bash
PYTHON=$(which python) \
CHECKPOINT=/tmp/decaf_ckpt.ckpt \
bash scripts/run_parallel_predict.sh \
    /path/to/queries \
    /path/to/output_decaf_mc_grad \
    4 10 5 --use_potentials --use_mc_grad --mc_grad_particles 4 --mc_grad_snr_ratio 10.0
```

#### Boltz1 baseline (200 steps)

```bash
PYTHON=$(which python) \
CHECKPOINT=~/.boltz/boltz1_conf.ckpt \
bash scripts/run_parallel_predict.sh \
    /path/to/queries \
    /path/to/output_boltz1 \
    4 200 5
```

#### MC-GRAD sampler (requires Decaf checkpoint + `--use_potentials`)

MC-GRAD (Holderrieth et al., 2025) corrects the Jensen gap bias in standard
FK steering. The standard approach approximates the value function as
`V(x_t) ≈ r(D_t(x_t))`, which is biased because it pushes the reward inside
a non-linear denoiser call. MC-GRAD corrects this by generating K renoised
particles at a slightly higher noise level σ' = √λ·σ_t and computing an
importance-weighted average:

1. **Renoise K times**: sample `z_k = x_t + √(σ'²−σ_t²) · ε_k` for k=1..K
2. **Flow map to x0**: `x0_k = z_k − σ' · φ(z_k, σ')` via the Decaf denoiser
3. **Importance weight**: logit for particle k is
   `v_k = r(x0_k) − ‖x_t−x0_k‖²/(2σ_t²) + γ_k + ½‖ε_k‖²`
   where `γ_k` is a score correction term (controlled by `--mc_grad_score_correction`)
4. **Weighted x0**: `x0_corrected = Σ_k softmax(v)[k] · x0_k`

The corrected x0 replaces the single-point denoiser estimate in the FK/SDE step.

```bash
python -m boltz.main predict \
    /path/to/queries \
    --checkpoint /tmp/decaf_ckpt.ckpt \
    --model boltz1 \
    --sampling_steps 10 \
    --diffusion_samples 5 \
    --recycling_steps 3 \
    --accelerator gpu \
    --out_dir /path/to/output \
    --no_kernels \
    --use_potentials \
    --use_mc_grad \
    --mc_grad_particles 4 \
    --mc_grad_snr_ratio 10.0
```

**MC-GRAD CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--use_mc_grad` | off | Enable Weighted MC-GRAD sampler (requires `--use_potentials`) |
| `--mc_grad_particles` | 4 | Number of particles K. Higher = lower variance but K× slower per step. |
| `--mc_grad_snr_ratio` | 10.0 | SNR ratio λ controlling the renoise level: σ' = √λ·σ_t. |
| `--no_mc_grad_gd` | off | Disable iterative GD guidance steps after mc_grad correction (pure mc_grad only). |

MC-GRAD improves the *reward estimator* — it corrects the Jensen gap in FK
steering by importance-weighting K particles, and requires `--use_potentials`.

### Output structure

```
output_dir/
  predictions/
    {target_name}/
      {target_name}_model_0.cif
      {target_name}_model_1.cif
      ...
      confidence_{target_name}_model_0.json
      ...
```

### Detecting Decaf vs Boltz1

Check the GPU logs for the line:
```
Detected Decaf checkpoint -- using DecafSampler for inference.
```
If absent, the run used the Boltz1 standard sampler.

## Evaluation Metrics

### 1. OpenStructure structural metrics (LDDT, TM-score, DockQ, RMSD, LDDT-PLI)

Uses `ost compare-structures` and `ost compare-ligand-structures`:

```bash
python scripts/eval/run_evals_local.py \
    /path/to/predictions \
    /path/to/reference_cifs \
    /path/to/evals_output \
    --num-samples 5
```

**Important:** The script uses the `ost` binary from the boltz env at
`ost`. If evals produce zero
output files, check that `ost` is accessible.

Reference CIFs are matched by PDB ID (first part of target name before `_`).
For example, target `5sb2_1k2` matches reference `5sb2.cif.gz`.

### 2. PoseBusters physical validity (pb_valid)

Runs PoseBusters checks in `redock_no_strain` mode (`PosebustersMode.REDOCK_NO_STRAIN`):

```bash
python scripts/eval/run_posebusters.py \
    --predictions /path/to/predictions \
    --queries /path/to/queries \
    --ref /path/to/reference_cifs \
    --num-samples 5 \
    --output /path/to/pb_results.csv
```

**Options:**
- `--ref`: Reference CIF directory. Enables `redock_no_strain` mode with identity checks (molecular_formula, molecular_bonds, double_bond_stereochemistry). Without `--ref`, falls back to `dock_no_strain`.
- `--benchmark`: PoseBusters benchmark directory with pre-split `{PDBID_CCD}_protein.pdb` and `{PDBID_CCD}_ligand.sdf` files. Uses these for protein context instead of extracting from prediction CIF. **Warning:** benchmark PDBs include crystal context (waters, cofactors) which can inflate failures.
- `--tools`: Compare multiple tools: `--tools boltz1:/path/a decaf:/path/b`

**pb_valid definition:**
- `pb_valid = 1.0` iff all `pbcheck_*` pass
- Excludes `mol_true_loaded`, `tetrahedral_chirality`, and `rmsd_<=_2a` from failure count

## Evaluation Workflow Example

Full workflow for a Decaf SDE run on PoseBusters pb_202109:

```bash
# 1. Predict
PYTHON=$(which python) \
CHECKPOINT=/tmp/decaf_ckpt.ckpt \
bash scripts/run_parallel_predict.sh \
    /path/to/data/pb_202109/queries \
    /path/to/data/pb_202109_decaf_sde \
    4 10 5

# 2. OST evals
python scripts/eval/run_evals_local.py \
    /path/to/data/pb_202109_decaf_sde/predictions \
    /path/to/data/pb_202109/ref \
    /path/to/data/pb_202109_decaf_sde_evals \
    --num-samples 5

# 3. PoseBusters
python scripts/eval/run_posebusters.py \
    --predictions /path/to/data/pb_202109_decaf_sde/predictions \
    --queries /path/to/data/pb_202109/queries \
    --ref /path/to/data/pb_202109/ref \
    --num-samples 5 \
    --output /path/to/data/pb_posebusters_decaf_sde.csv
```

## Key Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| LDDT | OST | Local Distance Difference Test (0-1, higher is better) |
| BB-LDDT | OST | Backbone LDDT |
| TM-score | OST | Template Modeling score (0-1) |
| RMSD | OST | Root Mean Square Deviation (lower is better) |
| DockQ | OST | Docking quality (0-1) |
| LDDT-PLI | OST | Protein-Ligand Interaction LDDT |
| L-RMSD | OST | Ligand RMSD |
| pb_valid | PoseBusters | Physical validity (0 or 1 per sample) |

**Aggregation modes:**
- **Single (top-1):** metrics from `model_0` only
- **Oracle (best-of-K):** best across K samples (max for most metrics, min for RMSD)
- **Average:** mean across K samples

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/run_parallel_predict.sh` | Multi-GPU parallel prediction |
| `scripts/eval/run_evals_local.py` | OpenStructure structural evals |
| `scripts/eval/run_posebusters.py` | PoseBusters physical validity checks |
| `scripts/eval/aggregate_evals.py` | General eval aggregation |
