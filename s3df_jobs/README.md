# S3DF Job Scripts

Scripts for running optimization jobs on S3DF SLURM cluster.

## Container

All jobs use the Singularity container:
```
/sdf/group/neutrino/images/develop.sif
```

To run commands manually with the container:
```bash
singularity exec -B /sdf,/fs,/sdf/scratch,/lscratch /sdf/group/neutrino/images/develop.sif python3 your_script.py
```

---

## Track Optimization Jobs

### 1. Create config files

Edit and run the config creation script (e.g., `nrays_config/create_configs.py`):

```bash
cd s3df_jobs/nrays_config
python3 create_configs.py
```

### 2. Submit jobs

Submit each config as a separate job:

```bash
cd s3df_jobs
for i in 0 1 2 3 4 5; do
  python3 submit_job.py \
    --config nrays_config/opt_config_${i}.json \
    --output /path/to/output/dir \
    --job-name myjob_${i} \
    --submit
done
```

Output pkl files will be saved directly to the output directory as `config_X.pkl`.

---

## Tau Hyperparameter Tuning

The `tau_vtx` parameter in the vertex loss function depends on Nrays and Energy.
The optimal value is determined through a hyperparameter scan and fitting procedure.

### Workflow

#### 1. Run Hyperparameter Scan

```bash
python submit_tau_hyperparameter_tuning_job.py --output ../output/tau_scan --submit
```

This scans `tau_vtx` values across (Nrays, Energy) combinations and saves results to CSV.

**Scan grid:**
- Nrays: 50k, 150k, 250k
- Energy: 500, 1000, 1500 MeV
- tau_vtx: values adapted per (Nrays, Energy) combination

Total: 40 configurations with bootstrap confidence intervals.

#### 2. Analyze Results

Open and run `good_notebooks/analyze_tau_scan.ipynb` to:
- Visualize position error vs tau_vtx for each (Nrays, Energy)
- Find optimal tau_vtx per combination
- Fit weighted linear parametrization: `tau_vtx = a*Nrays + b*E + c`
- Get updated coefficients to paste into `tools/optimization/losses.py`

#### 3. Update Centralized Parameters

After fitting, update the coefficients in `tools/optimization/losses.py`:
```python
TAU_VTX_PARAM_A = ...  # coefficient for Nrays
TAU_VTX_PARAM_B = ...  # coefficient for Energy (MeV)
TAU_VTX_PARAM_C = ...  # intercept
```

#### 4. Use Parametrization

```python
from lucid.losses import get_optimal_tau_vtx

tau = get_optimal_tau_vtx(nrays=150000, energy_mev=1000)
```

The function can also be used inside JAX-traced code with `jax.lax.stop_gradient`
on the energy for dynamic tau_vtx during optimization.

---

## Scripts Overview

| Script | Description |
|--------|-------------|
| `submit_tau_hyperparameter_tuning_job.py` | Run tau_vtx hyperparameter scan |
| `submit_eval_with_parametrization.py` | Evaluate reconstruction with dynamic tau_vtx |
| `run_eval_with_parametrization.py` | Worker script for parametrized evaluation |
| `submit_job.py` | Generic job submission utility |
| `run_track_optimization.py` | Single track optimization |

---

## SLURM Usage

All submit scripts support:
- `--output DIR` - Output directory for results
- `--submit` - Actually submit (otherwise just generates script)
- `--partition` - SLURM partition (default: ampere)
- `--account` - SLURM account (default: mli:cider-ml)
- `--time` - Time limit HH:MM:SS

Example:
```bash
python submit_tau_hyperparameter_tuning_job.py \
    --output ../output/tau_scan \
    --n-events 25 \
    --time 12:00:00 \
    --submit
```

Monitor jobs:
```bash
squeue -u $USER
tail -f s3df_jobs/logs/<job_name>_<timestamp>.log
```
