# SIREN training scans (Stage 3)

Cluster fan-out (SLURM or HTCondor — see
[`../../../../docs/guides/production/cluster-abstraction.md`](../../../../docs/guides/production/cluster-abstraction.md))
for `lucid-train-siren` hyperparameter sweeps. Each scan config lists a
baseline + a list of explicit per-run overrides; the driver materializes
one output sub-folder + submit description per run under
`<output_root>/<scan_name>/<run_name>/`. Sub-folder names are auto-derived
from each run's diff against the baseline so the file tree is
self-documenting.

## Quick start

```bash
# 1. Configure user paths (shared across all stages).
cp ../user_paths.s3df.sh.template   ../user_paths.sh   # on S3DF, or
cp ../user_paths.lxplus.sh.template ../user_paths.sh   # on LXPLUS
vim ../user_paths.sh
# Only LUCID_DEV_PATH and LUCID_IMAGE_PATH are read here.

# 2. Prepare the scan (no submission)
./generate_jobs.sh -c configs/water_mu_v1.json

# 3. Submit a single test run (first entry in the runs list)
./generate_jobs.sh -c configs/water_mu_v1.json -t -s

# 4. Submit the full scan
./generate_jobs.sh -c configs/water_mu_v1.json -s
```

Runs whose `final_training_progress.pdf` already exists are skipped on
re-runs of the driver, so you can append new entries to the config and run
the same command without re-submitting the completed ones. Use
`--no-skip-existing` to force a re-emit.

## Config schema

```json
{
  "name": "water_mu_v1",          // becomes <output_root>/<name>/

  "baseline": {                   // applied to every run unless overridden
    "material":       "water",
    "particle":       "muon",
    "data_type":      "photon",
    "h5_path":        "/sdf/.../photon_lookup_table.h5",
    "num_steps":      30000,
    "learning_rate":  2e-4,
    "min_lr":         1e-6,
    "batch_size":     65536,
    "patience":       20,
    "zero_threshold": 1e-2,
    "val_split":      0.01
  },

  "runs": [                       // explicit list of override dicts
    {},                           //   -> folder "baseline"
    {"patience": 40},             //   -> folder "p40"
    {"zero_threshold": 1e-4}      //   -> folder "z1e-04"
  ],

  "slurm": {                      // optional, defaults shown
    "partition": "roma",
    "account":   "mli:cider-ml",
    "time":      "04:00:00",
    "memory":    "32000",
    "cpus":      "4",
    "gpus":      "1"
  },

  "output_root": "/sdf/data/neutrino/cjesus/CIDER/SIREN_files/training_tests"
}
```

`baseline` and `runs[*]` keys must match `FLAG_MAP` in `generate_jobs.py` —
unknown keys raise at config-load time so typos don't silently drop
hyperparameters. To add a new knob, extend `FLAG_MAP` (CLI flag) and optionally
`NAME_MAP` (short label for folder names) in `generate_jobs.py`.

## Nominal training prescription

When the task is "(re)train SIREN for `<material>/<particle>`, photon or
dE/dx", use the **nominal** scan configs:

| Config | Lookup table | Output folder |
|---|---|---|
| `water_mu_dedx_seed_scan_nominal.json` | `LUCiD/data/water/muon/dedx_lookup_table.h5` | `<output_root>/water_mu_dedx_seed_scan_nominal/` |
| (older suffix-named configs `..._p100_ti02_n150k.json` carry the same hyperparameters and are kept around for historical scans.) | | |

These all share one baseline that has been verified to converge for both
photon and dE/dx tables. A scan runs **10 seeds** on the `turing` partition;
seed yield is imperfect — typically ~4–6 of 10 seeds converge and the rest
diverge — so the 10-seed fan-out plus `rank_seeds.py` (below) is the intended
way to obtain one good model.

Baseline (json form, as the configs encode it):

```json
"baseline": {
  "num_steps":         150000,
  "learning_rate":     1e-4,
  "min_lr":            5e-7,
  "patience":          100,
  "grad_clip_norm":    1.0,
  "zero_threshold":    1e-3,
  "zero_keep_frac":    0.5,
  "energy_balance":    "uniform",
  "target_importance": 0.2,
  "batch_size":        25000,
  "val_split":         0.01
}
```

Equivalent `lucid-train-siren` CLI invocation for a single seed (this is
what `generate_jobs.py` writes into each `submit.sbatch` under the hood):

```bash
lucid-train-siren \
    --material water --particle muon --data-type dedx \
    --h5-path /sdf/.../LUCiD/data/water/muon/dedx_lookup_table.h5 \
    --num-steps 150000 \
    --learning-rate 1e-4 --min-lr 5e-7 \
    --patience 100 --grad-clip-norm 1.0 \
    --zero-threshold 1e-3 --zero-keep-frac 0.5 \
    --energy-balance uniform --target-importance 0.2 \
    --batch-size 25000 --val-split 0.01 \
    --seed <N>
```

`generate_jobs.py` is the recommended path — it materialises the 10 seeds,
the SLURM/HTCondor `submit.sbatch`, the per-run `config.json`, and skips
already-completed runs automatically. Bypass it only for one-off debugging.

Standard end-to-end for a (material, particle) where the upstream
`training_inputs/<material>/<particle>/<E>MeV/photonsim.root` aggregates
have just been (re)generated:

```bash
# 1. Rebuild the lookup table from the fresh Stage-1 inputs.
apptainer exec -B /sdf/data/neutrino "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/lucid-build-dedx-table \
        --data-dir /sdf/data/neutrino/cjesus/CIDER/SIREN_files/training_inputs \
        --material water --particle mu- \
        --output   /sdf/.../LUCiD/data/water/muon/dedx_lookup_table.h5

# 2. Submit the 10-seed nominal scan.
./generate_jobs.sh -c configs/water_mu_dedx_seed_scan_nominal.json -s

# 3. Once seeds finish, rank and promote.
apptainer exec -B /sdf/data/neutrino "$LUCID_IMAGE_PATH" /opt/conda/bin/python3 \
    rank_seeds.py <output_root>/water_mu_dedx_seed_scan_nominal
```

(For the photon-table variant, swap `dedx` → `photon`, use the matching
photon scan config, and point `lucid-build-photon-table` at the same
`training_inputs` root.)

## Output layout

```
<output_root>/<scan_name>/
  ├── baseline/
  │   ├── submit.sbatch
  │   ├── config.json
  │   ├── job-<slurm_id>.out / .err
  │   ├── final_training_progress.pdf       # loss / LR / val-loss
  │   ├── training_history.json
  │   ├── prediction_plots/step_NNNNNN.pdf  # truth-vs-prediction PDFs
  │   ├── checkpoint_step_NNNNNN.npz
  │   └── trained_model/photonsim_siren_{weights.npz,metadata.json}
  ├── p40/                                  # patience=40 run
  │   └── …same shape…
  └── z1e-04/                               # zero_threshold=1e-4 run
      └── …same shape…
```

`config.json` in each run dir records the resolved settings (baseline +
overrides) so you can always recover what produced the artifacts there.

## Comparing runs

Once the scan finishes, rank the seeds by loss to pick the model to promote:

```bash
apptainer exec -B /sdf/data/neutrino "$LUCID_IMAGE_PATH" /opt/conda/bin/python3 \
    rank_seeds.py <output_root>/<scan_name> [--sort final_val|best_val|final_train]
```

`rank_seeds.py` reads each `seed*/trained_model/*_metadata.json` and
`training_history.json`, prints a table of final train / val loss + best-ever
val loss, and names the best seed (default ranking: final val loss). Works for
both photon and dE/dx scans.

To eyeball every loss plot side-by-side from a notebook instead:

```python
from pathlib import Path
from pdf2image import convert_from_path
import matplotlib.pyplot as plt

root = Path('/sdf/data/neutrino/cjesus/CIDER/SIREN_files/training_tests/water_mu_v1')
for run in sorted(root.iterdir()):
    img = convert_from_path(run / 'final_training_progress.pdf', dpi=120)[0]
    plt.figure(figsize=(8, 4)); plt.imshow(img); plt.title(run.name); plt.axis('off')
```

Likewise, sweep through `prediction_plots/step_*.pdf` to see how each run's
SIREN converges across the (E, angle, s/s_max) grid.

## Defaults vs. PhotonSim jobs

The CPU-tuned defaults in `../user_paths.sh` (`DEFAULT_GPUS=0`,
`DEFAULT_TIME=23:00:00`, `DEFAULT_MEMORY=39936`) are wrong for training, so
this tool hardcodes its own SLURM defaults (1 GPU, 4 h, 32 GB, 4 CPU,
partition `roma`). Override any of them via the config's `slurm` block.
