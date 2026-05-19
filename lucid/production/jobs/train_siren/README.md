# SIREN training scans (Stage 3)

Cluster fan-out (SLURM or HTCondor — see
[`../../../../docs/CLUSTER_ABSTRACTION.md`](../../../../docs/CLUSTER_ABSTRACTION.md))
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
python3 generate_jobs.py -c configs/water_mu_v1.json

# 3. Submit a single test run (first entry in the runs list)
python3 generate_jobs.py -c configs/water_mu_v1.json -t -s

# 4. Submit the full scan
python3 generate_jobs.py -c configs/water_mu_v1.json -s
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

  "output_root": "/sdf/data/neutrino/cjesus/SIREN_files/training_tests"
}
```

`baseline` and `runs[*]` keys must match `FLAG_MAP` in `generate_jobs.py` —
unknown keys raise at config-load time so typos don't silently drop
hyperparameters. To add a new knob, extend `FLAG_MAP` (CLI flag) and optionally
`NAME_MAP` (short label for folder names) in `generate_jobs.py`.

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

Once the scan finishes, eyeball every loss plot side-by-side from a notebook:

```python
from pathlib import Path
from pdf2image import convert_from_path
import matplotlib.pyplot as plt

root = Path('/sdf/data/neutrino/cjesus/SIREN_files/training_tests/water_mu_v1')
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
