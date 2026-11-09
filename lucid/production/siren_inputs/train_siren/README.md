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
cp ../../platform/templates/user_paths.s3df.sh.template   ../../platform/user_paths.sh   # on S3DF, or
cp ../../platform/templates/user_paths.lxplus.sh.template ../../platform/user_paths.sh   # on LXPLUS
vim ../../platform/user_paths.sh

# The generator runs inside the LUCiD container; -s submits from the host.
RUN=../../platform/run_in_container.sh

# 2. Prepare the scan (no submission). Copy example_water_mu.json first and set
#    baseline.h5_path to your lookup table.
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/example_water_mu.json

# 3. Submit a single test run (first entry in the runs list)
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/example_water_mu.json -t -s

# 4. Submit the full scan
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/example_water_mu.json -s
```

Runs whose `final_training_progress.pdf` already exists are skipped on
re-runs of the driver, so you can append new entries to the config and run
the same command without re-submitting the completed ones. Use
`--no-skip-existing` to force a re-emit.

## Config schema

```json
{
  "name": "example_water_mu",     // becomes <output_root>/<name>/

  "baseline": {                   // applied to every run unless overridden
    "h5_path":        "/path/to/photon_lookup_table.h5",   // required
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

  "output_root": "/path/to/training_tests"   // optional; default
                                  //   <SIREN_OUTPUT_BASE_PATH>/training_tests
                                  //   (else <OUTPUT_BASE_PATH>/training_tests)
}
```

`runs[*]` keys must be keys of `FLAG_MAP` in `../train_planning.py` (each maps
to a `lucid-train-siren` flag); an unknown override key raises at config-load
time, so a typo cannot silently drop a hyperparameter. Baseline keys outside
`FLAG_MAP` are not passed to the trainer. The material, particle and kind
(photon or dE/dx) come from the table at `h5_path`. A key left out falls back to
the trainer's default, the nominal recipe. To add a knob, extend `FLAG_MAP`
(CLI flag) and optionally `NAME_MAP` (short label for folder names) there.

## Nominal training prescription

The nominal recipe is [`lucid/siren/training/nominal.json`](../../../siren/training/nominal.json),
which is also `lucid-train-siren`'s default; read the values there. It has
been verified to converge for both photon and dE/dx tables, but seed yield is
imperfect (typically ~4–6 of 10 seeds converge and the rest diverge), so the
intended way to obtain one good model is a 10-seed scan plus `rank_seeds.py`.
No nominal scan config is committed; one is a baseline with only the table
and one run per seed (folders `seed0` … `seed9`, which `rank_seeds.py` reads):

```json
{
  "name": "water_mu_dedx_seed_scan",
  "baseline": {"h5_path": "/path/to/LUCiD/data/water/muon/dedx_lookup_table.h5"},
  "runs": [{"seed": 0}, {"seed": 1}, {"seed": 2}, {"seed": 3}, {"seed": 4},
           {"seed": 5}, {"seed": 6}, {"seed": 7}, {"seed": 8}, {"seed": 9}],
  "slurm": {"partition": "turing"}
}
```

Each run becomes one `lucid-train-siren --output-dir <run_dir> --h5-path <table>
--seed <N>` call in its submit description. `--output-dir` is required, and a
directory that already holds an `emitter.json` is refused. Use the generator
rather than calling the trainer by hand: it writes the submit descriptions and
per-run `config.json` and skips completed runs.

Standard end-to-end for a (material, particle) whose Stage-1
`training_inputs/<material>/<particle>/<E>MeV/photonsim.root` aggregates have
just been (re)generated:

```bash
# 1. Rebuild the lookup table from the fresh Stage-1 inputs.
apptainer exec -B /sdf/data/neutrino "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/lucid-build-dedx-table \
        --data-dir "$SIREN_OUTPUT_BASE_PATH/training_inputs" \
        --material water --particle mu- \
        --output   /path/to/LUCiD/data/water/muon/dedx_lookup_table.h5

# 2. Submit the 10-seed scan (the config above).
../../platform/run_in_container.sh lucid.production.siren_inputs.train_siren.generate_jobs \
    -c configs/water_mu_dedx_seed_scan.json -s

# 3. Once seeds finish, rank and promote.
apptainer exec -B /sdf/data/neutrino "$LUCID_IMAGE_PATH" /opt/conda/bin/python3 \
    rank_seeds.py <output_root>/water_mu_dedx_seed_scan
```

(For the photon-table variant, use `lucid-build-photon-table` and the photon
table at the same `training_inputs` root.)

## Output layout

```
<output_root>/<scan_name>/
  ├── baseline/
  │   ├── submit.sbatch
  │   ├── config.json
  │   ├── job-<slurm_id>.out / .err
  │   ├── recipe.json                       # the resolved training recipe
  │   ├── final_training_progress.pdf       # loss / LR / val-loss
  │   ├── training_history.json
  │   ├── prediction_plots/step_NNNNNN.pdf  # truth-vs-prediction PDFs
  │   └── emitter.json + weights.npz        # the emitter (recipe and losses recorded)
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

`rank_seeds.py` reads each `seed*/emitter.json` (`provenance.losses`), prints a table of
final train / val loss + best-ever val loss, and names the best seed (default ranking:
final val loss). Works for both photon and dE/dx scans. To promote a seed, copy its
`emitter.json` + `weights.npz` to `data/<material>/<particle>/<cherenkov|dedx>/` (a Cherenkov
emitter needs its t0 fit, the `t0.json` from PhotonSim's `PhotonSim/tools/t0_correction/calculate_t0.py`, passed as `--t0`; and, for the wavelength regime, an `emission_band_nm` in its `emitter.json`, see [siren-training-inputs.md](../../../../docs/guides/production/siren-training-inputs.md)).

Each run's `final_training_progress.pdf` (loss / LR / val-loss) and
`prediction_plots/step_*.pdf` show how it converged across the
(E, angle, s/s_max) grid.

## Defaults vs. PhotonSim jobs

The resource defaults in `../../platform/user_paths.sh` are set for the
PhotonSim jobs, so this tool uses its own SLURM defaults (`SLURM_DEFAULTS` in
`generate_jobs.py`: 1 GPU, 4 h, 32 GB, 4 CPU, partition `roma`, account
`mli:cider-ml`). `roma` is a CPU partition on S3DF, so a training scan needs a
GPU partition in the config's `slurm` block (`example_water_mu.json` sets
`turing`). Override any of them there.
