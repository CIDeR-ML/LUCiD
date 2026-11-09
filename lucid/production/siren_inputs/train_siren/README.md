# SIREN training sweeps

Cluster fan-out for `lucid-train-siren` (Slurm or HTCondor; see
[cluster abstraction](../../../../docs/guides/production/cluster-abstraction.md)). A scan config
gives a baseline and a list of per-run overrides; the generator writes one run directory and
submit file per run under `<output_root>/<scan_name>/<run_name>/`, naming each run after its
difference from the baseline. The training command itself is described in
[`lucid/siren/README.md`](../../../siren/README.md).

## Quick start

```bash
# 1. Paths (shared by all stages)
cp ../../platform/templates/user_paths.s3df.sh.template ../../platform/user_paths.sh   # or .lxplus
vim ../../platform/user_paths.sh

# The generator runs in the container; -s submits from the host.
RUN=../../platform/run_in_container.sh

# 2. Copy configs/example_water_mu.json, set baseline.h5_path to your lookup table, and prepare
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/my_scan.json

# 3. Submit only the first run as a test, then the whole scan
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/my_scan.json -t -s
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/my_scan.json -s
```

Runs whose `final_training_progress.pdf` exists are skipped, so entries can be appended to
`runs` and the same command rerun; `--no-skip-existing` regenerates them. `--output-root`
overrides the output root.

## Config

```json
{
  "name": "my_scan",                // becomes <output_root>/my_scan/
  "baseline": {                     // applied to every run
    "h5_path":        "/path/to/photon_lookup_table.h5",   // required
    "num_steps":      30000,
    "patience":       40,
    "zero_threshold": 1e-2
  },
  "runs": [                         // one run per entry
    {},                             //   -> baseline/
    {"patience": 100},              //   -> p100/
    {"zero_threshold": 1e-4}        //   -> z1e-4/
  ],
  "slurm": {"partition": "turing"}, // optional; overrides SLURM_DEFAULTS
  "output_root": "/path/to/training_tests"   // optional
}
```

- Each key maps to a `lucid-train-siren` flag through `FLAG_MAP` in `../train_planning.py`. An
  override key not in `FLAG_MAP` raises; a baseline key not in it is ignored. Keys left out take
  the trainer's default, the nominal recipe (`lucid/siren/training/nominal.json`). To add a key,
  extend `FLAG_MAP` and, for a short folder label, `NAME_MAP`.
- The material, particle and kind (photon or dE/dx) come from the table at `h5_path`.
- `slurm` defaults to `SLURM_DEFAULTS` in `generate_jobs.py`: partition `roma`, account
  `mli:cider-ml`, 4 h, 32 GB, 4 CPUs, 1 GPU. `roma` is a CPU partition on S3DF, so set a GPU
  partition (the example config uses `turing`).
- `output_root` defaults to `$SIREN_OUTPUT_BASE_PATH/training_tests`, else
  `$OUTPUT_BASE_PATH/training_tests`.

## Training a production emitter

Not every seed of the nominal recipe converges, so the way to get one good emitter is a scan
over seeds and a ranking. A seed scan is a baseline with only the table and one run per seed
(folders `seed0` ... `seed9`, which `rank_seeds.py` reads):

```json
{
  "name": "water_mu_dedx_seed_scan",
  "baseline": {"h5_path": "/path/to/LUCiD/data/water/muon/dedx_lookup_table.h5"},
  "runs": [{"seed": 0}, {"seed": 1}, {"seed": 2}, {"seed": 3}, {"seed": 4},
           {"seed": 5}, {"seed": 6}, {"seed": 7}, {"seed": 8}, {"seed": 9}],
  "slurm": {"partition": "turing"}
}
```

After regenerating the training inputs (`training_inputs/<material>/<particle>/<E>MeV/photonsim.root`):

```bash
# 1. Build the lookup table
apptainer exec -B /sdf/data/neutrino "$LUCID_IMAGE_PATH" \
    /opt/conda/bin/lucid-build-dedx-table \
        --data-dir "$SIREN_OUTPUT_BASE_PATH/training_inputs" \
        --material water --particle mu- \
        --output   /path/to/LUCiD/data/water/muon/dedx_lookup_table.h5

# 2. Submit the seed scan
../../platform/run_in_container.sh lucid.production.siren_inputs.train_siren.generate_jobs \
    -c configs/water_mu_dedx_seed_scan.json -s

# 3. Rank the seeds
apptainer exec -B /sdf/data/neutrino "$LUCID_IMAGE_PATH" /opt/conda/bin/python3 \
    rank_seeds.py <output_root>/water_mu_dedx_seed_scan [--sort final_val|best_val|final_train]
```

For a photon table use `lucid-build-photon-table` instead. `rank_seeds.py` reads each
`seed*/emitter.json` (`provenance.losses`), prints the final train, final validation and best
validation losses, and names the best seed (by final validation loss unless `--sort` says
otherwise).

To install the chosen seed, copy its `emitter.json` and `weights.npz` to
`data/<material>/<particle>/<cherenkov|dedx>/`. The generator has no key for `--t0`, so a
Cherenkov emitter trained here carries no t0 fit, and the simulator refuses it until one is
added: a `t0` block in its `emitter.json`, in the form `lucid-train-siren --t0` writes from the
`t0.json` of `PhotonSim/tools/t0_correction/calculate_t0.py` (`_t0_block` in
`lucid/siren/training/trainer.py`). A Cherenkov emitter used in wavelength mode
also needs `emission_band_nm` in its `emitter.json`
([SIREN training inputs](../../../../docs/guides/production/siren-training-inputs.md)).

## Output

```
<output_root>/<scan_name>/
  baseline/
    submit.sbatch
    config.json                         # the resolved settings of this run
    job-<jobid>.out/.err
    recipe.json                         # the resolved training recipe
    training_history.json
    final_training_progress.pdf         # loss, learning rate, validation loss
    prediction_plots/step_NNNNNN.pdf    # prediction vs table over (E, angle, s/s_max)
    emitter.json, weights.npz           # the emitter, with its recipe and losses
  p100/
  z1e-4/
```
