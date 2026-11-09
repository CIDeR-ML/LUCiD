# SIREN training sweeps

Cluster fan-out for `lucid-train-siren` (Slurm or HTCondor; see
[cluster abstraction](../../../../docs/guides/production/cluster-abstraction.md)), one GPU job
per run. A scan config gives a baseline and a list of per-run overrides; the generator writes one
run directory and submit file per run under `<output_root>/<scan_name>/<run_name>/`, naming each
run after its difference from the baseline. The training command itself is described in
[`lucid/siren/README.md`](../../../siren/README.md), the whole pipeline in
[Training a SIREN emitter](../../../../docs/guides/production/siren-training-inputs.md).

## Quick start

```bash
# 1. Paths (shared by all stages)
cp ../../platform/templates/user_paths.s3df.sh.template ../../platform/user_paths.sh   # or .lxplus
vim ../../platform/user_paths.sh

# The generator runs in the container; -s submits from the host.
RUN=../../platform/run_in_container.sh

# 2. Prepare a seed scan, submit its first run as a test, then the whole scan
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/water_mu_cherenkov_seeds.json
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/water_mu_cherenkov_seeds.json -t -s
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/water_mu_cherenkov_seeds.json -s
```

Runs whose `final_training_progress.pdf` exists are skipped, so entries can be appended to
`runs` and the same command rerun; `--no-skip-existing` regenerates them. `--output-root`
overrides the output root, `-P` the partition.

## Site settings

From `user_paths.sh`; a missing one is an error that names it.

| setting | use |
|---|---|
| `LUCID_IMAGE_PATH` | the container |
| `LUCID_ENV_BASE` | the Python user site with the GPU jaxlib (the image's is CPU-only), exported as `PYTHONUSERBASE`; the template says how to make one |
| `SIREN_TRAIN_PARTITION`, `SLURM_ACCOUNT` | Slurm: the GPU partition and the account |
| `CONDOR_TRAIN_FLAVOUR` | HTCondor: the job flavour (default `testmatch`) |
| `LUCID_DEV_PATH` | optional: run this checkout's trainer, bound over `/opt/LUCiD`; unset, the image's |
| `SIREN_OUTPUT_BASE_PATH` | the output root is `<it>/training_tests` (else `OUTPUT_BASE_PATH`) |
| `SIREN_INPUTS_BASE_PATH` | optional: what `${SIREN_INPUTS_BASE_PATH}` in a config means; default `SIREN_OUTPUT_BASE_PATH` |

## Config

```json
{
  "name": "my_scan",                // becomes <output_root>/my_scan/
  "baseline": {                     // applied to every run
    "h5_path":        "${SIREN_INPUTS_BASE_PATH}/training_inputs/water/lookup_tables/muon/photon_lookup_table.h5",
    "num_steps":      30000,
    "patience":       40,
    "zero_threshold": 1e-2
  },
  "runs": [                         // one run per entry
    {},                             //   -> baseline/
    {"patience": 100},              //   -> p100/
    {"zero_threshold": 1e-4}        //   -> z1e-4/
  ],
  "resources": {"time": "08:00:00"},       // optional: time, memory, cpus, gpus, flavour
  "output_root": "/path/to/training_tests" // optional
}
```

- `h5_path` is required; it may name `user_paths.sh` settings as `${VAR}`, and must exist. The
  material, particle and kind (photon or dE/dx) come from the table.
- Each key maps to a `lucid-train-siren` flag through `FLAG_MAP` in `../train_planning.py`. An
  override key not in `FLAG_MAP` raises; a baseline key not in it is ignored. Keys left out take
  the trainer's default, the nominal recipe (`lucid/siren/training/nominal.json`). To add a key,
  extend `FLAG_MAP` and, for a short folder label, `NAME_MAP`.
- `resources` defaults to `RESOURCES` in `generate_jobs.py`: 4 h, 32 GB, 4 CPUs, 1 GPU (150k
  steps took 30 min on a turing GPU). Partition and account are the site's, never the config's.
- Tracked here: `example_water_mu.json` (a hyperparameter scan) and the four production seed
  scans, `water_{mu,el}_{cherenkov,dedx}_seeds.json`. Other configs in `configs/` stay untracked.

## Training a production emitter

Not every seed of the nominal recipe converges, so a production emitter is the best of a seed
scan: a baseline with only the table and one run per seed (folders `seed0` ... `seed9`, which
`rank_seeds.py` reads). The four `*_seeds.json` are the scans the shipped water emitters were
chosen from.

```bash
# 1. Submit the seed scan (the tables: stage 2 of the pipeline)
$RUN lucid.production.siren_inputs.train_siren.generate_jobs -c configs/water_mu_cherenkov_seeds.json -s

# 2. Rank the seeds
apptainer exec -B "$APPTAINER_BINDS" "$LUCID_IMAGE_PATH" /opt/conda/bin/python3 \
    rank_seeds.py $SIREN_OUTPUT_BASE_PATH/training_tests/water_mu_cherenkov_seeds [--sort final_val|best_val|final_train]

# 3. Install the best one (a Cherenkov emitter with its t0 fit and band)
lucid-siren-promote --run $SIREN_OUTPUT_BASE_PATH/training_tests/water_mu_cherenkov_seeds/seed1 \
    --smax-fit $PHOTONSIM_DEV_PATH/data/water/mu-/smax_fit.csv \
    --t0 $SIREN_OUTPUT_BASE_PATH/t0_parametrization/water/mu-/timing_parameters.json \
    --emission-band-nm 274.91 673.83 --selection "water_mu_cherenkov_seeds/seed1: lowest final val_loss"
```

`rank_seeds.py` reads each `seed*/emitter.json` (`provenance.losses`), prints the final train,
final validation and best validation losses, and names the best seed (by final validation loss
unless `--sort` says otherwise). The t0 fit is the t0 stage's ([`../t0/`](../t0/README.md)); a
dE/dx emitter takes neither `--t0` nor `--emission-band-nm`.

## Output

```
<output_root>/<scan_name>/
  baseline/
    submit.sbatch
    config.json                         # the resolved settings and resources of this run
    job-<jobid>.out/.err
    recipe.json                         # the resolved training recipe
    training_history.json
    final_training_progress.pdf         # loss, learning rate, validation loss
    prediction_plots/step_NNNNNN.pdf    # prediction vs table over (E, angle, s/s_max)
    emitter.json, weights.npz           # the run, with its recipe and losses (no t0, no band)
  p100/
  z1e-4/
```
