# Command-line tools

`pip install -e .` installs five commands. The scripts in `scripts/` run from the repository
root with `python scripts/<name>.py`. Every tool prints its full option list with `--help`.

## `lucid-run-job`

Produces one batch of a dataset: writes a GEANT4 macro, runs PhotonSim (the binary named by
`$PHOTONSIM_BIN`), then the LUCiD simulation, digitizer and writer. A config with
`"primary_source": "genie"` runs GENIE (`gevgen`, `gntpc`) first; a supernova config runs
sntools first. Guides: [local](../guides/production/local.md),
[container](../guides/production/container.md).

```bash
export PHOTONSIM_BIN=/path/to/PhotonSim/build/PhotonSim
lucid-run-job --config lucid/production/dataprod/configs/GeV/02_mu.json \
              --detector SK_WAND --n-events 1000 --job-id 1 --master-seed 42 --output-dir out/
```

| flag | meaning |
|---|---|
| `--config` | required. The dataset config (JSON) |
| `--output-dir` | required. The dataset directory; the batch goes to `sensor/`, `hits/`, `step/`, `labl/` under it |
| `--job-id` | required. 1-based; the batch's file index is `job_id − 1` |
| `--detector` | selects `config/<detector>_{geom,physics}_config.json`. Required unless the config declares `detector` (then it must match) or with `--skip-lucid`. The readout comes from the dataset config, else from this physics config; only `SK_WAND` and `HK_WAND` carry one ([digitizer and trigger](digitizer-and-trigger.md)) |
| `--n-events` | events in this job (default: the config's `n_events_per_job`, else 100) |
| `--test` | 2 events; overrides `--n-events` |
| `--master-seed` | seed for PhotonSim/GEANT4 and, unless `--lucid-seed` is given, the LUCiD draws (default: random) |
| `--lucid-seed` | seed for the LUCiD draws only (vertex, rotation, t0, propagation, smearing), to re-place the same GEANT4 sample |
| `--override-energy-MeV` | monoenergetic primaries at this energy (MeV), for energy scans |
| `--keep-root` | keep the intermediate PhotonSim ROOT file (otherwise `cleanup_root_files` in the config decides) |
| `--skip-genie` | skip the generator step (GENIE or supernova); no effect for particle-gun configs |
| `--skip-photonsim`, `--skip-lucid` | run only the other half, reusing or leaving the ROOT file |
| `--sn-model`, `--sn-ordering` | a supernova config's model and mass ordering, one entry each of `supernova.models` and `supernova.orderings`; optional when the list has one entry |

It ends by printing a line starting `OK` and exits non-zero on failure. Output:
[dataset schema](dataset-schema.md).

## `lucid-build-photon-table`, `lucid-build-dedx-table`

Build the SIREN training tables from a PhotonSim energy scan
([SIREN training](../guides/production/siren-training-inputs.md)).

```bash
lucid-build-photon-table --data-dir SCAN --material water --particle mu- --output photon_lookup_table.h5
lucid-build-dedx-table   --data-dir SCAN --material water --particle mu- --output dedx_lookup_table.h5
```

| flag | meaning |
|---|---|
| `--data-dir` | required. Holds `<material>/<particle>/<E>MeV/photonsim.root`; every `<E>MeV` directory found is used |
| `--material`, `--particle` | required. The scan cell and its `smax_fit.csv` |
| `--output` | required. Output `.h5` (file or directory) |
| `--photonsim-dir` | the PhotonSim checkout holding `data/<material>/<particle>/smax_fit.csv` (default: `$PHOTONSIM_DEV_PATH`, or a sibling of `--data-dir`) |
| `--require-list CFG` | fail unless every energy in the config's `energy_list_MeV` is on disk |
| `--no-validate` | skip the check that re-reads the written table |

## `lucid-train-siren`

Trains a SIREN emitter on one table. The kind (Cherenkov or dE/dx), material and particle come
from the table.

```bash
lucid-train-siren --h5-path photon_lookup_table.h5 --output-dir runs/water_muon --seed 1
```

| flag | meaning |
|---|---|
| `--h5-path` | required. The table |
| `--output-dir` | required. Must not already hold an `emitter.json` |
| `--recipe` | recipe JSON (default `lucid/siren/training/nominal.json`, the recipe of the shipped emitters) |
| `--<field>` | override one recipe field, e.g. `--seed 1`, `--num-steps 20000`, `--learning-rate 1e-4` |
| `--init` | `weights.npz` to start from (an emitter or a checkpoint) |
| `--checkpoint-every N` | write `checkpoint_step_<N>.npz` every N steps |
| `--no-prediction-plots` | skip the prediction-against-table PDFs (`--prediction-plot-every`, default 10000 steps) |

It writes `emitter.json` and `weights.npz` (the emitter), `recipe.json`,
`training_history.json`, `final_training_progress.pdf` and `prediction_plots/`. Judge
convergence by the validation loss (`val_loss` in `training_history.json`), not the training
loss: the learning rate drops when it plateaus for `patience` validations (100 in the nominal
recipe), and a flat `val_loss` with no further drops means it is done. The run is installed
with `lucid-siren-promote`.

## `lucid-siren-promote`

Installs a training run as an emitter in `data/<material>/<particle>/<kind>/`, adding what the
simulator needs and the trainer cannot know
([stage 5](../guides/production/siren-training-inputs.md#stage-5-promote)).

```bash
lucid-siren-promote --run runs/water_muon --smax-fit PhotonSim/data/water/mu-/smax_fit.csv \
    --t0 timing_parameters.json --emission-band-nm 274.91 673.83
```

| flag | meaning |
|---|---|
| `--run` | required. The training run (`emitter.json`, `weights.npz`) |
| `--smax-fit` | required. The `smax_fit.csv` its table was built with; the run's `s_max_mm` must equal it |
| `--t0` | Cherenkov, required: the t0 fit, `timing_parameters.json` of `PhotonSim/tools/t0_correction/calculate_t0.py` |
| `--emission-band-nm LO HI` | Cherenkov, required: the band (nm) PhotonSim counted N_photons(E) over; water 274.91 673.83 |
| `--selection` | how the run was chosen, recorded in the provenance |
| `--dest` | output directory (default: `data/<material>/<particle>/<kind>/` of this checkout) |
| `--force` | replace an emitter already at `--dest` |

It checks the result with `validate_record` and records the run, the selection, the `s_max` fit,
the time and the commit in `provenance`.

## Scripts

| script | does | main options |
|---|---|---|
| `download_data.py` | installs the example data into `data/` ([install](../getting-started/install.md#download-the-example-data)) | `--from-dir DIR`, `--verify-only`, `--paper` |
| `visualize_detector.py` | prints a geometry's sensor count and extent, writes a 3D HTML view (and a 2D PNG for cylinders); needs `[viz]` | `--geom`, `--out` |
| `geofile_to_npz.py` | converts a WCSim geofile to a [PMT `.npz`](pmt-npz-schema.md) | `GEOFILE OUT.npz` |
| `find_optimal_k.py` | the smallest scattering count `K` that keeps a target charge fraction | `--detector`, `--source`, `--target` (0.999), `--nphot`, `--k-max` |
| `inspect_dataset.py` | a text summary of a dataset batch | `--path`, `--batch`, `--event` |
| `benchmark_forward.py` | wall time of the reconstruction and calibration forward and gradient against photon count and `K` | `--what`, `--geom`, `--physics`, `--n`, `--k` |
| `check_gradients.py` | autodiff against finite-difference gradients of the calibration loss | `--geom`, `--physics`, `--nphot`, `--rel-h` |
