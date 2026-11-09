# CLI reference

LUCiD installs four console entry points (defined in `pyproject.toml`). Beyond these,
`scripts/` carries portable tools for inspection, visualization, benchmarking, and gradient
checks — see `scripts/README.md`.

## `lucid-train-siren` — train the emission surrogate
Trains the SIREN Cherenkov/dE-dx emitter on PhotonSim lookup tables.
```bash
lucid-train-siren --h5-path data/water/muon/photon_lookup_table.h5 --output-dir runs/water_muon --seed 1
```
Trains on the lookup table (the kind, material and particle come from it) with the committed
recipe `lucid/siren/training/nominal.json` (`--recipe` for another; every recipe field has its
flag) and writes the emitter (`emitter.json` + `weights.npz`, recipe and losses recorded),
`recipe.json` (the resolved recipe, written at the start), `training_history.json`,
`final_training_progress.pdf` and `prediction_plots/step_NNNNNN.pdf` under the required
`--output-dir`, which must not already hold an `emitter.json`. `--t0 t0.json` records a Cherenkov
emitter's t0 fit (PhotonSim's `PhotonSim/tools/t0_correction/calculate_t0.py` writes it) and
`--init weights.npz` warm-starts. Installing the result (where it goes, the t0 fit and the
`emission_band_nm` the simulator needs) is in the
[SIREN training inputs guide](../guides/production/siren-training-inputs.md#how-inference-consumes-this).
Run `lucid-train-siren --help` for the full set.
Judge convergence from the held-out validation loss (`val_split`, default 1%),
not the training loss: watch `val_loss` in `training_history.json` (also plotted
to `final_training_progress.pdf`) flatten, and check the in-training
prediction-vs-truth plots track the surrogate against the table. The LR schedule
is patience-based (`--patience`, nominal 100), so a plateauing `val_loss` with no
further LR drops is the signal to stop.
Module: `lucid.siren.training.train`. Build its input tables first with the two table builders below.

## `lucid-build-photon-table` / `lucid-build-dedx-table` — SIREN training inputs
Build the photon lookup / dE-dx tables from PhotonSim ROOT output.
```bash
lucid-build-photon-table --data-dir <scan> --material water --particle mu- --output photon_lookup_table.h5
lucid-build-dedx-table   --data-dir <scan> --material water --particle mu- --output dedx_lookup_table.h5
```
`--data-dir`, `--material`, `--particle` and `--output` are required; `<scan>` holds
`<material>/<particle>/<E>MeV/photonsim.root`. Module:
`lucid.siren.training.photonsim_data.build_tables` (`main_photon` / `main_dedx`). The other
flags are in the
[SIREN training inputs guide](../guides/production/siren-training-inputs.md#stage-2-building-the-h5).

## `lucid-run-job` — produce a dataset
Single-job production: generates a GEANT4 macro, runs the external **PhotonSim** binary (path from
`PHOTONSIM_BIN`), and writes the four parallel HDF5 batch files. GENIE-flux configs additionally chain
`gevgen`→`gntpc`. Configs live in block subdirectories (`GeV/`, `SN/`, `Solar/`, `Test/`).
```bash
export PHOTONSIM_BIN=/path/to/PhotonSim/build/PhotonSim
lucid-run-job --config lucid/production/dataprod/configs/GeV/02_mu.json \
              --detector SK_WAND \
              --n-events 1000 --job-id 1 --master-seed 42 --output-dir out/
```
Writes `sensor/ hits/ step/ labl/` batch files under `--output-dir` (plus the intermediate
PhotonSim ROOT unless the config sets `cleanup_root_files`).

| flag | meaning |
|---|---|
| `--config` | dataset JSON (required) |
| `--output-dir` | dataset directory for this config (required) |
| `--job-id` | **1-based** job id; maps to `file_index = job_id - 1` (required) |
| `--detector` | geometry to simulate against — selects `config/<detector>_{geom,physics}_config.json` (required unless the dataset config declares `detector`, or `--skip-lucid`). A declared `detector` must match the flag, else the job exits before any step. The digitizer and trigger come from the dataset config, else from that physics config, and only `SK_WAND` / `HK_WAND` carry them: with any other detector a config that states no readout stops before PhotonSim runs (`"digitizer": "basic"` and `"trigger": null` opt out) |
| `--n-events` | events in this job (otherwise the config's `n_events_per_job`, which no shipped config sets, so 100) |
| `--master-seed` | seed of the job: PhotonSim / Geant4 and, unless `--lucid-seed` is given, the LUCiD-side draws (default: random) |
| `--lucid-seed` | seed for the LUCiD-side per-event draws (vertex translation, rotation, t0, propagation, smearing) only; PhotonSim stays on `--master-seed`, so the same Geant4 sample is re-placed and re-oriented |
| `--test` | cap `n_events` to 2 (overrides `--n-events`) |
| `--override-energy-MeV` | force monoenergetic primaries (for energy-scan fan-outs) |
| `--keep-root` | keep the intermediate PhotonSim ROOT even if the config cleans up |
| `--skip-genie` | skip the generator step (GENIE / supernova input); no-op for particle-gun configs |
| `--skip-photonsim` / `--skip-lucid` | run only the other half of the chain (resume/debug) |
| `--sn-model` / `--sn-ordering` | supernova model + mass-ordering subcase: an entry of the config's `supernova.models` / `supernova.orderings` (the fan-out passes one per subcase); each may be omitted only when its list has one entry. `supernova.orderings` is required, with no default |

Module: `lucid.production.dataprod.run_job`. See the [dataset schema](dataset-schema.md),
[digitizer & trigger](digitizer-and-trigger.md),
[cluster abstraction](../guides/production/cluster-abstraction.md), and the local
[quickstart](../guides/production/local.md).
