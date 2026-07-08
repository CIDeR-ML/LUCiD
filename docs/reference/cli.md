# CLI reference

LUCiD installs five console entry points (defined in `pyproject.toml`). Beyond these,
`scripts/` carries portable tools for inspection, visualization, benchmarking, and gradient
checks — see `scripts/README.md`.

## `lucid-optimize` — config-driven reconstruction
Runs the full track reconstruction (seed search → Fisher-Gauss-Newton fit) from a JSON config.
```bash
lucid-optimize lucid/optimization/configs/example_water_mu.json
```
Prints per-event seed → fit progress and writes one `.npz` per event (fitted 9-vector,
direction, winning seed, loss trajectory) to the config's output directory. Module: `lucid.optimization.run`. For an interactive walk-through
see the `track_optimization` tutorial or `examples/hello_reconstruct.py` /
`examples/seed_reconstruct.py`.

## `lucid-train-siren` — train the emission surrogate
Trains the SIREN Cherenkov/dE-dx emitter on PhotonSim lookup tables (needs the `[training]`
extra / torch).
```bash
lucid-train-siren --material water --particle muon --data-type photon
```
Writes `.npz` checkpoints, `training_history.json`, and the trained model
(`trained_model/<name>_metadata.json` + `<name>_weights.npz`) under the material/particle
data directory, printing per-epoch losses. (The `siren_params.json` the simulator loads is
a separate hand-maintained pointer config — update its `path` to switch to a new model.) The training loop has ~30 further flags (architecture, LR schedule,
energy-balance sampling, validation plots) — run `lucid-train-siren --help` for the full set.
Module: `lucid.siren.train`. Build its input tables first with the two table builders below.

## `lucid-build-photon-table` / `lucid-build-dedx-table` — SIREN training inputs
Build the photon lookup / dE-dx tables from PhotonSim ROOT output.
```bash
lucid-build-photon-table    # -> siren_training input .h5
lucid-build-dedx-table      # -> dedx_siren_training input .h5
```
Modules: `lucid.siren.training.photonsim_data.build_photon_table` / `build_dedx_table`.
Their shared flag set is documented in the
[SIREN training inputs guide](../guides/production/siren-training-inputs.md).

## `lucid-run-job` — produce a dataset
Single-job production: generates a GEANT4 macro, runs the external **PhotonSim** binary (path from
`PHOTONSIM_BIN`), and writes the four-file HDF5 dataset. GENIE-flux configs additionally chain
`gevgen`→`gntpc`. Configs live in block subdirectories (`GeV/`, `SN/`, `Solar/`, `Test/`).
```bash
export PHOTONSIM_BIN=/path/to/PhotonSim/build/PhotonSim
lucid-run-job --config lucid/production/configs/GeV/01_mu.json \
              --n-events 1000 --job-id 1 --master-seed 42 --output-dir out/
```
Writes `sensor/ hits/ step/ labl/` batch files under `--output-dir` (plus the intermediate
PhotonSim ROOT unless the config sets `cleanup_root_files`).

| flag | meaning |
|---|---|
| `--config` | dataset JSON (required) |
| `--output-dir` | dataset directory for this config (required) |
| `--job-id` | **1-based** job id; maps to `file_index = job_id - 1` (required) |
| `--detector` | geometry to simulate against — selects `config/<detector>_{geom,physics}_config.json` (default `SK_like`) |
| `--n-events` | override the config's events-per-job |
| `--master-seed` | JAX PRNG seed (default: random) |
| `--test` | cap `n_events` to 2 (overrides `--n-events`) |
| `--override-energy-MeV` | force monoenergetic primaries (for energy-scan fan-outs) |
| `--keep-root` | keep the intermediate PhotonSim ROOT even if the config cleans up |
| `--skip-genie` | skip the generator step (GENIE / supernova input); no-op for particle-gun configs |
| `--skip-photonsim` / `--skip-lucid` | run only the other half of the chain (resume/debug) |
| `--sn-model` / `--sn-ordering` | supernova model + mass-ordering subcase selection |

Module: `lucid.production.run_job`. See the [dataset schema](dataset-schema.md),
[digitizer & trigger](digitizer-and-trigger.md),
[cluster deployment](../guides/production/cluster-abstraction.md), and the local
[quickstart](../guides/production/local.md).
