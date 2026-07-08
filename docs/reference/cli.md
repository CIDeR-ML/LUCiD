# CLI reference

LUCiD installs five console entry points (defined in `pyproject.toml`).

## `lucid-optimize` — config-driven reconstruction
Runs the full track reconstruction (seed search → Fisher-Gauss-Newton fit) from a JSON config.
```bash
lucid-optimize lucid/optimization/configs/example_water_mu.json
```
Module: `lucid.optimization.run`. For an interactive walk-through see the `track_optimization`
tutorial or `examples/hello_reconstruct.py` / `examples/seed_reconstruct.py`.

## `lucid-train-siren` — train the emission surrogate
Trains the SIREN Cherenkov/dE-dx emitter on PhotonSim lookup tables (needs the `[training]`
extra / torch).
```bash
lucid-train-siren --material water --particle muon --data-type photon
```
Module: `lucid.siren.train`. Build its input tables first with the two table builders below.

## `lucid-build-photon-table` / `lucid-build-dedx-table` — SIREN training inputs
Build the photon lookup / dE-dx tables from PhotonSim ROOT output.
```bash
lucid-build-photon-table    # -> siren_training input .h5
lucid-build-dedx-table      # -> dedx_siren_training input .h5
```
Modules: `lucid.siren.training.photonsim_data.build_photon_table` / `build_dedx_table`.

## `lucid-run-job` — produce a v3 dataset
Single-job production: generates a GEANT4 macro, runs the external **PhotonSim** binary (path from
`PHOTONSIM_BIN`), and writes the v3 HDF5 dataset. GENIE-flux configs additionally chain
`gevgen`→`gntpc`.
```bash
export PHOTONSIM_BIN=/path/to/PhotonSim/build/PhotonSim
lucid-run-job --config lucid/production/configs/dataprod_01_mu.json \
              --n-events 1000 --job-id 0 --master-seed 42 --output-dir out/
```
Module: `lucid.production.run_job`. See the [v3 dataset schema](../LUCID_DATASET.md),
[cluster deployment](../CLUSTER_ABSTRACTION.md), and the local
[quickstart](../QUICKSTART_LOCAL.md).
