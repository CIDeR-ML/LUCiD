# scripts/

Runnable tools beyond the console entry points (`lucid-run-job`, `lucid-optimize`,
`lucid-train-siren`, `lucid-build-*-table`) and the hello-world spine in
[`../examples/`](../examples/). Run from the repo root (they use `config/…` / `data/…` paths).

## Portable user/dev tools

| Script | What | Typical use |
|--------|------|-------------|
| `download_data.sh` | Fetch the example SIREN emitter + PhotonSim tables from CERNBox (wires up wbls/ice symlinks) | run once after install |
| `benchmark_forward.py` | Forward + `value_and_grad` wall-time vs (N photons, K), track & calibration | perf sizing / claims |
| `generate_example_events.py` | Write simple per-event HDF5 (Q/T + truth) from a PhotonSim ROOT (not the full four-file dataset) | tutorials / tests / quick studies |
| `geofile_to_npz.py` | Convert a WCSim-style detector geofile (`.txt`) → a PMT-array `.npz` (schema `lucid/geometry/PMT_NPZ_SCHEMA.md`) | add/reproduce a measured detector |
| `visualize_detector.py` | Render any `*_geom_config.json` (3D scatter for all geometries; 2D unrolled for cylinders) | sanity-check a geometry |
| `inspect_dataset.py` | Text summary of a dataset batch (sensor/hits/step/labl) | inspect produced data |
| `check_gradients.py` | AD vs finite-difference gradient check (pathwise params); reports the score-estimated params | verify differentiability |
| `train_siren_pipeline.py` | Single-node SIREN training pipeline: build tables → train (photon + dE/dx) → validate (optional PhotonSim generation) | train the emitter end-to-end |
| `reconstruct_dataset.py` | Reconstruct a whole PhotonSim ROOT (two-start seed → Fisher-GN) and report vtx/dir/E/t0 resolution | recon study at scale |
| `calibrate_campaign.py` | Calibrate optical params from a source set over N random starts; recovery vs the Cramér-Rao bound | calibration study |
| `find_optimal_k.py` | Minimum scatter-iteration count `K` for a target charge fraction | geometry tuning |
| `compare_method_ab.py` | Compare two propagator methods (expected-value vs Bernoulli sampling) | method validation |
| `run_string_siren_tracks.py`, `run_string_cascades.py` | Batch neutrino-telescope (IceCube-string) track / cascade sims | telescope studies |

Reconstruction / calibration at the single-event or single-config level: see
[`../examples/`](../examples/) (`hello_reconstruct`, `seed_reconstruct`, `hello_calibrate`,
`hello_telescope`) and `lucid-optimize`.

## Research / campaign (internal, git-ignored)

- `campaign/` — the calibration research campaign (CRB, per-PMT QE, reflection, spectral,
  timing, shot-noise studies). SK-hardcoded, env-var-driven; the portable subset is
  `calibrate_campaign.py` above.
- `campaign_recon/` — the reconstruction research campaign + diagnostics; the portable subset is
  `reconstruct_dataset.py` above.

These are kept on disk for the team but are **git-ignored** (personal research scratch). Cluster
deployment lives in `../lucid/production/jobs/` (portable SLURM/HTCondor/NERSC) and `../s3df_jobs/`
(SLAC-specific).
