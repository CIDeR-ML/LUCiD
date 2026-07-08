# 05 — Scripts, CLI & the real "how to run LUCiD" map

**Status:** proposal / understanding doc. Captures how the runnable surfaces are actually
used, so the docs describe real workflows and the repo cleanly separates public from internal.

## The key finding

Three parallel user-facing surfaces tell the same 4–5 stories: `examples/*.py` (scripts),
`tutorials/*.ipynb` (notebooks), and the older `good_notebooks/`. They call the *same*
`lucid.fitting` seams. Resolution (matches the one-notebook-set decision):

- **`examples/` = short runnable scripts — the spine.** Verified API-current (every symbol the 4
  scripts call resolves against the live code), dependency-light (no PhotonSim/GEANT4 for
  simulate/calibrate/reconstruct), honest about demo-vs-physics scope.
- **`tutorials/` = the narrated notebooks** (see `02`).
- **`good_notebooks/` retired.**

The docs pair them: each workflow guide shows the ≤20-line `examples/` script, with the
`tutorials/` notebook as the narrated companion.

## Script surfaces: public vs internal

### Public — document these
| Surface | What / how invoked | Workflow |
|---|---|---|
| `examples/hello_simulate.py` | forward + display; `python examples/hello_simulate.py` | (a) simulate |
| `examples/hello_reconstruct.py`, `seed_reconstruct.py` | track fit; honest seeder→fit | (b) reconstruct |
| `examples/hello_calibrate.py` | GN optics fit vs CRB | (c) calibrate |
| `examples/hello_telescope.py` | **new** — string/IceCube geometry + cascade source; promote/trim `scripts/run_string_siren_tracks.py` | (a/b) neutrino-telescope entry point |
| `scripts/download_data.sh` | fetch example PhotonSim ROOTs + trained SIREN nets (CERNBox) | data step for all |
| `lucid-optimize` (`lucid/optimization/run.py`) | config-driven recon; `lucid-optimize <cfg.json>` | (b) reconstruct at scale |
| `lucid-train-siren` (`lucid/siren/train.py`) + `lucid-build-{photon,dedx}-table` | train the emitter; ROOT→training tables | (d) train SIREN |
| `lucid-run-job` (`lucid/production/run_job.py`) | PhotonSim/GENIE → v3 HDF5 batch | (e) produce a dataset |
| `lucid/production/jobs/…` → `cluster_common/` | general SLURM/HTCondor/NERSC fan-out | (f) cluster (portable) |
| `lucid/production/photon_shotgun/run.py` | synthetic per-photon dataset generator | research tool |
| `lucid/production/visualize_particle_events.py` | single-event static HTML view | inspect |

### Internal — do NOT document publicly (keep, relocate, or remove)
| Surface | Why internal |
|---|---|
| `scripts/campaign/`, `scripts/campaign_recon/` | research campaigns (methodology/recon validation) + `*_RESULTS.md`; multi-GPU env-driven. Relocate to `studies/` or remove (see `01`). |
| `baseline_scripts/` | migration-era regression harness (`LUCID_BACKEND=lucid|tools`); `tools.*` is gone and it's **not in CI** → stale. Remove (see `01`). |
| `s3df_jobs/` | SLAC-only (hardcoded `.sif`/`/sdf/data/…`/partitions); `CLUSTER_ABSTRACTION.md` already disowns it. Keep internal / relocate. |
| `ci_tests/` | CI plumbing (`render_ci_displays.py`, `speed_test.py`); document only in a short "CI" page. |
| `scripts/compare_method_ab.py`, `find_optimal_k.py`, `run_string_*.py` | dev studies. Keep internal. |
| `lucid/siren/validate*.py`, `plot_training_results.py` | dev/validation tools; mention in the SIREN runbook. |
| `lucid/production/cluster_common/`, `jobs/` | cluster ops plumbing (imported/orchestration). Keep internal. |

## The 6 canonical workflows (the docs "how to run" backbone)

**(a) Just simulate / see something — no PhotonSim, no external deps**
```bash
pip install -e .
./scripts/download_data.sh                # fetches the SIREN weights (NOT in git)
python examples/hello_simulate.py         # or tutorials/00_quickstart.ipynb
```

**(b) Reconstruct a track**
```bash
python examples/hello_reconstruct.py                                  # self-contained demo
python examples/seed_reconstruct.py                                   # honest seed → fit
lucid-optimize lucid/optimization/configs/example_water_mu.json       # config-driven
```

**(c) Calibrate optical parameters**
```bash
python examples/hello_calibrate.py        # or tutorials/calibration_optimization.ipynb
```

**(d) Train the SIREN emitter**
```bash
./scripts/download_data.sh
lucid-build-photon-table                   # ROOT → training .h5
lucid-train-siren --material water --particle muon --data-type photon
```

**(e) Produce a v3 dataset (PhotonSim / GENIE → HDF5)**
```bash
export PHOTONSIM_BIN=/path/to/PhotonSim/build/PhotonSim
lucid-run-job --config lucid/production/configs/dataprod_01_mu.json \
              --n-events 1000 --job-id 0 --master-seed 42 --output-dir out/
# canonical guide: docs/QUICKSTART_LOCAL.md ; read the output with tutorials/work_with_a_dataset
```

**(f) Run on a cluster**
```bash
lucid/production/jobs/dataprod/generate_jobs.sh   # portable fan-out (SLURM/HTCondor/NERSC)
# SLAC-only, internal/undocumented: s3df_jobs/submit_job.py
```

## Docs implication

The Getting Started + Workflow Guides sections in `03` should be organized around these 6
workflows, each naming the exact `examples/` script or CLI. Fix the README (see `04`): it
currently leads its Quick start with the heaviest step (`lucid-run-job`) and never mentions
`examples/` — the front door should be workflow (a).
