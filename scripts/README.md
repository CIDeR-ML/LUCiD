# scripts/

Runnable tools beyond the console entry points (`lucid-run-job`, `lucid-train-siren`,
`lucid-build-*-table`) and the scripts in [`../examples/`](../examples/). Run them from the repo
root (they use `config/…` / `data/…` paths).

`python scripts/X.py` puts `scripts/` first on `sys.path`, so `lucid` comes from the installed
package, not from the checkout the script sits in. With one `pip install -e .` the two are the
same; with a second checkout, run its scripts as `PYTHONPATH=<that checkout> python scripts/X.py`.

| Script | What | Typical use |
|--------|------|-------------|
| `download_data.py` | Install the example data (the `lucid-data-v1` release asset: SIREN weights + GEANT4 events) into `data/`, sha256-checked; `--from-dir`, `--verify-only` | once after install |
| `benchmark_forward.py` | Warm wall time vs (N photons, K) on any detector of what the fits run: the track model's forward, loss and gradient (`ReconModel`), the calibration forward and its Jacobian (`CalibrationForward`); `--what track\|calibration\|both` | perf sizing |
| `check_gradients.py` | AD vs finite-difference gradient check of the calibration loss | verify differentiability |
| `find_optimal_k.py` | Minimum scatter-iteration count `K` for a target charge fraction | geometry tuning |
| `geofile_to_npz.py` | Convert a WCSim-style detector geofile (`.txt`) → a PMT-array `.npz` (schema `docs/reference/pmt-npz-schema.md`) | add/reproduce a measured detector |
| `inspect_dataset.py` | Text summary of a dataset batch (sensor/hits/step/labl) | inspect produced data |
| `visualize_detector.py` | Render any `*_geom_config.json` (3D scatter for all geometries; 2D unrolled for cylinders; needs the `[viz]` extra) | sanity-check a geometry |

Cluster deployment lives in `../lucid/production/` (`platform/` for the cluster adapters, `dataprod/` for the jobs).
