# good_notebooks — curated LUCiD notebooks (new-user journey)

Thin notebooks that **import the library seams and tell a story** (machinery lives in `lucid/`,
not inline — see `STRUCTURE.md`). Paths are relative to each notebook's folder (`../../config`,
`../../lucid`). Start at `00_quickstart`, then follow the journey.

| step | notebook | seam |
|------|----------|------|
| 0 | [`00_quickstart`](00_quickstart.ipynb) | forward + `visualization.create_detector_display` |
| 1 | [`reconstruction/two_start_reconstruction`](reconstruction/) ⭐ | `fitting.fit_track_multistart` |
| 1b | [`reconstruction/recon_anatomy`](reconstruction/) | `+ fitting.analysis` (why it works) |
| 2 | [`calibration/calibrate_optics`](calibration/) ⭐ | `fitting.build_calibration_problem`/`fit`/`crb` |
| 3 | [`gradients/loss_landscapes`](gradients/) ⭐ | `gradient_analysis` sweeps over `ReconModel.loss` |
| 4 | [`displays/event_displays`](displays/) ⭐ | `visualization` (2D + animation + 3D) |
| — | [`infrastructure/`](infrastructure/) | benchmarks, SIREN training, data-vs-pred |

⭐ = the canonical entry for that workflow. `calibration/` and `gradients/` also hold deeper
scenario/study notebooks. `extend`: `../examples/` (≤20-line scripts) + `lucid/fitting/contracts.py`.
Superseded fat notebooks → `archive/`. Full per-notebook reference: `NOTEBOOK_ANALYSIS.md`.
