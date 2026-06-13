# good_notebooks — curated LUCiD notebooks (by workflow)

Reorganized into workflow folders (unification, 2026-06). All paths are relative to the
notebook's folder (`../../config`, `../../lucid`). Canonical APIs: forward =
`setup_event_simulator`; reconstruction = `lucid.fitting.fit_track_multistart` (+ `seed_vertex_time`);
calibration = `lucid.fitting.build_calibration_problem`/`fit`/`crb`; losses = `lucid.losses`;
parameter sweeps = `lucid.gradient_analysis`.

| Folder | What | Canonical API |
|--------|------|---------------|
| [reconstruction/](reconstruction/) | track fits (vertex/dir/energy/t0) | `lucid.fitting.fit_track_multistart` |
| [calibration/](calibration/) | detector params (scatter/abs/refl/QE/λ/timing) | `lucid.fitting.build_calibration_problem`/`fit`/`crb` |
| [gradients/](gradients/) | loss landscapes + 1D/2D parameter scans | `lucid.losses` + `lucid.gradient_analysis` |
| [visualization/](visualization/) | event / detector displays (shared) | `setup_event_simulator` + geometry display |
| [infrastructure/](infrastructure/) | benchmarks, SIREN training, data-vs-pred | various |
| [archive/](archive/) | superseded, kept for reference | — |

`NOTEBOOK_ANALYSIS.md` / `NOTEBOOK_DOCUMENTATION.md` = the full per-notebook reference.
