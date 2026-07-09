# reconstruction/ — track reconstruction

Fit the 9-vector `[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]` to one event's (charge, first-arrival
time). **Canonical API: `lucid.fitting.fit_track_multistart`** (charge-grid ‖ time-multilateration
two-start, 1% margin) + `seed_vertex_time`.

- **two_start_reconstruction** ⭐ START HERE — the canonical fit: forward → event → seeds →
  `fit_track_multistart` → convergence. (GPU-validated.)
- **recon_anatomy** — *why* it works: the two complementary seeds, the longitudinal loss basin,
  convergence, and the residual split via `lucid.fitting.analysis`. (GPU-validated.)
- **track_optimization_visualization** — post-hoc convergence / longitudinal-transverse residual stats.
- **optimization_vs_variables** — recon resolution vs Nrays / energy / #sensors (analysis-only).
- **visualize_3D_track_optimization** — recon on the **JUNO sphere** (the non-cylinder geometry
  example), 3D Plotly. (Legacy inline optimizer; port to `fit_track_multistart` when convenient.)

Superseded inline-Adam notebooks (5-stage, GIF) are in `../archive/`.
