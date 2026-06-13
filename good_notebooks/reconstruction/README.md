# reconstruction/ — track reconstruction

Fit the 9-vector `[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]` to one event's (charge, first-arrival
time). **Canonical API: `lucid.fitting.fit_track_multistart`** (charge-grid ‖ time-multilateration
two-start, 1% margin) + `seed_vertex_time`. Validated vs exact gun truth: vtx ~12 cm, dir ~1.0°,
energy unbiased, t0 ~0.5 ns RMS, 0 wanderers (`../../campaign_recon/RESULTS.md`).

- **two_start_reconstruction** ⭐ START HERE — the canonical fit: forward → event → seeds →
  `fit_track_multistart` → convergence. (GPU-validated.)
- **recon_anatomy** — *why* it works: the two complementary seeds, the longitudinal loss basin,
  convergence, and the residual split via `lucid.fitting.analysis`. (GPU-validated.)
- **track_optimization_visualization** — post-hoc convergence / longitudinal-transverse residual stats.
- **optimization_vs_variables** — recon resolution vs Nrays / energy / #sensors (analysis-only).

Superseded inline-Adam notebooks (5-stage, GIF, JUNO sphere) are in `../archive/`.
