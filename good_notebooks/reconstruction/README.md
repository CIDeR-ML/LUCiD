# reconstruction/ — track reconstruction

Fit the 9-vector `[E, x,y,z, sinθ,cosθ, sinφ,cosφ, t0]` to one event's (charge, first-arrival
time). **Canonical API: `lucid.fitting.fit_track_multistart`** (charge-grid ‖ time-multilateration
two-start, 1% loss margin) + `seed_vertex_time`. Validated vs exact gun truth: vtx ~12 cm, dir
~1.0°, energy unbiased, t0 ~0.5 ns RMS, 0 wanderers (`../../campaign_recon/RESULTS.md`).

- **tracking_opt_development_likelihood** — full 5-stage likelihood pipeline (reference; port target = `fit_track_multistart`).
- **tracking_opt_with_gif** — animated convergence GIF from a smeared start (→ seed with `seed_vertex_time`).
- **visualize_3D_track_optimization** — recon on the JUNO sphere geometry, 3D Plotly.
- **track_optimization_visualization** — post-hoc convergence + longitudinal/transverse residual stats.
- **optimization_vs_variables** — recon resolution vs Nrays / energy / #sensors (analysis-only).

Legacy heuristic-loss 5-stage version: `../archive/tracking_opt_development`.
