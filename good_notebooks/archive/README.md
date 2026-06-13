# archive/ — superseded notebooks (reference only)

Kept for the unique scenario/analysis content; their machinery (inline Adam optimizers, inline
loss assembly, inline display functions) is superseded by the library seams + the thin canonical
notebooks. Not maintained; relative paths may need a tweak to run.

**Reconstruction (inline 5-stage Adam → use `../reconstruction/two_start_reconstruction`):**
- `tracking_opt_development` — heuristic counts-loss 5-stage.
- `tracking_opt_development_likelihood` — likelihood-loss 5-stage.
- `tracking_opt_with_gif` — convergence GIF (animation now in `../displays/event_displays`).
- `visualize_3D_track_optimization` — **JUNO sphere** recon (the only non-cylinder example —
  worth re-porting to `fit_track_multistart` if a multi-geometry demo is wanted).

**Displays (inline display helpers → use `../displays/event_displays`):**
- `cylinder_2D_displays` — 2D + pred-vs-data comparison.
- `geometry_and_events_3D_visualization` — 3D discs, multi-detector.
- `event_hit_animation` — the original GIF animator.
