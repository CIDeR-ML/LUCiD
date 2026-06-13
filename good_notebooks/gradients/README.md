# gradients/ — gradient variations & loss landscapes

How the loss + its gradient vary with each parameter (the "gradient variations" set). Forward =
`setup_event_simulator`; losses in `lucid.losses`; sweeps via `lucid.gradient_analysis`
(`SweepParam`/`sweep_1d`/`sweep_2d`). Coverage = {1D, 2D} × {counts, likelihood}.

- **parameter_scans_1D** — 1D scans of all 7 track params, counts loss + multi-event zero-crossing resolution.
- **parameter_scans_1D_likelihood** — same, likelihood loss + photon-budget (Nphot) scaling.
- **parameter_scans_1D_v2** — 1D via the `gradient_analysis` sweep library (cleaner; supersedes `_1D` methodologically).
- **grad_loss_and_opt_in_2D** — 2D landscapes (6 param pairs) + multi-start optimization traces, counts loss.
- **grad_loss_and_opt_in_2D_likelihood** — 2D landscapes, likelihood loss (topology vs counts).
