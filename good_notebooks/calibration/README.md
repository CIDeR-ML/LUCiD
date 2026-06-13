# calibration/ — detector calibration

Fit detector properties from controlled sources. **Canonical API:
`lucid.fitting.build_calibration_problem` → `fit` / `crb`** (Gauss-Newton + per-PMT Schur,
×√12-honest CRB).

- **grad_param_calibration_multi_init_no_qe** — 4 global scalars (scatter, wall/sensor reflection, absorption), single laser, multi-start.
- **detector_grad_qe_convergence_multi_source** — per-PMT QE map (~11k-dim) from 15 isotropic sources (geometric diversity breaks degeneracy).
- **wavelength_calibration** — 4-param per-λ (350–500 nm) identifiability; 405 nm is the sweet spot.
- **per_sensor_tau_analysis** — timing τ-smoothing per sensor vs Nₛ regime (reference; mask Nₛ<1).
- **laser_source_grad_analysis** — calibration loss landscapes (1D + 2D pairs) via `gradient_analysis`.
- **wavelength_calibration_findings.md** — findings digest.
