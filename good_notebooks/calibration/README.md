# calibration/ — detector calibration

Fit detector properties from controlled sources. **Canonical API:
`lucid.fitting.build_calibration_problem` → `fit` / `crb`** (Gauss-Newton + per-PMT Schur,
×√12-honest CRB).

- **calibrate_optics** ⭐ — START HERE. The canonical notebook on the new API: 7 global optical
  params + per-PMT QE (Schur) recovered from a 4-source set; `build_calibration_problem` → `crb` →
  `fit`, with fit-vs-CRB plot. GPU-validated.
- **fit_absorption_field** — fit a *spatial* absorption field μ(x): configure a low-dim `poly`
  field (`absorption_field='poly'`), generate data from a known field, recover `field_params` by
  Adam through the differentiable transport, and visualise truth vs. fit (r–z maps). GPU-validated.
- **grad_param_calibration_multi_init_no_qe** — 4 global scalars (scatter, wall/sensor reflection, absorption), single laser, multi-start.
- **detector_grad_qe_convergence_multi_source** — per-PMT QE map (~11k-dim) from 15 isotropic sources (geometric diversity breaks degeneracy).
- **wavelength_calibration** — 4-param per-λ (350–500 nm) identifiability; 405 nm is the sweet spot.
- **per_sensor_tau_analysis** — timing τ-smoothing per sensor vs Nₛ regime (reference; mask Nₛ<1).
- **laser_source_grad_analysis** — calibration loss landscapes (1D + 2D pairs) via `gradient_analysis`.
- **wavelength_calibration_findings.md** — findings digest.
