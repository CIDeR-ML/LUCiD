# Phase 3.3 / 3.4 / Step 4 / Campaign — implementation plan

Status anchor: Steps 1, 2, 3.1 (optical λ-curves), 3.2 (reflection seam + angular) DONE on
branch `unify-calibration`. Guards: fast `pytest tests/ -q` (351) + the 5 slow step files
(`--slow`, 62) + `test_reflection_integration.py` (slow, 2). #73 = repair the remaining
pre-existing stale slow tests (sk_like_integration + ~6) — agent running.

## 3.3 — Charge moments (variance) + per-PMT/response wiring

Validated compound-Poisson moments (mie_hunter/chargedata.py):
- rate μ[s] = Σ_photons flat_weights·qe·qe_corrections[s]   (the existing `total_charge`, PE counts)
- E[Q][s]   = gain[s] · μ[s]
- Var[Q][s] = gain[s]² · (1 + w²) · μ[s]      (w = spe_width)
- v/m² = (1+w²)/μ measures the RATE μ → breaks QE↔gain (mean only gives the product); v/m = g(1+w²).

Timing (mie_hunter/timingcal.py): first_arrival[s] = t_geo[s] + t0[s] + TTS·E[min_{n~Poisson(occ)}N(0,1)].
- t0[s] additive per-PMT offset (dominant TQ-map term).
- TTS occupancy-bias: expected min of `occ` standard normals, a deterministic fn of occ=μ[s] (negative,
  ~−√(2 ln occ) for large occ, ≈0 for occ<1). In the differentiable soft-min path this is an additive
  correction `+ tts · zmin_bias(μ)`.
- walk[s]: keep as an optional charge-dependent additive term (default 0); not in the validated t0+TTS
  engine, so leave neutral/placeholder unless the campaign needs it.

### 3.3a — moments make_hits (commit)
- New `make_hits_moments(flat_weights, flat_indices, flat_times, num_detectors, qe, qe_corrections,
  gain, spe_width, t0, threshold, temperature)` returning **(mean_charge, var_charge, measured_time)**.
  - μ = segment_sum(flat_weights·qe·qe_corrections[idx])
  - mean = gain·μ ; var = gain²·(1+w²)·μ
  - time = existing soft-min first-arrival + t0[s]
  - Neutral defaults gain=1, w=0, t0=0 ⇒ mean=μ, var=μ, time unchanged ⇒ byte-identical to
    make_hits_simulation's (charge, time) [var is an extra output].
- New `hit_mode='moments'` in setup_event_simulator dispatch; thread gain/spe_width/t0 from
  dp.per_pmt.gain / dp.response.spe_width / dp.per_pmt.t0 into the make_hits call inside
  `_common_propagation`. The make_hits_fn signature there gains these (default-neutral so the other
  modes stay byte-identical — pass through or ignore).
- Guard: fast 351 + 5 slow step files 62 unchanged; new unit test for the moment formulas
  (mean=gain·μ, var=gain²(1+w²)μ, byte-identical at neutral defaults) + a slow e2e moments forward.

### 3.3b — TTS occupancy-bias in the differentiable first-arrival (commit)
- Add `tts` (dp.response.tts) to make_hits_moments: measured_time += tts · zmin_bias(μ).
- `zmin_bias(occ)` = a smooth approx of E[min of occ iid N(0,1)] as a fn of the continuous occ=μ
  (e.g. −√(2 ln(1+occ)) damped, or a fitted curve); ≈0 at occ→0, byte-identical at tts=0.
- Guard: tts=0 byte-identical; unit test that tts>0 shifts the time earlier monotonically in occ.

## 3.4 — Source / Spectrum abstraction
- A `Spectrum` abstraction owning λ-sampling (separated from optical_model, which only evaluates):
  `Monochromatic(λ)`, `PowerLaw(p=2)` (bare Cherenkov 1/λ²), and the production-only `QEWeighted`
  (importance-sampled, density-estimate — NOT for fitting qe, per the calibration findings).
- Sources (laser/isotropic/Cherenkov) carry a Spectrum; the simulator asks the Spectrum for per-photon λ
  instead of the inline `wavelength_sampling` branch in `_get_optical_arrays`. Default behaviour
  reproduces the current 'cherenkov'/'cherenkov_qe' sampling ⇒ byte-identical.
- Keep it minimal (the user said don't overcomplicate): a small NamedTuple/callable, registered like
  reflection models.

## Step 4 — lucid/fitting/
- `gauss_newton.py`: consistent FIXED-dataset GN (H/g/loss on ONE dataset; min‖g‖ readout) +
  FD/√-MSE-GN Hessian recomputed every N iters + median ridge + (CLIP=0 additive ridge default for
  calibration; positive-eigen-clip option for the track/recon basin) + per-PMT Schur complement with
  gauge mean(log k)=0 (and mean(t0)=0 for timing). Merge gnrec (recon) + gn_fast (calib) into one.
- `fisher.py`: Fisher CRB at θ_true = ¼(JᵀJ)⁻¹ via FD/HVP (AD Hessian broken for geometry; jacfwd blocked
  by the DiCE custom_vjp) with the ×√12 honesty factor; log-params→fractional; k-block Schur collapse.
- A declarative PARAM REGISTRY keyed by DetectorParams leaf name: per-field transform (log/linear-by-nominal),
  gauge, prior, which-observable — replaces every bespoke asm()/normalize.
- ravel_pytree(dp) ↔ flat θ bridges the pytree to the optimizer vector.

## Campaign — re-run all calibration combinations on the unified framework
Reproduce the mie_hunter campaign numbers (CRB / bias / recovery) with the unified base:
- Sources: laser (single + multi-position) × isotropic × mixed; broadband Cherenkov.
- Wavelengths: the usable band (~380–440nm) + the SK laser anchors; mono vs spectral.
- Param classes: optical (L_abs, L_R, L_M, g) ; reflection (scalar wall/sensor ; angular R0w/pw/nr/nk) ;
  per-PMT QE k (closed-form Q/M + bake, gauge mean(log k)=0) ; charge variance (gain, w) ;
  timing (t0, TTS) ; spectral λ-deviation curves.
- Recipe (CONSOLIDATED_FINDINGS §7): τ-smoothed √-MSE loss + consistent fixed-data GN (CLIP=0 ridge) +
  source diversity to break L_M↔k + closed-form per-PMT k=Q/M with one bake + ×√12 CRB.
- Output: a table comparing recovered value, bias, and √(CRB) per (source, λ, param) to the recorded
  mie_hunter results; flag any mismatch and explain.
