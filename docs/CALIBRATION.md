# Calibration — recipe & findings

How LUCiD calibrates detector optics — per-PMT quantum efficiency `k` plus the global optical
parameters (absorption `L_abs`, Rayleigh `L_R`, Mie `L_M`, wall/sensor reflectivity, QE spectrum,
`n_real`) — by gradient descent through the differentiable forward.(API / partition-combine), `examples/hello_calibrate.py` (runnable end-to-end), `docs/WAVELENGTH_DESIGN.md`.

## What is measured

`DetectorParams` is a JAX pytree with normalize/denormalize + bounds, so every optical property
is directly optimizable. Two tiers:
- **~19 global optical params** (scalars or λ-curves): `L_abs`, `L_R`, `L_M`, `wall_refl`,
  `sensor_refl`, `qe` spectral multipliers, `n_real`, SPE width `w`, etc.
- **per-PMT `k`** (QE map, ~10⁴ sensors).

## Recipe (per-sensor QE + all globals — the validated path)

1. **Source diversity is the lever.** Mix wall lasers (several λ + positions) + an isotropic
   volume source. Diversity breaks the key degeneracies — `L_M↔k`, `L_abs↔qe`, wall↔sensor —
   and makes `L_M` measurable with a plain absorption loss (no cross-term gymnastics).
2. **Globals: τ-smoothed √-MSE loss**, NOT Poisson-NLL (which biases `L_abs`/`qe` ~1.3%). The
   τ-smoothing is a spatial-frequency projector: smooth optical fields are low-freq, per-PMT `k`
   is white/high-freq, so τ-smoothing isolates the globals. Optimizer: **consistent
   fixed-dataset Gauss-Newton, `CLIP=0` additive ridge**, min‖g‖ readout → reaches near-CRB on
   6/7 parameter classes.
3. **per-PMT `k`: closed-form `k = Q/M`** (τ=0, isotropic source, slope 1.000, scatter ~1/√N).
4. **One bake alternation.** Without baking `k̂`, white per-PMT `k` leaks into the flattest global
   direction (e.g. `wall_refl`) and inflates its floor; baking `k̂` (estimate bakes as well as the
   oracle) restores it. Gauge: fix `mean(log k)=0` (else a `qe↔mean(k)` offset ~exp(ε²/2)).
   ⚠️ A SMOOTH (position-correlated) QE trend is the dangerous case — it lives in the optical band
   and, ignored, DIVERGES the reflectivities; per-sensor `k̂=Q/M` captures it, but needs a
   bootstrap `k̂` from a rough θ (naive `k̂=1` first pass diverges).

## Observable complementarity

- **Charge** — most parameters, especially anything coupling to total light + the longitudinal/
  spatial charge shape. Per-PMT `k` is pure charge.
- **Timing (first-arrival)** — geometric, ⊥ to absorption; it IS the lever for the
  scattering/reflection *geometry*: low-occupancy single-PE timing recovers `wall_refl` ~4–5×,
  `qe` ~1.4×, blue `L_abs` ~1.3×. It CANNOT measure `L_M` (Mie is optically thin → charge-
  invisible AND timing-invisible at the per-event level).
- **Charge variance** — breaks the per-PMT QE↔gain degeneracy + measures the SPE width `w`.

## Wavelength consistency (shared with reconstruction)

Calibration runs `wavelength_mode=True` and applies `qe_fn(λ)` per photon. The same
band-consistency principle from reconstruction applies: a broadband Cherenkov `qe(λ)` fit must
sample the **bare 1/λ² spectrum over the physical emission band** and apply differentiable
`qe(λ)` — the production `cherenkov_qe` importance-sampler bakes the unknown QE into the λ
distribution and cannot be used to FIT qe. Broadband sources measure only the spectrum integral
(amplitudes + `k`); **monochromatic lasers anchor `qe(λ_i)`**. The QE knot range is [294, 648] nm;
the medium grid is [300, 700] nm.

## Uncertainty: quote the CRB, not the sim toy-MC

Fit uncertainty = inverse Poisson Fisher at θ_true, `¼(JᵀJ)⁻¹` (autodiff jvp, log-params →
fractional). Representative: `L_M` σ ~5%, corr `L_abs↔qe` ≈ −0.78. ⚠️ The implicit-capture
(expected-value) engine is ~√12 quieter than real Poisson shot noise → noise-free fits look √12
too good. Use `sample_engine.py` (exact Bernoulli per-photon quanta) for honest shot-noise
validation; otherwise quote the CRB. Per-PMT `k` reaches ~1% at a few×10⁸ photons; smooth
λ-curves are systematics/prior-limited (flat in N), only per-PMT `k` is photon-limited (1/√N).

## Frontier

The biggest untapped lever is the **timing observable** for the reflectivities + blue `L_abs`
(all charge-only work to date zeroes reflected/late-photon time). The flexible-curve engine
(`wl_engine2`) supports free λ-deviation curves (28/30 DOF) for beyond-SK spectral calibration.
