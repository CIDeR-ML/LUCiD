# Calibration

How LUCiD calibrates detector optics — per-PMT quantum efficiency `k` plus the global optical
parameters (absorption `L_abs`, Rayleigh `L_R`, Mie `L_M`, wall/sensor reflectivity, QE
spectrum, `n_real`) — by gradient descent through the differentiable forward. See also:
`lucid.fitting` (`build_calibration_problem` / `fit` / `crb`), `examples/hello_calibrate.py`
(runnable end-to-end), and [wavelength physics](concepts/wavelength.md).

## What is measured

`DetectorParams` is a JAX pytree with normalize/denormalize + bounds, so every optical property
is directly optimizable. Two tiers:

- **Global optical params** (scalars or λ-curves): `L_abs`, `L_R`, `L_M`, `wall_refl`,
  `sensor_refl`, `qe` spectral multipliers, `n_real`, SPE width, etc.
- **per-PMT `k`** (the QE map — one multiplier per sensor, ~10⁴ of them).

## Recipe (per-sensor QE + all globals)

1. **Source diversity is the lever.** Mix wall lasers (several wavelengths and positions) with
   an isotropic volume source. Diversity breaks the key degeneracies — `L_M↔k`, `L_abs↔qe`,
   wall↔sensor reflectivity — and makes the scattering lengths measurable with a plain charge
   loss.
2. **Globals: a smoothed square-root-MSE loss**, not Poisson-NLL (the NLL's weighting biases
   the absorption/QE point at finite photon counts). The spatial smoothing acts as a frequency
   projector: smooth optical fields are low-frequency across the sensor array while per-PMT `k`
   is white, so smoothing isolates the globals. Optimizer: a consistent fixed-dataset
   Gauss-Newton with an additive ridge and a min-gradient readout.
3. **per-PMT `k`: closed-form `k = Q/M`** — the ratio of observed to predicted charge per
   sensor under an isotropic source.
4. **One bake alternation.** Without baking the estimated `k̂` back into the forward, white
   per-PMT variation leaks into the flattest global direction (typically a reflectivity) and
   inflates its uncertainty; one alternation restores it. Fix the gauge with `mean(log k)=0`,
   otherwise a global QE↔mean(k) offset is unconstrained. A **smooth, position-correlated QE
   trend is the dangerous case** — it mimics the optical fields, and if ignored it drags the
   reflectivities away; the per-sensor `k̂=Q/M` step captures it, but needs a bootstrap `k̂`
   from a rough global fit first.

## Observable complementarity

- **Charge** — constrains most parameters, especially anything coupling to total light or the
  spatial charge shape. Per-PMT `k` is pure charge.
- **Timing (first-arrival)** — geometric and largely orthogonal to absorption; it is the lever
  for the scattering/reflection *geometry*: low-occupancy single-PE timing constrains the wall
  reflectivity far better than charge alone. It cannot measure `L_M` at typical detector
  scales (Mie scattering is optically thin — nearly invisible to both charge and per-event
  timing).
- **Charge variance** — breaks the per-PMT QE↔gain degeneracy and measures the SPE width.

## Wavelength consistency (shared with reconstruction)

Calibration runs `wavelength_mode=True` and applies `qe_fn(λ)` per photon. The same
band-consistency principle from [reconstruction](RECONSTRUCTION.md) applies: a broadband
Cherenkov `qe(λ)` fit must sample the **bare 1/λ² spectrum over the physical emission band**
and apply differentiable `qe(λ)` — an importance sampler that bakes the assumed QE into the λ
distribution cannot be used to *fit* QE. Broadband sources measure only the spectrum integral
(amplitudes + `k`); **monochromatic lasers anchor `qe(λᵢ)`**. The QE knot range is
[294, 648] nm; the medium grid is [300, 700] nm.

## Uncertainty: quote the CRB, not the sim toy-MC

Fit uncertainty comes from the inverse Poisson Fisher information at the fitted point
(`lucid.fitting.crb`; autodiff JVPs, log-params give fractional errors). One caveat: the
expected-value (implicit-capture) engine is quieter than real Poisson shot noise, so noise-free
closure fits look better than reality — validate shot noise with sampled per-photon quanta, or
simply quote the CRB. Per-PMT `k` improves as 1/√N with photon statistics; smooth λ-curves
become systematics-limited rather than photon-limited.

## Frontier

The biggest untapped lever is the **timing observable** for the reflectivities and blue-end
absorption — charge-only calibration ignores the arrival-time information that reflected and
late photons carry. Free per-λ deviation curves on the optical properties extend the same
machinery to beyond-nominal spectral calibration.
