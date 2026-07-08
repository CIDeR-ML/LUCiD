# Calibration

How LUCiD calibrates detector optics — per-PMT quantum efficiency `k` plus the global optical
parameters (absorption `L_abs`, Rayleigh `L_R`, Mie `L_M`, wall/sensor reflectivity, QE
spectrum, `n_real`) — by gradient descent through the differentiable forward. See also:
`lucid.fitting` (`build_calibration_problem` / `fit` / `crb`), `examples/hello_calibrate.py`
(runnable end-to-end), and [wavelength physics](../concepts/wavelength.md).

!!! note "Runnable entry points"
    `examples/hello_calibrate.py` and the `calibration_optimization` tutorial fit **scalar**
    optics (`wavelength_mode=False`). The spectral λ-deviation machinery described below is
    code-complete (`lucid.wavelength.optical_model`), but a wavelength-mode calibration
    notebook does not exist yet.

## Minimal calibration

Build the problem, take the Cramér-Rao bound at truth, fit — the condensed shape of
`examples/hello_calibrate.py`. Calibration uses lasers + a flasher (no SIREN emitter), so it
needs no downloaded emitter weights:

```python
import jax, jax.numpy as jnp, numpy as np
from lucid.geometry import generate_detector
from lucid.simulation import setup_event_simulator
from lucid.sources import laser_source, isotropic_source
from lucid.detector_params import DetectorParams
from lucid.fitting import build_calibration_problem, fit, crb

GEOM = 'config/SK_like_geom_config.json'
det = generate_detector(GEOM); NS = len(det.all_points)
top, bot, R = det.H/2 - .1, -det.H/2 + .1, det.r

# truth optics; per-PMT qe_corrections is the k map (all-ones here)
dp = DetectorParams.from_flat(scatter_length=70., mie_scatter_length=3000., g=0.9,
                              absorption_length=60., wall_reflection_rate=.2,
                              sensor_reflection_rate=.2, qe=0.07, qe_corrections=jnp.ones(NS))

# source diversity is the lever: wall lasers (several positions) + an isotropic flasher
sources = [laser_source(position=[0, 0, top], direction=[0, 0, -1], intensity=1e6),
           laser_source(position=[0, 0, bot], direction=[0, 0,  1], intensity=1e6),
           laser_source(position=[R-.1, 0, 0], direction=[-1, 0, 0], intensity=1e6),
           isotropic_source(position=[0, 0, 0], intensity=1e6)]
sim = setup_event_simulator(GEOM, 1_000_000, temperature=None, K=8, is_calibration=True,
                            hit_mode='aggregated', wavelength_mode=False)

# globals to recover; the ~10^4 per-PMT k are marginalized analytically (Schur complement)
FIELDS = ['g', 'scatter_length', 'mie_scatter_length', 'absorption_length',
          'wall_reflection_rate', 'sensor_reflection_rate', 'qe']
prob  = build_calibration_problem(sim, sources, dp, FIELDS, key=jax.random.PRNGKey(1))
sigma = crb(prob['source_models'], prob['theta_true'], NS)['sigma']       # Cramer-Rao bound, per field
start = prob['theta0'] + np.random.default_rng(0).uniform(-.15, .15, prob['theta0'].shape)
res   = fit(prob['source_models'], prob['truth_charge'], start, NS, steps=100, refresh=15, nb_h=2)
# res['theta'] = fitted globals; compare to np.exp(prob['theta0']) (truth) and sigma (the bound).
# theta0/theta_true are in log space; res['k'] holds the recovered per-PMT map.
```

## What is measured

`DetectorParams` is a JAX pytree with normalize/denormalize + bounds, so every optical property
is directly optimizable. Two tiers:

- **Global optical params** (scalars or λ-curves): `L_abs`, `L_R`, `L_M`, `wall_refl`,
  `sensor_refl`, `qe` spectral multipliers, `n_real`, SPE width, etc.
- **per-PMT `k`** (the QE map — one multiplier per sensor, ~10⁴ of them).

The shorthand above maps to the physics-config / `DetectorParams.from_flat` field names:

| shorthand | config key |
|-----------|-----------|
| `L_abs` | `absorption_length` |
| `L_R` (Rayleigh) | `scatter_length` |
| `L_M` (Mie) | `mie_scatter_length` (with asymmetry `g`) |
| `wall_refl` | `wall_reflection_rate` |
| `sensor_refl` | `sensor_reflection_rate` |
| `qe` | `qe` |
| per-PMT `k` | `qe_corrections` |

With `reflection_model='angular'`, reflection gains fittable levers beyond the two scalar
rates: the reflectance *magnitude* is angle- and λ-dependent (Schlick blacksheet wall +
multilayer-Fresnel cathode), and the reflected *direction* is a specular/diffuse mixture with
per-surface fractions `wall_fspec`/`sensor_fspec` (the discrete specular-vs-diffuse branch is
carried by a DiCE score). The angular model reads the `DetectorParams.reflection` fields
(`wall_R0`/`wall_p`/`wall_fspec`, `cathode_nr`/`cathode_nk`/`sensor_fspec`) and **requires
`wavelength_mode=True`** for the λ-dependent magnitude; the `*_fspec` fields default to 0 (fully
diffuse) and are inert under the default `reflection_model='scalar'`.

## Recipe (per-sensor QE + all globals)

1. **Source diversity is the lever.** Mix wall lasers (several wavelengths and positions) with
   an isotropic volume source. Diversity breaks the key degeneracies — `L_M↔k`, `L_abs↔qe`,
   wall↔sensor reflectivity — and makes the scattering lengths measurable with a plain charge
   loss.
2. **Globals: a smoothed square-root-MSE loss**, not Poisson-NLL (the NLL's weighting biases
   the absorption/QE point at finite photon counts). The spatial smoothing acts as a frequency
   projector: smooth optical fields are low-frequency across the sensor array while per-PMT `k`
   is white, so smoothing isolates the globals. Optimizer: a consistent fixed-dataset
   Gauss-Newton with an additive ridge (optional Polyak tail-averaging of the iterates).
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

In wavelength mode (`wavelength_mode=True` — the spectral path noted at the top of this
page), calibration applies `qe_fn(λ)` per photon. The same
band-consistency principle from [reconstruction](reconstruction.md) applies: a broadband
Cherenkov `qe(λ)` fit must sample the **bare 1/λ² spectrum over the physical emission band**
and apply differentiable `qe(λ)` — i.e. keep `setup_event_simulator`'s default
`wavelength_sampling='cherenkov'` (λ ~ 1/λ², per-photon weight `qe_fn(λ)`). The alternative
`wavelength_sampling='cherenkov_qe'` importance-samples λ ~ QE(λ)/λ² and collapses the weight to
a scalar ⟨QE⟩, baking the assumed QE into the λ distribution — so it **cannot be used to *fit*
QE**. Broadband sources measure only the spectrum integral
(amplitudes + `k`); **monochromatic lasers anchor `qe(λᵢ)`**. The bundled SK QE curve
spans ≈294–648 nm; the medium grid is [300, 700] nm.

## Uncertainty: quote the CRB, not the sim toy-MC

Fit uncertainty comes from the inverse Poisson Fisher information at the fitted point
(`lucid.fitting.crb`; autodiff JVPs, log-params give fractional errors). One caveat: the
expected-value (implicit-capture) engine is quieter than real Poisson shot noise, so noise-free
closure fits look better than reality — validate shot noise with sampled per-photon quanta, or
simply quote the CRB. Per-PMT `k` improves as 1/√N with photon statistics; smooth λ-curves
become systematics-limited rather than photon-limited.

## If the calibration misbehaves

- **A degenerate fit** (parameters trade off, huge CRB on some direction) — add **source
  diversity**. A single source leaves `L_M↔k`, `L_abs↔qe`, and wall↔sensor reflectivity
  degenerate; mixing wall lasers (several positions/wavelengths) with an isotropic flasher breaks
  them and makes the scattering lengths measurable.
- **per-PMT `k` runs away** (a global QE↔mean-`k` offset drifts) — fix the gauge with
  `mean(log k)=0` (the `gauge_k` step, on by default). If white per-PMT variation is leaking into
  the flattest global (usually a reflectivity), run **one bake alternation** (`bake_k=True`) to
  fold `k̂` back into the forward.
- **CRB disagrees with a sim toy-MC** — expected: the expected-value (implicit-capture) engine is
  quieter than real Poisson shot noise (the `crb` bound carries a ×√12 honesty factor). Validate
  with sampled per-photon quanta (`use_expected_value=False`), or just quote the CRB.
- **Poisson-NLL biases `L_abs`/`qe`** — use the smoothed square-root-MSE loss instead (the NLL's
  finite-count weighting biases the absorption/QE point); the smoothing also isolates the
  low-frequency globals from the white per-PMT `k`.

## Frontier

The biggest untapped lever is the **timing observable** for the reflectivities and blue-end
absorption — charge-only calibration ignores the arrival-time information that reflected and
late photons carry. Free per-λ deviation curves on the optical properties extend the same
machinery to beyond-nominal spectral calibration.
