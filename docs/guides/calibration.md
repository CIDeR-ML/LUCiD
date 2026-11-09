# Calibration

How LUCiD calibrates detector optics — per-PMT quantum efficiency `k` plus the global optical
parameters (absorption `L_abs`, Rayleigh `L_R`, Mie `L_M`, wall/sensor reflectivity, QE
spectrum) — by gradient-based fitting through the differentiable forward. See also:
`lucid.fitting` (`CalibrationForward` / `CalibrationProblem` / `calibrate` / `crb`),
`examples/hello_calibrate.py` and `examples/hello_calibrate_own_optimiser.py` (runnable
end-to-end), and [wavelength physics](../concepts/wavelength.md).

!!! note "Runnable entry points"
    `examples/hello_calibrate.py` fits **scalar** optics (`wavelength_mode=False`). The spectral
    λ-deviation machinery described below is code-complete (`lucid.wavelength.optical_model`),
    but no wavelength-mode calibration has been run end to end yet.

## Minimal calibration

One model, one problem, any solver. A **model** (`CalibrationForward`) is the simulator, the
sources and the parameterisation: what the detector predicts. A **problem**
(`CalibrationProblem`) adds the observed charge and the estimator. `calibrate` runs a solver on
the problem, and `crb` takes the bound from the model alone. Calibration uses lasers and a
flasher (no SIREN emitter), so it needs no downloaded emitter weights. The core of
`examples/hello_calibrate.py`, included from the file:

```python
--8<-- "examples/hello_calibrate.py:calibrate"
```

The example's `steps`, `refresh`, `max_step`, `polyak=0` and `jacobian_draws=2` are chosen for
speed; without them `calibrate` and `CalibrationProblem` run the library defaults (below).

Run the closure first on any new detector: the truth is known because you set it, so a poor
recovery is a fact about the configuration (sources, photon budget, parameterisation) and not
about the data. Average the closure data over several draws (`closure_data`'s `n_draws=8`, the
published value): the Neyman weight is built from the data, so one noisy draw at ~1.6 PE per
sensor biases the fit.

`calibrate`'s defaults are 600 steps, a Jacobian refresh every 20, a 0.5 log-space step clip and a
50-iterate Polyak readout, with 8 Jacobian draws on the problem; pass `steps=` and `refresh=` to make
a first look cheaper. The paper states its own working point, `CALIB` in
`analysis/paper/utils/calibration.py`: the same estimator at **1200 steps** per seed, because the
diffuse wall and sensor reflectances reach their truth only by ~1200 steps (600 leaves them about
10% and 22% off; every other parameter settles by ~300). To hold some parameters where they are, fit
fewer fields (`FieldParams(dp, ['absorption_length'])`), or keep the layout and pass
`calibrate(problem, theta0, fix=[...])`. `CalibrationParams` is the published estimand: the
per-wavelength optics plus shared reflection the paper fits.

## Choosing a solver

The problem is the published estimator; the solver is an argument.

The snippets are from `examples/hello_calibrate_own_optimiser.py`, which runs all four on one
problem from one `start` and prints them against the bound.

**The default**, damped Gauss-Newton with the per-PMT gains profiled (another damping:
`tx=gauss_newton_tx(lam, mu)`, from `lucid.fitting`):

```python
--8<-- "examples/hello_calibrate_own_optimiser.py:gn"
```

**Any optax rule**, on the same driver (step clip, Polyak readout and `fix` still apply). A rule
that takes no curvature ignores the metric it is handed:

```python
--8<-- "examples/hello_calibrate_own_optimiser.py:adam"
```

**Your own loop.** `grad_metric_loss(theta, step)` returns the published gradient `Jᵀr` (gains
profiled, the Jacobian on a stream independent of the residual's), the metric `JᵀJ` and the loss;
`step` selects the Monte-Carlo draw:

```python
--8<-- "examples/hello_calibrate_own_optimiser.py:own"
```

**scipy**, on one fixed draw (common random numbers, so the objective is deterministic):

```python
--8<-- "examples/hello_calibrate_own_optimiser.py:scipy"
```

First-order rules crawl along the stiff directions (Mie, `g`, the reflectivities); that is why
Gauss-Newton is the default.

## What is measured

`DetectorParams` is a JAX pytree with normalize/denormalize + bounds, so every optical property
is directly optimizable. Two tiers:

- **Global optical params** (scalars or λ-curves): `L_abs`, `L_R`, `L_M`, `wall_refl`,
  `sensor_refl`, `qe` spectral multipliers, etc. The refractive index is a fixed
  argument, not a parameter ([parameters vs arguments](../concepts/detector-params-vs-args.md)).
- **per-PMT `k`** (the QE map, `PerPmtParams.qe_corrections` — one multiplier per sensor, ~10⁴
  of them; "gains" below means these multipliers).

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
(`wall_R0`/`wall_p`/`wall_fspec`, `cathode_nr`/`cathode_nk`/`sensor_fspec`; a physics config
states the first four in an `"angular_reflection"` block) and evaluates the PMT-glass dispersion at
one wavelength, the setup's `reflection_wavelength` (default 400 nm; exact for a monochromatic
laser). The **default** model is
`reflection_model='scalar_mix'`, which keeps the scalar rate *magnitudes* but adds the
specular/diffuse `*_fspec` direction split — so those fractions become fittable **without**
`wavelength_mode`, which is the cheaper lever for a scalar calibration.

## Recipe (per-sensor QE + all globals)

1. **Source diversity is the lever.** Mix wall lasers (several wavelengths and positions) with
   an isotropic volume source. Diversity breaks the key degeneracies — `L_M↔k`, `L_abs↔qe`,
   wall↔sensor reflectivity — and makes the scattering lengths measurable with a plain charge
   loss.
2. **Globals: a Neyman χ² residual**, `(k·M − Q)/√max(Q, q_floor)`. Its weight depends on the
   data alone, so the residual stays linear in the Monte-Carlo forward, which is redrawn every
   step; see *A residual that is not linear in the model* below. Optimizer: by default the
   damped Gauss-Newton shared with reconstruction (`gauss_newton_tx(lam, mu)`: Marquardt `lam`,
   Levenberg `mu`), with Polyak tail-averaging of the iterates (`polyak=`); see *Choosing a
   solver*.
3. **per-PMT `k`: profiled, `k = ΣQ/ΣM`**, solved in closed form at every step and gauged to
   `mean(log k)=0` (`gauge='linear'` for `mean(k)=1`). The gains never enter the optimizer. The
   profiled gain is nonlinear in the forward, so it carries a bias that grows with the forward's
   Monte-Carlo noise; averaging forward draws (`CalibrationProblem(n_forward_draws=)`) reduces it
   (`tests/fitting/test_fitting_estimator_unbiased.py` measures it).
4. **A smooth, position-correlated QE trend is the dangerous case.** It mimics the optical
   fields, and if ignored it drags the reflectivities away.

## Observable complementarity

- **Charge** — constrains most parameters, especially anything coupling to total light or the
  spatial charge shape. Per-PMT `k` is pure charge.
- **Timing (first-arrival)** — geometric and largely orthogonal to absorption; it is the lever
  for the scattering/reflection *geometry*: low-occupancy single-PE timing constrains the wall
  reflectivity far better than charge alone. It cannot measure `L_M` at typical detector
  scales (Mie scattering is optically thin — nearly invisible to both charge and per-event
  timing).
- **Charge spread** is not a fitted observable: the only per-PMT factor is `qe_corrections` (the
  profiled `k`), and `DetectorParams` has no gain or SPE-width field. The charge resolution of
  simulated data is the simulator's `charge_resolution` argument (sampled hits) or the production
  digitizer's SPE model ([digitizer & trigger](../reference/digitizer-and-trigger.md)).

## Wavelength consistency (shared with reconstruction)

In wavelength mode (`wavelength_mode=True` — the spectral path noted at the top of this
page), calibration applies `qe_fn(λ)` per photon. A QE fit keeps `setup_event_simulator`'s
default `wavelength_sampling='cherenkov'`: `'cherenkov_qe'` folds the assumed QE into the
sampling law and leaves the gradient nothing to act on
([wavelength physics](../concepts/wavelength.md#wavelength_sampling-bare-spectrum-vs-qe-weighted)).
Broadband sources measure only the spectrum integral (amplitudes + `k`); **monochromatic lasers
anchor `qe(λᵢ)`**.

Wavelength-dependent optics have two representations, and they answer different questions:

- **Effective scalar optics per wavelength**: `CalibrationParams(n_wavelengths=W)` evaluates the
  same sources once per laser wavelength, each row with its own scalar scattering, absorption and
  QE, sharing the gains and the reflection. This is what the paper fits (five SK laser lines).
- **The spectral curve**: `FieldParams(dp, ['abs_dev', ...])` with `wavelength_mode=True` and
  monochromatic sources (`laser_source(position, wavelength=405)`) fits the λ-deviation curves as array
  leaves. Code-complete, not yet run end to end.

## Uncertainty: quote the CRB, not the sim toy-MC

Fit uncertainty comes from the inverse Poisson Fisher information of the model at a point,
`crb(model, theta)['sigma']` (autodiff JVPs; log-params give fractional errors). It needs no
data, works for `FieldParams` and `CalibrationParams` alike, and marginalises the per-PMT gains
rather than profiling them, because a bound must integrate over a nuisance where a fit may profile
it. One caveat: the
expected-value (implicit-capture) engine is quieter than real Poisson shot noise, so noise-free
closure fits look better than reality — validate shot noise with sampled per-photon quanta, or
simply quote the CRB. Per-PMT `k` improves as 1/√N with photon statistics; smooth λ-curves
become systematics-limited rather than photon-limited.

## If the calibration misbehaves

- **A degenerate fit** (parameters trade off, huge CRB on some direction) — add **source
  diversity**. A single source leaves `L_M↔k`, `L_abs↔qe`, and wall↔sensor reflectivity
  degenerate; mixing wall lasers (several positions/wavelengths) with an isotropic flasher breaks
  them and makes the scattering lengths measurable.
- **CRB disagrees with a sim toy-MC** — expected; see *Uncertainty* above (the `crb` bound
  carries a ×√12 honesty factor for it). Validate with sampled per-photon quanta
  (`transport='sampled', detection='sampled'`), or just quote the CRB.
- **A residual that is not linear in the model** — the forward is a Monte-Carlo estimate redrawn
  every step, so any residual nonlinear in it has `E[f(M)] ≠ f(E[M])` and a permanently displaced
  fixed point; no amount of iteration recovers. The fitter therefore runs the **Neyman** χ²,
  `(k·M − Q)/√Q`, the member of the family whose weight depends on the data alone.

## Frontier

The biggest untapped lever is the **timing observable** for the reflectivities and blue-end
absorption — charge-only calibration ignores the arrival-time information that reflected and
late photons carry. Free per-λ deviation curves on the optical properties extend the same
machinery to beyond-nominal spectral calibration.
