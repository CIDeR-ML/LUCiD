# Calibration

Calibration fits the detector's optical parameters (scattering and absorption lengths,
reflectivities, QE) and one efficiency per PMT to the charge seen from known light sources,
by gradient-based fitting through the forward model. It uses lasers and flashers, not a particle
emitter.

Runnable: `examples/hello_calibrate.py` (seven parameters and 10764 per-PMT gains, about 4
minutes on a GPU) and `examples/hello_calibrate_own_optimiser.py` (one problem, four solvers).
Both fit scalar optics (`wavelength_mode=False`).

## Minimal calibration

The core of `examples/hello_calibrate.py`:

```python
--8<-- "examples/hello_calibrate.py:calibrate"
```

The pieces, all in `lucid.fitting`:

- `FieldParams(dp, fields)`: which `DetectorParams` fields to fit, in log space; the rest of
  `dp` stays fixed. `params.theta_from_physical()` is `dp` as a fit vector, `params.to_real(theta)`
  the reverse.
- `CalibrationForward(sim, sources, params)`: the model, the predicted charge of every source.
  `sim` is a `source='light_source'` simulator.
- `closure_data(model, theta, gains, n_draws=8)`: pseudo-data from the model at known
  parameters and gains. With real data, use the observed charge, shape `(n_sources, n_sensors)`.
- `CalibrationProblem(model, data)`: the model, the data and the estimator.
- `calibrate(problem, theta0)`: the fit. It returns a dict with the fitted parameters
  (`'real'`) and the per-PMT gains (`'gains'`).
- `crb(model, theta)`: the Cramér-Rao bound, from the model alone.

Run the closure test first on a new detector: you set the truth, so a poor recovery is a
property of the setup (sources, photon count, parameterisation) and not of the data. Average the
pseudo-data over several draws (`n_draws=8`, the default): the estimator weights each sensor by
its observed charge, and one noisy draw at about 1.6 photoelectrons per sensor biases the fit.

`calibrate` defaults: 600 steps, a Jacobian rebuild every 20 (`refresh`), a 0.5 clip on each
log-space step (`max_step`), and the mean of the last 50 iterates as the result (`polyak`).
`CalibrationProblem` averages 8 Jacobian draws (`jacobian_draws`). The example lowers these for
speed. The paper's working point (`CALIB` in `analysis/paper/utils/calibration.py`) runs 1200
steps: the diffuse wall and sensor reflectances need about that many, while the other
parameters settle by about 300.

To hold parameters fixed, fit fewer fields, or keep the layout and pass
`calibrate(problem, theta0, fix=[...])`.

## Choosing a solver

The snippets are from `examples/hello_calibrate_own_optimiser.py`, which runs all four from one
start and prints them against the bound.

The default, damped Gauss-Newton with the per-PMT gains profiled (`tx=gauss_newton_tx(lam, mu)`
changes the damping):

```python
--8<-- "examples/hello_calibrate_own_optimiser.py:gn"
```

Any optax rule, on the same driver (step clip, Polyak readout and `fix` still apply):

```python
--8<-- "examples/hello_calibrate_own_optimiser.py:adam"
```

Your own loop: `problem.grad_metric_loss(theta, step)` returns the gradient `Jᵀr` (gains
profiled), the metric `JᵀJ` and the loss; `step` selects the random draw:

```python
--8<-- "examples/hello_calibrate_own_optimiser.py:own"
```

scipy, on one fixed draw so the objective is deterministic:

```python
--8<-- "examples/hello_calibrate_own_optimiser.py:scipy"
```

First-order rules such as Adam move slowly along stiff directions (Mie, `g`, the
reflectivities), which is why Gauss-Newton is the default.

## What can be fitted

Any leaf of `DetectorParams` ([parameters](../reference/detector-params.md)) can be named in
`FieldParams`. The usual ones are `scatter_length` (Rayleigh), `mie_scatter_length` with `g`,
`absorption_length`, `wall_reflection_rate`, `sensor_reflection_rate` and `qe`. The per-PMT
gains are `qe_corrections`, profiled rather than fitted (below). The refractive index is not a
parameter.

The reflection model is a simulator argument. The default `'scalar_mix'` adds the
specular/diffuse fractions `wall_fspec` and `sensor_fspec` as fittable parameters without
needing wavelength mode. `'angular'` makes the reflectance depend on angle and wavelength
(Schlick wall `wall_R0`, `wall_p`; Fresnel cathode `cathode_nr`, `cathode_nk`), evaluated at
`reflection_wavelength` (default 400 nm).

## The estimator

- **Sources.** Mix lasers at several positions (and wavelengths) with an isotropic flasher. One
  source alone leaves pairs such as Mie length and gains, absorption and QE, or wall and sensor
  reflectivity degenerate.
- **Residual.** The Neyman χ² residual `(k·M − Q)/√max(Q, q_floor)`, with `M` the predicted and
  `Q` the observed charge. Its weight depends on the data only, so it is linear in the
  Monte-Carlo forward, which is redrawn every step. A residual nonlinear in the forward would
  converge to a biased point.
- **Gains.** The per-PMT gains `k = ΣQ/ΣM` are solved in closed form at every step and fixed to
  `mean(log k) = 0` (`gauge='linear'` gives `mean(k) = 1`). They never enter the optimiser. The
  profiled gain has a bias that grows with the forward's noise; `CalibrationProblem(...,
  n_forward_draws=N)` averages draws to reduce it.
- **Correlated QE trends.** A smooth, position-dependent QE pattern mimics the optical
  parameters and can pull the reflectivities off.

Charge constrains most parameters. First-arrival timing is nearly independent of absorption and
constrains reflection and scattering geometry, but at detector scale it does not measure the
Mie length. `lucid.fitting.calibrate_timing` recovers per-PMT `t0` and the global TTS from
per-PMT first-arrival means and variances over many flashes.

## Per-wavelength optics

`CalibrationParams(n_wavelengths=W)` fits a separate scattering length, absorption length and
QE at each of `W` laser wavelengths, with the gains and the four reflection parameters shared.
This is the paper's estimand (five SK laser lines). Mie is held fixed in it.

Fitting the wavelength deviation curves instead (`FieldParams(dp, ['abs_dev', ...])` with
`wavelength_mode=True` and monochromatic sources, `laser_source(..., wavelength=405)`) is
implemented but has not been validated end to end. A QE fit must keep
`wavelength_sampling='cherenkov'` ([optics](../concepts/optics.md#how-wavelengths-are-sampled)).

## Uncertainty

`crb(model, theta)['sigma']` is the fractional Cramér-Rao bound per parameter, from the
autodiff Fisher information of the Poisson likelihood; it needs no data and marginalises the
per-PMT gains. The expected-value forward has less variance than real shot noise, so the bound
is inflated by √12 by default (`honesty=`; `cov_raw` is the uninflated covariance). For the same
reason, a closure fit on expected-value pseudo-data looks better than a real one; to test against
shot noise, generate the data with `transport='sampled', detection='sampled'`.

## If the calibration misbehaves

- Parameters trade off, or the bound is huge in some direction: add source diversity.
- The fit settles at a biased point on closure data: average more draws (`n_draws`), raise the
  photon count, or average forward draws (`n_forward_draws`).
- The diffuse reflectances have not converged: run more steps (the paper uses 1200).
