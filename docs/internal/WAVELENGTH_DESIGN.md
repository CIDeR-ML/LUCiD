# Wavelength handling: now vs. how it should be (DetectorParams / pytree design)

## 1. How it is NOW — two mutually-exclusive modes, neither fittable-and-λ-dependent

`DetectorParams` (NamedTuple pytree) holds **scalar** optical fields: `scatter_length, g, mie_scatter_length,
wall_reflection_rate, sensor_reflection_rate, absorption_length, qe` (each a SINGLE value) + one per-PMT array
`qe_corrections (NS,)`. `setup_event_simulator._get_optical_arrays(n, dp, key, wavelengths)` then branches:

- **`wavelength_mode=False`** → broadcast the DetectorParams **scalars** to all photons. These are FITTABLE, but
  λ-INDEPENDENT (one value, projected from a curve at a 400 nm reference via `_project_missing_scalars`).
- **`wavelength_mode=True`** → per-photon `scatter/mie/absorption = 1/medium.coeff(λ)` interpolated from the **medium
  model** at each photon's λ. Full λ-dependence, but **FIXED** (the medium is an external file; DetectorParams' optical
  scalars are IGNORED). The QE curve is likewise external/fixed.

**The gap:** there is NO representation for a *fittable, wavelength-dependent* optical property. You get either
(fittable scalar, no λ) XOR (full λ-curve, not fittable). Calibration needs both at once — that is the whole point of
the flexible-curve work. Reflection is worse: only a scalar `wall/sensor_reflection_rate`, no angle and no λ.

Contrast with calibration mode (`mie_hunter/wl_engine2`): a `C=(6,5)` array of control-point deviations + a `(4,)`
scalar array, passed POSITIONALLY, decoded by a bespoke `asm()`, with bounds done by inline `clip`. Flexible and
λ-fittable, but un-named, not a bounded pytree, fragile.

## 2. How it SHOULD be — one decomposition: [fixed reference shape] × [fittable deviation]

Every optical property = **reference(λ)** (known physics, FIXED, external) **× deviation(λ; DetectorParams)** (FITTABLE).
This is exactly what the calibration work established physically:
- **Reference shapes (FIXED, NOT in DetectorParams):** Rayleigh 1/λ⁴, pure-water absorption (Pope–Fry), glass Sellmeier
  dispersion, Mie phase shape, bench bialkali QE(λ), n_imag. These live in the medium/optical-model as constants.
- **Fittable deviations (IN DetectorParams):** an **amplitude** (1 scalar) where the shape is known (Rayleigh, Mie),
  or a **control-point curve** (`(n_ctrl,)` array at the laser anchor λ's) where the shape is impurity/material-set
  (absorption, cathode n_real, blacksheet R0, QE deviation).

So the per-photon optical property is `prop(λ) = ref(λ) · interp(λ, control_λ, deviation_field)`, with `control_λ` a
FIXED grid (the laser anchors, SK 337/375/398/405/445). ONE representation, λ-fittable, and it collapses to the scalar
case when the deviation is a length-1 amplitude or all photons share one λ.

## 3. The pytree representation (extend DetectorParams cleanly)

Keep the NamedTuple style — it already proves arrays work (`qe_corrections (NS,)`) and the bounds/normalize/mask/
grad-scale helpers are all `jax.tree.map` (element-wise, so array fields just work). DetectorParams becomes a flat
pytree whose leaves are EXACTLY the fittable DOF:

```
class DetectorParams(NamedTuple):
    # --- scattering (amplitude on known shapes) ---
    rayleigh_amp: f32[]            mie_amp: f32[]            g: f32[]
    # --- absorption (FREE curve) ---
    abs_curve: f32[n_ctrl]        # deviation × pure-water shape at control λ
    # --- QE (bench shape × free deviation × per-PMT) ---
    qe_amp: f32[]   qe_curve: f32[n_ctrl]   qe_corrections: f32[NS]
    # --- blacksheet reflection ---
    wall_R0_curve: f32[n_ctrl]    wall_p: f32[]   wall_fspec: f32[]
    # --- PMT reflection (multilayer) ---
    cathode_nr_curve: f32[n_ctrl] cathode_nk: f32[]  sensor_fspec: f32[]
    # --- per-PMT charge/timing (calibration) ---
    gain: f32[NS]   t0: f32[NS]   walk: f32[NS]   spe_width: f32[]   tts: f32[]
```

The `control_λ` grid is a module constant / config, NOT a field. Reference shapes are NOT fields. This makes
DetectorParams == the optimization space, and the whole bounds/normalize/mask machinery already handles it.

**Reconciling with the mie_hunter array style:** `jax.flatten_util.ravel_pytree(dp)` → the flat θ the GN/Schur fitter
wants (exactly mie_hunter's vector); unravel → the named pytree the forward wants. The `asm()` becomes
(unravel + per-field transform). Per-field log/linear transform + gauge + prior + which-observable are declared ONCE in
a small **param registry** keyed by field name (replaces every bespoke `asm()`). Best of both: named/bounded/composable
pytree for the model, flat vector for the optimizer.

## 4. The optical-model boundary (code cleanliness) — pre-evaluate λ, keep the step λ-resolved-scalar

`_get_optical_arrays` should become a standalone, testable `optical_model(dp, wavelengths, medium, control_λ) ->
OpticalArrays` (a small NamedTuple of per-photon arrays: `scatter_len, mie_len, abs_len, R0_wall, nr_cathode, qe`, each
`(n_photon,)`). Computed ONCE before the scan (optical props are constant over a photon's life — elastic). The
photon_step then takes per-photon **scalars** (clean vmap), and computes only the **angle-dependent** part in-step.
Eliminate the `wavelength_mode` True/False branch: ALWAYS evaluate the optical model at the photon λ (monochromatic =
all photons one λ). One path, not two.

**Reflection is the subtle one — split λ (pre-scan) from angle (in-step).** R0_wall(λ) and n_real(λ) are pre-evaluated
per-photon; the angle `cth_inc = |dir·sg(normal)|` is only known IN the step. So pass per-photon `R0(λ)`/`nr(λ)` into the
step and compute `R(θ,λ)` there. Make the reflection a **pluggable function** (your point #2):
`reflection_fn(cth_inc, R0_λ, wall_p, nr_λ, nk, fspec, hit_sensor) -> (refl_prob, refl_dir)`. Scalar / Schlick /
multilayer-Fresnel are interchangeable functions; the step is agnostic. (The magnitude params stay pathwise-exact since
`cth_inc` uses `sg(normal)` — verified in refl_check2.)

## 5. Specific cases (the design must handle all of these the same way)

1. **Monochromatic laser @440** — all photons λ=440; `optical_model` evaluates the curves at 440 (interp picks the 440
   control point + neighbours). Only the relevant control points get gradient. The pytree holds the full curve; the data
   constrains the local part. Same code path.
2. **Broadband Cherenkov iso (1/λ²)** — per-photon λ varies; `optical_model` is vmapped over λ → the full curve is
   constrained by the spread. Same code path.
3. **Reconstruction (SIREN track)** — detector is KNOWN (fixed DetectorParams), free params = track. `optical_model`
   just EVALUATES the (fixed) curves at the track's λ — single λ for speed, or per-photon. Recon pays nothing extra and
   uses the SAME DetectorParams + optical model. The wavelength machinery is a no-op cost when not fitted.
4. **Data mode (ROOT photons, external λ)** — wavelengths supplied → evaluate the curves at them. Same path; no
   sampling. (Matches the existing "ignored when caller supplies wavelengths" contract.)
5. **A genuinely FIXED property (glass dispersion, Mie g-shape, n_imag)** — NOT a DetectorParams field; a constant in
   the optical model. DetectorParams stays exactly the fittable DOF; nothing fixed leaks into the optimizer.
6. **Per-PMT × λ (QE)** — composes as a product: `qe(p,λ) = bench(λ)·qe_amp·interp(λ,ctrl,qe_curve)·qe_corrections[p]`.
   Per-PMT factor is λ-flat (NS scalars), λ-shape is global (n_ctrl). They multiply; no NS×n_ctrl blowup (SK doesn't fit
   per-PMT λ-curves either).
7. **The medium is the reference, the curve is the deviation** — `abs_len(λ) = (1/medium.abs_coeff(λ)) /
   interp(λ,ctrl,abs_curve)`; truth `abs_curve≈1` reproduces pure-water; the fit learns the impurity deviation. Rayleigh
   `abs_curve`→amplitude on 1/λ⁴; the small dispersion residual can be a fixed-or-free 1-or-n_ctrl field.

## 6. custom_vjp — where it is genuinely needed

The `_bwd` blanket `nan_to_num` on all cotangents is a 2nd-order-AD safety net for the RECON path (the FD/HVP Hessian
hits sqrt/normalize/log singularities). For CALIBRATION (1st-order GN with `jvp` Jacobians), it is mostly NOT needed —
**if the forward is NaN-clean** (the real sources are already fixed: `CYL_SQRT_EPS`, SAFE sqrt-norm in the simulator,
the mie/g loader projection, the `first_arrival_nll` empty-segment floor). Plan: KEEP the `custom_vjp` (it's the place
to do principled gradient surgery), but (a) make the `nan_to_num` **opt-in + logged** rather than a silent blanket zero —
a zeroed NaN gradient is a silent wrong answer for calibration; (b) drive NaNs to zero at the SOURCE (the sqrt/log
guards) so sanitization is a backstop, not the mechanism; (c) it's the natural hook for the reflection-direction
score / variance reduction later. So: understand it as the *recon 2nd-order backstop*, not a forward-correctness device.

## 7. One-line takeaway
Today optical params are either fittable-scalars (no λ) or fixed-medium-curves (not fittable). Unify both as
`prop(λ) = ref(λ) · interp(λ, control_λ, deviation)`, put the **deviation** (amplitude scalar OR control-point curve)
as named leaves in DetectorParams (== the optimization space), keep reference shapes/glass/g/n_imag as fixed model
constants, pre-evaluate everything λ-dependent into a per-photon `OpticalArrays` before the scan, compute only the
angle-dependent reflection in-step via a pluggable `reflection_fn`, and use `ravel_pytree` to bridge the named pytree
(model) ↔ flat θ (GN/Schur). Recon and calibration then share the SAME DetectorParams + optical model; only the free
set differs.
