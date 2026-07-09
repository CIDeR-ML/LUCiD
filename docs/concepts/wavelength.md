# Wavelength physics

LUCiD runs in one of two regimes, selected by `wavelength_mode` on
`setup_event_simulator`:

- **Scalar (monochromatic) mode** — every photon carries the same optical properties: the
  fittable scalars in `DetectorParams` (`scatter_length`, `mie_scatter_length`,
  `absorption_length`, `qe`, …). Fast, and sufficient whenever the wavelength dependence is
  not the question being asked.
- **Wavelength mode** — every photon carries its own λ, and scattering/absorption/QE are
  evaluated per photon from medium reference curves. This is the physically faithful mode
  used for spectral calibration and wavelength-consistent reconstruction.

## The medium reference

`lucid.wavelength.medium.make_medium(material, wavelength_grid)` builds the reference curves
from `config/materials/<material>.json`:

- **Rayleigh scattering** — a 1/λ⁴ law anchored to the material's measured scattering length
  (Mie scattering adds a second length and an asymmetry parameter `g` controlling how
  forward-peaked it is).
- **Absorption** — for water: the SK-calibration power law in the blue joined onto the
  Pope & Fry (1997) measured data in the red, blended smoothly across the ~464 nm seam so
  gradients stay kink-free. On top of *both* branches rides an always-on `P0·P1/λ⁴` term
  (`alpha_abs = P0·P1/λ⁴ + C` in `medium.py`, `C` = the blue-power-law-or-Pope-Fry piece) — the
  short-wavelength absorption tail is part of SK's own model, not a separate addition.
- **Refractive index** — currently a single constant `n` per material (no dispersion):
  the Cherenkov angle and photon timing use one `n`/light-speed per event. A per-λ index
  is a natural extension the medium format already reserves space for.

The medium grid spans [300, 700] nm by default, but is narrowed to the loaded PMT QE
curve's support when one is set — `[max(300, qe_lo), min(700, qe_hi)]` — so for the bundled
SK curve the effective grid is ≈[300, 648] nm. Materials are composable: `water`, `wbls`, and `ice`
each have their own JSON (WbLS inherits water's bulk optics; the bundled ice file is an
acknowledged placeholder using the water functional form).

## Reference × deviation: what is fittable

Per-photon evaluation happens in one place — `lucid.wavelength.optical_model.
evaluate_optical_model` — under a single reference-and-deviation decomposition: the fixed medium reference sets the
λ-shape, and a short fittable deviation curve rescales it. The deviation multiplies the
*interaction strength*, so for QE the per-photon weight is

```
qe(λ)      = qe_ref(λ)  ×  dev(λ)            # more deviation → more detected light
L_prop(λ)  = L_ref(λ)   /  dev(λ)            # more deviation → shorter length (Rayleigh, Mie, absorption)
```

Each optical property carries its `*_dev` curve in `DetectorParams` (`rayleigh_dev`,
`mie_dev`, `abs_dev`, `qe_dev`) — one
value per control wavelength, anchored at the SK calibration-laser lines
(337, 375, 398, 405, 445 nm — though 398 nm is SK's timing/diffuser laser rather than one of the
transparency lasers) and interpolated to each photon's λ with flat extrapolation
outside the grid. A deviation curve of all ones reproduces the pure reference exactly, so
calibration fits *departures* from known physics rather than refitting the physics itself.

## QE and Cherenkov sampling

- **QE**: in wavelength mode each photon is weighted by `qe_fn(λ)` built from the PMT's
  measured QE curve (`config/pmt/`; e.g. the bundled SK curve spans ≈294–648 nm); in scalar mode a single QE
  enters at hit-making instead.
- **Cherenkov spectrum**: track sources sample the 1/λ² Cherenkov spectrum over the physical
  emission band (`cherenkov_emission_band` on `setup_event_simulator`) and weight each photon
  by `qe_fn(λ)`. Note that a photon's λ is *clamped to the medium grid* before the QE lookup, so
  a λ falling outside the grid takes the nearest grid-edge QE value rather than a zero weight —
  keep the emission band inside the grid/QE support if you want out-of-band light genuinely
  suppressed.
- **Scalar projection**: configs that choose a scalar representation for a property get it
  projected from the referenced λ-curve at `scalar_ref_wavelength` (400 nm by default), so
  the two regimes agree at the reference wavelength by construction.

## `wavelength_sampling`: bare spectrum vs QE-weighted

When LUCiD samples the wavelengths itself (track mode, or calibration with `source.wavelength=None`),
`wavelength_sampling` on `setup_event_simulator` chooses *how*:

- **`'cherenkov'`** (default) — sample λ from the bare `1/λ²` Cherenkov spectrum and give each
  photon the explicit per-photon weight `qe_fn(λ)`. The QE curve is carried *outside* the sampler,
  so it stays visible as a distinct factor in every photon's weight.
- **`'cherenkov_qe'`** — sample λ from `QE(λ)/λ²` directly (inverse-CDF), so the per-photon weight
  collapses to the scalar mean `⟨QE⟩_C`. This is variance-optimal in expected-value mode (it puts
  samples where they are detectable), but it *folds QE into the sampling distribution*. Rejected at
  setup when `wavelength_mode=False`, no QE curve is loaded, or `is_data=True`.

The practical consequence: **fitting QE requires the bare spectrum (`'cherenkov'`)**. Once QE is
baked into the sampling law (`'cherenkov_qe'`), the per-photon weight no longer carries `qe(λ)` as a
free factor, so a gradient of the loss w.r.t. the QE curve has nothing to act on. Use
`'cherenkov_qe'` for variance-reduced forward densities; use `'cherenkov'` whenever QE (or its
`qe_dev` deviation curve) is being calibrated.

See [Parameters (DetectorParams)](detector-params-vs-args.md) for how the fittable leaves
are organized, and [Calibration](../guides/calibration.md) for how the deviation curves are
constrained by laser + isotropic source data.
