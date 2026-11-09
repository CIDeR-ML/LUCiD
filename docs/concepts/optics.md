# Optics: media and wavelength

## Media

A detector's medium is the `material` of its geometry config, read from
`config/materials/<material>.json`. Three ship:

| material | file | contents |
|---|---|---|
| water | `config/materials/water.json` | Rayleigh and Mie scattering and absorption from the SK calibration (arXiv:1307.0162); above 464 nm the absorption follows Pope & Fry (1997), blended smoothly across the join. Refractive index 1.33 |
| WbLS | `config/materials/wbls.json` | 10% water-based liquid scintillator: water's bulk optics plus a scintillation block (below) |
| ice | `config/materials/ice.json` | a placeholder: water's functional forms with ice coefficients and index 1.309. Fine for geometry and workflow studies, not for ice-optics precision |

All three use the water SIREN emitters (`"siren": "water"`). The refractive index is one
constant per medium: the Cherenkov angle and the light speed `c/n` used for timing have no
dispersion.

## Scalar and wavelength mode

`setup_event_simulator(..., wavelength_mode=...)` picks one of two regimes:

- `wavelength_mode=False` (scalar): every photon sees the same optics, the scalar fields of
  `DetectorParams` (`scatter_length`, `mie_scatter_length`, `absorption_length`, `qe`, ...).
  Faster, and enough when the wavelength dependence is not the question.
- `wavelength_mode=True` (the default): each photon carries a wavelength, and scattering,
  absorption and QE are evaluated for it from the medium's curves and the PMT's QE curve. The
  medium covers 300 to 700 nm.

A physics config that omits a scalar optical length, `g` or `qe` gets it from the medium and QE
curves at 400 nm, so the two regimes agree there. The optical lengths given under
`scalar_mode` in a physics config are used only in scalar mode.

### Reference and deviation

In wavelength mode, each optical property is a fixed reference curve (from the medium or QE file)
times a fittable deviation curve in `DetectorParams`: `rayleigh_dev`, `mie_dev`, `abs_dev` and
`qe_dev`. Each has one value per control wavelength (337, 375, 398, 405 and 445 nm, the SK
laser lines), interpolated to the photon's wavelength and held flat outside that range. The
deviation multiplies the interaction strength:

```
qe(λ) = qe_ref(λ) × dev(λ)
L(λ)  = L_ref(λ) / dev(λ)      # Rayleigh, Mie and absorption lengths
```

All-ones deviations reproduce the reference, so calibration fits departures from it.

### QE

`qe` in `DetectorParams` is the QE at 400 nm, for light arriving at the PMT with its reflection
included, as PMT QE is quoted. In wavelength mode each photon is weighted by
`qe × qe_fn(λ) / qe_fn(400 nm)`, where `qe_fn` is the QE curve named by the physics config's
`qe_curve` (`config/pmt/`; the SK curve spans 294 to 648 nm). The photon step reflects part of
the light off the PMT and converts the rest at `QE / (1 − R0)`, with `R0` the sensor's
normal-incidence reflectance, so the net detection is the quoted QE. Setup rejects parameters
with a peak QE above `1 − R0`.

### How wavelengths are sampled

Track sources sample the Cherenkov `1/λ²` spectrum over the emitter's `emission_band_nm`, the
band its photon yield `N_photons(E)` was counted over (274.91 to 673.83 nm for the shipped
emitters). Photons outside the QE curve or the medium range count nothing, as they should.
`cherenkov_emission_band` overrides the band.

When LUCiD samples wavelengths (a track, or a calibration source with `wavelength=None`),
`wavelength_sampling` decides how:

- `'cherenkov'` (default): sample `1/λ²` and weight each photon by `qe_fn(λ)`.
- `'cherenkov_qe'`: sample `QE(λ)/λ²` directly, so every photon has the same weight. Lower
  variance for an expected forward, but the QE is then inside the sampling distribution and a
  gradient with respect to the QE curve has nothing to act on. Do not use it when fitting QE or
  `qe_dev`. It is rejected with `wavelength_mode=False`, without a QE curve, and for
  `source='photons'`.

## WbLS scintillation

Scintillation needs `wavelength_mode=True`; setup raises for a scintillating medium in scalar
mode. The physics config inherits `S`, `kB`, `C`, `tau_rise` and `tau_fall` from the material
file unless it sets them.

**Light yield**, Chou form (Birks with `C = 0`):

```
dL/dx = S · (dE/dx) / [1 + kB · (dE/dx) + C · (dE/dx)²]
```

The bundled file is 10% WbLS: `S = 1387.3` photons/MeV, `kB = 1.65e-5` mm/keV,
`C = 1.33e-9` mm²/keV².

- `S` is linear in the WbLS mass fraction `c` over 1 to 10%:
  `S(c) = (127.9 ± 17.0)·c + (108.3 ± 51.0)` photons/MeV (Caravaca 2020, arXiv:2006.00173;
  CHESS, cosmic muons and ⁹⁰Sr, scintillation separated from Cherenkov).
- `kB` and `C` were measured for 5% WbLS on 2 to 20 MeV proton recoils (Callaghan 2023,
  arXiv:2210.03876, Table 6; oxygenated sample):

| form | kB | C |
|---|---|---|
| Chou | (1.65 ± 0.81) × 10⁻⁵ mm/keV | (1.33 ± 0.27) × 10⁻⁹ mm²/keV² |
| Birks only | (5.95 ± 0.43) × 10⁻⁵ mm/keV | — |

The S-kB correlation in the Chou fit is 87.4% (full covariance in Callaghan 2023, Table 7).

**Emission time.** The measured profile has two decay components with a shared rise time:

```
p(t) = R₁ · g(t; τ_r, τ₁) + (1 − R₁) · g(t; τ_r, τ₂)
g(t; τ_r, τ_d) = (e^(−t/τ_d) − e^(−t/τ_r)) / (τ_d − τ_r),   t ≥ 0
```

| parameter | 1% WbLS | 5% WbLS | 10% WbLS |
|---|---|---|---|
| τ_r [ns] | 0.00 ± 0.06 | 0.06 ± 0.11 | 0.13 ± 0.12 |
| τ₁ [ns] | 2.25 ± 0.15 | 2.35 ± 0.13 | 2.70 ± 0.16 |
| τ₂ [ns] | 15.10 ± 7.47 | 23.21 ± 3.28 | 27.05 ± 4.20 |
| R₁ | 0.96 ± 0.01 | 0.94 ± 0.01 | 0.94 ± 0.01 |

Uncertainties are statistical plus systematic (Caravaca 2020, Table 2; ⁹⁰Sr betas). LUCiD
simulates the fast component only: one rise and one fall time (`tau_rise = 0.13`,
`tau_fall = 2.7` ns for 10%). The slow component, 6% of the light, is not simulated.

The emission spectrum is a Moyal distribution, given in the material file.
