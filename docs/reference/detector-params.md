# Parameters: `DetectorParams` and `ParticleParams`

`lucid.detector_params` holds the two parameter sets a fit can move. Both are JAX pytrees
(NamedTuples), so `jax.grad` and the optimisers work on them directly.

- `ParticleParams(energy, position, theta, phi, t0)`: one track. Energy in MeV (kinetic),
  position in m, polar and azimuthal direction angles in rad, start time in ns.
  `ParticleParams.from_cartesian(energy, position, direction, t0=0.0)` takes a direction
  vector instead.
- `DetectorParams`: the detector's optical and response state, grouped by physics.

Reconstruction moves `ParticleParams`; calibration moves (a subset of) `DetectorParams`.
Everything else, such as geometry, the medium's reference curves, the refractive index, the
reflection model, `n_photons`, `K` and `temperature`, is a fixed argument of
`setup_event_simulator`.

## `DetectorParams`

```
DetectorParams(
  scattering    = ScatteringParams(scatter_length, mie_scatter_length, g,
                                   rayleigh_dev[5], mie_dev[5]),
  absorption    = AbsorptionParams(absorption_length, abs_dev[5]),
  reflection    = ReflectionParams(wall_reflection_rate, sensor_reflection_rate,
                                   wall_R0, wall_p, wall_fspec,
                                   cathode_nr, cathode_nk, sensor_fspec),
  response      = ResponseParams(qe, tts, qe_dev[5]),
  per_pmt       = PerPmtParams(qe_corrections[NS], t0[NS]),
  scintillation = ScintillationParams(S, kB, C, tau_rise, tau_fall),
)
```

`NS` is the number of sensors; `[5]` is one value per control wavelength
([optics](../concepts/optics.md#reference-and-deviation)).

| field | unit | meaning |
|---|---|---|
| `scatter_length` | m | Rayleigh scattering length (scalar mode) |
| `mie_scatter_length` | m | Mie scattering length (scalar mode) |
| `g` | | Henyey-Greenstein asymmetry of Mie scattering, [0, 1] |
| `absorption_length` | m | absorption length (scalar mode) |
| `rayleigh_dev`, `mie_dev`, `abs_dev`, `qe_dev` | | wavelength-mode deviation curves; all ones is the reference |
| `wall_reflection_rate`, `sensor_reflection_rate` | | reflectance of the wall and of a sensor (scalar models) |
| `wall_fspec`, `sensor_fspec` | | specular fraction of a reflection (the rest is diffuse) |
| `wall_R0`, `wall_p` | | Schlick wall reflectance at normal incidence and its angular exponent (`'angular'` model) |
| `cathode_nr`, `cathode_nk` | | real and imaginary refractive index of the PMT cathode (`'angular'` model) |
| `qe` | | quantum efficiency at 400 nm |
| `tts` | ns | transit-time spread of a PMT (time smearing of each photon) |
| `qe_corrections` | | per-sensor efficiency multiplier, the per-PMT gain calibration fits |
| `t0` | ns | per-sensor time offset; carried but not applied by the hits functions |
| `S`, `kB`, `C` | photons/MeV, mm/keV, mm²/keV² | scintillation yield and quenching; zero for a non-scintillating medium |
| `tau_rise`, `tau_fall` | ns | scintillation rise and decay times |

Each reflection model reads only the fields it needs: `'scalar'` the two rates, `'scalar_mix'`
the rates and the two `*_fspec`, `'angular'` `wall_R0`, `wall_p`, `cathode_nr`, `cathode_nk` and
the `*_fspec`.

The charge resolution is not a parameter: it is the simulator's `charge_resolution` argument,
or the digitizer's SPE model in production.

## Building and moving parameters

- `load_detector_params(physics_config, num_sensors=None)` reads a physics config
  ([keys](config.md#physics-config)).
- `DetectorParams.from_flat(**fields)` builds one from flat field names; omitted fields take
  neutral values that leave the forward unchanged.
- `default_bounds(num_sensors)` gives physical ranges for every leaf, and `normalize_params` /
  `denormalize_params` map to and from [0, 1].
- In `lucid.fitting`, `FieldParams(dp, ['absorption_length', 'qe', ...])` fits any named
  leaves in log space, holding the rest of `dp` fixed
  ([calibration](../guides/calibration.md)).
