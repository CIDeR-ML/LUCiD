# Configuration reference

Each detector is defined by two JSON files in `config/`. They are parsed by
`lucid/detector_params.py` (`load_detector_params`, `load_physics_config`) and
`lucid/geometry/` (`DetectorGeometry.from_config`).

## Geometry config — `*_geom_config.json`

| key | meaning |
|-----|---------|
| `material` | `"water"`, `"wbls"`, `"ice"`, … — selects the medium model and the SIREN emitter path |
| `detector_type` | `cylinder` / `sphere` / `box` / `string` — selects the registry class |
| `geometry_definitions` | shape & sensor placement (fields depend on `detector_type`) |

`geometry_definitions` by type:

- **cylinder**: `radius`, `height`, `n_sensors`, `sensor_radius`
- **sphere**: `radius`, `n_sensors`, `sensor_radius`
- **box**: `length`, `width`, `height`, `n_sensors`, `sensor_radius`
- **string**: `npz_file_path` (DOM positions; see `pmt-npz-schema.md`)

Measured cylinders (SK/HK/WCTE) come from a PMT-array `.npz` via `Cylinder.from_pmt_file` — see
`pmt-npz-schema.md`.

## Physics config — `*_physics_config.json`

A **flat** set of optical properties; each is independently scalar or wavelength-dependent.
Common keys:

| key | meaning |
|-----|---------|
| `scatter_length`, `mie_scatter_length` | Rayleigh / Mie scattering lengths (m) |
| `g` | Mie asymmetry parameter |
| `absorption_length` | absorption length (m) |
| `wall_reflection_rate`, `sensor_reflection_rate` | scalar reflectivities |
| `wall_R0`, `wall_p`, `wall_fspec` | angular wall-reflection model (Schlick base reflectivity, exponent, specular fraction) — used when `reflection_model='angular'`; `'scalar_mix'` combines the scalar rates with the `*_fspec` specular/diffuse split |
| `cathode_nr`, `cathode_nk`, `sensor_fspec` | angular sensor/cathode reflection (Fresnel n, k; specular fraction) |
| `qe` | global quantum efficiency |
| `qe_corrections`, `gain`, `t0`, `walk` | per-PMT response arrays (QE multiplier, charge gain, time offset, TQ-walk slope) |
| `spe_width`, `tts` | single-photoelectron charge width; transit-time spread (per-PMT timing jitter, ns) |
| `S`, `kB`, `C`, `tau_rise`, `tau_fall`, `moyal_*` | scintillation block (usually inherited from the material JSON — see [materials](../concepts/materials.md)) |

Deviation-curve leaves (`rayleigh_dev`, `mie_dev`, `abs_dev`, `qe_dev`) are normally left at
their neutral defaults here and fitted by calibration — see
[wavelength physics](../concepts/wavelength.md).

Production dataset configs (`lucid/production/configs/`) additionally carry `digitizer`,
`trigger`, and `selection` blocks — those are electronics/DAQ settings, not `DetectorParams`;
see [digitizer & trigger](digitizer-and-trigger.md).

**How each value is interpreted:**

- a **number** → scalar;
- a **list** → inline per-element array;
- `null` / missing → *projected* from a referenced λ-curve evaluated at `scalar_ref_wavelength`
  (default 400 nm), if the property has an associated curve. Optional properties otherwise
  fall back to a neutral default; the essential three (`scatter_length`,
  `absorption_length`, `qe`) instead raise a `ValueError` if nothing can supply them;
- `"path/to/file.json"` → loaded from JSON; `"__array__:file.npy"` → loaded from a companion `.npy`.

Two extra keys are resolved relative to the config directory (and reference model files rather
than being stored on `DetectorParams`):

- `medium_model` → a medium-model JSON (e.g. under `config/materials/`)
- `qe_curve` → a PMT QE-curve JSON (e.g. under `config/pmt/`)

These configs map onto the `DetectorParams` pytree — see
[DetectorParams vs args](../concepts/detector-params-vs-args.md) for which fields are fittable and how the
scalar/wavelength choice is represented in code.
