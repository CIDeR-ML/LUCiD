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

- **cylinder**: `radius`, `height`, `n_sensors`, `sensor_radius` (algorithmic), or
  `npz_file_path` (a measured PMT array)
- **sphere**: `radius`, `n_sensors`, `sensor_radius`
- **box**: `length`, `width`, `height`, `n_sensors`, `sensor_radius`
- **string**: `npz_file_path` (DOM positions; see the [PMT `.npz` schema](pmt-npz-schema.md))

Measured cylinders (SK/HK/WCTE) come from a PMT-array `.npz` via `Cylinder.from_pmt_file` — see
the [PMT `.npz` schema](pmt-npz-schema.md).

## Physics config — `*_physics_config.json`

A **flat** set of optical properties, keyed by `DetectorParams` leaf name. The loader is strict:
a key it does not know raises (naming the closest known key), and so does a required key left
out. `_`-prefixed keys are comments.

| key | meaning |
|-----|---------|
| `scalar_mode` | a block holding `scatter_length`, `absorption_length`, `mie_scatter_length` (Rayleigh / Mie scattering and absorption lengths, m). Only scalar mode (`wavelength_mode=False`) reads them — wavelength mode evaluates the medium's curves — so they are not accepted at the top level |
| `g` | Mie asymmetry parameter |
| `wall_reflection_rate`, `sensor_reflection_rate` | scalar reflectivities (required) |
| `wall_fspec`, `sensor_fspec` | wall / sensor specular fraction (required): the default `'scalar_mix'` reflection combines the scalar rates with this specular/diffuse split |
| `angular_reflection` | a block holding `wall_R0`, `wall_p` (Schlick wall reflectance at normal incidence, exponent) and `cathode_nr`, `cathode_nk` (the PMT cathode's complex index). Only `reflection_model='angular'` reads them, so they are not accepted at the top level; it requires all four when it runs on params baked from this config. Without the block they hold reference values no other model reads |
| `qe` | global quantum efficiency, per photon arriving at a PMT with its reflection included (as PMT QE is quoted); at most `1 − R0`, the sensor's non-reflected fraction at normal incidence |
| `qe_corrections`, `t0` | per-PMT response arrays (QE multiplier, first-arrival time offset, not yet applied by the hits); `qe_corrections` is required |
| `tts` | transit-time spread (per-PMT timing jitter, ns) |
| `S`, `kB`, `C`, `tau_rise`, `tau_fall` | scintillation block (usually inherited from the material JSON — see [materials](../concepts/materials.md)) |
| `digitizer`, `trigger` | production's readout blocks (electronics/DAQ, not `DetectorParams`); `lucid-run-job` reads them, the simulator does not. Of the bundled detectors only `SK_WAND` and `HK_WAND` carry them — see [digitizer & trigger](digitizer-and-trigger.md) |

Deviation-curve leaves (`rayleigh_dev`, `mie_dev`, `abs_dev`, `qe_dev`) are normally left at
their neutral defaults here and fitted by calibration — see
[wavelength physics](../concepts/wavelength.md).

Production dataset configs (`lucid/production/dataprod/configs/`) carry a `selection` block, and
may carry their own `digitizer` and `trigger` blocks, which then win over the detector's (the
`Solar`, `SN` and `Test` configs do; the `GeV` configs take them from the detector); see
[digitizer & trigger](digitizer-and-trigger.md). The `selection` block picks an event-keeping
`mode`: `"trigger"` defers to the real readout trigger, while `"min_physics_hits"` (with `"n": N`)
is a cheap truth-level cut keeping interactions with at least `N` real hits.

**How each value is interpreted:**

- a **number** → scalar;
- a **list** → inline per-element array;
- `"path/to/file.json"` → loaded from JSON;
- omitted → `scatter_length`, `absorption_length`, `mie_scatter_length` and `g` are *projected*
  from the `medium_model` at the scalar reference wavelength (400 nm), and `qe` from the
  `qe_curve`; with nothing to project from, loading raises. The other optional leaves take
  their identity value (no TTS smearing, zero offset, no λ deviation; the
  scintillation values come from a scintillating material, else zero). `null` is not a value.

Two extra keys are resolved relative to the config directory (and reference model files rather
than being stored on `DetectorParams`); each file is read once per load:

- `medium_model` → a medium-model JSON (e.g. under `config/materials/`)
- `qe_curve` → a PMT QE-curve JSON (e.g. under `config/pmt/`)

These configs map onto the `DetectorParams` pytree — see
[DetectorParams vs args](../concepts/detector-params-vs-args.md) for which fields are fittable and how the
scalar/wavelength choice is represented in code.
