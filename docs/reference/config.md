# Configuration files

A detector is a geometry config and a physics config in `config/`. Paths inside either file
(`npz_file_path`, `medium_model`, `qe_curve`, `"file.json"` values) are resolved relative to the
file's own directory. Lengths are in metres; there is no unit field.

## Geometry config

`<name>_geom_config.json`, read by `lucid.geometry.generate_detector(path)`:

```json
{
  "material": "water",
  "detector_type": "cylinder",
  "geometry_definitions": {"radius": 16.9, "height": 36.2, "n_sensors": 11000, "sensor_radius": 0.25}
}
```

| key | meaning |
|---|---|
| `material` | `water`, `wbls` or `ice`: the medium file `config/materials/<material>.json` and its SIREN emitters |
| `detector_type` | `cylinder`, `sphere`, `box` or `string` |
| `geometry_definitions` | the shape and sensors, by type (below) |

| `detector_type` | `geometry_definitions` |
|---|---|
| `cylinder` | `radius`, `height`, `n_sensors`, `sensor_radius`; or `npz_file_path` alone, for measured PMT positions ([PMT `.npz`](pmt-npz-schema.md)) |
| `sphere` | `radius`, `n_sensors`, `sensor_radius` |
| `box` | `length`, `width`, `height`, `n_sensors`, `sensor_radius` |
| `string` | `npz_file_path`, the DOM positions ([string `.npz`](pmt-npz-schema.md#string-detectors)) |

The cylinder axis is `z`, with caps at `z = ±height/2`. `n_sensors` is a target: generated
placement rounds down per surface, so `SK_like` asks for 11000 and gets 10764. Read the real
count from `len(generate_detector(path).all_points)`.

## Physics config

`<name>_physics_config.json`, read by `lucid.detector_params.load_physics_config` and
`load_detector_params`. Keys are `DetectorParams` field names
([parameters](detector-params.md)) plus the blocks below. An unknown key raises, naming the
closest known one; keys starting with `_` are comments.

```json
{
  "medium_model": "materials/water.json",
  "qe_curve": "pmt/SK_QE.json",
  "wall_reflection_rate": 0.05,
  "wall_fspec": 0.55,
  "sensor_reflection_rate": 0.25,
  "sensor_fspec": 0.90,
  "qe_corrections": 1.0
}
```

| key | meaning |
|---|---|
| `wall_reflection_rate`, `sensor_reflection_rate` | required. Reflectance of the wall and a sensor |
| `wall_fspec`, `sensor_fspec` | required. Specular fraction of each reflection |
| `qe_corrections` | required. Per-sensor efficiency multiplier: a number (all sensors) or a list or file of `n_sensors` values |
| `medium_model` | a material file, e.g. `materials/water.json`, of the geometry's `material` (a mismatch raises). The scalar optics are projected from it at 400 nm when omitted |
| `qe_curve` | a PMT QE curve, e.g. `pmt/SK_QE.json`; `qe` is projected from it at 400 nm when omitted |
| `scalar_mode` | block with `scatter_length`, `absorption_length`, `mie_scatter_length` (m). Read only when `wavelength_mode=False`; these keys are refused at the top level |
| `g` | Mie asymmetry |
| `qe` | QE at 400 nm, at most `1 − R0` (the sensor's normal-incidence non-reflected fraction) |
| `tts` | transit-time spread, ns (default 0) |
| `t0` | per-sensor time offset, ns (default 0) |
| `angular_reflection` | block with all four of `wall_R0`, `wall_p`, `cathode_nr`, `cathode_nk`. Required by `reflection_model='angular'`; these keys are refused at the top level |
| `rayleigh_dev`, `mie_dev`, `abs_dev`, `qe_dev` | deviation curves, 5 values (default all ones) |
| `S`, `kB`, `C`, `tau_rise`, `tau_fall` | scintillation; default from a scintillating material, else 0 |
| `digitizer`, `trigger` | readout blocks read by `lucid-run-job`, not by the simulator ([digitizer and trigger](digitizer-and-trigger.md)) |

A value is a number, an inline list, or a `"file.json"` holding a list. `null` is not accepted.

`scatter_length`, `absorption_length`, `mie_scatter_length` and `g` must each be given or be
projectable from `medium_model`, and `qe` given or projectable from `qe_curve`; otherwise loading
raises and names the missing fields. Wavelength mode takes the scattering and absorption from
the medium curves (`medium_model`, else the geometry's `material`); the `scalar_mode` numbers do
not enter it.

## Material and QE files

`config/materials/<material>.json` holds a medium: refractive index, the scattering and
absorption parametrisations, the emitter set to use (`"siren"`), and for WbLS a
`scintillation` block ([optics](../concepts/optics.md)). `config/pmt/*.json` holds QE curves as
`wavelengths_nm` and `qe_percent`.
