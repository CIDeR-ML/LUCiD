# Geometry & configuration

## Detectors are registry-dispatched

Geometries register themselves with a decorator (`@register_detector('cylinder')`) and are built
by `generate_detector(config)`, which looks up the class from the config's `detector_type`. Each
geometry has a matching propagator in `lucid/propagation/`.

| `detector_type` | class | examples |
|-----------------|-------|----------|
| `cylinder` | `Cylinder` | SK, HK, WCTE (algorithmic `*_like`, or measured PMTs via `.npz`) |
| `sphere` | `Sphere` | JUNO, TAO |
| `box` | `Box` | test / segmented geometries |
| `string` | `StringTelescope` | IceCube-style neutrino telescopes (DOMs on vertical strings) |

Cylinders can be built **algorithmically** (`SK_like`, `WCTE_like`) or from **measured PMT
positions** via `Cylinder.from_pmt_file(npz)` (`SK`, `HK`, `WCTE` from public WCSim geofiles). The
`.npz` schema is documented in `lucid/geometry/PMT_NPZ_SCHEMA.md`; convert a geofile to a
schema `.npz` with `scripts/geofile_to_npz.py`.

## Two JSON files per detector

A detector is defined by a **geometry** config and a **physics** config:

**`*_geom_config.json`** — shape & sensor placement:
```json
{ "material": "water", "detector_type": "cylinder",
  "geometry_definitions": { "radius": 2.0, "height": 3.0, "n_sensors": 200, "sensor_radius": 0.1 } }
```
`detector_type` selects the registry class; `geometry_definitions` are the shape/placement fields
(they differ per type — a sphere has `radius`; a string has `npz_file_path`; etc.).

**`*_physics_config.json`** — optical properties, **flat and composable**. Each property is
independently a scalar or a wavelength-dependent curve:
```json
{ "scatter_length": 50.0, "absorption_length": 50.0,
  "wall_reflection_rate": 0.2, "sensor_reflection_rate": 0.2,
  "qe": 0.2, "qe_corrections": 1.0 }
```
Each value may be: a **number** (scalar); a **list** (inline array); `null`/missing (projected
from a referenced λ-curve at a reference wavelength, default 400 nm); or a `"path.json"` /
`"__array__:file.npy"` reference. Two extra keys, `medium_model` and `qe_curve`, point at
material and PMT-QE model files (`config/materials/`, `config/pmt/`).

See the [configuration reference](../reference/config.md) for the field-by-field details.

## Detectors and materials

The same forward call runs across geometries by swapping only these config files. Bundled
examples include `SK_like`, `JUNO`, `MidBox`, `IceCube86_*`, and their `_wbls` / `ice` variants —
the `event_displays` tutorial shows one muon across cylinder / sphere / box / string and
water / WbLS / ice.
