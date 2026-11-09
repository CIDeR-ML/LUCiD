# Model your own detector

This guide takes you from "I have a tank/sphere/telescope I want to simulate" to a
running, differentiable LUCiD forward model for it. It assumes you've already
[installed LUCiD](../getting-started/install.md) and run `python scripts/download_data.py`.

A LUCiD detector is nothing but **two small JSON files** plus, optionally, one `.npz` of
measured sensor positions — no code. Everything downstream (simulation, reconstruction,
calibration) reads the same two files, so getting them right is the whole job: the same forward
call runs across geometries by swapping only these files
(`analysis/paper/fig_charge_displays.py` draws one muon across several bundled detectors).

Throughout LUCiD, **sensor** is the generic term for a light detector; **PMT** (photomultiplier
tube) is the tank-geometry instance and **DOM** (digital optical module) the string-telescope
instance of the same concept.

## 1. Choose your geometry family

LUCiD dispatches on a `detector_type` string in the geometry config to one of four
registered classes ([architecture](../concepts/architecture.md#geometry-registry-dispatched)
says how):

| `detector_type` | class | examples |
|-----------------|-------|----------|
| `cylinder` | `Cylinder` | SK, HK, WCTE (algorithmic `*_like`, or measured PMTs via `.npz`) |
| `sphere` | `Sphere` | JUNO, TAO |
| `box` | `Box` | test / segmented geometries |
| `string` | `StringTelescope` | IceCube-style neutrino telescopes (DOMs on vertical strings) |

If your detector's shape doesn't match any of these, it isn't supported yet — there's no
generic mesh importer. `cylinder`, `sphere`, and `box` all get their sensors placed for
you (see below); `string` always loads DOM positions from a file, since a telescope has
no simple parametric layout.

**Algorithmic vs. measured placement.** For `cylinder` you have a choice:

- **Algorithmic** — you give LUCiD `radius`, `height`, and a target `n_sensors`, and it
  tiles the barrel + caps for you. Good for a first pass, a generic/hypothetical tank, or
  when you don't have (or don't yet need) real PMT coordinates.
- **Measured** — you have an actual PMT survey / WCSim-style geofile and want LUCiD to
  use the exact positions. This is how SK, HK, and WCTE are built from public WCSim
  geofiles (Section 3).

`sphere` and `box` are algorithmic-only today. `string` is always measured (there's no
"algorithmic telescope"). For a tour of the bundled example detectors across all four
families, see [Detectors](../getting-started/detectors.md).

## 2. Write the geometry config

A detector's shape lives in `config/<name>_geom_config.json`. Two real, minimal examples
already in the repo:

```json title="config/MidBox_geom_config.json (box)"
{
  "material": "water",
  "detector_type": "box",
  "geometry_definitions": {
    "length": 6.0,
    "width":  12.0,
    "height": 5.0,
    "n_sensors": 9000,
    "sensor_radius": 0.075
  }
}
```

```json title="config/SK_like_geom_config.json (cylinder, algorithmic)"
{
  "material": "water",
  "detector_type": "cylinder",
  "geometry_definitions": {
    "radius": 16.9,
    "height": 36.2,
    "n_sensors": 11000,
    "sensor_radius": 0.25
  }
}
```

Three top-level keys, always:

- **`material`** — `"water"`, `"wbls"`, or `"ice"`. Selects the medium model
  ([concepts: wavelength](../concepts/wavelength.md)) and which SIREN emitter tables get
  loaded.
- **`detector_type`** — the registry key from Section 1.
- **`geometry_definitions`** — shape + placement fields, which differ **by type** (a string
  needs only `npz_file_path`, Section 3); the
  [configuration reference](../reference/config.md) lists them.

All lengths are **meters** — there is no unit field and no conversion at this layer
(`self.r = radius`, verbatim). If you designed your tank in cm, divide by 100 first, or
you'll get a detector 100× the wrong size with no error to warn you.

Here's a complete, made-up example — a small 3 m-radius, 6 m-tall water tank with
algorithmic sensor placement:

```json title="config/MyTank_geom_config.json"
{
  "material": "water",
  "detector_type": "cylinder",
  "geometry_definitions": {
    "radius": 3.0,
    "height": 6.0,
    "n_sensors": 500,
    "sensor_radius": 0.1
  }
}
```

This loads cleanly through `generate_detector()` (the same call `setup_event_simulator`
makes internally):

```python
from lucid.geometry import generate_detector
det = generate_detector('config/MyTank_geom_config.json')
print(type(det).__name__, len(det.all_points), 'sensors placed')
# Cylinder 452 sensors placed
```

**Gotcha: `n_sensors` is a target, not a guarantee.** Placement splits sensors
proportionally across barrel/top-cap/bottom-cap by area, then rounds down per face, so
the actual placed count is always a bit under what you asked for — 500 requested becomes
452 placed above; the bundled SK-like config asks for 11000 and gets 10764; `MidBox`
asks for 9000 and gets 8932. This is normal. Always read the actual count back from
`generate_detector()` (or `scripts/visualize_detector.py`, Section 5) rather than
assuming the config number.

## 3. Measured PMT positions

If you have a real PMT survey or a WCSim-style geofile, load it instead of placing
sensors algorithmically. The two paths for the *same* cylinder class:

```json title="config/WCTE_like_geom_config.json (algorithmic)"
{ "material": "water", "detector_type": "cylinder",
  "geometry_definitions": { "radius": 2, "height": 4, "n_sensors": 2500, "sensor_radius": 0.040 } }
```

```json title="config/WCTE_geom_config.json (measured)"
{ "material": "water", "detector_type": "cylinder",
  "geometry_definitions": { "npz_file_path": "wcte_geometry.npz" } }
```

Presence of `npz_file_path` in `geometry_definitions` is the switch: `Cylinder.from_config`
sees it and calls `Cylinder.from_pmt_file(npz_path)` instead of the algorithmic
constructor — no other key changes. The path is resolved **relative to the config
file's own directory**, so `wcte_geometry.npz` above means "next to
`WCTE_geom_config.json` in `config/`".

**Converting a geofile.** `scripts/geofile_to_npz.py` turns a WCSim-style `.txt` geofile
(cm units, one header block + one row per PMT) into a schema-conformant `.npz`. The WCSim dumps
behind the bundled `.npz` (`geofile_SuperK.txt` and the HK / WCTE ones) are not in the repository;
they ship in the `detector-geometry-sources` asset
([`tools/assets/detector-geometry-sources.manifest`](https://github.com/CIDeR-ML/LUCiD/blob/main/tools/assets/detector-geometry-sources.manifest)):

```bash
python scripts/geofile_to_npz.py geofile_SuperK.txt config/mytank_geometry.npz
```

It prints the parsed radius/height/sensor_radius and a surface-label breakdown so you can
eyeball it before trusting it. Scope note straight from the script's docstring: it's
validated against single-PMT-type geofiles (like SK's); detectors with mPMT domes +
outer-detector PMTs (HyperK, WCTE/NuPRISM) need extra `inactive_*` arrays the converter
doesn't emit yet — those bundled `.npz` were built by hand and just ship pre-built.

**The schema** ([full reference](../reference/pmt-npz-schema.md)) requires, at minimum:

| key | shape | meaning |
|-----|-------|---------|
| `positions_mm` | `(N,3)` | PMT positions, **millimeters**, z vertical |
| `directions` | `(N,3)` | PMT viewing-direction unit vectors |
| `surfaces` | `(N,)` | `'barrel'` / `'top'` / `'bottom'` |
| `pmt_id` | `(N,)` | unique PMT id |
| `radius`, `height`, `sensor_radius` | scalar | cylinder envelope (**meters**) |

Note the unit switch: the per-PMT arrays are millimeters (`from_pmt_file` converts to
meters on load), but the envelope scalars are already meters — mixing these up is a
common source of a detector that's 1000× too big or too small. By default,
`from_pmt_file(..., snap_to_wall=True)` projects barrel PMTs onto `r_xy = radius` and cap
PMTs onto `z = ±height/2`, since real geofiles leave PMTs a few cm off the nominal
surface and the ray-tracer wants them exactly on it.

Then reference the file in your geom config exactly like WCTE does above.

`string` telescopes use a different, DOM-per-string `.npz` schema (`dom_xyz`,
`n_dom_per_str`, `envelope_radius`, `envelope_z_min/max` — see `IceCube86_full_geom_config.json`
+ `icecube86_full.npz`), not the PMT schema above; there's no algorithmic path for it.

## 4. Write the physics config

The optical properties live in a sibling `config/<name>_physics_config.json`. It's a
**flat, composable** bag of properties — each one independently a plain scalar or a
reference to a wavelength-dependent curve ([concepts: materials](../concepts/materials.md),
full field table in the [configuration reference](../reference/config.md)). Two real
styles already in the repo:

```json title="config/EOS_physics_config.json (all scalar)"
{
  "scalar_mode": {"scatter_length": 50.0, "absorption_length": 50.0, "mie_scatter_length": 1.0e9},
  "g": 0.0,
  "wall_reflection_rate": 0.05,
  "wall_fspec": 0.55,
  "sensor_reflection_rate": 0.25,
  "sensor_fspec": 0.90,
  "qe": 0.2,
  "qe_corrections": 1.0
}
```

```json title="config/SK_like_physics_config.json (curve-referenced)"
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

!!! note "Give every reflecting surface a specular fraction"
    The default reflection model is `scalar_mix`, which needs **four** numbers: a rate and a
    specular fraction for each of the wall and the sensor; the loader requires all four.
    The values above are SK's: a black PET liner reflects ~5% and a PMT face ~25%, which is why
    they are not equal. A surface with rate `0` never reflects, so its fraction is unused.

The optical lengths of the first style sit in a `scalar_mode` block because only scalar mode
(`wavelength_mode=False`) reads them: wavelength mode evaluates the medium's curves
(without a `medium_model`, the geometry's material, `config/materials/<material>.json`).
The second style omits them, `g` and `qe` entirely — they get **projected** from the referenced curves (`medium_model`, `qe_curve`) at a reference
wavelength (400 nm by default) instead of being typed in by hand. `medium_model` points
at a material file under `config/materials/`; `qe_curve` at a PMT QE curve under
`config/pmt/`. Both are resolved relative to the physics config's own directory, same as
the npz path above.

For `MyTank`, mixing the two styles works exactly as you'd expect — reuse the water
medium/QE curve, but pin your own reflectivities:

```json title="config/MyTank_physics_config.json"
{
  "medium_model": "materials/water.json",
  "qe_curve": "pmt/SK_QE.json",
  "wall_reflection_rate": 0.2,
  "wall_fspec": 0.55,
  "sensor_reflection_rate": 0.2,
  "sensor_fspec": 0.90,
  "qe_corrections": 1.0
}
```

This loads through `lucid.detector_params.load_detector_params()` without errors (the
same call `setup_event_simulator(..., default_detector_params=True)` makes for you), and
projects `scatter_length` / `absorption_length` / `qe` from `materials/water.json` and
`pmt/SK_QE.json` at 400 nm.

**What happens if you get it wrong.** The loader is strict: an unknown key (a typo) raises and
names the closest known key, a required key left out raises, and so does an optical length at
the top level instead of in `scalar_mode`. The optical lengths, `g` and `qe` must each be a
literal number or projectable from `medium_model` / `qe_curve`; if neither is available,
loading fails with a named error — e.g. dropping `medium_model` from the config above gives:

```
ValueError: Physics config 'config/MyTank_physics_config.json' has no ['scatter_length',
'absorption_length', 'mie_scatter_length', 'g'] and no 'medium_model' to project from at λ=400.0nm.
```

Everything else (per-PMT response fields, λ-deviation curves, scintillation parameters) is
optional and takes its identity value (deviation curves all-ones, no smearing) if you omit
it — you only have to supply what you actually care about.

## 5. Sanity-check before physics

Before running any physics, look at the geometry alone with `scripts/visualize_detector.py`:

```bash
python scripts/visualize_detector.py --geom config/MyTank_geom_config.json
```

It prints the actual placed sensor count and the bounding box, and writes an interactive
3D scatter (`MyTank_geom_config_geometry.html`) plus, for cylinders, a 2D unrolled PNG.
Check, in order:

1. **Sensor count** — close to but a bit under your requested `n_sensors` (Section 2's
   gotcha), not wildly off (a huge shortfall usually means `sensor_radius` is too large
   for the surface area you gave it, so cap placement collapses).
2. **Physical extent** — the printed `x/y/z` ranges should match your intended
   dimensions **in meters**; if they're 100× or 1000× off, you mixed up cm/mm somewhere
   in Section 2 or 3.
3. **Orientation** — `z` is vertical, the barrel sits in the `x`/`y` plane at
   `r = radius`, and the two caps are flat discs at `z = ±height/2`. If your detector is
   supposed to look different, this is where you'd catch it.

## 6. First event

Point the runnable example at your new configs. Open `examples/hello_simulate.py` and
change the two path constants at the top:

```python title="examples/hello_simulate.py (edit these two lines)"
GEOM, PHYS = 'config/MyTank_geom_config.json', 'config/MyTank_physics_config.json'
```

then run it:

```bash
python examples/hello_simulate.py
```

**A healthy run** prints a finite, positive total charge and your detector's sensor count
(`charge.size`: every placed sensor, lit or not), and writes `hello_simulate.png` — a 2D
unrolled event display you can eyeball for a plausible hit pattern (a single track lights only a
cone/ring, so a ring for a horizontal muon, roughly, if your geometry is cylinder-like).

**Common failure modes**, in the order you're likely to hit them:

- **`FileNotFoundError` pointing at `data/water/muon/cherenkov/weights.npz`**
  — this is *not* about your detector config. It means the SIREN emitter weights
  haven't been installed yet; run `python scripts/download_data.py` once (any detector using
  `material: "water"`, `particle='muon'` reuses the same bundled weights — there's
  nothing per-detector to train). If you picked a different `material`/particle
  combination with no bundled weights under `data/<material>/<particle>/`, training a
  new SIREN emitter is a separate task, out of scope here.
- **Zero total charge** — almost always the track: the vertex/direction you chose doesn't
  send any Cherenkov light at a placed sensor (e.g. firing along the cylinder axis in a very
  short, wide tank). The propagation grid does not drop sensors: `create_propagator` gives
  every grid cell room for all the sensors that overlap it (raising `max_candidates_per_ray`,
  with a warning), and checks the map it builds. A setup warning starting `Sensor map:` (a
  sensor in no cell, low cell coverage) names a geometry problem worth fixing before physics;
  the grid is sized automatically from the sensor spacing, or by explicit cell counts as
  `hello_simulate.py` passes them (`n_cap`, `n_angular`, `n_height` for a cylinder).
- **A different `ValueError` at setup time** — almost always a physics config problem
  from Section 4 (a missing essential scalar with no curve to project from); the message
  names the exact field and file.
- **`FileNotFoundError` on your own config/npz/material file** — paths in
  `geometry_definitions.npz_file_path`, `medium_model`, and `qe_curve` are resolved
  relative to the *config file's* directory, not your current working directory; double
  check the file actually sits next to the JSON that references it.

## 7. Next steps

With `MyTank_geom_config.json` / `MyTank_physics_config.json` simulating events, the same
two files drive everything else in LUCiD unchanged:

- [Reconstruction](reconstruction.md) — fit `(energy, vertex, direction, t0)` on your
  detector's simulated events.
- [Calibration](calibration.md) — fit the optical parameters (and per-PMT QE) back out
  of simulated or real data from your detector.

Once you're happy with it, consider adding it to the gallery in
[Detectors](../getting-started/detectors.md) alongside the bundled examples — the same
`scripts/visualize_detector.py` + `examples/hello_simulate.py` checks from Sections 5–6
are a reasonable checklist before sharing a new config.
