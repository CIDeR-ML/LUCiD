# Model your own detector

A detector is two JSON files, plus a `.npz` if you have measured sensor positions. No code is
needed, and simulation, reconstruction and calibration all read the same two files. This page
builds a small water tank, `MyTank`. Every key is in the
[configuration reference](../reference/config.md).

## 1. Pick a shape

| `detector_type` | use for | sensor placement |
|---|---|---|
| `cylinder` | SK, HK, WCTE-like tanks | generated, or measured from a `.npz` |
| `sphere` | JUNO, TAO-like detectors | generated |
| `box` | rectangular tanks | generated |
| `string` | neutrino telescopes (DOMs on vertical strings) | measured from a `.npz` |

Other shapes are not supported (there is no mesh import).

## 2. Write the geometry config

Save as config/MyTank_geom_config.json: a 3 m radius, 6 m tall tank with generated placement:

```json
{
  "material": "water",
  "detector_type": "cylinder",
  "geometry_definitions": {"radius": 3.0, "height": 6.0, "n_sensors": 500, "sensor_radius": 0.1}
}
```

All lengths are metres. A tank written in cm comes out 100 times too large without any error.

Check what was placed:

```python
from lucid.geometry import generate_detector
det = generate_detector('config/MyTank_geom_config.json')
print(type(det).__name__, len(det.all_points))   # Cylinder 452
```

`n_sensors` is a target: placement splits it over the barrel and caps by area and rounds down,
so 500 becomes 452.

### Measured positions

For a cylinder with a PMT survey or a WCSim geofile, convert it and point the config at the
result instead of the four numbers:

```bash
python scripts/geofile_to_npz.py geofile.txt config/mytank_geometry.npz
```

```json
{"material": "water", "detector_type": "cylinder",
 "geometry_definitions": {"npz_file_path": "mytank_geometry.npz"}}
```

The converter prints the parsed radius, height and sensor radius and a count per surface. Note
the units in the file: PMT positions in mm, envelope scalars in m
([`.npz` layout](../reference/pmt-npz-schema.md)). A string detector always uses a `.npz`, in a
different layout.

## 3. Write the physics config

Save as config/MyTank_physics_config.json. Reusing the water medium and the SK QE curve, with your own
reflectivities:

```json
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

The scattering and absorption lengths, `g` and `qe` are left out, so they are taken from the
medium and the QE curve at 400 nm. To set them by hand instead (as `config/EOS_physics_config.json`
does), give `g` and `qe` at the top level and the three lengths in a `scalar_mode` block; they
are then used in scalar mode (`wavelength_mode=False`).

The four reflection keys and `qe_corrections` are required. For reference, SK's black liner
reflects about 5% and a PMT face about 25%.

The loader is strict: a misspelled key raises and suggests the closest one, and a missing
optical value with nothing to project it from raises and names it, for example:

```
ValueError: Physics config 'config/MyTank_physics_config.json' has no ['scatter_length',
'absorption_length', 'mie_scatter_length', 'g'] and no 'medium_model' to project from at λ=400.0nm.
```

## 4. Look at the geometry

```bash
python scripts/visualize_detector.py --geom config/MyTank_geom_config.json
```

This needs the `[viz]` extra. It prints the sensor count and the x, y, z ranges in metres, and
writes `MyTank_geom_config_geometry.html` (3D) and, for a cylinder, an unrolled 2D PNG. Check
that the count is close to what you asked for (a large shortfall usually means `sensor_radius`
is too big for the surface), that the extent is right in metres, and that `z` is the axis.

## 5. Simulate an event

Edit the two paths at the top of `examples/hello_simulate.py`:

```python
GEOM, PHYS = 'config/MyTank_geom_config.json', 'config/MyTank_physics_config.json'
```

and run it. It prints a finite, positive total charge over your sensor count and writes the
display. The unrolled display draws cylinders only; for another shape, use the printed charge
or `lucid.visualization.event_display_3d`. Also drop the cylinder grid keywords
(`n_cap`, `n_angular`, `n_height`) for a non-cylinder.

If it fails:

- `FileNotFoundError` for `data/water/muon/cherenkov/weights.npz`: the example data is not
  installed (`python scripts/download_data.py`). Emitters exist for muons and electrons; water,
  WbLS and ice all use the water ones.
- `FileNotFoundError` for your `.npz`, material or QE file: those paths are relative to the
  config file's directory, not the working directory.
- `ValueError` at setup: usually the physics config; the message names the field and file.
- Zero total charge: the track sends no light at a sensor (a vertex outside the tank, or a
  track along the axis of a short, wide tank). A setup warning starting `Sensor map:` points at
  a geometry problem.

## Next

The same two files drive [reconstruction](reconstruction.md) and [calibration](calibration.md).
To produce GEANT4 datasets with a detector, it also needs a readout
([digitizer and trigger](../reference/digitizer-and-trigger.md)).
