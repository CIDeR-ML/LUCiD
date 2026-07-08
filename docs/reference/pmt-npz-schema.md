# Unified PMT-array `.npz` schema

A cylindrical detector whose PMT positions come from an external file
(measured layout, WCSim geofile export, ROOT TTree, …) is stored on
disk as a NumPy `.npz` whose contents follow the schema described
here. :py:meth:`lucid.geometry.Cylinder.from_pmt_file` reads any
file that conforms to this schema; conversion scripts that produce
new files (the portable converter is `scripts/geofile_to_npz.py`; `config/scripts/` is a local scratch area) must respect it.

## Coordinate convention

* All positions are in **millimetres**.
* `z` is the **vertical** axis; `(x, y)` is the cylinder cross-section.
* The cylinder is centred at the origin: caps at `z = ±height/2`,
  barrel at `r_xy = radius`.
* `directions` are unit 3-vectors pointing in the PMT viewing
  direction. For an inward-facing barrel PMT this is the inward
  radial direction; for an inward-facing top-cap PMT it is `-ẑ`.

## Required arrays — active sensors

These define the photosensors that participate in propagation. They
must all be present and aligned (`positions_mm[i]`, `directions[i]`,
`surfaces[i]`, `pmt_id[i]` all describe the same PMT). `N` is the
number of active sensors.

| Key             | Shape    | dtype       | Meaning                                                  |
|-----------------|----------|-------------|----------------------------------------------------------|
| `positions_mm`  | `(N, 3)` | `float64`   | PMT positions, mm, z-vertical                            |
| `directions`    | `(N, 3)` | `float64`   | PMT viewing-direction unit vectors                       |
| `surfaces`      | `(N,)`   | unicode str | One of `'barrel'`, `'top'`, `'bottom'`                   |
| `pmt_id`        | `(N,)`   | `int32/64`  | Unique PMT identifier (cable id, sequential, …)          |

## Required scalars — cylinder envelope

| Key             | Shape  | dtype     | Meaning                                |
|-----------------|--------|-----------|----------------------------------------|
| `radius`        | scalar | `float64` | Cylinder wall radius (m)               |
| `height`        | scalar | `float64` | Cylinder height (m)                    |
| `sensor_radius` | scalar | `float64` | Active-PMT radius (m)                  |

## Optional — inactive PMTs

Some geofiles include PMTs that are *not* active sensors (e.g. HK's
mPMT sub-PMTs and OD PMTs are kept in the source for reference but
not used in propagation). When present they live under the
`inactive_` prefix and are loaded as opaque attributes on the
detector instance — they are *not* used by the propagator.

| Key                        | Shape    | dtype     | Meaning                              |
|----------------------------|----------|-----------|--------------------------------------|
| `inactive_positions_mm`    | `(M, 3)` | `float64` | Inactive PMT positions, mm           |
| `inactive_directions`      | `(M, 3)` | `float64` | Inactive PMT directions              |
| `inactive_surfaces`        | `(M,)`   | str       | `'barrel'/'top'/'bottom'` (or other) |
| `inactive_pmt_id`          | `(M,)`   | int       | Inactive PMT ids                     |
| `inactive_<anything-else>` | `(M, …)` | any       | Opaque per-inactive-PMT metadata     |

`M` need not equal `N`. If no inactive block is present the loader
exposes nothing and the detector behaves as if every PMT in the file
is active.

## Optional — per-detector metadata

Any additional array whose first dimension equals `N` is treated as
per-active-PMT metadata: it is reordered to match `all_points`
(barrel → top cap → bottom cap) and exposed as an instance attribute
named after the key.

Examples currently in use:

| Detector       | Key                  | Meaning                                         |
|----------------|----------------------|-------------------------------------------------|
| WCTE           | `mpmt_id`            | mPMT module index (0..105)                      |
| WCTE           | `pmt_id_in_mpmt`     | PMT slot within the dome (0..18; 0 = central)   |
| WCTE           | `mpmt_kind`          | `'ME'` / `'FD'`                                 |
| SK_official    | `pmtflag`            | PMT status flag (e.g. dead/alive)               |
| SK_official    | `hutnum`             | Electronics hut number                          |
| SK_official    | `group`              | Electronics group                               |
| SK_official    | `oldhv`              | High-voltage value                              |
| SK / HK        | `pmt_type`           | WCSim type code (0 for standard ID PMT)         |

Scalars and arrays whose first dimension is anything other than `N`
or `M` are stored on the instance as-is (no reordering).

## Reserved attribute names

The loader sets the following on the returned `Cylinder` instance,
so per-detector metadata keys **must not collide with**:

```
all_points, barr_points, tcap_points, bcap_points,
ID_to_position, ID_to_case, pmt_id_to_idx,
pmt_directions, raw_positions,
r, H, S_radius, n_sensors, C, npz_file_path, snap_to_wall,
_n_cap, _n_angular, _n_height
```

## String / DOM detectors — a separate `.npz` layout

String telescopes (`detector_type: "string"`, IceCube-like) use a **different**
`.npz` schema, read by :py:meth:`lucid.geometry.StringTelescope.from_npz`
(`lucid/geometry/string.py`). It stores per-DOM positions grouped by string
rather than a flat PMT list, and — unlike the cylinder schema above — its
positions are in **metres**, not millimetres. There is no `positions_mm`-style
suffix precisely because the units differ; do not mix the two conventions.

### Required arrays / scalars

| Key              | Shape                  | dtype       | Units | Meaning                                                       |
|------------------|------------------------|-------------|-------|---------------------------------------------------------------|
| `dom_xyz`        | `(N_str, max_dom, 3)`  | `float64`   | m     | Per-DOM positions, grouped by string; **NaN-padded** to `max_dom` |
| `n_dom_per_str`  | `(N_str,)`             | `int32`     | —     | Actual DOM count on each string (rows past this are the NaN pad) |
| `sensor_radius`  | scalar                 | `float64`   | m     | DOM glass-sphere radius                                       |
| `envelope_radius`| scalar                 | `float64`   | m     | Radius of the bounding cylinder used for ray entry/exit clipping |
| `envelope_z_min` | scalar                 | `float64`   | m     | Bottom of the bounding cylinder (`z`)                        |
| `envelope_z_max` | scalar                 | `float64`   | m     | Top of the bounding cylinder (`z`)                          |

`N_str` is the number of strings, `max_dom` the longest string's DOM count
(shorter strings are NaN-padded to it). There is no closed detector surface —
the envelope cylinder is only a clipping/inside-detector volume, not a
reflective wall.

Files in the repo may also carry informational scalars such as `n_strings`,
`max_dom_per_str`, and `total_doms`; the loader does **not** read them (they are
derivable from `dom_xyz` / `n_dom_per_str`) and they can be omitted.

### No converter script yet

The cylinder path has a portable converter (`scripts/geofile_to_npz.py`); there
is **no equivalent string-npz generator** in the repo yet. String `.npz` files
(e.g. `config/icecube86_simple.npz`) are produced ad hoc — write the six keys
above directly with `numpy.savez`.

## Versioning

If/when this schema needs a breaking change, add a `schema_version`
scalar and bump it. Loaders should treat absence as version 1.
