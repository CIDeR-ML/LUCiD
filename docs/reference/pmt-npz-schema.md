# Sensor position files (`.npz`)

Measured sensor layouts are NumPy `.npz` files named by `npz_file_path` in a geometry config.
Cylinders and strings use different layouts and different units.

## Cylinder PMT arrays

Read by `lucid.geometry.Cylinder.from_pmt_file(path, snap_to_wall=True)`.
`scripts/geofile_to_npz.py` writes this layout from a WCSim geofile.

Coordinates: positions in **millimetres**, `z` vertical, the cylinder centred at the origin
(caps at `z = ±height/2`, barrel at `r_xy = radius`). Directions are unit vectors along the PMT's
viewing direction (inward radial for a barrel PMT, `-z` for a top-cap PMT).

Required, aligned per active PMT (`N` of them):

| key | shape | dtype | meaning |
|---|---|---|---|
| `positions_mm` | `(N, 3)` | float64 | PMT positions, mm |
| `directions` | `(N, 3)` | float64 | viewing-direction unit vectors |
| `surfaces` | `(N,)` | str | `'barrel'`, `'top'` or `'bottom'` |
| `pmt_id` | `(N,)` | int | unique PMT identifier |

Required scalars, in **metres**:

| key | meaning |
|---|---|
| `radius` | wall radius |
| `height` | cylinder height |
| `sensor_radius` | PMT radius |

With `snap_to_wall=True` (the default), barrel PMTs are moved onto `r_xy = radius` and cap PMTs
onto `z = ±height/2`; geofiles often leave them a few cm off the surface.

Optional:

- `inactive_positions_mm`, `inactive_directions`, `inactive_surfaces`, `inactive_pmt_id` and any
  other `inactive_*` array (`M` rows): PMTs kept for reference (for example HK's outer-detector
  PMTs). They are stored on the detector object and not simulated.
- any other array of length `N`: per-PMT metadata, reordered like the sensors (barrel, top cap,
  bottom cap) and stored as an attribute of the same name. The shipped files carry `mpmt_id` and
  `pmt_id_in_mpmt` (`config/wcte_geometry.npz`) and `pmt_type` (`config/sk_geometry.npz`,
  `config/hk_geometry.npz`).
- anything else is stored as-is.

Metadata keys must not reuse these attribute names: `all_points`, `barr_points`,
`tcap_points`, `bcap_points`, `ID_to_position`, `ID_to_case`, `pmt_id_to_idx`,
`pmt_directions`, `raw_positions`, `r`, `H`, `S_radius`, `n_sensors`, `C`, `npz_file_path`,
`snap_to_wall`.

`scripts/geofile_to_npz.py` handles single-PMT-type geofiles such as SK's. It does not write the
`inactive_*` arrays that HK and WCTE (multi-PMT modules, outer detector) need; their files ship
pre-built in `config/`. The WCSim geofiles behind the shipped files are the
`detector-geometry-sources` release asset ([studies and archives](../studies.md)).

## String detectors

Read by `lucid.geometry.StringTelescope.from_npz(path)`. Positions in **metres**.

| key | shape | meaning |
|---|---|---|
| `dom_xyz` | `(N_str, max_dom, 3)` | DOM positions per string, NaN-padded to the longest string |
| `n_dom_per_str` | `(N_str,)` | DOMs on each string |
| `sensor_radius` | scalar | DOM radius |
| `envelope_radius` | scalar | radius of the bounding cylinder used to clip rays |
| `envelope_z_min`, `envelope_z_max` | scalar | bottom and top of the bounding cylinder |

The bounding cylinder only limits the volume; it is not a reflecting wall. Extra keys
(`n_strings`, `max_dom_per_str`, `total_doms` in the shipped files) are ignored. There is no
converter: write the keys with `numpy.savez`.
