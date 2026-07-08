# Bundled detectors

A detector in LUCiD is configuration, not code: a `*_geom_config.json` (shape,
sensor placement) plus a `*_physics_config.json` (optical properties) is enough
to point the same simulation engine at a different tank, sphere, box, or
neutrino telescope. See [geometry & configuration](../concepts/geometry.md) for
how the two files are read and dispatched, and
[model your own detector](../guides/model-your-own-detector.md) for writing a
new pair.

## The list

| Name | Geometry class | Sensor placement | Material(s) | Config pair |
|------|----------------|-------------------|-------------|--------------|
| `SK` | cylinder | measured PMT positions (`.npz`) | water | `SK_geom_config.json` / `SK_physics_config.json` |
| `SK_like` | cylinder | algorithmic | water | `SK_like_geom_config.json` / `SK_like_physics_config.json` |
| `SK_like_wbls` | cylinder | algorithmic | WbLS | `SK_like_wbls_geom_config.json` / `SK_like_wbls_physics_config.json` |
| `HK` | cylinder | measured PMT positions (`.npz`) | water | `HK_geom_config.json` / `HK_physics_config.json` |
| `BigHK` | cylinder | algorithmic | water | `BigHK_geom_config.json` / `BigHK_physics_config.json` |
| `WCTE` | cylinder | measured PMT positions (`.npz`) | water | `WCTE_geom_config.json` / `WCTE_physics_config.json` |
| `WCTE_like` | cylinder | algorithmic | water | `WCTE_like_geom_config.json` / `WCTE_like_physics_config.json` |
| `IWCD` | cylinder | algorithmic | water | `IWCD_geom_config.json` / `IWCD_physics_config.json` |
| `JUNO` | sphere | algorithmic | water | `JUNO_geom_config.json` / `JUNO_physics_config.json` |
| `JUNO_wbls` | sphere | algorithmic | WbLS | `JUNO_wbls_geom_config.json` / `JUNO_wbls_physics_config.json` |
| `TAO` | sphere | algorithmic | water | `TAO_geom_config.json` / `TAO_physics_config.json` |
| `IceCube86_full` | string | measured DOM positions (`.npz`) | ice | `IceCube86_full_geom_config.json` / `IceCube86_ice_physics_config.json` |
| `IceCube86_simple` | string | measured DOM positions (`.npz`) | water | `IceCube86_simple_geom_config.json` / `IceCube86_physics_config.json` |
| `MidBox` | box | algorithmic | water | `MidBox_geom_config.json` / `MidBox_physics_config.json` |
| `EOS` | cylinder | algorithmic | water | `EOS_geom_config.json` / `EOS_physics_config.json` |
| `nuSCOPE` | box | algorithmic | water | `nuSCOPE_geom_config.json` / `nuSCOPE_physics_config.json` |

All config pairs live under `config/` at the repo root. "Algorithmic" sensor
placement means the sensor positions are generated from a few numbers (radius,
height, sensor count, ...) in `geometry_definitions`; "measured" means the
positions are read from a `.npz` geometry file (real PMT or DOM survey data
converted to LUCiD's schema).

A few notes on the real-experiment names above:

- **SK** = Super-Kamiokande, **HK** = Hyper-Kamiokande — large cylindrical
  water-Cherenkov detectors in Japan. `SK_like` reproduces their scale
  algorithmically (no PMT survey file needed); `SK` and `HK` use the real
  measured PMT layouts.
- **WCTE** = Water Cherenkov Test Experiment, a smaller test-bench tank used to
  validate water-Cherenkov reconstruction techniques; `WCTE` uses the real
  measured PMT layout, `WCTE_like` an algorithmic stand-in of similar scale.
- **JUNO** = Jiangmen Underground Neutrino Observatory, a large spherical
  liquid-scintillator detector; **TAO** is its companion near detector.
  `JUNO_wbls` swaps in water-based liquid scintillator (WbLS) instead of pure
  water, and `SK_like_wbls` does the same for the SK-like cylinder.
- **IceCube** is the cubic-kilometre neutrino telescope at the South Pole:
  strings of **DOMs** (digital optical modules) frozen into the ice.
  `IceCube86_full` uses the real 86-string DOM layout in an ice medium;
  `IceCube86_simple` is a lighter-weight string layout used for propagator
  cross-checks, run in a water medium.

`BigHK`, `IWCD`, `EOS`, `MidBox`, and `nuSCOPE` are smaller or larger synthetic
tanks used for testing and demos rather than a specific real experiment.

## What the tutorials and examples use

`SK_like` (cylinder, water, algorithmic) is the default across almost every
`examples/hello_*.py` script and tutorial notebook (`00_quickstart`,
`calibration_optimization`, `calibration_gradients`, `track_gradients`,
`data_vs_prediction`, and the SK-like leg of `track_optimization`). The
`event_displays` tutorial additionally tours `IceCube86_full`/`IceCube86_ice`
and `SK_like_wbls` to show a cylinder / sphere / box / string and
water / WbLS / ice all through the same call; `track_optimization` also runs a
`JUNO` (sphere) example; and `examples/hello_telescope.py` uses
`IceCube86_full` with the ice physics config as the neutrino-telescope
entry point.
