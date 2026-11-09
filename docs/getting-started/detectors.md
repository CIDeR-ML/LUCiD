# Bundled detectors

A detector is two JSON files in `config/`: `<name>_geom_config.json` (shape and sensors) and
`<name>_physics_config.json` (optical properties). Pass their paths to
`setup_event_simulator(geom, ..., physics_config=phys)`. To add one, see
[Model your own detector](../guides/model-your-own-detector.md); every key is in the
[configuration reference](../reference/config.md).

| name | shape | sensors | placement | medium |
|---|---|--:|---|---|
| `SK` | cylinder | 11096 | measured (`sk_geometry.npz`) | water |
| `SK_WAND` | cylinder | 11096 | measured (`sk_geometry.npz`) | water |
| `SK_like` | cylinder | 10764 | generated | water |
| `SK_like_wbls` | cylinder | 10764 | generated | WbLS |
| `HK` | cylinder | 19746 | measured (`hk_geometry.npz`) | water |
| `HK_WAND` | cylinder | 19746 | measured (`hk_geometry.npz`) | water |
| `BigHK` | cylinder | 19650 | generated | water |
| `WCTE` | cylinder | 1995 | measured (`wcte_geometry.npz`) | water |
| `WCTE_like` | cylinder | 2444 | generated | water |
| `IWCD` | cylinder | 6892 | generated | water |
| `EOS` | cylinder | 194 | generated | water |
| `JUNO` | sphere | 10000 | generated | water |
| `JUNO_wbls` | sphere | 10000 | generated | WbLS |
| `TAO` | sphere | 4000 | generated | water |
| `MidBox` | box | 8932 | generated | water |
| `nuSCOPE` | box | 4928 | generated | water |
| `IceCube86_full` | string | 5160 | measured (`icecube86_full.npz`) | ice |
| `IceCube86_simple` | string | 4680 | measured (`icecube86_simple.npz`) | water |

The config pair of each is `config/<name>_geom_config.json` and
`config/<name>_physics_config.json`, except `IceCube86_full`, whose physics config is
`config/IceCube86_ice_physics_config.json`.

"Generated" placement tiles sensors from a few numbers (radius, height, sensor count, sensor
radius); "measured" reads sensor positions from a `.npz` file converted from a survey or WCSim
geometry. Generated placement places a little fewer sensors than the config asks for.

Notes on the names:

- `SK` (Super-Kamiokande) and `HK` (Hyper-Kamiokande) use the real PMT layouts; `SK_like` is a
  generated tank at SK scale and is what most examples use.
- `SK_WAND` and `HK_WAND` are fixed descriptions for dataset production. They are the only
  configs with a readout (`digitizer` and `trigger` blocks), which `lucid-run-job` needs
  ([digitizer and trigger](../reference/digitizer-and-trigger.md)).
- `WCTE` is the Water Cherenkov Test Experiment layout; `WCTE_like` a generated tank of similar
  size.
- `JUNO_wbls` and `SK_like_wbls` fill the JUNO-sized sphere and the SK-sized tank with WbLS.
  `JUNO` and `TAO` are spheres at JUNO and TAO scale filled with water.
- `IWCD`, `EOS` and `nuSCOPE` are generated tanks at the scale of those experiments, not models
  of them. `BigHK` and `MidBox` are synthetic.
- `IceCube86_full` is the 86-string IceCube DOM layout in ice (used by
  `examples/hello_telescope.py`). `IceCube86_simple` is a lighter string layout in water.
