# lucid.optimization

Seed search + the config-driven reconstruction entry point.

- **`grid_search.py`** — hierarchical charge-weighted position grid search and direction
  cone search; provides the reconstruction seeds (`hierarchical_position_grid_search`,
  `hierarchical_direction_search_cone`, `energy_scan_optimization`,
  `get_detector_bounds`).
- **`run.py`** — the `lucid-optimize` console entry point: full track reconstruction
  (seeding → `lucid.fitting.fit_track_multistart` Fisher-Gauss-Newton fit) driven by a
  JSON config; see `configs/example_water_mu.json`.

```bash
lucid-optimize lucid/optimization/configs/example_water_mu.json
```

The fit machinery itself lives in `lucid/fitting/` (`ReconModel`, `fit_track`,
`fit_track_multistart`). Docs: `docs/guides/reconstruction.md` and
`docs/reference/cli.md`.
