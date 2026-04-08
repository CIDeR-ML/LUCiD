# LUCiD Notebook & Module Cleanup Suggestions

These are improvements to make AFTER the format migration is complete.
Not blocking — the code works without these — but they improve quality and maintainability.

---

## Code Quality Issues

### 1. Auxiliary return value in `tools/optimization/losses.py:268`
```python
# Current (wrong auxiliary, correct loss):
return combined, (vertex_loss_val, counts_loss_val, vertex_loss_val)
# Should be:
return combined, (vertex_loss_val, counts_loss_val, time_loss_val)
```
The loss scalar is correct (gradients fine). Only the monitoring/logging tuple is wrong —
returns vertex_loss twice instead of time_loss. Any code that unpacks `aux[2]` as "time_loss"
is actually getting vertex_loss.

### 2. Docstring in `tools/optimization/losses.py:239`
Says "product of vertex loss, counts loss, and energy loss" — should say "time loss"
(uses `cone_time_loss`, not `energy_loss`).

### 3. `stop_gradient` inconsistency on position in `origin_time_loss` call
- `losses.py:259`: `origin_time_loss(jax.lax.stop_gradient(position), ...)`
- `single_track_optimization.py:173`: `origin_time_loss(position, ...)` (no stop_gradient)
Decide which is correct and make consistent.

### 4. Dead code: `direction = spherical_to_cartesian(theta, phi)` in loss functions
Computed but never used inside `combined_product_loss` in notebooks:
- `parameter_scans_1D.ipynb`
- `grad_loss_and_opt_in_2D.ipynb`
- `visualize_3D_track_optimization.ipynb`

### 5. Dead code: `energy` assigned but unused in data mode
`_simulation_with_data_impl` line 764: `energy = particle_params.energy` — never used after assignment.

### 6. TRUE_T0 cheat in `tracking_opt_development.ipynb`
Line 1207: `TRUE_T0, #stage1_results['best_t0'],`
Uses ground truth t0 instead of estimated value. The commented-out code shows the intended behavior.

---

## Duplication

### 7. `counts_loss` ≡ `poisson_nll`
`tools/optimization/losses.py:counts_loss` and `tools/losses.py:poisson_nll` are identical
functions with different names. Should have one canonical version.

### 8. Photon data padding + rotation boilerplate (~30 lines)
Copy-pasted in 8+ locations:
- `geometry_and_events_3D_visualization.ipynb`
- `cylinder_2D_displays.ipynb`
- `parameter_scans_1D.ipynb` (x3 copies!)
- `data_vs_pred_hit_predictions.ipynb` (has local helper + inline copy)
- `grad_loss_and_opt_in_2D.ipynb`
- `visualize_3D_track_optimization.ipynb`
- `tracking_opt_with_gif.ipynb`
- `tracking_opt_development.ipynb`
- `tools/optimization/single_track_optimization.py:generate_event_data()`

Should extract to utility functions:
```python
def pad_photon_data(photon_data, target_size=1_000_000): ...
def set_photon_transform(photon_data, track): ...
```
Note: `parameter_scans_1D_v2` already uses `set_photon_transform` — verify if it exists in tools/
or is notebook-local.

### 9. `create_combined_loss_function` exists in 3 places
- `tools/optimization/losses.py` (the module version)
- `tools/optimization/single_track_optimization.py` (different signature, has nrays-dependent tau)
- Several notebooks define their own inline copies

Should consolidate to one canonical version.

---

## Dead Imports (per notebook)

### `geometry_and_events_3D_visualization.ipynb`
- `from tools.utils import generate_random_params` — never used

### `parameter_scans_1D.ipynb`
- `from tools.utils import load_single_event, save_single_event, generate_random_params, print_particle_params` — all unused
- `from scipy.interpolate import interp1d` — unused
- `from functools import partial` — unused

### `data_vs_pred_hit_predictions.ipynb`
- `import plotly.graph_objects as go` — unused
- `import plotly.subplots as sp` — unused
- `from datetime import datetime` — unused

### `grad_loss_and_opt_in_2D.ipynb`
- `from tools.optimization.optimize import get_detector_params_from_config` — imported but unused
- `import uproot` — unused
- `from scipy.interpolate import interp1d` — unused
- `from functools import partial` — unused
- Massive import redundancy: `jax`, `jnp`, `np`, `plt`, `pickle` imported 3-5 times across cells

### `visualize_3D_track_optimization.ipynb`
- `from tools.generate import read_photon_data_from_photonsim` — imported but unused (only loads from pkl)

---

## Hardcoded Values

### Speed of light
```python
C_MEDIUM = 0.299792 / 1.33  # in 4+ notebooks
```
Should use `from tools.utils import get_speed_of_light_in_material`

### Inconsistent detector params across notebooks
Different scatter_length, reflection_rate, absorption_length values without justification.
Using `load_detector_params` or `default_detector_params=True` would standardize.

### Inconsistent cone_time_loss tau
- `parameter_scans_1D`: `tau=0.23`
- `parameter_scans_1D_v2`: `tau=0.12`
- `grad_loss_and_opt_in_2D`: default (0.12)
- `single_track_optimization.py`: `tau=0.12`

### Photon padding target `1_000_000`
Hardcoded in 8+ locations, coupled to internal simulation constant.
Should be a named constant or derived from `n_photons`.

---

## Structural Improvements

### 10. `generate_random_params` returns old 3-tuple
```python
return energy, position, direction_angles  # should return ParticleParams
```

### 11. `generate_random_event_params` returns old 3-tuple
```python
return position, direction, energy  # should return ParticleParams
```

### 12. `print_particle_params` / `print_propagation_params` expect old tuples
Should accept ParticleParams / DetectorParams with named field access.

### 13. `save_single_event` / `load_single_event` use index-based access
```python
particle_params[0], particle_params[1], particle_params[2]  # should use named fields
sensor_params[0], sensor_params[1], sensor_params[2], sensor_params[3]
```

### 14. `get_detector_params_from_config` returns old 4-tuple
Returns `(scatter_length, reflection_rate, absorption_length, qe)` —
should return DetectorParams or be deprecated in favor of `load_detector_params`.

### 15. Missing `default_particle_bounds(detector_bounds)`
Analogous to `default_bounds(num_sensors)` for DetectorParams.
Would enable normalization of ParticleParams in optimizer (currently not done).

---

## Dead Code in Supporting Modules

### 16. Dead `true_data` parameter in `combined_product_loss`
Both `losses.py` and `single_track_optimization.py` versions have `true_data` in the
function signature but never read it. Observation data comes from the separate args
(`hit_detector_positions`, `observed_times`, `observed_counts`).

### 17. Dead params in `single_track_optimization.py`'s `create_combined_loss_function`
- `vertex_weight_scale` — in signature, never used in body
- `counts_weight_scale` — in signature, never used in body
- `c_medium` — in signature, never used in body

### 18. Dead `cone_time_loss_parametric` in `single_track_optimization.py`
Defined inside `create_combined_loss_function` but never called.
The actual loss uses `cone_time_loss` from `losses.py` with hardcoded `tau=0.12`.
The entire nrays-dependent tau infrastructure (loads `tau_parametrization.pkl`,
computes `a_energy`, `b_log10nrays`, `c_intercept`, `log10_nrays`) is also dead.

### 19. Dead `test_params` flat array in `energy_scan_optimization` (`utils/functions.py`)
```python
test_params = jnp.array([position[0], position[1], position[2],
                          initial_t0, theta, phi, energy])
```
Constructed but never used — leftover from old commented-out code.

---

## Interface Improvements (post-migration)

### 20. Index-based detector_params printing in notebooks
- `tracking_opt_development.ipynb`: prints `detector_params[0]`, `detector_params[1]`, etc.
- `tracking_opt_with_gif.ipynb`: same pattern
Should use named field access (`.scatter_length`, `.wall_reflection_rate`, etc.).

### 21. Adopt v2 notebook loss function pattern
The `parameter_scans_1D_v2` notebook uses a cleaner loss interface:
```python
combined_product_loss(particle_params, true_data, key)
# where true_data = (hit_counts, hit_times), detector_points captured in closure
```
Current optimizer uses separate args:
```python
combined_product_loss(track, hit_detector_positions, observed_times, observed_counts, true_data, key)
```
The v2 pattern is cleaner but requires changing all caller code.

### 22. Tracking optimizer: normalize/denormalize in [0,1] space
Calibration notebooks use `normalize_params`/`denormalize_params` to handle scale differences.
Tracking optimizer could benefit from same approach.
Requires implementing `default_particle_bounds(detector_bounds)` (#15).

### 23. Tracking optimizer: `optax.multi_transform` with warmup
Calibration notebooks use `multi_transform` with `'train'/'freeze'` labels and
`warmup_constant_schedule`. Could replace manual damping factor and energy freezing
with more idiomatic optax patterns. Only applicable after normalization is added (#22).

### 24. Bundle true params in `run_complete_optimization_adam`
Currently takes `true_energy, true_position, true_direction, TRUE_T0` as separate args.
Could accept a single `true_track: ParticleParams` instead.

### 25. Optimization config JSON files use old field names
Have `reflection_rate` (single value) instead of `wall_reflection_rate`/`sensor_reflection_rate`.
Missing `qe_corrections` field.
Should be updated to match `DetectorParams` field names or replaced by physics config paths.
