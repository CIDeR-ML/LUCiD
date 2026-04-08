# LUCiD Notebook Migration Analysis

## Context

After merging v2 simulation features into LUCiD's `tools/simulation.py`, all notebooks in
`good_notebooks/` and several supporting Python modules need updating to use the new API.
This document tracks the complete analysis and required changes.

---

## New API Reference

### DetectorParams (NamedTuple — JAX pytree)
```python
class DetectorParams(NamedTuple):
    scatter_length: jnp.ndarray           # scalar, meters
    wall_reflection_rate: jnp.ndarray     # scalar, [0, 1]
    sensor_reflection_rate: jnp.ndarray   # scalar, [0, 1]
    absorption_length: jnp.ndarray        # scalar, meters
    qe: jnp.ndarray                       # scalar, base QE [0, 1]
    qe_corrections: jnp.ndarray           # (num_sensors,), per-sensor multipliers
```

### ParticleParams (NamedTuple — JAX pytree)
```python
class ParticleParams(NamedTuple):
    energy: jnp.ndarray      # scalar, MeV
    position: jnp.ndarray    # (3,), meters
    theta: jnp.ndarray       # scalar, polar angle radians
    phi: jnp.ndarray         # scalar, azimuthal angle radians
    t0: jnp.ndarray          # scalar, vertex time offset (ns)

    @classmethod
    def from_cartesian(cls, energy, position, direction, t0=0.0): ...

    @property
    def direction(self):  # computed from (theta, phi)
```

### Source Types (callable NamedTuples)
```python
IsotropicSource(position, intensity)        # via isotropic_source()
LaserSource(position, intensity, direction, fiber_NA)  # via laser_source()
```

### setup_event_simulator() Signature
```python
def setup_event_simulator(
    json_filename,
    n_photons=1_000_000,
    temperature=0.2,
    K=7,
    is_data=False,
    is_calibration=False,
    max_sensors_per_cell=4,
    detector_type='Cylinder',
    use_expected_value=True,
    particle='muon',
    apply_smearing=True,
    physics_config=None,
    default_detector_params=False,
)
```

### Returned Simulator Call Signatures

| Mode | default_detector_params=False | default_detector_params=True |
|------|-------------------------------|------------------------------|
| Track | `(particle_params, detector_params, key)` | `(particle_params, key)` |
| Data  | `(particle_params, detector_params, key, photon_data)` | `(particle_params, key, photon_data)` |
| Calibration | `(source, detector_params, key)` | `(source, key)` |

### Key Helpers (from tools.detector_params)
```python
load_detector_params(filepath, num_sensors=None) → DetectorParams
save_detector_params(params, filepath)
create_default_detector_params(num_sensors) → DetectorParams
create_default_particle_params() → ParticleParams
normalize_params(params, bounds_min, bounds_max)
denormalize_params(normalized, bounds_min, bounds_max)
default_bounds(num_sensors) → (bounds_min, bounds_max)
make_optimization_mask(params, trainable_fields)
default_gradient_scales(num_sensors) → DetectorParams
```

---

## Notebook Inventory (18 notebooks)

### Already Fully Migrated (4 notebooks — no changes needed to simulation API)
| Notebook | Notes |
|----------|-------|
| `parameter_scans_1D_v2.ipynb` | Uses `ParticleParams`, `default_detector_params=True`, `physics_config`, `sweep_1d` |
| `detector_grad_qe_convergence_multi_source.ipynb` | Uses `DetectorParams`, `isotropic_source`, `load_detector_params`, `WC_loss` |
| `laser_source_grad_analysis.ipynb` | Uses `DetectorParams`, `laser_source`, `load_detector_params`, `sweep_1d`/`sweep_2d` |
| `grad_param_calibration_multi_init_no_qe.ipynb` | Uses `DetectorParams`, `laser_source`, `normalize_params`, `make_optimization_mask` |

### No Simulation API Changes Needed (3 notebooks)
| Notebook | Notes |
|----------|-------|
| `train_siren.ipynb` | Only `%run` external scripts (train.py, validate.py). No direct simulation calls. |
| `optimization_vs_variables.ipynb` | Pure post-processing from pickle files. No simulator calls. |
| `track_optimization_visualization.ipynb` | Only loads pkl results and plots. No simulator or loss function calls despite importing simulation. |

### Need Full Migration (11 notebooks)
| Notebook | Modes | Key Issues |
|----------|-------|------------|
| `geometry_and_events_3D_visualization.ipynb` | Standard + Data + Calibration | All 3 simulator variants, source_params tuple |
| `cylinder_2D_displays.ipynb` | Standard + Data | 2 simulators, photon data boilerplate |
| `parameter_scans_1D.ipynb` | Standard + Data | Loss function, scan arrays, duplicated photon boilerplate x3 |
| `data_vs_pred_hit_predictions.ipynb` | Standard + Data | Loop creating multiple simulators, metric comparison |
| `computational_performance_evaluation.ipynb` | Standard + Calibration | Benchmark loops, source_params tuple |
| `grad_loss_and_opt_in_2D.ipynb` | Standard + Data | Loss function, Adam optimizer, 2D scans, random t0 |
| `visualize_3D_track_optimization.ipynb` | Standard + Data | Loss function, Adam optimizer, 3D plotly, JUNO config |
| `tracking_opt_with_gif.ipynb` | Standard + Data | Loss function, Adam optimizer, GIF generation |
| `tracking_opt_development.ipynb` | Standard + Data | Full 5-stage pipeline, calls optimization utility functions |

---

## Detailed Per-Notebook Analysis

### 1. `geometry_and_events_3D_visualization.ipynb`

**Purpose:** Generates events for multiple detector geometries (SK, JUNO, MidBox) and renders 3D visualizations with event displays. Tests all 3 simulation modes.

**Imports needing change:**
```python
from tools.utils import generate_random_params      # never used — remove
from tools.utils import load_single_event, save_single_event  # use old tuple format
```

**detector_params (old 4-tuple):**
```python
detector_params = (
    jnp.array(100),      # scatter_length
    jnp.array(0.2),      # reflection_rate (single)
    jnp.array(1000),     # absorption_length
    jnp.array(0.001)     # dead tau_gs
)
```

**track_params (old 3-tuple):**
```python
track_params = (
    jnp.array(1050.0, dtype=jnp.float32),
    jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
    jnp.array([jnp.pi/2, jnp.pi/6], dtype=jnp.float32)
)
```

**source_params (old 2-tuple):**
```python
source_params = (
    jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
    jnp.array(1.0, dtype=jnp.float32)
)
```

**3 simulator calls with old format:**
```python
# Standard mode:
single_event = simulator(track_params, detector_params, key)
# Calibration mode:
single_event = simulator(source_params, detector_params, key)
# Data mode:
data_single_event = data_simulator(data_track_params, detector_params, data_event_key, photon_data)
```
Where `data_track_params = (energy, position, direction_vector)` — Cartesian direction.

**Photon data boilerplate:** Full padding + rotation + translation inline code.

**t0 usage:** None.

**Recommendation:** Use `default_detector_params=True` with `physics_config`. Convert track_params to `ParticleParams`. Convert source to `isotropic_source()`. Factor out photon data setup.

---

### 2. `cylinder_2D_displays.ipynb`

**Purpose:** Data vs prediction comparison using cylindrical 2D display format. Loads ROOT photons and compares with SIREN predictions.

**detector_params (old 4-tuple):**
```python
detector_params = (
    jnp.array(50.),      # scatter_length
    jnp.array(0.2),      # reflection_rate
    jnp.array(50.),      # absorption_length
    jnp.array(0.001)     # dead
)
```

**Particle params — dual format:**
```python
# Prediction (spherical):
spherical_params = (true_energy, true_position, jnp.array([theta, phi]))
# Data (Cartesian):
true_params = (true_energy, true_position, true_direction)
```

**Photon data boilerplate:** Full inline rotation + translation setup.

**t0 usage:** None. Times compared directly without offset.

**Recommendation:** Use `default_detector_params=True`. Use `ParticleParams` for prediction, `ParticleParams.from_cartesian()` for data.

---

### 3. `parameter_scans_1D.ipynb`

**Purpose:** 1D parameter scans across 7 parameters (X, Y, Z, t0, theta, phi, energy) with loss landscape and gradient visualization.

**Dead imports (never used):**
```python
from tools.utils import load_single_event, save_single_event, generate_random_params, print_particle_params
from scipy.interpolate import interp1d
from functools import partial
```

**detector_params (old 4-tuple):**
```python
detector_params = (jnp.array(50.), jnp.array(0.2), jnp.array(50.), jnp.array(0.001))
```

**Loss function (old format):**
```python
@jit
def combined_product_loss(params, hit_detector_positions, observed_times, observed_counts,
                          true_data, detector_params, key):
    position = params[:3]
    t0 = params[3]
    theta = params[4]
    phi = params[5]
    energy = params[6]
    track_params = (energy, position, jnp.array([theta, phi]))  # OLD
    simulated_data = prediction_simulator(track_params, detector_params, key)  # OLD
    ...
    direction = spherical_to_cartesian(theta, phi)  # DEAD CODE — computed but never used
    ...
```

**Hardcoded:**
- `C_MEDIUM = 0.299792/1.33` — should use `get_speed_of_light_in_material('water')`
- `tau=0.23` in `cone_time_loss` call
- `true_t0 = 0.0` — hardcoded
- Photon padding boilerplate duplicated 3 times across cells

**t0 usage:** Hardcoded to 0.0 everywhere. Used in loss function.

**Recommendation:** Use `default_detector_params=True`. Rewrite loss to use `ParticleParams`. Remove dead code/imports. Use `parameter_scans_1D_v2` as the template — that notebook is the clean replacement.

---

### 4. `data_vs_pred_hit_predictions.ipynb`

**Purpose:** Compare predicted simulator output vs data mode across different n_photons values.

**detector_params (old 4-tuple, duplicated twice):**
```python
# Cell 4:
sensor_params = (jnp.array(50.0), jnp.array(0.1), jnp.array(100.0), jnp.array(0.001))
# Cell 14 (duplicate):
sensor_params = (jnp.array(50.0), jnp.array(0.1), jnp.array(100.0), jnp.array(0.001))
```

**Dead imports:**
```python
import plotly.graph_objects as go
import plotly.subplots as sp
from datetime import datetime
```

**Has local `prepare_photon_data` helper (good pattern, should go to `tools/`):**
```python
def prepare_photon_data(photon_data, track_position, track_direction):
    pd = dict(photon_data)
    original_direction = jnp.array([0.0, 0.0, 1.0])
    true_direction_norm = track_direction / (jnp.linalg.norm(track_direction) + 1e-8)
    rotation_axis = jnp.cross(original_direction, true_direction_norm)
    ...
    return pd
```

**Dual-format particle params:**
```python
prediction_params = (track_energy, track_position, direction_angles)   # spherical
data_params = (track_energy, track_position, track_direction)          # Cartesian
```

**Notable:** Inconsistent K values — CONFIG['K']=6 for first section, K=7 for second section (separate analyses merged into one notebook).

**t0 usage:** None.

**Recommendation:** Use `default_detector_params=True`. Move `prepare_photon_data` to `tools/utils.py` or `tools/generate.py`.

---

### 5. `computational_performance_evaluation.ipynb`

**Purpose:** Benchmarks forward simulation and gradient computation time across N and K values.

**detector_params (old 4-tuple):**
```python
detector_params = (jnp.array(4.), jnp.array(0.2), jnp.array(6.), jnp.array(0.001))
```

**track_params (old 3-tuple):**
```python
track_params = (
    jnp.array(800.0, dtype=jnp.float32),
    jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
    jnp.array([jnp.pi/3, jnp.pi/4], dtype=jnp.float32)
)
```

**Calibration source (old 2-tuple):**
```python
other_params = (source_origin, 1000)  # for is_calibration=True
```

**Simulator calls in benchmark loop:**
```python
result = simulate_event(other_params, detector_params, subkey)
```

**t0 usage:** None.

**Recommendation:** Use `default_detector_params=True`. Convert to `ParticleParams` and `isotropic_source()`.

---

### 6. `grad_loss_and_opt_in_2D.ipynb`

**Purpose:** 2D parameter scan landscapes with gradient overlays, then Adam optimization from random initial guesses with trajectory visualization.

**Massive import redundancy:** `jax`, `jnp`, `np`, `plt`, `pickle` imported 3-5 times each across cells.

**Dead imports:**
```python
from tools.optimization.optimize import get_detector_params_from_config  # imported but never used
import uproot          # never used
from scipy.interpolate import interp1d  # never used
from functools import partial  # never used
```

**detector_params (old 4-tuple):**
```python
detector_params = (jnp.array(50.), jnp.array(0.2), jnp.array(50.), jnp.array(0.065))
```

**Loss function (old format, returns auxiliary):**
```python
@jit
def combined_product_loss(params, hit_detector_positions, observed_times, observed_counts,
                          true_data, detector_params, key):
    ...
    track_params = (energy, position, jnp.array([theta, phi]))
    simulated_data = prediction_simulator(track_params, detector_params, key)
    ...
    direction = spherical_to_cartesian(theta, phi)  # DEAD CODE
    ...
    return combined_loss, (vertex_loss_val, counts_loss_val, time_loss_val)
```

**Notable — random t0:**
```python
true_t0 = jax.random.uniform(key, shape=(), minval=-15.0, maxval=15.0)
hit_times += true_t0  # applied post-simulation
```
This is the correct pattern for testing t0 recovery. Other notebooks hardcode t0=0.

**Adam optimizer receives `detector_params` and `prediction_simulator` as args:**
```python
def run_complete_optimization_adam_from_guess(
    initial_params, hit_detector_positions, ...,
    prediction_simulator, detector_params, combined_grad_fn, qe, ...):
```

**Recommendation:** Use `default_detector_params=True`. Rewrite loss to use `ParticleParams`. Remove dead imports and dead `direction` computation. Clean up import redundancy.

---

### 7. `visualize_3D_track_optimization.ipynb`

**Purpose:** Creates 3D interactive plotly visualizations showing detector surface, optimization trajectory arrows, and final reconstructed vs true tracks. Uses JUNO detector.

**detector_params (old 4-tuple):**
```python
detector_params = (jnp.array(50.), jnp.array(0.2), jnp.array(50.), jnp.array(0.065))
```

**Loss function — identical to grad_loss_and_opt_in_2D but captures `detector_params` in closure:**
```python
@jit
def combined_product_loss(params, hit_detector_positions, observed_times, observed_counts,
                          true_data, key):  # NOTE: no detector_params arg — captured in closure
    ...
    simulated_data = prediction_simulator(track_params, detector_params, key)
```

**Hardcoded JUNO config:**
```python
det_json_filename = '../config/JUNO_geom_config.json'
```

**Photon data boilerplate:** Full inline padding + rotation + translation.

**t0 usage:** Random t0 applied post-simulation (same as grad_loss_and_opt_in_2D).

**Recommendation:** Use `default_detector_params=True`. Rewrite loss to use `ParticleParams`. Factor out photon data setup.

---

### 8. `tracking_opt_with_gif.ipynb`

**Purpose:** Adam optimization from smeared initial guess (skipping grid search), generates animated GIF of convergence progression.

**detector_params — loaded from config (still returns old 4-tuple):**
```python
detector_params = get_detector_params_from_config(config)  # returns OLD 4-tuple
```

**Loss function via local `create_combined_loss_function`:**
```python
def create_combined_loss_function(prediction_simulator, detector_params):
    @jit
    def combined_product_loss(params, ...):
        track_params = (energy, position, jnp.array([theta, phi]))
        simulated_data = prediction_simulator(track_params, detector_params, key)
        ...
```

**Smeared initial guess generator:**
```python
def generate_smeared_initial_params(true_position, true_direction, true_energy, TRUE_T0,
                                     position_sigma=5.0, direction_sigma=15.0, ...):
    # Uses Rodrigues rotation for direction smearing
    # Returns jnp.array([x, y, z, t0, theta, phi, energy])
```

**Missing `detector_type` in `setup_event_simulator` call** — defaults to `'Cylinder'` which may be wrong.

**Uses global `DETECTOR_R`, `DETECTOR_H`** in smeared initial guess — should be passed as args.

**Recommendation:** Replace `get_detector_params_from_config` with `load_detector_params`. Use `default_detector_params=True`. Rewrite loss to use `ParticleParams`.

---

### 9. `tracking_opt_development.ipynb`

**Purpose:** Full 5-stage hierarchical optimization pipeline development (Stage 0-4). The most comprehensive optimization notebook.

**detector_params — loaded from config (old 4-tuple):**
```python
detector_params = get_detector_params_from_config(config)
```

**Calls optimization utility functions with old detector_params:**
```python
# Stage 0 — energy scan:
energy_scan_optimization(prediction_simulator, detector_params, position=..., ...)

# Stage 2 — direction search:
hierarchical_direction_search_cone(prediction_simulator, detector_params, optimal_position, ...)

# Stage 3 — energy refinement:
energy_scan_optimization(prediction_simulator, detector_params, optimal_position, ...)
```
These utility functions in `tools/optimization/utils/functions.py` also use old tuple format internally.

**BUG found — uses ground truth t0 as initial guess:**
```python
initial_params = jnp.array([
    stage1_results['best_position'][0],
    ...
    TRUE_T0,  # cheating -- should be stage1_results['best_t0']
    ...
])
```
The commented-out code shows `stage1_results['best_t0']` should be used instead.

**Missing `detector_type` in `setup_event_simulator` call.**

**Recommendation:** Requires updating utility functions first (`energy_scan_optimization`, `hierarchical_direction_search_cone`). Use `default_detector_params=True`. Fix TRUE_T0 bug.

---

## Finalized Decisions

- **Bake in detector params:** All simulators use `default_detector_params=True` with `physics_config`.
  Simulator call signature: `(particle_params, key)` for track, `(particle_params, key, photon_data)` for data,
  `(source, key)` for calibration. `detector_params` is removed from all downstream function signatures.
- **Delete `get_detector_params_from_config`:** Dead code with baked-in params.
- **Full optimizer redesign:** Flat `jnp.array([x,y,z,t0,theta,phi,energy])` → `ParticleParams` pytree.
  Single `optax.adam` + manual per-field scaling via `ParticleParams(...)` constructor (preserves exact
  current optimization dynamics).
- **Defer photon data utilities:** `pad_photon_data` / `set_photon_transform` creation deferred to cleanup.
  Inline boilerplate preserved as-is in this iteration.
- **No logic changes:** All transformations are mechanical type substitutions. Dead code, dead params,
  bugs, inconsistencies are preserved exactly as they are. See `notebook_cleanup_suggestions.md`.
- **File format compatibility:** `save_single_event`/`load_single_event` keep old HDF5 structure
  (Cartesian direction, old field names). No backward compatibility concern.

---

## Migration Rules (Mechanical Translation)

### Rule 1: Old 3-tuple track params → ParticleParams
```python
# OLD (prediction/SIREN mode — spherical angles):
track_params = (energy, position, jnp.array([theta, phi]))

# NEW:
track = ParticleParams(energy=energy, position=position, theta=theta, phi=phi, t0=t0)
```

```python
# OLD (data mode — Cartesian direction):
true_params = (true_energy, true_position, true_direction)

# NEW:
true_track = ParticleParams.from_cartesian(energy=true_energy, position=true_position,
                                            direction=true_direction, t0=TRUE_T0)
```

### Rule 2: Old 4-tuple detector params → baked in
```python
# OLD:
detector_params = (jnp.array(50.), jnp.array(0.2), jnp.array(50.), jnp.array(0.065))
simulator = setup_event_simulator(json_file, Nphot, temperature, K=K, is_data=False)
result = simulator(track_params, detector_params, key)

# NEW:
simulator = setup_event_simulator(json_file, Nphot, temperature, K=K, is_data=False,
                                   physics_config=PHYSICS_CONFIG,
                                   default_detector_params=True)
result = simulator(track, key)
```

### Rule 3: Old 2-tuple source → callable NamedTuple
```python
# OLD:
source_params = (jnp.array([0., 0., 0.]), jnp.array(1.0))
result = simulator(source_params, detector_params, key)

# NEW:
source = isotropic_source(position=[0., 0., 0.], intensity=1.0)
result = simulator(source, key)
```

### Rule 4: Flat array unpacking → named field access
```python
# OLD:
position = params[:3]
t0 = params[3]
theta = params[4]
phi = params[5]
energy = params[6]

# NEW:
position = track.position
t0 = track.t0
theta = track.theta
phi = track.phi
energy = track.energy
```

### Rule 5: Flat array optimizer → ParticleParams pytree optimizer
```python
# OLD:
initial_params = jnp.array([x, y, z, t0, theta, phi, energy])
optimizer = optax.adam(lr)
opt_state = optimizer.init(initial_params)
# ...
updates, opt_state = optimizer.update(grad, opt_state, current_params)
scaled_updates = updates * update_scales * damping
if iteration < 25:
    scaled_updates = scaled_updates.at[-1].set(0.)
current_params = optax.apply_updates(current_params, scaled_updates)

# NEW:
track = ParticleParams(energy=jnp.asarray(energy), position=jnp.array([x, y, z]),
                        theta=jnp.asarray(theta), phi=jnp.asarray(phi), t0=jnp.asarray(t0))
optimizer = optax.adam(lr)
opt_state = optimizer.init(track)
# ...
updates, opt_state = optimizer.update(grads, opt_state, track)
scaled_updates = ParticleParams(
    energy=updates.energy * energy_scale * damping,
    position=updates.position * position_scale * damping,
    theta=updates.theta * direction_scale * damping,
    phi=updates.phi * direction_scale * damping,
    t0=updates.t0 * t0_scale * damping,
)
if iteration < 25:
    scaled_updates = scaled_updates._replace(energy=jnp.zeros_like(scaled_updates.energy))
track = optax.apply_updates(track, scaled_updates)
```

### Rule 6: Flat array constraints → _replace with clip
```python
# OLD:
current_params = jnp.array([
    jnp.clip(current_params[0], -R*0.95, R*0.95),
    jnp.clip(current_params[1], -R*0.95, R*0.95),
    jnp.clip(current_params[2], -H/2*0.95, H/2*0.95),
    jnp.clip(current_params[3], -20.0, 20.0),
    current_params[4], current_params[5],
    jnp.clip(current_params[6], 300.0, 2000.0)
])

# NEW:
track = track._replace(
    position=jnp.array([
        jnp.clip(track.position[0], -R*0.95, R*0.95),
        jnp.clip(track.position[1], -R*0.95, R*0.95),
        jnp.clip(track.position[2], -H/2*0.95, H/2*0.95),
    ]),
    t0=jnp.clip(track.t0, -20.0, 20.0),
    energy=jnp.clip(track.energy, 300.0, 2000.0),
)
```

### Rule 7: Gradient NaN handling and norm for pytrees
```python
# OLD:
if jnp.any(jnp.isnan(grad)):
    grad = jnp.nan_to_num(grad, nan=0.0)
grad_norm = jnp.linalg.norm(grad)

# NEW:
has_nan = any(jnp.any(jnp.isnan(g)) for g in jax.tree.leaves(grads))
if has_nan:
    grads = jax.tree.map(lambda g: jnp.nan_to_num(g, nan=0.0), grads)
grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
```

### Rule 8: Parameter extraction for error reporting
```python
# OLD:
current_position = current_params[:3]
current_direction = spherical_to_cartesian(current_params[4], current_params[5])

# NEW:
current_position = track.position
current_direction = spherical_to_cartesian(track.theta, track.phi)
```

### Rule 9: Remove detector_params from function signatures
When baking in, `detector_params` is removed from all functions that only used it to pass
to the simulator. The function signatures shrink. All callers must stop passing it.

### Rule 10: Keep all dead code, dead params, bugs, and inconsistencies
- `true_data` param in loss functions: keep (dead but preserved)
- `vertex_weight_scale`, `counts_weight_scale`, `c_medium` in factory: keep (dead but preserved)
- `cone_time_loss_parametric`: keep (dead but preserved)
- Return value bug in `losses.py`: keep (documented in cleanup #1)
- `stop_gradient` inconsistency: keep each version as-is (documented in cleanup #3)
- Dead `direction = spherical_to_cartesian(...)` in notebooks: keep (documented in cleanup #4)

---

## Supporting Modules: Exact Changes

### 1. `tools/optimization/losses.py`

**`create_combined_loss_function`** (line 232):

| Change | Old | New |
|--------|-----|-----|
| Import | — | `from tools.detector_params import ParticleParams` |
| Factory arg | `(prediction_simulator, detector_params)` | `(prediction_simulator)` |
| Inner fn first arg | `params` (flat array) | `track` (ParticleParams) |
| Unpack position | `params[:3]` | `track.position` |
| Unpack t0 | `params[3]` | `track.t0` |
| Unpack theta | `params[4]` | `track.theta` |
| Unpack phi | `params[5]` | `track.phi` |
| Unpack energy | `params[6]` | `track.energy` |
| Simulator call | `prediction_simulator(track_params, detector_params, key)` | `prediction_simulator(track, key)` |
| Old tuple construction | `track_params = (energy, position, jnp.array([theta, phi]))` | removed |

**Preserved exactly:** `stop_gradient` on position, `true_data` in signature, return value bug,
docstring error, all loss computation logic.

### 2. `tools/optimization/single_track_optimization.py`

**`create_combined_loss_function`** (line 121):

Same pattern as losses.py version, except:
- Factory has extra dead params `(vertex_weight_scale, counts_weight_scale, ..., c_medium, nrays)` — keep all
- `cone_time_loss_parametric` dead function inside — keep
- No `stop_gradient` on position — preserve (no stop_gradient)
- Correct return value — preserve

| Change | Old | New |
|--------|-----|-----|
| Factory args | `(..., prediction_simulator, detector_params, c_medium, nrays)` | `(..., prediction_simulator, c_medium, nrays)` |
| Inner fn / simulator call | same pattern as losses.py | same pattern as losses.py |

**`run_complete_optimization_adam`** (line 190):

| Change | Old | New |
|--------|-----|-----|
| Arg | `detector_params` | removed |
| Arg | `combined_grad_fn` | `combined_grad_fn` (same, but takes ParticleParams now) |
| Initial params | `jnp.array([x, y, z, t0, theta, phi, energy])` | `ParticleParams(energy=..., position=..., theta=..., phi=..., t0=...)` |
| Optimizer init | `optimizer.init(initial_params)` | `optimizer.init(track)` |
| Grad call | `combined_grad_fn(current_params, ...)` | `combined_grad_fn(track, ...)` |
| NaN check | `jnp.any(jnp.isnan(grad))` | pytree version (Rule 7) |
| Grad norm | `jnp.linalg.norm(grad)` | pytree version (Rule 7) |
| Update scaling | `updates * update_scales * damping` | `ParticleParams(energy=... * scale * damping, ...)` (Rule 5) |
| Energy freeze | `.at[-1].set(0.)` | `._replace(energy=jnp.zeros_like(...))` |
| Apply updates | `optax.apply_updates(current_params, ...)` | `optax.apply_updates(track, ...)` |
| Constraints | Rebuild flat array with clips | `._replace(...)` with clips (Rule 6) |
| Param extraction | `current_params[:3]`, `current_params[3]`, etc. | `track.position`, `track.t0`, etc. |
| History | `current_params.copy()` | `track` (immutable NamedTuple) |
| Return values | `current_params[:3]`, `current_params[4]`, etc. | `track.position`, `track.theta`, etc. |

**Preserved exactly:** All separate true_* args, all loss logic, damping factor, iteration count,
verbosity, history structure, return dict structure.

**`generate_event_data`** (line 536):

| Change | Old | New |
|--------|-----|-----|
| Arg | `detector_params` | removed |
| True params | `true_params = (true_energy, true_position, true_direction)` | `true_track = ParticleParams.from_cartesian(energy=true_energy, position=true_position, direction=true_direction, t0=TRUE_T0)` |
| Simulator call | `data_simulator(true_params, detector_params, key, photon_data)` | `data_simulator(true_track, key, photon_data)` |

**Preserved exactly:** All photon data boilerplate (padding, rotation, translation), random
vertex/direction generation, all return dict fields.

**`main()`** (line 677):

| Change | Old | New |
|--------|-----|-----|
| Simulator setup | No `physics_config`, no `default_detector_params` | Add both |
| `detector_params` | `get_detector_params_from_config(config)` | removed entirely |
| `create_combined_loss_function` call | passes `detector_params` | does not pass it |
| `generate_event_data` call | passes `detector_params` | does not pass it |
| `run_complete_optimization_adam` call | passes `detector_params` | does not pass it |

### 3. `tools/optimization/utils/functions.py`

**`hierarchical_direction_search_cone`** (line 112):

| Change | Old | New |
|--------|-----|-----|
| Import | — | `from tools.detector_params import ParticleParams` |
| Arg | `detector_params` | removed |
| Old tuple | `track_params = (energy_guess, position, jnp.array([theta, phi]))` | `track = ParticleParams(energy=..., position=position, theta=..., phi=..., t0=...)` |
| Simulator call | `prediction_simulator(track_params, detector_params, search_key)` | `prediction_simulator(track, search_key)` |

**`energy_scan_optimization`** (line 414):

| Change | Old | New |
|--------|-----|-----|
| Arg | `detector_params` | removed |
| Old tuple | `current_track_params = (energy, position, jnp.array([theta, phi]))` | `track = ParticleParams(energy=..., position=position, theta=..., phi=..., t0=...)` |
| Simulator call | `prediction_simulator(current_track_params, detector_params, scan_key)` | `prediction_simulator(track, scan_key)` |

**Preserved exactly:** Dead `test_params` flat array construction, all search logic,
all return dict structure.

### 4. `tools/optimization/optimize.py`

| Change | Old | New |
|--------|-----|-----|
| `get_detector_params_from_config` | exists (line 62) | **deleted** |

No other changes needed — this file's other functions (`hierarchical_position_grid_search`,
`evaluate_loss_batch`, etc.) operate on raw position arrays and `origin_time_loss`, not on
the simulator. They don't need ParticleParams or DetectorParams.

Note: `optimize.py` does NOT contain `create_combined_loss_function` (earlier analysis was wrong).

### 5. `tools/utils.py`

These functions are used by notebooks (migrated later) but should be updated in Phase 1
for module consistency:

**`generate_random_params`** (line 491):
```python
# OLD: return energy, position, direction_angles
# NEW: return ParticleParams(energy=energy, position=position,
#                             theta=direction_angles[0], phi=direction_angles[1],
#                             t0=jnp.array(0.0))
```

**`generate_random_event_params`** (line 1888):
```python
# OLD: return position, direction, energy
# NEW: return ParticleParams.from_cartesian(energy=energy, position=position,
#                                            direction=direction, t0=jnp.array(0.0))
```

**`print_particle_params`** (line 566):
```python
# OLD: energy, position, direction_angles = trk_params
# NEW: use trk_params.energy, trk_params.position, trk_params.theta, trk_params.phi
```

**`print_propagation_params`** (line 585):
- Currently unpacks old 4-tuple including `sim_temperature`
- `DetectorParams` has no `sim_temperature` field (temperature is now in physics_config)
- Mechanical translation: access named fields, drop temperature printing

**`save_single_event`** (line 325):
```python
# OLD: particle_params[0], particle_params[1], particle_params[2]
# NEW: particle_params.energy, particle_params.position, np.array(particle_params.direction)
# (direction property gives Cartesian — matches old HDF5 format)
```

**`load_single_event`** (line 406):
```python
# OLD: particle_params = (track_energy, track_origin, track_direction)
# NEW: particle_params = ParticleParams.from_cartesian(
#          energy=track_energy, position=track_origin,
#          direction=track_direction, t0=jnp.array(0.0))
```

For `sensor_params`, the old HDF5 has `scatter_length, reflection_rate, absorption_length,
sim_temperature` — 4 fields. With baked-in params, detector params are rarely loaded from
event files. Keep old HDF5 read logic but return what's available (not a full DetectorParams
since the file doesn't have all 6 fields).

### 6. `tools/losses.py` — No changes needed
Calibration losses operate on raw sensor arrays. No simulator calls, no ParticleParams.

### 7. `tools/optimization/losses.py` — Primitive losses: No changes needed
`origin_time_loss`, `cone_time_loss`, `counts_loss`, `energy_loss`, `grid_origin_time_loss`
all operate on raw arrays. Only the `create_combined_loss_function` factory changes.

---

## Bugs and Inconsistencies (preserved, not fixed)

See `notebook_cleanup_suggestions.md` for the full list. Key items:
1. Return value bug in `losses.py` `create_combined_loss_function` (cleanup #1)
2. `stop_gradient` inconsistency between `losses.py` and `single_track_optimization.py` (cleanup #3)
3. Dead `true_data` parameter (cleanup #16)
4. Dead factory params in `single_track_optimization.py` (cleanup #17)
5. Dead `cone_time_loss_parametric` (cleanup #18)

---

## Migration Order

### Phase 1: Supporting modules

1. **`tools/optimization/optimize.py`** — delete `get_detector_params_from_config`
2. **`tools/optimization/losses.py`** — update `create_combined_loss_function`
3. **`tools/optimization/utils/functions.py`** — update `energy_scan_optimization`, `hierarchical_direction_search_cone`
4. **`tools/optimization/single_track_optimization.py`** — update `create_combined_loss_function`, `run_complete_optimization_adam`, `generate_event_data`, `main()`
5. **`tools/utils.py`** — update `generate_random_params`, `generate_random_event_params`, `print_particle_params`, `print_propagation_params`, `save_single_event`, `load_single_event`

### Phase 2: Verify already-migrated notebooks

6. `parameter_scans_1D_v2.ipynb` — verify clean
7. `detector_grad_qe_convergence_multi_source.ipynb` — verify clean
8. `laser_source_grad_analysis.ipynb` — verify clean
9. `grad_param_calibration_multi_init_no_qe.ipynb` — verify clean

### Phase 3: Medium-complexity notebooks

10. `computational_performance_evaluation.ipynb`
11. `data_vs_pred_hit_predictions.ipynb`
12. `cylinder_2D_displays.ipynb`
13. `geometry_and_events_3D_visualization.ipynb`

### Phase 4: High-complexity optimization notebooks

14. `parameter_scans_1D.ipynb`
15. `grad_loss_and_opt_in_2D.ipynb`
16. `visualize_3D_track_optimization.ipynb`
17. `tracking_opt_with_gif.ipynb`
18. `tracking_opt_development.ipynb`

### Phase 5: Verification

19. `train_siren.ipynb` — verify still works
20. `optimization_vs_variables.ipynb` — verify still works
21. `track_optimization_visualization.ipynb` — verify still works (only reads pickles)
