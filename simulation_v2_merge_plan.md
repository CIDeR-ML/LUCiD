# LUCiD simulation.py Merge Plan — v2 Feature Absorption

## Context

LUCiD's `tools/simulation.py` is the primary differentiable photon simulation engine.
diffCherenkov's `tools/simulation_v2.py` introduced significant architectural and physics
improvements that need to be merged back into LUCiD. After the merge, the diffCherenkov
`_v2` files will be retired. LUCiD is the canonical codebase going forward.

---

## Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| `LUCiD/tools/detector_params.py` | **CREATE** | Port from diffCherenkov — DetectorParams, ParticleParams, IsotropicSource, LaserSource, all helpers |
| `LUCiD/tools/simulation.py` | **REWRITE** | Merge v2 features into LUCiD's simulation engine |
| `LUCiD/tools/utils.py` | **MODIFY** | Add `smear_times` and `smear_charges_SK_like` (moved from simulation.py) |
| `LUCiD/tools/generate.py` | **MODIFY** | Fix import: `smear_times`/`smear_charges_SK_like` now from `utils` not `simulation` |
| `LUCiD/config/SK_physics_config.json` | **CREATE** | Default DetectorParams values for load_detector_params() |

---

## All Decisions Made

### Architecture
- **Primary codebase**: LUCiD absorbs v2 features
- **Target file**: `LUCiD/tools/simulation.py`
- **Backward compatibility**: None. Clean break — old tuple-based API removed entirely
- **Post-merge**: Retire diffCherenkov `simulation_v2.py` and all `_v2` notebooks

### Parameter Interface
- **DetectorParams**: NamedTuple with 6 fields (scatter_length, wall_reflection_rate, sensor_reflection_rate, absorption_length, qe, qe_corrections)
- **ParticleParams**: NamedTuple with 4 fields (energy, position, theta, phi) + `.direction` computed property
- **ParticleParams rationale**: LUCiD non-data notebooks consistently use `(energy, position, [theta, phi])`. SIREN mode needs theta/phi natively. Data mode can compute angles from direction when needed.
- **Port all of detector_params.py**: Including IsotropicSource, LaserSource, normalize/denormalize, save/load, default_bounds, make_optimization_mask, default_gradient_scales — but keep flexible, no commitment to optax.masked pattern

### Physics Model
- **Reflection**: Dual model — wall (diffuse/cosine-hemisphere) + sensor (specular). Uses `inside_sensor` from propagation results and `hit_sensor = jnp.max(inside_sensor, axis=0)`
- **tau_gs**: Remove entirely (dead parameter, unused in both codebases)
- **Speed of light**: LUCiD's configurable `get_speed_of_light_in_material(material)` from config — NOT v2's hardcoded `SOL = 0.299792 / 1.33`
- **Material config**: Keep `get_material_from_config(json_filename)` and `unpack_photonsim_params(particle, material)` with material parameter
- **Per-sensor QE**: Yes — `qe_corrections` array in DetectorParams, applied as `qe * qe_corrections[flat_indices]`
- **Calibration sources**: Callable duck-typed objects (IsotropicSource, LaserSource) instead of fixed (origin, intensity) tuples
- **K default**: 7 (from v2)
- **apply_smearing default**: True (from LUCiD)

### Gradient & Memory
- **NaN handling**: Custom VJP only (v2 approach). Remove verbose ray-level NaN checking from _common_propagation
- **jax.remat**: Apply to both photon_update_fn and propagation_step in scan loop
- **n_grad_iters**: Yes, port from v2. Default 0 for reconstruction, 2 for calibration. Controls selective stop_gradient on position/direction
- **default_detector_params**: Yes — setup_event_simulator can bake DetectorParams into closure via physics_config parameter

### Code Quality
- **normalize()**: Use v2's `v / jnp.maximum(norm, epsilon)` instead of LUCiD's `v / (norm + epsilon)`
- **Dead code removal**: Remove `gumbel_softmax()` (unused in both codebases)
- **Smearing functions**: Move `smear_times` and `smear_charges_SK_like` from simulation.py to utils.py
- **SIREN grid function**: Already correctly in `siren/core.py` (verified via git history). No move needed
- **Module docstring**: Comprehensive (v2 style), describing all features
- **qe_corrections validation**: Add shape check in setup_event_simulator — `assert len(qe_corrections) == NUM_SENSORS`

### Sampling & Generation
- **generate.py**: No changes now (except fixing the smear import). LUCiD's stratified sampling is superior to diffCherenkov's Gaussian noise
- **LUCiD generate.py is a superset**: Has multi-particle events, label processing, VMAP optimization — none of this exists in diffCherenkov

---

## Edge Cases and Critical Findings

### 1. QE vs tau_gs Inconsistency (Existing Bug — Harmless)

**Finding**: LUCiD's `optimize.py:get_detector_params_from_config()` puts QE as the 4th tuple element, but `simulation.py:_common_propagation()` unpacks it as `tau_gs`. This means QE values (0.065) are being passed where tau_gs is expected.

**Impact**: None — tau_gs is dead code (unused in photon iteration functions despite being in the signature). The merge fixes this properly by removing tau_gs and giving QE its own named field in DetectorParams.

### 2. qe_corrections Shape Mismatch Risk

**Problem**: `qe_corrections` must have shape `(NUM_SENSORS,)` but NUM_SENSORS varies per detector config JSON. If a physics_config JSON is loaded with mismatched qe_corrections, indexing will silently produce wrong results or crash.

**Solution**: Add validation in `setup_event_simulator()`:
```python
if len(detector_params.qe_corrections) != NUM_SENSORS:
    raise ValueError(f"qe_corrections has {len(detector_params.qe_corrections)} elements "
                     f"but detector has {NUM_SENSORS} sensors")
```

**Also at risk**: `load_detector_params()` loading stale `.npy` files after detector geometry changes.

### 3. ParticleParams .direction Property and JAX

**Finding**: NamedTuple `@property` works with JAX. `jax.tree.leaves()` returns only the 4 stored fields (energy, position, theta, phi). The `.direction` property is NOT a pytree leaf — it's computed on access via `spherical_to_cartesian(theta, phi)`.

**Gradient behavior**: Gradients flow through theta/phi (the stored fields), not through the derived direction. This is correct for optimization — theta and phi are the parameters being optimized.

**Data mode consideration**: Data mode receives Cartesian direction from ROOT files. The simulation code should convert to theta/phi when constructing ParticleParams for data mode, or handle direction vectors directly in the data-mode path (as both LUCiD and v2 currently do).

### 4. inside_sensor Availability Across All Geometries

**Verified**: All three propagation modules (cylinder.py, sphere.py, box.py) return `inside_sensor` in their result dict with identical structure. The merge can safely access `prop_results['inside_sensor']` regardless of detector_type.

### 5. Propagation Result Dict Structure (All 3 Geometries)

All return:
```python
{
    'times': sensor_times,              # Ray parameter in meters
    'sensor_weights': weights,          # Detection weights
    'sensor_indices': sensor_indices,   # Sensor indices
    'per_sensor_positions': ...,        # Hit positions on sensors
    'positions': hit_positions,         # Intersection points
    'normals': final_normals,           # Surface normals
    'sensor_normals': sensor_normals,   # Sensor normals
    'inside_sensor': inside_sensor      # Bool: hit sensor vs wall
}
```

### 6. make_hits Functions Are Internal Only

**Verified**: `make_hits_simulation`, `make_hits_simulation_min`, and `make_hits_data` are only called within simulation.py (via wrapper closures). No external callers. Safe to modify signatures freely.

### 7. photon_iteration Functions Are Internal Only

**Verified**: `photon_iteration_sample` and `photon_iteration_update_factors` are only used within simulation.py. No external imports. Safe to change signatures (add hit_sensor, remove tau_gs, etc.).

### 8. SIREN Grid Function — Already in Correct Location

**Verified via git history**: `create_photonsim_siren_grid` was moved from `simulation.py` to `siren/core.py` in commit `89ac95d`, then removed from `simulation.py` in commit `265b587`. The code is byte-for-byte identical (only a docstring was added). Exported via `siren/__init__.py`. Imported by simulation.py via `from tools.siren.core import *`.

### 9. generate.py Import of Smearing Functions

**Finding**: `LUCiD/tools/generate.py` imports `smear_charges_SK_like` and `smear_times` from `tools.simulation`. When these move to `tools.utils`, this import will break.

**Fix**: Update the import line in generate.py (included in merge scope).

### 10. Production Scripts Use Relative Imports

**Finding**: `tools/production/generate_events.py` and `generate_events_with_labels.py` use `from simulation import setup_event_simulator` (no `tools.` prefix). They run from within `tools/` directory. The merged simulation.py keeps the same function name, so the import path still works — but the function signature changes (DetectorParams instead of tuples), so these scripts will break at runtime.

**Deferred**: Updated alongside notebooks in a follow-up pass.

---

## Complete List of All Callers That Will Break

### Python Modules (break at runtime, deferred updates)
| File | What It Does | How It Uses detector_params |
|------|---|---|
| `tools/optimization/single_track_optimization.py` | Main optimization script | Gets 4-tuple from `get_detector_params_from_config()`, passes to simulator |
| `tools/optimization/optimize.py` | `get_detector_params_from_config()` returns 4-tuple | Constructs `(scatter_length, reflection_rate, absorption_length, qe)` |
| `tools/optimization/losses.py` | `create_combined_loss_function()` | Captures detector_params in closure, forwards to simulator |
| `tools/optimization/utils/functions.py` | `hierarchical_direction_search_cone()`, `energy_scan_optimization()` | Passes detector_params through to simulator |
| `tools/production/generate_events.py` | Batch event generation | Hardcodes 4-tuple sensor_params |
| `tools/production/generate_events_with_labels.py` | Label-based event generation | Hardcodes 4-tuple sensor_params |
| `tools/generate.py` | `generate_events_from_root()`, `generate_events_from_photonsim()` | Receives sensor_params from callers, forwards to simulator |
| `s3df_jobs/temperature_opt_config/create_temp_configurations.py` | Job config creation | Programmatically creates detector_params dict |

### Notebooks (break at runtime, deferred updates)
All 11 good_notebooks:
1. `geometry_and_events_3D_visualization.ipynb`
2. `cylinder_2D_displays.ipynb`
3. `parameter_scans_1D.ipynb`
4. `computational_performance_evaluation.ipynb`
5. `data_vs_pred_hit_predictions.ipynb`
6. `tracking_opt_development.ipynb`
7. `train_siren.ipynb`
8. `optimization_vs_variables.ipynb`
9. `grad_loss_and_opt_in_2D.ipynb`
10. `track_optimization_visualization.ipynb`
11. `visualize_3D_track_optimization.ipynb`

### How Each Notebook Currently Constructs Parameters

**detector_params (all notebooks, 4-tuple):**
```python
detector_params = (
    jnp.array(50.),     # scatter_length
    jnp.array(0.2),     # reflection_rate (single rate)
    jnp.array(50.),     # absorption_length
    jnp.array(0.065)    # tau_gs / qe (inconsistent across notebooks)
)
```

**Will become:**
```python
from tools.detector_params import DetectorParams
detector_params = DetectorParams(
    scatter_length=jnp.array(50.),
    wall_reflection_rate=jnp.array(0.15),
    sensor_reflection_rate=jnp.array(0.1),
    absorption_length=jnp.array(50.),
    qe=jnp.array(0.065),
    qe_corrections=jnp.ones(NUM_SENSORS),
)
```

**particle_params — non-data mode (most notebooks):**
```python
track_params = (
    jnp.array(1050.0),                              # energy
    jnp.array([0.0, 0.0, 0.0]),                     # position
    jnp.array([jnp.pi/2, jnp.pi/6])                 # [theta, phi]
)
```

**Will become:**
```python
from tools.detector_params import ParticleParams
particle_params = ParticleParams(
    energy=jnp.array(1050.0),
    position=jnp.array([0.0, 0.0, 0.0]),
    theta=jnp.array(jnp.pi/2),
    phi=jnp.array(jnp.pi/6),
)
```

**particle_params — data mode (cylinder_2D_displays, data_vs_pred):**
```python
true_params = (true_energy, true_position, true_direction)  # Cartesian direction
```

**Will become:**
```python
# Convert Cartesian direction to spherical
theta = jnp.arccos(jnp.clip(true_direction[2], -1.0, 1.0))
phi = jnp.arctan2(true_direction[1], true_direction[0])
particle_params = ParticleParams(
    energy=true_energy,
    position=true_position,
    theta=theta,
    phi=phi,
)
```

**calibration source_params (geometry_and_events_3D_visualization):**
```python
source_params = (
    jnp.array([0.0, 0.0, 0.0]),  # position
    jnp.array(1.0)                 # intensity
)
```

**Will become:**
```python
from tools.detector_params import isotropic_source
source = isotropic_source(position=[0.0, 0.0, 0.0], intensity=1.0)
```

---

## What Stays From LUCiD (Do Not Change)

- `get_speed_of_light_in_material(material)` — configurable, not hardcoded
- `get_material_from_config(json_filename)` — reads material from detector JSON
- `unpack_photonsim_params(particle, material)` — keeps material parameter
- `unpack_t0_params(particle, material)` — keeps material parameter
- `create_photonsim_siren_grid` in `siren/core.py` — already in correct location
- All of generate.py (stratified sampling, label processing, VMAP optimization)
- `spherical_to_cartesian` in utils.py — already exists
- `base_dir_path` in utils.py — already exists
- `inside_detector` bounds checking in _common_propagation

## What Comes From v2 (Port to LUCiD)

- DetectorParams / ParticleParams / IsotropicSource / LaserSource pytree classes
- All detector_params.py helpers (normalize, denormalize, bounds, mask, scales, save, load)
- Wall (diffuse) vs sensor (specular) reflection distinction
- `sample_cosine_hemisphere(normal, rng_key)` for diffuse wall reflection
- Custom VJP wrapper `photon_iteration_update_factors_safe` with NaN gradient sanitization
- `jax.remat` on photon_update_fn and propagation_step
- `n_grad_iters` parameter for selective gradient flow control
- Per-sensor QE corrections in make_hits functions
- `default_detector_params` / `physics_config` parameters in setup_event_simulator
- Cleaner _common_propagation (no verbose NaN ray checking, no commented debug prints)
- `normalize()` with `jnp.maximum` denominator

## What Gets Removed

- `tau_gs` parameter — everywhere (function signatures, detector_params, vmap in_axes)
- `gumbel_softmax()` function — unused dead code
- `smear_times()` and `smear_charges_SK_like()` — moved to utils.py, removed from simulation.py
- Verbose NaN ray checking in _common_propagation (problematic_rays mask, safe replacements)
- All commented-out debug prints (`jax.debug.print`)
- `make_hits_simulation_min` (if redundant with make_hits_simulation — both exist in LUCiD, v2 has both too)

---

## Detailed Implementation Plan

### Step 1: Add smearing functions to utils.py

Move `smear_times` and `smear_charges_SK_like` from `simulation.py` to `utils.py`.
Keep the detailed docstrings with arXiv references from LUCiD's versions.

### Step 2: Fix generate.py import

Change:
```python
from tools.simulation import smear_charges_SK_like, smear_times
```
To:
```python
from tools.utils import smear_charges_SK_like, smear_times
```

### Step 3: Create detector_params.py

Port from `/home/oalterka/desktop_linux/diffWC/diffCherenkov/tools/detector_params.py` with these modifications:
- Keep ParticleParams as (energy, position, theta, phi) with .direction property
- Keep all helpers: normalize_params, denormalize_params, default_bounds, make_optimization_mask, default_gradient_scales
- Keep IsotropicSource and LaserSource callables
- Keep save/load functions
- Ensure imports reference LUCiD's utils (spherical_to_cartesian, etc.)

### Step 4: Create SK_physics_config.json

Create default physics config with reasonable SK-like values:
```json
{
    "scatter_length": 50.0,
    "wall_reflection_rate": 0.2,
    "sensor_reflection_rate": 0.1,
    "absorption_length": 50.0,
    "qe": 0.065,
    "qe_corrections": null
}
```
(null qe_corrections = all-ones, initialized at load time based on detector size)

### Step 5: Rewrite simulation.py

This is the core of the merge. The new simulation.py combines:

**Imports**:
- Add: `from tools.detector_params import DetectorParams, ParticleParams, load_detector_params`
- Add: `from tools.utils import smear_times, smear_charges_SK_like`
- Keep: `from tools.geometry import generate_detector, get_material_from_config`
- Keep: `from tools.utils import unpack_t0_params, unpack_photonsim_params, get_speed_of_light_in_material, spherical_to_cartesian, base_dir_path`
- Keep: `from tools.siren.core import *` (gets create_photonsim_siren_grid)
- Keep: `from tools.generate import get_isotropic_rays, photonsim_differentiable_get_rays, predict_t0, generate_laser_photons, setup_calibration_generator`

**Helper functions** (from v2):
- `normalize()` — use `jnp.maximum` denominator
- `compute_reflection_direction()` — same as both
- `sample_cosine_hemisphere()` — NEW from v2
- `create_local_frame()` — same as both
- `sample_scatter_distance()` — same as both
- `solve_rayleigh_inverse_cdf()` — same as both
- `compute_scatter_direction()` — same as both
- `jax_normalize()` and `jax_rotate_vector()` — for data mode rotation

**Photon iteration functions** (new signatures):
```python
def photon_iteration_sample(
    position, direction, time, surface_distance,
    normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
    absorption_length,
    hit_sensor, rng_key, speed_of_light):
```

```python
def photon_iteration_update_factors(
    position, direction, time, surface_distance,
    normal, scatter_length, wall_reflection_rate, sensor_reflection_rate,
    absorption_length,
    hit_sensor, rng_key, speed_of_light):
```

**Custom VJP wrapper**:
```python
@jax.custom_vjp
def photon_iteration_update_factors_safe(...):
    return photon_iteration_update_factors(...)
# _fwd and _bwd with nan_to_num on all 6 gradient components
```

**make_hits functions** (with per-sensor QE):
- `make_hits_simulation()` — add `qe_corrections` parameter
- `make_hits_data()` — keep existing (no per-sensor QE in data mode)

**setup_event_simulator** (merged signature):
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
    default_detector_params=False):
```

**_common_propagation** (merged):
- Add `n_grad_iters` parameter
- Use DetectorParams named field access (not tuple unpacking)
- Extract `inside_sensor` from prop_results, compute `hit_sensor`
- vmap with 12 args: position, direction, time, surface_distance, normal, scatter_length, wall_reflection_rate, sensor_reflection_rate, absorption_length, hit_sensor, rng_keys, SPEED_OF_LIGHT_MATERIAL
- `in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0, 0, None)`
- Selective stop_gradient via `n_grad_iters`
- `jax.remat` on propagation_step
- Only inside_detector bounds check (no verbose NaN ray masking)

**Speed of light**: Use LUCiD's configurable approach:
```python
material = get_material_from_config(json_filename)
SPEED_OF_LIGHT_MATERIAL = get_speed_of_light_in_material(material)
```

**Photon update function selection**:
```python
if is_data:
    photon_update_fn = photon_iteration_sample
elif use_expected_value is False:
    photon_update_fn = photon_iteration_sample
else:
    photon_update_fn = jax.remat(photon_iteration_update_factors_safe)
```

**Mode-specific simulation functions**:
- `_simulation_with_data_impl()` — uses ParticleParams fields
- `_simulation_without_data_impl()` — uses ParticleParams fields, SIREN path
- `_simulation_sensor_calibration_impl()` — callable source interface

**Return logic with default_detector_params baking** (from v2).

---

## Verification Plan

### After Implementation
1. **Import test**: `python -c "from tools.simulation import setup_event_simulator; from tools.detector_params import DetectorParams, ParticleParams"`
2. **Instantiation test**: Create DetectorParams and ParticleParams, verify fields
3. **JAX pytree test**: `jax.tree.leaves(detector_params)` returns 6 items
4. **Shape validation test**: Verify setup_event_simulator rejects mismatched qe_corrections

### After Notebook Updates (future)
1. Run each notebook cell-by-cell
2. Compare simulation outputs (charges, times) with pre-merge results for same parameters
3. Verify gradient computation works with new DetectorParams
4. Test calibration mode with IsotropicSource and LaserSource

---

## Vmap in_axes Reference

### LUCiD current (11 args):
```python
in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0, None)
# pos, dir, time, dist, normal, scatter, refl, abs, tau_gs, rng, SOL
```

### Merged version (12 args):
```python
in_axes=(0, 0, 0, 0, 0, None, None, None, None, 0, 0, None)
# pos, dir, time, dist, normal, scatter, wall_refl, sensor_refl, abs, hit_sensor, rng, SOL
```

Key changes: tau_gs removed, hit_sensor added (per-ray, axis=0), wall/sensor reflection split.

---

## File Dependency Graph (Post-Merge)

```
detector_params.py (NEW)
    └── imports from: tools.utils (spherical_to_cartesian)
    └── imports from: tools.generate (get_isotropic_rays, generate_laser_photons)

simulation.py (REWRITTEN)
    └── imports from: tools.detector_params (DetectorParams, ParticleParams, load_detector_params)
    └── imports from: tools.utils (smear_times, smear_charges_SK_like, unpack_t0_params, ...)
    └── imports from: tools.geometry (generate_detector, get_material_from_config)
    └── imports from: tools.generate (get_isotropic_rays, photonsim_differentiable_get_rays, ...)
    └── imports from: tools.siren.core (create_photonsim_siren_grid via *)
    └── imports from: tools.propagation.{cylinder,sphere,box}

utils.py (MODIFIED)
    └── adds: smear_times, smear_charges_SK_like

generate.py (IMPORT FIX)
    └── changes: smear import from tools.utils instead of tools.simulation
```
