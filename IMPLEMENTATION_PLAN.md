# LUCiD Refactor — Implementation Plan

**Prerequisite:** Read `CODEBASE_ANALYSIS.md` for full context.

**Core principle:** Bottom-up. Each phase has a different risk profile and
requires different verification. No phase starts until the previous is verified.

---

## Notes for Implementers

### "Exact match" means bit-exact

JAX is deterministic for a given PRNGKey, device, and computation graph.
If a refactored function produces different output for the same seed on the
same device, the computation graph changed — investigate before proceeding.
All verification checks mean **bit-exact match**, not "within epsilon."

### Phase 2 sub-phases are sequential

2.1 → 2.2 → 2.3 → 2.4 → 2.5. They cannot run in parallel because later
sub-phases import from modules created in earlier ones (e.g., 2.2 imports
from `lucid.simulation` created in 2.1; 2.5 moves functions into
`sources/event_io.py` created in 2.2).

### Lazy imports inside function bodies

Several functions in `generate.py` have `from tools.X import Y` **inside the
function body** (not at the top of the file). These won't be caught by
top-level grep. When moving functions to new files, read each function body
for inline imports and update them. Known patterns:
- `from tools.detector_params import ParticleParams`
- `from tools.simulation import smear_times, smear_charges_SK_like`
- `from tools.production.voxelize import ...`

### Files that only need import updates (no structural changes)

These files are not split, moved, or restructured in any phase. They only
need `from tools.` → `from lucid.` updates in Phase 1:
- `lucid/overlap.py` — sensor overlap integrals
- `lucid/visualization.py` — 2D event displays (cylinder-only, kept as-is)
- `lucid/gradient_analysis/` — parameter sweep framework (migrated to notebooks in Phase 10)
- `lucid/siren/` — all files except core.py lazy torch (Phase 1.5)
- `lucid/production/` — all files (bare import fixes in Phase 1.3)

### Verifying non-runnable code

For functions requiring ROOT files, SIREN model data, or external datasets:
`git diff` the function body before and after. For Category A (moves), the
body must be byte-identical. For Category B (interface changes), manually
verify that field extraction from containers matches the original variable
usage line by line.

---

## Testing Strategy

Changes fall into two categories with fundamentally different verification needs:

### Category A: Pure structural moves (no code changes)

Function bodies are the same bytes in different files. The risk is only broken
imports. Verification:
1. **Diff function body** before/after — confirm identical
2. **Import every moved function** — confirm paths resolve
3. No functional testing needed — identical code is identical by definition

Applies to: Phase 1 (rename), Phase 2 (module splits), most of Phase 10 (cleanup)

### Category B: Code changes (logic or interface changes)

The actual code changes. Even with same intent, JAX may retrace differently.
Verification must be functional — call the code and compare outputs.

**For deterministic functions** (no RNG): fixed inputs → exact output match

**For stochastic functions** (RNG involved): fixed PRNGKey → exact output match
(JAX is deterministic for a given key, device, and computation graph)

**For code we cannot run** (needs ROOT files, external data): verify via code
review that only import paths changed, not function bodies. For interface changes,
carefully review that field extraction matches the original variable usage.

**When deduplicating functions that differ** (e.g., different epsilon values):
test every caller with both implementations. If outputs differ, preserve
per-caller behavior rather than silently changing numerical results.

---

## Phase 0: Preparation

### 0.1 Merge branches

Merge sk_geom → main, then likelihood → main. Single branch with all features.
This is the baseline code.

### 0.2 Identify testable code paths

Before writing any tests, catalog which functions we can actually call locally:
- Pure math functions (optics, coordinate transforms, losses) — all testable
- Photon iteration functions — testable with fixed inputs + seed
- Propagator construction + call — testable with small detector configs
- Full simulation pipeline — testable if SIREN model data is available
- Production/event generation — requires ROOT files, test via code review only
- HPC scripts — test via import resolution only

### 0.3 Write pytest test suite (grows with each phase)

Create `tests/` with modular pytest files from the start:
- `tests/conftest.py` — shared fixtures (small detector config, fixed keys, fixed inputs)
- `tests/test_optics.py` — normalize, reflection, scattering, local frame
- `tests/test_losses.py` — all loss functions
- `tests/test_params.py` — DetectorParams/ParticleParams construction, field access
- `tests/test_geometry.py` — detector construction per config, sensor positions
- `tests/test_propagation.py` — propagator output with small detector, fixed rays
- `tests/test_photon_step.py` — photon iteration functions (sample + update_factors)
- `tests/test_sensor_response.py` — make_hits_* functions

Small, fast fixtures. Fixed PRNGKey for stochastic functions. Reference values
hardcoded after first run. `JAX_PLATFORM_NAME=cpu` for determinism.

This test suite **grows with each phase** — new tests are added for each
phase's changes. Not a one-shot fixture file; an evolving verification suite.

---

## Phase 1: Package rename + pyproject.toml

**What changes:** `tools/` → `lucid/`, all import paths, pyproject.toml created.
**Category:** A (pure structural move), with two minor Category B exceptions:
  wildcard→explicit import (step 4) and lazy torch (step 5)
**Risk:** Low — mechanical find-and-replace

### Steps

1. Create `pyproject.toml` (dependencies, build system, setuptools-scm — **no entry points yet**, added in Phase 2.3)
2. Rename `tools/` → `lucid/`
3. `from tools.` → `from lucid.` across all `.py` files, `good_notebooks/`, s3df_jobs scripts.
   Also fix bare imports in `production/generate_events.py` and
   `production/generate_events_with_particles.py` (`from generate import ...` → `from lucid.generate import ...`).
   Leave `notebooks/`, `time_diagnosis/`, and `time_problem/` broken (archived in Phase 10).
4. Clean up imports: replace wildcard `from tools.siren.core import *` with explicit imports — `create_photonsim_siren_grid` in simulation.py, `SIREN, create_photonsim_siren_grid` in generate.py (generate.py uses `SIREN` directly in `photonsim_differentiable_get_rays`). Remove 3 unused imports from simulation.py (`get_isotropic_rays`, `generate_laser_photons`, `setup_calibration_generator` — imported but never called). Add `__all__` to `siren/core.py` to prevent future wildcard leakage.
5. Make `import torch` lazy in `siren/core.py` — move inside functions that use it (torch_to_jax, convert_pytorch_to_jax, load_siren_jax). Without this, `from lucid import setup_event_simulator` fails without torch.
6. Remove all `sys.path.insert` calls
7. Create `lucid/__init__.py` with public API exports
8. `pip install -e .` — verify installable

### Verification

- Every `.py` file imports without error
- `pip install -e .` succeeds
- `from lucid import setup_event_simulator` works **without torch installed**
- Diff confirms: only import strings changed, no function bodies touched (except siren/core.py torch move)
- Add `tests/test_imports.py` — smoke-import every module in `lucid/`
- Run all Phase 0 pytest tests (import paths updated)

---

## Phase 2: Module splits

**What changes:** Functions move between files within `lucid/`.
**Category:** A (pure structural move)
**Risk:** Low — same code, different file locations

### 2.1 Split `simulation.py` → `lucid/simulation/`

**Dependency order** (no circular imports — verified by agent analysis):
```
optics.py          → imports: jax only
photon_step.py     → imports: optics
sensor_response.py → imports: jax, lucid.utils (smearing)
simulator.py       → imports: all above + propagation + geometry + sources
```

Move functions per the verified assignment (see CODEBASE_ANALYSIS.md §4.4).
`simulation/__init__.py` re-exports `setup_event_simulator`.

### 2.2 Split `generate.py` → `lucid/sources/`

```
siren_rays.py          → imports: jax, lucid.siren.core, lucid.utils
                         Contains: SIREN ray generation (photonsim_differentiable_get_rays,
                         cone vectors, normalization) + predict_t0
calibration_sources.py → imports: jax, lucid.utils, lucid.detector_params
                         Contains: isotropic/laser ray generators (get_isotropic_rays,
                         generate_laser_photons) + IsotropicSource/LaserSource +
                         setup_calibration_generator
event_io.py            → imports: uproot, h5py, lucid.utils, lucid.detector_params,
                         lucid.production.voxelize (lazy imports inside functions)
                         Contains: ROOT I/O, event loading, event generation
```

Shared math utilities (`normalize`, `generate_orthonormal_basis`,
`jax_rotate_vector_local`) move to `utils.py` — both siren_rays.py and
calibration_sources.py import from there.

Move `IsotropicSource`/`LaserSource` from `detector_params.py` to
`calibration_sources.py`. Move `predict_t0`/`predict_t0_wrapper` to
`siren_rays.py`. Update imports.

### 2.3 Split `single_track_optimization.py`

- `pipeline.py` — `create_combined_loss_function`, `run_complete_optimization_adam`, `generate_event_data`
- `run.py` — `parse_arguments`, `load_config`, `main`
- Rename `optimize.py` → `grid_search.py`

### 2.4 Consolidate losses

Move all from `lucid/optimization/losses.py` into `lucid/losses.py`.
Update all imports across codebase.

### 2.5 Slim `utils.py`

Move ROOT I/O functions to `sources/event_io.py`.

### Verification (all of Phase 2)

- Diff each moved function: body is byte-identical to original
- Import every moved function from its new location
- For functions we can call: run with fixed inputs, confirm identical output
  to pre-move values from Phase 0 fixtures

---

## Phase 3: Deduplication

**What changes:** Delete duplicate function copies, redirect imports.
**Category:** B (code changes — callers now use a different implementation)
**Risk:** Medium — duplicates may have subtle differences

### Approach: one deduplication per commit

For each duplicate group:
1. **Read both implementations side by side** — note every difference
2. **If implementations are identical:** delete one, update imports, done
3. **If implementations differ** (epsilon, keepdims, jit decorator, np vs jnp):
   - Identify every caller of each copy
   - For callers we can run: test with both implementations, compare outputs
   - If outputs differ: decide whether to parameterize (e.g., epsilon as arg)
     or keep the caller-specific value
   - If outputs are identical despite code differences: safe to consolidate

### 3.1 `spherical_to_cartesian` (3 copies)

All compute identical math:
- `utils.py` (line 257)
- `optimization/utils/functions.py` (line 261) — has `@jit`
- `optimization/utils/visualization.py` (line 29) — has `@jit`

The `@jit` decorator doesn't affect output, only performance.
Keep one in `lucid/utils.py`, remove the other two, update imports.

### 3.2 `normalize` (2 copies)

**These differ:**
- `simulation/optics.py` (was simulation.py): `v / jnp.maximum(norm, 1e-6)` with `keepdims=True`
- `utils.py` (was generate.py, moved in Phase 2.2): `v / (norm + 1e-8)` without keepdims

**Decision:** Use `norm + epsilon` approach everywhere (`jnp.maximum` can zero
gradients for near-zero vectors). Keep `keepdims=True` and `axis=-1` for batch
support. Make epsilon a parameter.

**Final location:** `lucid/utils.py` (already has the generate.py copy from Phase 2.2).
Delete the copy in `simulation/optics.py`. Update optics callers to
`from lucid.utils import normalize`.

Callers of the optics.py version: reflection, scattering, photon iteration
Callers of the utils.py version: ray generation, SIREN code

**Must test:** Every caller with the unified `norm + epsilon` implementation to
confirm outputs don't change for realistic inputs. The simulation path has
a custom VJP NaN sanitizer as a safety net, so the `jnp.maximum` gradient
cutoff was a redundant second layer.

### 3.3 `jax_rotate_vector` (2 copies)

Compare implementations, test callers.

### 3.4 Dead code handling

- Commented-out `first_arrival_nll` versions → move to `lucid/losses_archived.py`
- Dead `create_combined_loss_function` — already deleted in Phase 2.4
- Unused loss functions (compute_simple_loss, compute_loss_with_time, compute_softmin_loss,
  compute_simplified_loss, hellinger_loss, WC_loss, WC_smooth_loss variants) — move to
  `lucid/losses_archived.py` (recoverable without git history)
- `bo_leap/` directory → move to `archived/bo_leap/`
- `propagation/geometry.py` — keep (unused by production code, but useful reference)
- `propagation/base.py:calculate_linear_index_base` — keep (unused, related to Phase 8)
- Final decision on what to actually remove at end of refactor

### Verification (per deduplication)

- Each is a separate commit
- For each: callers produce identical output with the surviving implementation
- After all: run full set of Phase 0 fixtures to catch accumulated issues

---

## Phase 4: Unify propagation paths

**What changes:** `_common_propagation` and `_common_propagation_likelihood` merge
into one function.
**Category:** B (code change)
**Risk:** Low — only 2 lines differ

### Implementation

Both functions share the same direction stop_gradient mechanism:
- `next_dir = jnp.where(i < n_grad_iters, new_dirs, stop_gradient(new_dirs))` — identical in both

The two functions differ in exactly:
- Position stop_gradient: `i < K` (standard) vs `i < 0` (likelihood = always stop)
- Hit function: `_make_hits_fn` vs `_make_hits_likelihood_fn`

Unified function takes the position stop_gradient threshold and hit function
as parameters (both already selected at factory time in the closure).
Direction stop_gradient via n_grad_iters is unchanged (same in both).

### Verification

- Call with standard-mode parameters: output matches old `_common_propagation`
  baseline (fixed seed)
- Call with likelihood-mode parameters: output matches old
  `_common_propagation_likelihood` baseline (fixed seed)
- Gradient values match for both modes

---

## Phase 5: Wavelength module

**What changes:** Create `lucid/wavelength/` with code ported from `wavelength_dependency` branch.
**Category:** A (new files, no changes to existing code)
**Risk:** Low — purely additive, no existing code modified

### Implementation

1. Create `lucid/wavelength/__init__.py`
2. Create `lucid/wavelength/medium.py`:
   - `MediumProperties` NamedTuple (material, refractive_index, speed_of_light,
     wavelength_grid, scatter_coeff, absorption_coeff, refractive_index_curve,
     mie_scatter_coeff, mie_asymmetry — wavelength arrays Optional)
   - `MediumProperties.from_material()` — loads scalar + wavelength data if available
   - `compute_effective_properties()` — derives per-photon effective scatter/absorption/QE
     from DetectorParams scalars × medium corrections × detector QE curve
3. Create `lucid/wavelength/spectrum.py`:
   - `sample_cherenkov_wavelengths(medium, qe_curve, beta, n_photons, key)` —
     samples from Cherenkov spectrum weighted by detector QE for importance sampling
   - Uses medium n(λ) dispersion curve + detector QE(λ) spectral response
4. Create `lucid/wavelength/scattering.py` — Rayleigh phase function,
   Mie/Henyey-Greenstein phase function (ported, available for external use,
   not called in simulation loop)
5. Create `lucid/wavelength/data/water.json` — SK water model parameters
6. Create `lucid/wavelength/data/sk_qe.csv` — SK R3600 PMT QE curve
   (detector hardware data, loaded during DetectorGeometry construction)

Port from `origin/wavelength_dependency` branch `tools/wavelength.py`,
restructured into the above files. Port everything except Mie integration
into the simulation loop (Mie functions ported and available, Mie fields
on MediumProperties populated, but not called during propagation).

**Branch data notes:** On the branch, optical properties (Rayleigh coefficients,
Pope & Fry absorption data) are hardcoded Python constants, not JSON. Extract
these into `water.json` during porting. SK_QE.csv is missing from the branch
commit — source this data separately (SK R3600 PMT spectral response, ~294-648nm).
Cherenkov spectrum sampling (`sample_wavelengths_cherenkov`) is in generate.py
on the branch — port to `spectrum.py`.

This phase creates the standalone module only. Integration happens in:
- Phase 6: `DetectorGeometry.medium` and `DetectorGeometry.qe_curve` fields,
  always-on effective property derivation in simulator
- Phase 7: `PhotonRays.wavelengths` (Optional)

### Verification

- Add `tests/test_wavelength.py`:
  - `MediumProperties.from_material('water')` loads and returns valid scalar data
  - When wavelength data available: arrays loaded, shapes correct
  - `compute_effective_properties()` with monochromatic (wavelengths=None) returns
    DetectorParams scalars unchanged
  - `compute_effective_properties()` with wavelengths returns correct per-photon arrays
  - `sample_cherenkov_wavelengths()` returns valid wavelength distribution
  - Rayleigh/Mie phase functions produce normalized distributions
- No existing tests affected (purely additive)

---

## Phase 6: Container types

**What changes:** `DetectorGeometry`, `SimConfig`, `ParticleModel` NamedTuples
introduced. `setup_event_simulator` signature changes from 13 flat args to
containers. `bounds_check` added to Detector subclasses. Always-on effective
property derivation added to simulator.
**Category:** B (interface change)
**Risk:** Medium — must ensure containers reconstruct exactly the same internal state

**Parameter distribution** (the old 13 flat args map to containers):
- `json_filename`, `detector_type`, `temperature`, `max_sensors_per_cell` → `DetectorGeometry.from_config()`
- `n_photons`, `K`, `is_data`/`is_calibration` (→ `mode`), `use_expected_value`, `apply_smearing` → `SimConfig`
- `particle` → `ParticleModel.particle`
- `physics_config`, `default_detector_params` → eliminated (DetectorParams passed explicitly)
- `n_grad_iters` → settable on SimConfig, default derived from mode (track=0, calibration=2)

### Implementation

1. Add `bounds_check` method to each Detector subclass (Cylinder, Sphere, Box).
   Extract from the inline closures in `setup_event_simulator`
   (`get_inside_detector_flag`). This is a prerequisite — `DetectorGeometry`
   uses `detector.bounds_check` instead of if/elif closures.
2. Define `DetectorGeometry` in `lucid/geometry/`, `SimConfig` in `lucid/simulation/`,
   `ParticleModel` in `lucid/sources/` — each near its primary consumers.
   - `DetectorGeometry` imports `MediumProperties` from `lucid.wavelength`
   - `DetectorGeometry` has `qe_curve: Optional[jnp.ndarray]` (PMT spectral response)
   - `ParticleModel` does NOT store `material` — validates at load time, raises on mismatch
3. `DetectorGeometry.from_config()` does everything the old inline code in
   `setup_event_simulator` did: load config, create detector, create
   `MediumProperties.from_material()`, load QE curve from detector config,
   build propagator, derive speed_of_light from medium
4. `ParticleModel.from_config(particle, material)` loads SIREN model, t0 params,
   normalization. Validates that SIREN model was trained for `material`.
   Does not store material on the resulting object.
5. Refactor `setup_event_simulator` to extract fields from containers,
   use `detector.bounds_check` instead of inline closures.
   Add always-on effective property derivation: propagation loop and sensor
   response consume per-photon effective arrays (eff_scatter, eff_absorption,
   eff_qe) instead of raw DetectorParams scalars. For monochromatic
   (wavelengths=None): effective = scalar passthrough (corrections = 1.0).
6. Update all callers

### Verification

- Add `tests/test_containers.py`:
  - Per geometry: `detector.bounds_check()` on test positions matches old
    inline closure output exactly
  - `DetectorGeometry.from_config()` produces identical propagator/sensor config
  - `MediumProperties.from_material()` returns valid medium with correct speed_of_light
  - `ParticleModel.from_config('muon', 'water')` loads correctly;
    `ParticleModel.from_config('muon', 'ice')` raises ValueError if no ice SIREN model
  - `SimConfig(mode='track')` has n_grad_iters=0 by default
- Construct containers from same config used in Phase 0 baselines
- Call `setup_event_simulator` with containers
- Compare output against Phase 0 baseline values (fixed seed) — note that
  the calling interface changed but the internal computation must be identical.
  Phase 0 fixtures need wrapper functions that construct containers from the
  original flat args, then call the new interface.
- Verify monochromatic effective properties: with wavelengths=None,
  effective arrays equal DetectorParams scalars exactly
- Verify geometry reuse: same `DetectorGeometry`, different `SimConfig` →
  no propagator rebuild, identical outputs

---

## Phase 7: Pipeline NamedTuples

**What changes:** Raw dicts and positional tuples replaced with
`PropagationResult`, `PhotonState`, `PhotonStepResult`, `PhotonRays`.
**Category:** B (interface change)
**Risk:** Medium — JAX retraces when pytree structure changes

### Implementation order

Each is a separate commit:
1. `PhotonRays` — replace 3-tuple return from ray generation.
   `wavelengths` field is Optional (None for monochromatic). Ray generators
   return `PhotonRays(dirs, origins, weights, wavelengths=None)` initially.
   When wavelength mode is active, the simulator populates wavelengths
   via `sample_cherenkov_wavelengths()` after ray generation.
2. `PropagationResult` — replace dict return from propagator.
   Include all 8 fields from current dict (6 used in loop + `per_sensor_positions`
   and `sensor_normals` for debugging/visualization).
3. `PhotonStepResult` — replace 6-tuple return from photon iteration
   (also simplify custom VJP backward to `jax.tree.map`)
4. `PhotonState` — replace 5-tuple carry in `jax.lax.scan`

### Verification (per NamedTuple)

- Extend existing pytest files per NamedTuple:
  - `test_sources.py`: PhotonRays field values match old 3-tuple
  - `test_propagation.py`: PropagationResult field-by-field match against old dict
  - `test_photon_step.py`: PhotonStepResult + custom VJP gradient check
  - `test_propagation.py`: PhotonState lax.scan with NamedTuple carry works
- After all 4: full pipeline baseline comparison (fixed seed)

---

## Phase 8: Geometry registry

**What changes:** If/elif dispatch → decorator-based registry. Bounds check
moves to Detector subclass methods. Casing standardized to lowercase.
**Category:** B (dispatch mechanism change)
**Risk:** Medium — bounds_check closure context changes

### Implementation

1. Create `lucid/geometry/registry.py` with decorator
2. Add `@register_detector` to each subclass
3. Add `from_config(geom_def)` classmethods to each Detector subclass.
   Note: this is a DIFFERENT method than `DetectorGeometry.from_config(config_path)`
   from Phase 6. `Detector.from_config(geom_def)` constructs a single geometry
   object; `DetectorGeometry.from_config(config_path)` is the top-level entry
   point that calls `generate_detector()` which uses the registry to dispatch
   to the correct `Detector.from_config()`.
4. Refactor `generate_detector` to use registry lookup
5. Standardize casing to lowercase throughout
   (`bounds_check` already added in Phase 6, `setup_event_simulator`
   already uses `detector.bounds_check` via DetectorGeometry)
6. Note: `base.py` also has a `_add_detector_surface()` dispatcher
   (`_add_cylinder_surface`, `_add_sphere_surface`, `_add_box_surface`)
   that should become a subclass override for consistency
7. Refactor `geometry/utils.py:calculate_surface_normals()` — currently uses
   `isinstance(detector, Cylinder)` checks. Should use `detector.compute_normal()`
   (the Phase 9 abstract method) once available, or at minimum use registry-style
   dispatch instead of isinstance

### Verification

Per geometry type (cylinder, sphere, box):
- `from_config` produces detector with identical `all_points`, `ID_to_case`
- `bounds_check` on a grid of test positions (inside, outside, boundary)
  matches the old inline closure output exactly
- Full pipeline through registry matches baseline (fixed seed)

---

## Phase 9: Propagation framework (Option A — full ABC)

**What changes:** Detector ABC gains 6 abstract methods. Shared `create_propagator()`
replaces 3 geometry-specific factories. Each geometry implements methods that
encapsulate its specific math.
**Category:** B (significant restructure)
**Risk:** High — the propagation math must produce identical results
**Expected reduction:** ~20-25% (~2115 → ~1600 lines). Geometry-specific math
(intersection, normals, grid indexing) is ~500-600 lines per file and cannot
be unified. Shared boilerplate (find_intersected_sensors, create_propagator,
sensor assignment loops) is ~200-250 lines per file.

### The 6 abstract methods each geometry must implement

```
intersect_ray(origin, direction)       — ray-geometry intersection + grid info
compute_normal(intersection_point)     — surface normal at hit
bounds_check(points)                   — containment test (already added in Phase 6)
point_to_grid_cell(intersection_info)  — map intersection to linear grid cell index (NEW)
assign_sensor_to_cells(sensor_pos, sensor_radius) — which grid cells a sensor overlaps (NEW)
grid_cell_centers()                    — (n_cells, 3) array (NEW)
```

`bounds_check` already exists from Phase 6. The 5 new methods are
`intersect_ray`, `compute_normal`, `point_to_grid_cell`,
`assign_sensor_to_cells`, `grid_cell_centers`.

### The shared framework handles

```
create_propagator(detector, sensor_positions, sensor_radius, temperature, max_sensors_per_cell):
    1. Call detector.assign_sensor_to_cells() for all sensors  → geometric assignments
    2. Call detector.grid_cell_centers()                        → cell centers
    3. Call find_closest_sensors(centers, positions, K)         → distance assignments
    4. Build inverted sensor map from both assignment types     → (n_cells, max_sensors)
    5. Create overlap_prob from temperature + sensor_radius
    6. Return JIT-compiled propagate_photons() that internally:
       a. Calls detector.intersect_ray() via vmap              → intersection points + info
       b. Calls detector.point_to_grid_cell()                  → grid cell indices
       c. Looks up potential sensors from inverted map
       d. Calls compute_sensor_intersections_base() (existing) → per-sensor results
       e. Calls detector.compute_normal()                      → geometry normals
       f. Calls process_intersection_normals() (existing)      → final positions/normals
       g. Returns PropagationResult
```

### Shared utility functions (from current `propagation/base.py`)

Rename `propagation/base.py` → `propagation/intersection.py`. These functions
are used by the shared framework and stay as module-level utilities:
- `compute_sensor_intersections_base()` — core sensor intersection math
- `process_intersection_normals()` — final position/normal selection
- `find_closest_sensors()` — K-nearest sensor lookup
- `calculate_weighted_sensor_properties()` — weighted sensor processing
- `calculate_hit_properties()` — hit position/normal selection

### Implementation order (keep old code until verified)

1. Add abstract methods to Detector ABC (no implementations yet)
2. Implement methods on Cylinder by extracting from propagation/cylinder.py
   — the cylinder propagation file stays intact, methods are ADDITIONS
3. Write shared create_propagator() that uses the Cylinder methods
4. **VERIFY Cylinder** (see checks below) — old vs new must match exactly
5. Only after Cylinder passes: implement Sphere methods + verify
6. Only after Sphere passes: implement Box methods + verify
7. Only after ALL pass: remove old propagation factories

### Verification checks (PER GEOMETRY — cylinder, sphere, box)

Each check uses a small test detector with fixed config and fixed ray inputs.

**Layer 1 — Individual method outputs:**

| Check | What to compare | How |
|---|---|---|
| 1a | `assign_sensor_to_cells()` output | New method vs old standalone function, same sensor positions → exact match of assignment arrays |
| 1b | `grid_cell_centers()` output | New method vs old standalone function → exact match of (n_cells, 3) array |
| 1c | `bounds_check()` for test positions | New method vs old standalone function, grid of positions inside/outside/boundary → exact match of boolean array |
| 1d | `intersect_ray()` for fixed rays | New method vs old batch intersection function → exact match of intersection points, t-values, surface flags |
| 1e | `compute_normal()` at fixed intersection points | New method vs old normal calculation → exact match |
| 1f | `point_to_grid_cell()` for fixed intersection points | New method vs old inline `calculate_linear_index` → exact match of integer indices |

**Layer 2 — Shared infrastructure with geometry methods:**

| Check | What to compare | How |
|---|---|---|
| 2a | Inverted sensor map | Built via shared framework using new methods vs old geometry-specific builder → exact match of (n_cells, max_sensors) integer array |
| 2b | `find_intersected_sensors` output | Shared framework using new methods vs old geometry-specific function, same fixed rays → exact match of ALL output fields (weights, indices, times, positions, normals, inside_sensor) |

**Layer 3 — Full propagator:**

| Check | What to compare | How |
|---|---|---|
| 3a | `propagate_photons()` output | New shared propagator vs old geometry-specific propagator, same fixed rays → exact match of full PropagationResult |
| 3b | Gradient through propagator | `jax.grad` of a scalar reduction of propagator output (e.g., sum of weights) → exact match of gradient values |

**Layer 4 — Full pipeline:**

| Check | What to compare | How |
|---|---|---|
| 4a | Full simulation output | `setup_event_simulator` using new propagator vs old, fixed ParticleParams + DetectorParams + seed → exact match of all output values |
| 4b | Gradient through full pipeline | `jax.grad` of loss through full pipeline → exact match |

**Critical details to watch:**

- The `calculate_linear_index` is currently **inlined** inside `find_intersected_sensors`
  with closure over grid dimensions. It is NOT a standalone function — each geometry has
  a fundamentally different indexing scheme (cylinder: wall+2 caps with 3 surface types;
  sphere: theta+phi polar grid; box: 6 faces with per-face grids). The new
  `point_to_grid_cell` method must be written as a new standalone function per geometry,
  not just moved. Off-by-one errors here silently produce wrong sensor lookups.

- The `assign_sensors_to_grid` functions have different output shapes per geometry:
  cylinder returns (n_sensors, 4, 3), sphere returns (n_sensors, 4, 2), box returns
  (n_sensors, 4, 3). The shared inverted map builder must handle this.

- The `create_inverted_sensor_map` has a coordinate decoder that converts linear cell
  index back to grid coordinates for the geometric assignment check. This decoder is
  geometry-specific and must be part of the geometry's method, not the shared framework.

- The old cylinder and sphere inverted map builders use `@jax.jit`, box does not.
  The shared builder should be consistent — verify that adding or removing JIT doesn't
  change the output.

- The `batch_intersect_*_with_grid` functions return different sets of values per geometry
  (cylinder returns is_wall/is_top_cap/wall_indices/cap_indices; sphere returns theta_idx/phi_idx;
  box returns face/grid_indices). The `intersect_ray` abstract method needs a return type
  that accommodates all geometries while allowing `point_to_grid_cell` to consume it.
  Recommended approach: `intersect_ray` returns `(intersection_point, t_value, grid_info)`
  where `grid_info` is a geometry-specific NamedTuple or flat array that `point_to_grid_cell`
  knows how to decode. The shared framework treats `grid_info` as opaque — it just passes
  it from `intersect_ray` to `point_to_grid_cell`. Each geometry defines its own grid_info
  shape (cylinder: 5 values for surface type + wall/cap indices; sphere: 2 for theta/phi;
  box: 3 for face + 2D grid position).

- The coordinate decoder inside `create_inverted_sensor_map` (linear cell index → grid
  coordinates for geometric assignment matching) should become a method on the geometry
  class (e.g., `cell_index_to_coords(linear_idx)`) since it's the inverse of
  `point_to_grid_cell`. The shared framework calls it when building the inverted map.

---

## Phase 10: Cleanup and polish

**Category:** Mostly A (moves and archival)

### 10.1 Gradient analysis standardization

Before migrating notebooks to use `lucid/gradient_analysis/`, verify the module's
interface matches what notebooks currently do manually:
- Write toy tests that call `sweep_1d` / `sweep_2d` with small configs and a
  simple loss function (not the full simulation — just functional verification)
- Compare the output format (SweepResult1D/2D) against what notebooks expect
  for plotting
- Verify the plotting functions produce the same types of figures

Only after interface verification: migrate notebooks to use the module instead
of inline boilerplate.

### 10.2 Notebook restructuring

Move to categorized directories. Migrate 11 notebooks to new API.
Migrate paper_notebooks from diffCherenkov.

### 10.3 Archive dead directories

Move to `archived/`: `time_problem/`, `time_diagnosis/`, old `tests/` (scripts),
old `notebooks/` (stale), `figures/`, `plots/`, `output/`.
(bo_leap/ already archived in Phase 3.4)

### 10.4 Data organization

Ship small files (<1 MB) as package_data. Archive `siren_training_OLD/`.
Keep `scripts/download_data.sh` for ROOT file.

### 10.5 Final removal decision

Review all archived items. Decide what to permanently remove vs keep in
`archived/`. This is the only phase where actual deletion happens.

### 10.5 ~~Lazy torch import~~ — DONE in Phase 1.5

### 10.6 Configurable constants

Make `epsilon` (surface offset) and `TAU_TIME` parameters with current values
as defaults instead of hardcoded magic numbers.

### 10.7 Version management

Set up `setuptools-scm`, tag `v0.1.0`.

---

## Phase 11: Finalize test suite + CI/CD

By this point, the pytest suite has grown incrementally through every phase.
This phase finalizes and hardens it.

### 11.1 Audit test coverage

Review all test files accumulated through phases 0-10. Fill gaps:
- Unit tests per module covering key functions and edge cases found during refactoring
- Ensure every public API function has at least one test
- Add negative tests (invalid configs, material mismatches, degenerate inputs)

### 11.2 Integration tests

- Full simulation pipeline (track, calibration, data modes)
- Gradient flow end-to-end (loss → grad → parameter update)
- Wavelength mode: monochromatic vs wavelength-active produce correct results
- nbmake for notebooks

### 11.3 CI/CD

- GitHub Actions: CPU pytest on every PR
- `@pytest.mark.gpu` for tests that need GPU (run on S3DF or nightly)
- nbmake for notebook execution in CI

### Per-phase test additions (summary)

| Phase | Tests added |
|---|---|
| 0 | `conftest.py`, `test_optics.py`, `test_losses.py`, `test_params.py`, `test_geometry.py`, `test_propagation.py`, `test_photon_step.py`, `test_sensor_response.py` |
| 1 | `test_imports.py` (smoke-import every module) |
| 2 | `test_sources.py`, `test_pipeline.py` + import path updates in existing tests |
| 3 | Per-dedup caller tests (normalize near-zero gradient test) |
| 4 | Unified propagation mode tests in `test_propagation.py` |
| 5 | `test_wavelength.py` (MediumProperties, spectrum sampling, phase functions) |
| 6 | `test_containers.py` (bounds_check, from_config, effective properties) |
| 7 | NamedTuple field-by-field checks in existing test files |
| 8 | `test_registry.py` (lookup, from_config, case normalization) |
| 9 | 4-layer propagation verification per geometry |
| 10 | Notebook execution tests |
| 11 | Coverage audit, integration tests, CI/CD |

---

## Execution Notes

- **Each phase is a separate PR** that can be reviewed independently
- **No phase starts until the previous passes verification**
- **Within Category B phases, each discrete change is a separate commit** —
  easy to bisect if something breaks
- **For code we cannot run** (no ROOT files, no external data): verify via
  diff that function bodies are unchanged (Category A) or carefully review
  interface changes (Category B). The code review must confirm that field
  extraction from containers matches the original variable usage.
- **When duplicates differ:** test callers before choosing an implementation.
  Never silently change numerical behavior.
- **Fixed seeds produce identical JAX output** for the same computation graph
  on the same device. If a refactored function produces different output for
  the same seed, the computation graph changed — investigate before proceeding.

## Planning Granularity

`PHASE_0_TO_2_PLAN.md` provides step-by-step detail for Phases 0-2 (the
immediate work). Phases 3+ are described at a higher level in this document.
**Before starting each subsequent phase, write a similarly detailed plan**
(e.g., `PHASE_3_4_PLAN.md`) using this document as the spec and the codebase
as ground truth. The detailed plan for Phases 0-2 should serve as a template
for the level of specificity needed: exact file lists, line numbers, import
blocks, verification commands, and commit messages.

## Notebook Inventory

21 notebooks in `good_notebooks/` need import updates in Phase 1 and
API migration in Phase 10. For reference:
- Track reconstruction: tracking_opt_development.py, tracking_opt_development_likelihood.ipynb,
  tracking_opt_with_gif.ipynb, track_optimization_visualization.ipynb,
  visualize_3D_track_optimization.ipynb, optimization_vs_variables.ipynb
- Calibration: detector_grad_qe_convergence_multi_source.ipynb,
  grad_param_calibration_multi_init_no_qe.ipynb, laser_source_grad_analysis.ipynb
- Parameter analysis: parameter_scans_1D.ipynb, parameter_scans_1D_v2.ipynb,
  parameter_scans_1D_likelihood.ipynb, grad_loss_and_opt_in_2D.ipynb,
  grad_loss_and_opt_in_2D_likelihood.ipynb, per_sensor_tau_analysis.ipynb
- Visualization: geometry_and_events_3D_visualization.ipynb, cylinder_2D_displays.ipynb,
  event_hit_animation.ipynb, data_vs_pred_hit_predictions.ipynb
- Infrastructure: computational_performance_evaluation.ipynb, train_siren.ipynb

---

## Appendix: Execution Record

This section documents what was actually implemented, choices made, bugs found,
and current state. Written for context preservation.

### Repositories

Two separate repos are involved:

- **LUCiD** (`CIDeR-ML/LUCiD.git`) at `/home/oalterka/desktop_linux/diffWC/LUCiD/`:
  The canonical repository. Contains the `likelihood` branch (commit `2786118`) which is the
  **base code** before any refactoring. This is the reference baseline. The `refactor-v2`
  branch here is at an early state (merge only, no refactoring applied yet).

- **diffCherenkov** (`CIDeR-ML/diffCherenkov.git`) at `/home/oalterka/desktop_linux/diffWC/diffCherenkov/`:
  Shares commit history with LUCiD. The `refactor-v2` branch here was the initial working
  copy and has been transferred to the LUCiD repo.

- **Fixed baseline** at `/tmp/lucid-baseline/`: Worktree from `origin/likelihood`
  (commit `2786118`) with the normal convention fix applied (7-line change:
  outward geometry normals + `inward_normal = -normal` in photon_step + negated
  sensor normals in base.py).

**Working directory**: `/home/oalterka/desktop_linux/diffWC/LUCiD/` on branch `refactor-v2`.

**Note on tests/**: The LUCiD repo `tests/` directory contains both our refactor tests
AND some pre-existing test files (`test_tangent_gradients.py`, `test_all_fixes.py`, etc.).
Run refactor tests by listing them explicitly or by excluding the old ones.

### Phases Completed

**Phase 0** (commit `101f682`): Merged `origin/sk_geom` and `origin/likelihood` into
`origin/main` on `refactor-v2` branch. Created pytest suite with 73 baseline tests.

**Phase 1** (commit `551b066`): Renamed `tools/` → `lucid/`, created `pyproject.toml`,
updated all imports (`from tools.` → `from lucid.`), replaced wildcard imports,
made torch lazy in siren/core.py, removed sys.path.insert calls, created `lucid/__init__.py`.

**Phase 2** (commits `018a9d0` through `e050f22`, + audit fix `fab1b23`):
- 2.1: Split `simulation.py` → `lucid/simulation/` (optics, photon_step, sensor_response, simulator)
- 2.2: Split `generate.py` → `lucid/sources/` (siren_rays, calibration_sources, event_io)
- 2.3: Split `single_track_optimization.py` → pipeline.py + run.py, renamed optimize.py → grid_search.py
- 2.4: Consolidated all losses into `lucid/losses.py`
- 2.5: Moved ROOT I/O from utils.py to sources/event_io.py
- Audit fix: Incorrect lazy import in event_io.py (`from lucid.simulation` → `from lucid.utils` for smear functions)

**Phase 3** (commits `5deeee0` through `2a06cf7`):
- 3.1: Deduplicated `spherical_to_cartesian` (3 copies → 1 in utils.py)
- 3.2: Deduplicated `normalize` (2 copies → 1, unified to `norm + epsilon` with batch support, epsilon changed from 1e-6 to 1e-8)
- 3.3: Deduplicated `jax_rotate_vector` (2 copies → 1 in utils.py, with axis normalization)
- 3.4: Archived dead code (bo_leap/ → archived/, 6 unused loss functions → losses_archived.py)

**Phase 4** (commit `584943d`): Unified `_common_propagation` and `_common_propagation_likelihood`
into one function with `pos_grad_threshold` and `make_hits_fn` parameters. Net -73 lines.

**Phase 5** (commits `8023150`, `7737365`): Created `lucid/wavelength/` module:
- `medium.py`: MediumProperties NamedTuple, make_medium(), compute_effective_properties()
- `spectrum.py`: sample_cherenkov_wavelengths() (1/λ² inverse CDF)
- `scattering.py`: Rayleigh + Mie/HG phase function samplers
- `data/water.json`: SK water model parameters
- `data/sk_qe.csv`: Representative SK R3600 PMT QE data
- `load_qe_curve()`: JIT-compiled QE interpolator

**Phase 6** (commits `269d9f4` through `e23e001`):
- 6.1: Added `bounds_check()` to Cylinder/Sphere/Box, replaced inline closures in simulator
- 6.2: Defined container NamedTuples: SimConfig, DetectorGeometry, ParticleModel
- 6.3: Refactored setup_event_simulator internals to construct and use containers

**Phase 7** (commit `774b0e7`): Defined pipeline NamedTuples (PhotonRays, PropagationResult,
PhotonStepResult, PhotonState). Replaced 5-tuple carry in jax.lax.scan with PhotonState.

**Phase 8** (commit `d0fed13`): Geometry registry with `@register_detector` decorator.
Replaced if/elif dispatch in generate_detector. Case-insensitive lookup.

**Phase 9** (commits `4a9f8f2` through `0def541`, + grid fixes):
- Added 7 abstract methods to Detector ABC (intersect_ray, compute_normal, point_to_grid_cell, etc.)
- Implemented on Cylinder, Sphere, Box (delegating to existing propagation functions)
- Created shared `create_propagator()` in `lucid/propagation/shared.py`
- Verified bit-identical output vs old geometry-specific propagators for all 3 geometries
- DetectorGeometry.from_config now uses shared propagator
- Grid auto-derivation from sensor count/geometry to prevent overcrowding
- `validate_sensor_map()` raises ValueError on overcrowded cells
- Grid params flow through: `create_propagator(**grid_params)` → `detector.configure_grid(**grid_params)`
- `DetectorGeometry.from_config()` accepts `**grid_params` for explicit control

### Bugs Found and Fixed

**1. Normal convention bug** (commits `c3c6164`, `c9ba700`):
- **Discovery**: Git archaeology revealed cylinder wall normals were accidentally negated to
  inward in commit `511841e` (Apr 2025) during a SIREN merge rewrite. Original code had
  outward normals. Sphere, box, caps were always outward.
- **Impact**: With mixed conventions, epsilon offset and diffuse reflection behaved differently
  per geometry. Cylinder barrel: accidentally correct (inward normal + `+epsilon*normal` = pushed inward).
  Caps/sphere/box: wrong (outward normal + `+epsilon*normal` = pushed outward → photon killed by bounds check).
- **Fix**: All geometry normals now point OUTWARD (standard convention). In photon_step.py,
  `inward_normal = -normal` is computed once and used for epsilon offset and diffuse reflection.
  Specular reflection uses `normal` directly (sign-agnostic formula).
- **Sensor normals** (commit `c9ba700`): Also negated in `compute_sensor_intersections_base`
  so they point outward from the detector wall, consistent with geometry normals.
- **Verification**: For cylinder barrel path (the main path used by all SK notebooks),
  the change is net-zero: outward + negate = same as old inward. Cap/sphere/box paths
  are now FIXED (previously broken).
- **Fixed baseline**: Applied the same 7-line fix to the old `tools/` code at `/tmp/lucid-baseline/`.
  Comparison shows barrel path is bit-identical, cap path differs (now correct).

**2. Lazy import bug** (commit `fab1b23`): `event_io.py` had `from lucid.simulation import
smear_charges_SK_like, smear_times` — should be `from lucid.utils import`.

**3. Grid parameter defaults** (identified by old-vs-new diff analysis):
- Old `setup_event_simulator` used hardcoded grid params (cyl: 150/250/150, sph: n_divisions=100, box: 125/125/125)
- New code auto-derives from geometry. Different grid → different numerical results.
- **Decision**: Auto-derived is BETTER (prevents overcrowding). For exact matching, callers
  can pass explicit grid params via `**grid_params` to `from_config()` or `create_propagator()`.

**4. temperature=None handling**: Old propagator factories handled None as step-function
overlap (hard assignment). New shared propagator initially crashed on `None * sensor_radius`.
Fixed to replicate old behavior: `create_overlap_prob(None, sensor_radius)`.

### Design Choices Made

1. **Outward normal convention**: All geometry and sensor normals point outward. Consumer code
   (`photon_step.py`) negates with `inward_normal = -normal` for epsilon offset and diffuse reflection.
   Named variable for clarity.

2. **Grid auto-derivation**: `configure_grid(max_sensors_per_cell=4)` derives grid resolution
   from detector dimensions and sensor count to ensure no cell exceeds `max_sensors_per_cell`.
   Safety factors: cylinder 4x, sphere 8x (polar grid equatorial concentration), box 2x.

3. **validate_sensor_map raises ValueError**: Overcrowded cells (more geometric assignments than
   max_sensors_per_cell) are a hard error, not a warning. This catches configs that silently
   dropped sensors in the old code.

4. **normalize epsilon**: Changed from 1e-6 to 1e-8 during deduplication. Verified no impact
   on realistic inputs (L1 baseline 32/32 match).

5. **Backwards-compat shims**: `lucid/generate.py` re-exports all functions from sources/.
   `lucid/utils.py` re-exports I/O functions from sources/event_io.py.
   `detector_params.py` re-exports IsotropicSource/LaserSource from calibration_sources.py.

### Test Suite

341 tests across 19 test files:
- Reference value tests (optics, losses, params, geometry, sensor response, photon step)
- Physics tests (reflection law, Rayleigh PDF, cosine hemisphere E[cosθ]=2/3, Beer-Lambert,
  STE vs MC agreement, gradient sign correctness, HG E[cosθ]=g)
- Integration tests (cross-module chains, end-to-end gradients, normalize epsilon change)
- JIT compatibility tests (PhotonState in scan, bounds_check in jit/grad/vmap)
- Registry tests (lookup, case insensitivity, detector baseline verification)
- Propagator tests (output structure, determinism, ray intersection math per geometry)
- Differentiability tests (gradient through propagator, full pipeline, stop_gradient mechanisms)
- Sensor map validation tests (auto-grid, overcrowding detection)
- Shared propagator bit-identical tests (cylinder, sphere, box vs old factories)

### Baseline Comparison

**Level 1 (completed)**: `baseline_scripts/L1_capture.py` captures 32 reference values from
pure functions (optics, photon step, losses, gradients, propagator output). Comparison between
fixed baseline (`tools/`) and refactored code (`lucid/`): **32/32 PASS, bit-identical**.

**Level 1 Timing**: No performance regressions. Several operations faster in refactored code
(photon_step 0.64x, gradient 0.75x, poisson_nll 0.08x). No operation slower.

**Level 2 (in progress)**: Full optimization pipeline comparison scripts designed (6 scripts).
Requires SIREN model data + ROOT files. Design covers: track optimization (counts + likelihood),
1D parameter scans, calibration (4-param + QE), 2D loss landscapes.

### Current State and Next Steps

**Current blocker**: The `SK_geom_config.json` uses `superk` detector type requiring
`ConnectionTable_SK5.root` which is not available locally. All notebooks use SK geometry.
`SK_like_geom_config.json` is a cylinder approximation (same dimensions, ~11k sensors)
but its auto-derived grid can trigger the overcrowding validation. Options:
1. Obtain ConnectionTable_SK5.root
2. Use SK_like config with explicit grid params or increased max_sensors_per_cell
3. Use WCTE config (smaller, works, but doesn't match notebooks)

**Immediate next steps**:
1. Resolve SK config issue for L2 scripts
2. Implement and run L2 baseline scripts on both codebases
3. Fix any remaining differences found
4. Proceed to Phase 10 (cleanup) and Phase 11 (CI/CD)

### Comprehensive Notebook Analysis

All 18 notebooks in `good_notebooks/` have been thoroughly analyzed by 5 specialized agents.
Documentation at:
- `good_notebooks/NOTEBOOK_ANALYSIS.md` — structured reference by theme
- `good_notebooks/NOTEBOOK_DOCUMENTATION.md` — cell-by-cell walkthrough

Key finding: ALL notebooks use SK geometry with `SK_physics_config.json`.
Two loss paradigms: counts-based (`sqrt(v*c*t)`) and likelihood-based (3-term with stop_gradient).
The prediction simulator returns 2-tuple (counts-based) or 4-tuple (likelihood-based) depending
on which `_common_propagation` path is taken (standard vs likelihood).

### Old-vs-New Diff Analysis Results

Three agents performed exhaustive comparison. Findings:
- **No behavioral bugs** in the refactor — all differences are intentional
- Only numerical impact: normalize epsilon (1e-6→1e-8) and grid auto-derivation
- temperature=None handling and grid param passthrough were fixed after discovery
- All function signatures, loss formulas, optimizer setups, return types: identical
- Structural changes (unified propagation, NamedTuples, registry): behaviorally equivalent
