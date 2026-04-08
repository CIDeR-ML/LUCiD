# LUCiD Refactor — Phases 0-2 Detailed Plan

---

## Phase 0: Preparation

### 0.1 Merge branches

**Order:**
1. `git checkout main`
2. `git merge sk_geom` — clean merge, adds superk.py + config changes
3. `git merge likelihood` — adds likelihood losses, tau_vtx, pipeline integration

**Verify:** `git log --oneline --graph` shows single branch with all features.

### 0.2 Create a refactor branch

```bash
git checkout -b refactor-v2
```

All refactor work happens here. Main stays as the pre-refactor reference.

### 0.3 Write test fixtures

Create a proper pytest suite from the start. This grows incrementally with each
phase — new tests are added as code changes, not batched at the end.

**Directory structure:**
```
tests/
  conftest.py          — shared fixtures (small detector config, fixed keys, fixed inputs)
  test_optics.py       — normalize, reflection, scattering, local frame
  test_losses.py       — all loss functions
  test_params.py       — DetectorParams/ParticleParams construction, field access
  test_geometry.py     — detector construction per config, sensor positions
  test_propagation.py  — propagator output with small detector, fixed rays
  test_photon_step.py  — photon iteration functions (sample + update_factors)
  test_sensor_response.py — make_hits_* functions
```

**conftest.py shared fixtures:**
- Small cylinder config (100 sensors, r=5, h=10)
- Fixed input arrays for pure functions
- Fixed PRNGKey(42) for stochastic functions
- `JAX_PLATFORM_NAME=cpu` for determinism

**What the tests cover:**

```
tests/test_optics.py — DETERMINISTIC:
  1. normalize([3, 4, 0]) — both implementations
  2. jax_normalize — fixed vectors (different epsilon from normalize)
  3. compute_reflection_direction(incident, normal) — fixed vectors
  4. solve_rayleigh_inverse_cdf(0.5)
  5. create_local_frame([0, 0, 1])
  6. spherical_to_cartesian(theta, phi) — all 3 implementations
  7. jax_rotate_vector — fixed vector + rotation

tests/test_losses.py — DETERMINISTIC:
  8. poisson_nll(true, pred) — fixed arrays
  9. counts_loss(true, pred) — verify same as poisson_nll
  10. energy_loss(sim, true) — fixed arrays
  11. origin_time_loss — fixed inputs
  12. first_arrival_nll — fixed inputs
  13. segment_logsumexp — fixed inputs

tests/test_params.py — DETERMINISTIC:
  14. DetectorParams construction and field access
  15. ParticleParams construction, .direction property
  16. normalize_params / denormalize_params — round-trip test
  17. default_bounds — verify shape and values

tests/test_geometry.py — DETERMINISTIC:
  18. cylinder_bounds_check — grid of test positions
  19. sphere_bounds_check — grid of test positions
  20. box_bounds_check — grid of test positions
  21. Detector construction from each config (cylinder, sphere, box)
      → record all_points shape, first/last 5 sensor positions, ID_to_case sample
  22. calculate_surface_normals — per geometry, fixed sensor positions

tests/test_sensor_response.py — DETERMINISTIC:
  23. make_hits_likelihood — fixed flat_weights, flat_indices, flat_times
  24. make_hits_data — fixed inputs (no smearing)
  25. make_hits_simulation — fixed inputs
  26. smear_times — fixed inputs, verify Gaussian smearing
  27. smear_charges_SK_like — fixed inputs

tests/test_propagation.py — DETERMINISTIC:
  28. create_overlap_prob — small config, verify probability values

tests/test_optics.py — STOCHASTIC (exact match with fixed seed):
  29. sample_scatter_distance — PRNGKey(42), fixed D, S
  30. compute_scatter_direction — PRNGKey(42), fixed incident
  31. sample_cosine_hemisphere — PRNGKey(42), fixed normal

tests/test_photon_step.py — STOCHASTIC:
  32. photon_iteration_sample — PRNGKey(42), all 12 args fixed
  33. photon_iteration_update_factors — PRNGKey(42), all 12 args fixed

tests/test_propagation.py — STOCHASTIC:
  34. get_isotropic_rays — PRNGKey(42), fixed position/intensity/Nphot=100
  35. Propagator output — small cylinder (100 sensors, r=5, h=10),
      PRNGKey(42), 50 fixed rays → record full output dict
```

Reference values hardcoded after first run.

**What it does NOT test** (requires data files or full pipeline):
- Full setup_event_simulator (needs SIREN model)
- ROOT file I/O (needs .root file)
- Production scripts
- Notebook execution

### 0.4 Run pytest and record reference values

Run `pytest tests/ -v` on current code. Record all outputs.
These become the reference values hardcoded in the test assertions.

---

## Phase 1: Package rename + pyproject.toml

### 1.1 Create pyproject.toml

```toml
[build-system]
requires = ["setuptools>=64", "setuptools-scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "lucid-sim"
dynamic = ["version"]
description = "Differentiable photon simulation for optical particle detectors"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "flax>=0.8.0",
    "optax>=0.1.7",
    "numpy>=1.24",
    "scipy>=1.10",
    "h5py>=3.8",
    "tqdm>=4.60",
    "plotly>=5.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "uproot>=5.0",
]

[project.optional-dependencies]
training = ["torch>=2.0"]
dev = ["pytest>=7.0", "nbmake"]

# NOTE: Entry points added in Phase 2.3 after run.py is created.
# [project.scripts]
# lucid-optimize = "lucid.optimization.run:main"
# lucid-train-siren = "lucid.siren.train:main"

[tool.setuptools.packages.find]
include = ["lucid*"]

[tool.setuptools_scm]
```

### 1.2 Rename directory

```bash
# Clean stale bytecode first to avoid import confusion
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
git mv tools lucid
```

### 1.3 Update all internal imports

Find-and-replace `from tools.` → `from lucid.` and `import tools.` → `import lucid.`
in all files:

**Files to update (lucid/ — internal imports):**
- `lucid/simulation.py`
- `lucid/generate.py`
- `lucid/detector_params.py`
- `lucid/utils.py`
- `lucid/visualization.py`
- `lucid/overlap.py`
- `lucid/geometry/__init__.py`
- `lucid/geometry/detector.py`
- `lucid/geometry/base.py`
- `lucid/propagation/__init__.py`
- `lucid/siren/core.py`
- `lucid/siren/validate.py`
- `lucid/siren/plot_training_results.py`
- `lucid/siren/training/inference.py`
- `lucid/siren/training/trainer.py`
- `lucid/siren/training/dataset.py`
- `lucid/siren/training/analyzer.py`
- `lucid/siren/training/monitor.py`

Also update `diffCherenkov` → `lucid` in docstring examples:
- `lucid/siren/training/trainer.py` (line 62)
- `lucid/siren/training/analyzer.py` (line 27)
- `lucid/siren/training/monitor.py` (line 27)
- `lucid/optimization/single_track_optimization.py`
- `lucid/optimization/optimize.py`
- `lucid/optimization/losses.py`
- `lucid/optimization/utils/functions.py`
- `lucid/optimization/utils/visualization.py` — also remove duplicate import block
  (has both relative and absolute imports for the same symbols)
- `lucid/optimization/utils/geometry.py`
- `lucid/production/generate_events.py` — **NOTE: uses bare imports** (`from generate import ...`), not `from tools.`
- `lucid/production/generate_events_with_particles.py` — **NOTE: uses bare imports**, same pattern
- `lucid/production/particle_data_utils.py`
- `lucid/production/data_prod_utils.py`
- `lucid/production/visualize_particle_events.py`
- `lucid/production/voxelize.py`

**Files to update (outside lucid/):**
- `s3df_jobs/run_track_optimization.py`
- `s3df_jobs/run_eval_with_parametrization.py`
- `s3df_jobs/submit_eval_with_parametrization.py`
- `s3df_jobs/submit_tau_hyperparameter_tuning_job.py`
- `s3df_jobs/submit_job.py`
- `tests/` (all pytest files — update `from tools.` imports)
- All notebooks in `good_notebooks/` (`.ipynb` files — JSON, search for `"from tools.`)

**NOT updated (stale, archived in Phase 10):**
- `notebooks/` — stale notebooks, left broken until archival
- `time_diagnosis/` — 50+ files with `from tools.` imports, left broken until archival
- `time_problem/` — 16 files with `from tools.` imports, left broken until archival

### 1.4 Clean up imports

**Replace wildcard imports** with explicit imports:

In `lucid/simulation.py` (line 24):
```python
# was: from tools.siren.core import *
from lucid.siren.core import create_photonsim_siren_grid
```

In `lucid/generate.py` (line 10):
```python
# was: from tools.siren.core import *
from lucid.siren.core import SIREN, create_photonsim_siren_grid
```

simulation.py only uses `create_photonsim_siren_grid`.
generate.py also uses `SIREN` directly in `photonsim_differentiable_get_rays` (line 244).

**Remove unused imports** in `lucid/simulation.py` (lines 2, 5, 6):

`get_isotropic_rays`, `generate_laser_photons`, and `setup_calibration_generator`
are imported from generate.py but **never called** in simulation.py. They're used
by IsotropicSource/LaserSource (in detector_params.py), not by simulation.py
directly. Remove them.

### 1.5 Make torch import lazy in siren/core.py

Move `import torch` (line 6) from top-level into the functions that use it:
- `torch_to_jax()`
- `convert_pytorch_to_jax()`
- `load_siren_jax()`

Without this, `from lucid import setup_event_simulator` fails without torch
installed, defeating the purpose of `torch` being a `[training]` optional dep.

### 1.6 Remove sys.path.insert calls

Search and remove all instances of:
```python
sys.path.insert(0, ...)
sys.path.append(...)
project_root = Path(__file__).resolve().parent...
```

These exist in:
- `s3df_jobs/run_eval_with_parametrization.py`
- `s3df_jobs/submit_tau_hyperparameter_tuning_job.py` (inside generated script)
- `lucid/production/visualize_particle_events.py`
- Various other scripts

After `pip install -e .`, these are unnecessary.

### 1.7 Create lucid/__init__.py

```python
"""LUCiD — Light-based Unified Calibration and trackIng Differentiable simulation."""

from lucid.simulation import setup_event_simulator
from lucid.detector_params import DetectorParams, ParticleParams
from lucid.geometry import generate_detector

__all__ = [
    'setup_event_simulator',
    'DetectorParams',
    'ParticleParams',
    'generate_detector',
]
```

### 1.8 Verify

```bash
pip install -e .
python -c "from lucid import setup_event_simulator; print('OK')"
pytest tests/ -v
```

All tests must pass with identical values to Phase 0 baseline.
Add `tests/test_imports.py` — smoke-import every module in `lucid/`.

**Commit message:** `Rename tools/ to lucid/, add pyproject.toml`

---

## Phase 2: Module splits

Each sub-phase is a separate commit. Run `pytest tests/ -v`
after each to confirm nothing broke.

### 2.1 Split simulation.py → lucid/simulation/

**Step 1:** Create directory and __init__.py

```bash
mkdir lucid/simulation
```

**Step 2:** Create `lucid/simulation/optics.py`

Move these functions (cut from simulation.py, paste into optics.py):
- `normalize` (lines 33-36)
- `compute_reflection_direction` (lines 39-43)
- `sample_cosine_hemisphere` (lines 46-63)
- `create_local_frame` (lines 66-76)
- `sample_scatter_distance` (lines 79-83)
- `solve_rayleigh_inverse_cdf` (lines 86-112)
- `compute_scatter_direction` (lines 115-130)
- `jax_normalize` (lines 134-137)
- `jax_rotate_vector` (lines 140-147)

Add imports at top:
```python
import jax
import jax.numpy as jnp
```

**Step 3:** Create `lucid/simulation/photon_step.py`

Move these functions:
- `photon_iteration_sample` (lines 154-260)
- `photon_iteration_update_factors` (lines 263-365)
- `photon_iteration_update_factors_safe` (lines 372-382)
- `_fwd` (lines 385-398)
- `_bwd` (lines 401-412)
- Line 415: `photon_iteration_update_factors_safe.defvjp(_fwd, _bwd)`

Add imports:
```python
import jax
import jax.numpy as jnp
from lucid.simulation.optics import (
    normalize, compute_reflection_direction, sample_cosine_hemisphere,
    sample_scatter_distance, compute_scatter_direction
)
```

**Step 4:** Create `lucid/simulation/sensor_response.py`

Move these functions:
- `make_hits_simulation` (lines 422-449)
- `make_hits_data` (lines 452-489)
- `make_hits_likelihood` (lines 492-539)

Add imports:
```python
import jax
import jax.numpy as jnp
from lucid.utils import smear_times, smear_charges_SK_like
```

**Step 5:** Create `lucid/simulation/simulator.py`

This keeps `setup_event_simulator` and all its internal closures:
- `_common_propagation`, `_common_propagation_likelihood` (nested inside setup_event_simulator)
- `propagation_step` (nested inside _common_propagation)
- Mode-specific implementations: `_simulation_with_data_impl`,
  `_simulation_without_data_impl`, `_simulation_sensor_calibration_impl`

Update imports at top to reference new submodules:
```python
from lucid.simulation.optics import (
    normalize, jax_normalize, jax_rotate_vector
)
from lucid.simulation.photon_step import (
    photon_iteration_sample, photon_iteration_update_factors_safe
)
from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_data, make_hits_likelihood
)
```

Keep all other existing imports (from sources, propagation, geometry, etc.
still pointing to old locations — those move in phase 2.2).

**Step 6:** Create `lucid/simulation/__init__.py`

```python
from lucid.simulation.simulator import setup_event_simulator
from lucid.utils import smear_times, smear_charges_SK_like
```

Re-export smear functions so existing `from lucid.simulation import smear_times`
still works (generate.py imports them from simulation). Will be cleaned up in
Phase 2.2 when event_io.py imports directly from `lucid.utils`.

**Step 7:** Delete old `lucid/simulation.py`

It's now `lucid/simulation/` directory. The old file must be removed.

**Step 8:** Update all imports of `from lucid.simulation import ...`

The `__init__.py` re-exports `setup_event_simulator`, so `from lucid.simulation import setup_event_simulator` still works. But any direct imports of internal functions need updating:

Search for all `from lucid.simulation import` across the codebase and verify
each imported name is either re-exported from `__init__.py` or updated to
the specific submodule.

**Verification:**
- Diff each moved function body against the original — byte-identical
- `python -c "from lucid.simulation import setup_event_simulator"` — works
- `python -c "from lucid.simulation.optics import normalize"` — works
- `pytest tests/ -v` — all tests pass

**Commit message:** `Split simulation.py into lucid/simulation/ package`

---

### 2.2 Split generate.py → lucid/sources/

**Step 1:** Create directory

```bash
mkdir lucid/sources
```

**Step 2:** Move shared math utilities to `lucid/utils.py`

Move from generate.py to utils.py (append after existing functions):
- `normalize` (line 72) — generate.py's copy (`norm + 1e-8`, no keepdims).
  Deduplication with simulation/optics.py's copy happens in Phase 3.
- `generate_orthonormal_basis` (line 90)
- `jax_rotate_vector_local` (line 61)

These are general math utilities used by both siren_rays.py and
calibration_sources.py. Placing them in utils.py avoids cross-dependency
between sources/ submodules.

**Step 3:** Create `lucid/sources/siren_rays.py`

Move these SIREN-specific functions from generate.py:
- `generate_random_cone_vectors` (line 118)
- `denormalize_log_predictions` (line 157)
- `normalize_inputs_jit` (line 162)
- `photonsim_differentiable_get_rays` (line 191)
- `predict_t0` (line 558)
- `predict_t0_wrapper` (line 575)

Add imports:
```python
from lucid.siren.core import SIREN
from lucid.utils import normalize, generate_orthonormal_basis, jax_rotate_vector_local
```

`photonsim_differentiable_get_rays` uses `SIREN` directly for model inference (line 244).

**Step 4:** Create `lucid/sources/calibration_sources.py`

Move ray generators from generate.py (these are geometric, not SIREN-related):
- `get_isotropic_rays` (line 337)
- `get_isotropic_rays_random` (line 382)
- `generate_laser_photons` (line 396)
- `setup_calibration_generator` (line 470)
- `generate_random_direction` (line 508)
- `generate_random_vertex` (line 541)

Move from detector_params.py:
- `IsotropicSource` class
- `LaserSource` class
- `isotropic_source` factory function
- `laser_source` factory function

Add imports:
```python
from lucid.utils import normalize, generate_orthonormal_basis
```

Update `detector_params.py` to remove source classes and update its imports.

**Step 5:** Create `lucid/sources/event_io.py`

Move from generate.py only (utils.py functions move in Phase 2.5):
- `get_max_photons_per_particle` (line 16)
- `generate_events_from_root` (line 588)
- `generate_multi_folder_events` (line 858)
- `read_photon_data_from_photonsim` (line 947)
- `read_particle_data_from_photonsim` (line 1002)
- `generate_events_from_photonsim` (line 1238)
- `generate_events_from_photonsim_particles` (line 1481)

**NOTE:** Several event functions have **lazy imports inside function bodies**
(e.g., `from tools.detector_params import ParticleParams`, `from tools.simulation
import smear_charges_SK_like`). These won't be caught by top-level grep. After
moving to event_io.py, update these to `from lucid.detector_params import ...`
and `from lucid.utils import ...`.

**Step 6:** Create `lucid/sources/__init__.py`

```python
from lucid.sources.siren_rays import (
    photonsim_differentiable_get_rays,
    predict_t0,
)
from lucid.sources.calibration_sources import (
    IsotropicSource, LaserSource,
    get_isotropic_rays,
    generate_laser_photons,
    setup_calibration_generator,
)
```

**Step 7:** Delete old `lucid/generate.py`

**Step 8:** Update all imports across codebase

Key files that import from generate.py:
- `lucid/simulation/simulator.py` — update to `from lucid.sources.siren_rays import ...`
  Also remove unused imports: `get_isotropic_rays`, `generate_laser_photons`,
  `setup_calibration_generator` (imported but never called in simulation.py)
- `lucid/detector_params.py` — remove generate imports (sources moved out)
- `lucid/production/generate_events.py` — update to import from `lucid.sources.event_io`
- `lucid/production/generate_events_with_particles.py` — update (has lazy imports
  inside function bodies that also need updating: `from lucid.simulation import smear_*`
  → `from lucid.utils import smear_*`)
- `s3df_jobs/run_eval_with_parametrization.py` — update
- `s3df_jobs/submit_tau_hyperparameter_tuning_job.py` — update (including embedded script)
- `lucid/siren/validate.py` — update
- `good_notebooks/` — update

**Verification:**
- Diff each moved function — byte-identical
- All imports resolve
- `pytest tests/ -v` — all tests pass

**Commit message:** `Split generate.py into lucid/sources/ package`

---

### 2.3 Split single_track_optimization.py + rename optimize.py

**Step 1:** Create `lucid/optimization/pipeline.py`

Move from single_track_optimization.py:
- `create_combined_loss_function` (line 134)
- `run_complete_optimization_adam` (line 219)
- `generate_event_data` (line 548)

**Step 2:** Create `lucid/optimization/run.py`

Move from single_track_optimization.py:
- `parse_arguments` (line 73)
- `load_config` (line 103)
- `main` (line 690)

Update `main` to import from `pipeline`:
```python
from lucid.optimization.pipeline import (
    create_combined_loss_function,
    run_complete_optimization_adam,
    generate_event_data,
)
```

**Step 3:** Rename `lucid/optimization/optimize.py` → `lucid/optimization/grid_search.py`

Update all imports that reference `optimize`:
- `lucid/optimization/pipeline.py` (was single_track_optimization.py)
- `lucid/optimization/run.py`
- `lucid/optimization/utils/functions.py`
- `s3df_jobs/run_eval_with_parametrization.py`
- `s3df_jobs/submit_tau_hyperparameter_tuning_job.py` — **embedded script** (f-string)
  also imports `from lucid.optimization.optimize`, must update there too

**Step 4:** Delete old `lucid/optimization/single_track_optimization.py`

**Step 5:** Add entry points to `pyproject.toml`

Uncomment and add the entry points now that `run.py` exists:
```toml
[project.scripts]
lucid-optimize = "lucid.optimization.run:main"
lucid-train-siren = "lucid.siren.train:main"
```

Run `pip install -e .` again to register them.

**Verification:**
- Diff each moved function — byte-identical
- All imports resolve
- CLI entry point works: `lucid-optimize --help` (or `python -m lucid.optimization.run --help`)

**Commit message:** `Split optimization pipeline, rename optimize.py to grid_search.py`

---

### 2.4 Consolidate losses

**Step 1:** Move all functions from `lucid/optimization/losses.py` into `lucid/losses.py`

Append the optimization-specific losses after the existing functions:
- `energy_loss`
- `counts_loss`
- `grid_origin_time_loss`
- `softplus`, `smooth_pinball`
- `origin_time_loss`
- `cone_time_loss`
- `segment_logsumexp`
- `first_arrival_nll`
- `TAU_VTX_PARAM_A/B/C` constants
- `get_optimal_tau_vtx`
- `poisson_nll = counts_loss` alias
- `origin_time_loss_configurable` alias

**Do NOT move** `create_combined_loss_function` from optimization/losses.py — it is
dead code (superseded by the version in single_track_optimization.py). Delete it.

**Step 2:** Handle `poisson_nll` collision

`tools/losses.py:293` has a standalone `poisson_nll` function.
`tools/optimization/losses.py:417` has `poisson_nll = counts_loss` (alias).
Both compute the same result but are structured differently.
Keep `counts_loss` and the alias `poisson_nll = counts_loss`.
Delete the standalone function from the old losses.py.

**Step 3:** Delete `lucid/optimization/losses.py`

**Step 4:** Update all imports

Every file that does `from lucid.optimization.losses import ...` changes to
`from lucid.losses import ...`:
- `lucid/optimization/pipeline.py`
- `lucid/optimization/run.py`
- `lucid/optimization/grid_search.py`
- `lucid/optimization/utils/functions.py` — also replace wildcard
  `from lucid.optimization.losses import *` with explicit imports
- `s3df_jobs/run_eval_with_parametrization.py`
- `s3df_jobs/submit_tau_hyperparameter_tuning_job.py` (including embedded script)

**Verification:**
- Every loss function callable from `lucid.losses`
- `pytest tests/ -v` — all loss-related tests pass

**Commit message:** `Consolidate all loss functions into lucid/losses.py`

---

### 2.5 Slim utils.py

**Step 1:** Move ROOT I/O functions from `lucid/utils.py` to `lucid/sources/event_io.py`

Functions to move (if not already moved in 2.2):
- `read_photon_data_from_root`
- `get_random_root_entry_index`
- `save_single_event`
- `load_single_event`
- `save_single_event_with_extended_info`
- `save_single_event_with_particle_info`
- `merge_event_files`
- `read_multi_folder_events`
- `read_event_file`
- `analyze_event_directory`
- `analyze_event_kinematics`
- `print_event_kinematics`
- `analyze_loaded_particle`
- `extract_particle_properties`
- `momentum_to_angles_and_energy`

**Step 2:** Update imports

Files that import these functions from utils.py need updating:
- `lucid/production/generate_events.py`
- `lucid/production/generate_events_with_particles.py`
- `lucid/sources/event_io.py` (internal references)
- `s3df_jobs/` scripts
- Notebooks

**Verification:**
- Diff each moved function — byte-identical
- All imports resolve
- `pytest tests/ -v` — all tests pass

**Commit message:** `Move ROOT I/O from utils.py to sources/event_io.py`

---

## Phase 2 Gate

After all 5 sub-phases:

1. Run `pytest tests/ -v` — all tests pass
2. Run `pip install -e .` — clean install
3. Run `python -c "from lucid import setup_event_simulator, DetectorParams, ParticleParams"` — works
4. Verify directory structure matches CODEBASE_ANALYSIS.md §4.4
5. No remaining `from tools.` or `import tools.` in production code:
   ```bash
   grep -rn "from tools\.\|import tools\." lucid/ s3df_jobs/ production/ --include="*.py" | grep -v __pycache__
   grep -rn "from tools\." good_notebooks/ --include="*.ipynb"
   ```
   Both should return nothing. Also check for bare imports in production/:
   ```bash
   grep -rn "from generate import\|from simulation import\|from utils import\|from detector_params import" lucid/production/ --include="*.py"
   ```
   Should return nothing (all converted to `from lucid.X import`).
6. No remaining `sys.path.insert` in production code:
   ```bash
   grep -rn "sys.path.insert\|sys.path.append" lucid/ s3df_jobs/ --include="*.py" | grep -v __pycache__
   ```
   Should return nothing.

Only proceed to Phase 3 after all gates pass.
