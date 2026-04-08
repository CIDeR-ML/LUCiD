# LUCiD Codebase Analysis & Refactor Plan

**Date:** 2026-04-01
**Scope:** main, refactor, likelihood, sk_geom branches

---

## 1. Project Overview

LUCiD is a differentiable photon simulation framework for optical particle detectors.
It uses JAX for automatic differentiation through Monte Carlo ray-tracing, enabling
gradient-based optimization of detector calibration and particle track reconstruction.

**Core pipeline:**
```
Particle → Photon generation (SIREN / isotropic / laser / ROOT data)
         → Ray-tracing through detector (scattering, reflection, absorption)
         → Sensor response (overlap probability, QE, timing)
         → Loss computation (charge + timing likelihood)
         → Gradients via JAX autodiff → Parameter optimization
```

---

## 2. Current Architecture

### 2.1 Branch State

```
main (32cee34)
  +-- sk_geom (+2 commits) — SuperK real PMT positions from ROOT ConnectionTable
  +-- refactor (+25 commits) — NamedTuples, dual reflection, per-sensor QE
  +-- likelihood (+4 on refactor) — likelihood losses, tau_vtx, full pipeline integration
```

`likelihood` is a strict superset of `refactor`. All branches merge cleanly.
**Recommended order:** sk_geom → main, then likelihood → main.

Additionally, `origin/wavelength_dependency` branch exists (+2 commits on sk_geom, diverged
from main not likelihood). Adds `tools/wavelength.py` (Rayleigh + Mie scattering, QE curves,
Cherenkov spectrum sampling), extends `DetectorParams` with `asym_scatter_length` and `mie_g`.
Does NOT include likelihood/refactor changes. Wavelength features will be integrated into the
refactored structure — the module and data structures are brought over, Mie is not used in the
simulation loop for now.

**Latest likelihood commits (2786118):**
- Full pipeline integration with likelihood simulator output — direction search
  and energy scan now unpack the 4-tuple `(log_w, flat_times, flat_indices, total_charge)`
  instead of old 2-tuple
- Energy scan switched from `poisson_nll` to `energy_loss` (log ratio of total counts)
  to avoid tau_vtx bias during initial energy estimation
- Convergence check removed from Adam loop to ensure consistent history length
- Physics config path lookup fixed (`config['basic_config'].get('physics_config')`)
- Added JUNO physics config (same format as SK, 10K sensors)
- S3DF Singularity container documentation

### 2.2 Key Source Files (current, pre-rename from `tools/`)

| File | Lines | Role |
|---|---|---|
| `tools/simulation.py` | ~900 | Monolithic: photon physics, propagation loop, simulator factory |
| `tools/generate.py` | ~1500 | Photon generation (SIREN, isotropic, laser), ROOT I/O |
| `tools/utils.py` | ~1970 | Grab-bag: I/O, coordinates, smearing, material properties |
| `tools/losses.py` | ~780 | Loss functions (WC, Poisson, smoothed) |
| `tools/optimization/losses.py` | ~420 | Likelihood losses, tau_vtx, combined loss |
| `tools/optimization/single_track_optimization.py` | ~1000 | 5-stage optimization pipeline + CLI |
| `tools/propagation/{cylinder,sphere,box}.py` | ~2100 | Near-duplicate propagation per geometry |
| `tools/detector_params.py` | ~360 | DetectorParams, ParticleParams NamedTuples |
| `tools/geometry/{base,cylinder,sphere,box}.py` | ~800 | Detector classes + sensor placement |
| `tools/overlap.py` | ~300 | Sensor overlap probability (Gaussian kernel) |
| `tools/visualization.py` | ~464 | 2D event displays (cylinder-only) |
| `tools/propagation/geometry.py` | ~275 | Unified ray-geometry intersection math (used by propagation/__init__.py) |
| `tools/propagation/base.py` | ~275 | Shared propagation utilities (compute_sensor_intersections_base, etc.) |
| `tools/geometry/utils.py` | — | Geometry utilities (calculate_surface_normals, create_disc_mesh) |
| `tools/geometry/detector.py` | — | generate_detector, load_detector_geom, load_detector_config |
| `tools/siren/plot_training_results.py` | — | SIREN training results visualization |
| `spatial_overlap_integrals/` | — | Precomputed Gaussian overlap JSON files (ship as package_data) |

**Additional notes:**
- `tools/production/generate_events.py` and `generate_events_with_particles.py` use **bare imports** (`from generate import ...`) rather than `from tools.generate import ...`. These rely on sys.path hacks.
- `tools/siren/core.py` has a **top-level `import torch`** (line 6) that makes torch a hard dependency at import time. Must be made lazy before Phase 1 completes.
- `tools/simulation.py` and `tools/generate.py` both use `from tools.siren.core import *` — only `create_photonsim_siren_grid` is actually used from this wildcard.

### 2.3 Simulation Modes

| Mode | Photon source | Hit aggregation | Differentiable? |
|---|---|---|---|
| **Track** | SIREN NN | Per-photon likelihood | Yes (STE) |
| **Data** | ROOT file photons | Hard-min + Bernoulli QE | No (sampling) |
| **Calibration** | Isotropic/Laser source | Soft-min or likelihood | Yes (STE) |

**Likelihood is the confirmed path forward** for track reconstruction, replacing soft-min.

### 2.4 Differentiability Strategy

- **STE:** Expected-value photon updates with custom VJP for NaN gradient sanitization
- **jax.remat:** On photon step for memory efficiency
- **stop_gradient on directions:** After `n_grad_iters` iterations (physics-based per-mode: track=0 = always stop, calibration=2 = gradient flows for first 2 iters)
- **stop_gradient on positions:** After K iterations (standard mode) or always (likelihood mode, `i < 0`). This is the 2-line difference unified in Phase 4.
- **Per-sensor QE:** `qe * qe_corrections[sensor_idx]` — optimizable per-sensor calibration

---

## 3. Problems to Solve

### 3.1 Structural

1. **`simulation.py` is monolithic** — photon physics, hit aggregation, propagation loop, and factory all in one 900-line file with deeply nested closures
2. **`generate.py` mixes concerns** — SIREN photon generation, ROOT I/O, calibration sources, random vertex generation in 1500 lines
3. **`utils.py` is a 1970-line grab-bag** — I/O, coordinate math, material properties, smearing, printing all mixed together
4. **Losses split across two files** — `lucid/losses.py` and `optimization/losses.py` with no clear reason
5. **`single_track_optimization.py` does everything** — CLI, config loading, loss construction, 5-stage pipeline, event generation in 1000 lines
6. **3 near-identical propagation functions** — `_common_propagation`, `_common_propagation_likelihood`, data mode path share ~80% of logic

### 3.2 Extensibility

7. **8+ if/elif chains for detector type** across 6 files — adding a new geometry requires 10+ code locations
8. **Case inconsistency** — JSON uses `"cylinder"`, geometry uses `'cylinder'`, simulation requires `'Cylinder'`
9. **`load_detector_geom` returns different-length tuples** per geometry type
10. **2100 lines of near-duplicate propagation code** — cylinder (729), sphere (523), box (860) repeat the same algorithmic structure with different geometry math
11. **No protocol for propagation** — geometry-specific propagators selected by string matching

### 3.3 Data Structure Fragility

12. **Propagation returns raw dict** with string keys — `'times'` is actually meters not nanoseconds
13. **Photon iteration returns positional 6-tuple** — no named fields
14. **Photon data from ROOT uses implicit dict schema** — undocumented keys, silent typos
15. **13-argument factory function** mixes detector, simulation, and calibration concerns

### 3.4 Code Quality

16. **Duplicate functions:** `spherical_to_cartesian` (3 copies: utils.py, optimization/utils/functions.py, optimization/utils/visualization.py), `normalize` (2+), `jax_rotate_vector` (2), `get_particle_name` (2), `energy_loss` (in losses.py and duplicated inline in notebook)
17. **Wildcard import:** `from lucid.siren.core import *` in simulation.py
18. **No automated tests** — `tests/` contains only investigation scripts
19. **No `pyproject.toml`** — dependencies undocumented
20. **Commented-out code** — 2 deprecated `first_arrival_nll` versions in losses.py
21. **Hardcoded physics constants** — `c/1.33`, `41.2 degrees`, `epsilon=1e-4` scattered throughout
22. **11 of 24 notebooks** still use old tuple API

---

## 4. Refactor Plan

### 4.1 Container Types

Three build-time containers separate concerns clearly:

```python
class DetectorGeometry:
    """Physical detector + propagator. Built once, reused across simulations.
    Expensive work (grid construction, JIT compilation) happens at construction."""
    detector_type: str                # 'cylinder', 'sphere', 'box', 'superk'
    sensor_points: jnp.ndarray        # (num_sensors, 3)
    sensor_radius: float
    num_sensors: int
    speed_of_light: float             # m/ns (= medium.speed_of_light, convenience)
    medium: MediumProperties          # material physics (scattering, absorption, dispersion)
    qe_curve: Optional[jnp.ndarray]  # (n_λ,) PMT spectral response — detector hardware
    detector: Detector                # geometry object (dimensions, viz, sensor placement)
    propagator: Callable              # JIT-compiled, returns PropagationResult

    @classmethod
    def from_config(cls, config_path, temperature=0.2, max_sensors_per_cell=4):
        """Single entry point — builds everything from JSON config.
        Loads detector geometry, determines material, creates MediumProperties,
        loads QE curve from detector config, builds propagator."""


class SimConfig:
    """Simulation run parameters. Lightweight, no expensive construction."""
    n_photons: int                    # photons per event
    K: int                            # max scattering iterations
    mode: str                         # 'track', 'data', 'calibration'
    use_expected_value: bool           # STE (True) vs MC sampling (False)
    apply_smearing: bool              # SK-like charge/time smearing (data mode)
    n_grad_iters: int                 # settable; default derived from mode (track=0, calibration=2)


class ParticleModel:
    """Particle-specific learned models. Only needed for track mode.
    Interface: given particle params, produce PhotonRays."""
    siren_predictor: SIRENPredictor
    grid_data: jnp.ndarray            # pre-computed SIREN evaluation grid
    model_params: dict                # SIREN weights (JAX pytree)
    t0_params: tuple                  # timing parametrization (7 values)
    normalization: tuple              # (a, b, c) photon count power law
    num_seeds: tuple                  # (a, b, c) seed count scaling
    particle: str

    @classmethod
    def from_config(cls, particle, material):
        """Load SIREN model, t0 params, normalization from data/{material}/{particle}/.
        Validates that the SIREN model was trained for the requested material.
        Raises ValueError on material mismatch. Does not store material —
        authoritative material comes from DetectorGeometry.medium."""
```

**Call-time objects** (JAX pytrees, flow through JIT, differentiable):
- `DetectorParams` — scatter, reflection, absorption, QE (optimizable calibration)
- `ParticleParams` — energy, position, theta, phi, t0 (optimizable track)

**Simulator factory:**
```python
def setup_event_simulator(geometry: DetectorGeometry, config: SimConfig,
                          particle_model: Optional[ParticleModel] = None,
                          detector_params: Optional[DetectorParams] = None):
    """Build simulator. Uses geometry.propagator (already built).
    particle_model required for track mode, optional otherwise."""
```

**Usage:**
```python
geometry = DetectorGeometry.from_config('config/SK_geom_config.json')
model = ParticleModel.from_config('muon', 'water')
config = SimConfig(n_photons=100_000, K=7, mode='track', ...)

sim = setup_event_simulator(geometry, config, particle_model=model)
charges, times = sim(particle_params, detector_params, key)

# Reuse geometry with different config (no propagator rebuild)
config2 = SimConfig(n_photons=500_000, K=10, mode='track', ...)
sim2 = setup_event_simulator(geometry, config2, particle_model=model)
```

### 4.2 Pipeline NamedTuples

Replace raw dicts and positional tuples with structured types:

```python
class PropagationResult(NamedTuple):
    """Output of propagator."""
    sensor_weights: jnp.ndarray       # (max_sensors_per_cell, n_rays)
    sensor_indices: jnp.ndarray       # (max_sensors_per_cell, n_rays)
    times_meters: jnp.ndarray         # (max_sensors_per_cell, n_rays)
    positions: jnp.ndarray            # (n_rays, 3)
    normals: jnp.ndarray              # (n_rays, 3)
    inside_sensor: jnp.ndarray        # (max_sensors_per_cell, n_rays)
    per_sensor_positions: jnp.ndarray # (max_sensors_per_cell, n_rays, 3) — for debugging/viz
    sensor_normals: jnp.ndarray       # (max_sensors_per_cell, n_rays, 3) — for debugging/viz

class PhotonState(NamedTuple):
    """Carry state through jax.lax.scan."""
    positions: jnp.ndarray            # (n_rays, 3)
    directions: jnp.ndarray           # (n_rays, 3)
    times: jnp.ndarray                # (n_rays,) ns
    survival: jnp.ndarray             # (n_rays,)
    key: jnp.ndarray

class PhotonStepResult(NamedTuple):
    """Output of single photon iteration."""
    position: jnp.ndarray             # (3,)
    direction: jnp.ndarray            # (3,)
    time: jnp.ndarray                 # ns
    detect_probability: jnp.ndarray   # [0,1]
    reflection_attenuation: jnp.ndarray  # [0,1]
    continuing_factor: jnp.ndarray    # [0,1]

class PhotonRays(NamedTuple):
    """Output of ray generation."""
    directions: jnp.ndarray           # (n_rays, 3)
    origins: jnp.ndarray              # (n_rays, 3) meters
    weights: jnp.ndarray              # (n_rays,)
    wavelengths: Optional[jnp.ndarray]  # (n_rays,) nm — None for monochromatic
```

All are JAX pytrees — work with `vmap`, `lax.scan`, `grad`.
Note: `PhotonRays.wavelengths` is Optional. When None (monochromatic mode),
effective properties equal DetectorParams scalars directly. When present,
per-photon effective properties are derived. The structure is fixed at setup
time (not toggled mid-simulation), so the one-time rejit cost is acceptable.

### 4.3 Geometry Registry + Propagation Framework

**Registry replaces 8+ if/elif chains:**

```python
DETECTOR_REGISTRY = {}

def register_detector(name):
    def decorator(cls):
        DETECTOR_REGISTRY[name] = cls
        return cls
    return decorator

@register_detector('cylinder')
class Cylinder(Detector): ...

@register_detector('superk')
class SuperK(Cylinder):
    """Inherits cylinder propagation, overrides sensor placement.""" ...
```

**Detector ABC provides geometry-specific methods for shared propagation:**

```python
class Detector(ABC):
    @abstractmethod
    def intersect_ray(self, origin, direction): ...
    @abstractmethod
    def compute_normal(self, intersection_point): ...
    @abstractmethod
    def bounds_check(self, points): ...
    @abstractmethod
    def point_to_grid_cell(self, point) -> int: ...
    @abstractmethod
    def assign_sensor_to_cells(self, sensor_pos, sensor_radius): ...
    @abstractmethod
    def grid_cell_centers(self): ...
    @abstractmethod
    def place_photosensors(self): ...

    @classmethod
    def from_config(cls, geom_def: dict): ...
```

**Shared `create_propagator()` uses these methods:**
- Build inverted sensor map from `assign_sensor_to_cells()`
- Distance-based assignment via `grid_cell_centers()`
- `find_intersected_sensors_differentiable()` via `intersect_ray()` + `compute_normal()`
- Returns JIT-compiled closure producing `PropagationResult`

Each geometry implements ~6 methods. Shared framework handles inverted sensor map,
find_intersected_sensors, and factory. **~20-25% reduction (2115 → ~1600 lines).**
Geometry-specific math cannot be unified; savings come from deduplicating shared
boilerplate. Requires thorough per-geometry, per-layer output verification.

### 4.4 Target Directory Structure

```
LUCiD/
  pyproject.toml

  lucid/
    __init__.py                   — public API exports
    detector_params.py            — DetectorParams, ParticleParams
    losses.py                     — ALL losses consolidated
    overlap.py                    — sensor overlap integrals
    utils.py                      — slimmed: coordinates, material, smearing, printing
    visualization.py              — 2D event displays

    simulation/
      __init__.py
      optics.py                   — light transport primitives
      photon_step.py              — photon iteration + custom VJP
      sensor_response.py          — make_hits_data, make_hits_likelihood
      simulator.py                — setup_event_simulator + unified propagation loop

    sources/
      __init__.py
      siren_rays.py               — SIREN-based photon generation (photonsim_differentiable_get_rays, predict_t0)
      calibration_sources.py      — ray generators (isotropic, laser) + IsotropicSource/LaserSource + setup_calibration_generator
      event_io.py                 — ROOT I/O, event loading

    geometry/
      __init__.py
      detector.py                 — generate_detector, load_detector_geom, load_detector_config
      registry.py                 — DETECTOR_REGISTRY, register_detector decorator
      base.py                     — Detector ABC with propagation methods
      cylinder.py, sphere.py, box.py, superk.py

    propagation/
      __init__.py
      propagator.py               — shared create_propagator() framework
      intersection.py             — sensor intersection math (from current base.py)
      geometry.py                 — unified ray-geometry intersection math (existing;
                                    may become redundant after Phase 9 when each geometry
                                    implements intersect_ray() — evaluate then)

    wavelength/
      __init__.py
      medium.py                   — MediumProperties NamedTuple, from_material(),
                                    compute_effective_properties()
      spectrum.py                 — Cherenkov spectrum sampling
                                    (sample_cherenkov_wavelengths), wavelength utilities
      scattering.py               — phase functions: Rayleigh, Mie/HG (ported,
                                    available for external use, not used in sim loop)
      data/
        water.json                — SK water model parameters (P0-P8, Pope & Fry)
        sk_qe.csv                 — SK R3600 PMT QE curve (detector data, loaded
                                    during DetectorGeometry construction)

    gradient_analysis/            — parameter sweep framework (sweep.py, plotting.py)
    siren/                        — unchanged (lazy torch import in core.py)
    production/                   — unchanged

    optimization/
      pipeline.py                 — 5-stage optimization engine
      grid_search.py              — hierarchical position/direction search
      run.py                      — CLI entry point
      utils/                      — search algorithms, viz, Cherenkov geometry

  config/                         — geometry + physics JSON configs
  data/                           — SIREN models, PhotonSim tables
  spatial_overlap_integrals/      — precomputed Gaussian overlap JSONs (ship as package_data)
  scripts/                        — download_data.sh
  s3df_jobs/                      — HPC job scripts

  notebooks/
    track_reconstruction/         — 11 notebooks
    calibration/                  — 4 notebooks
    event_display/                — 4 notebooks
    reference/                    — 5 notebooks (benchmarks, training, paper figures)

  tests/                          — new pytest suite
```

### 4.5 Structural Changes

1. **Rename `tools/` → `lucid/`** — proper installable package name. All imports change from `from tools.X` to `from lucid.X`. `pyproject.toml` declares `lucid` as the package.
2. Split `simulation.py` → `lucid/simulation/` (optics, photon_step, sensor_response, simulator)
3. Split `generate.py` → `lucid/sources/` (siren_rays, calibration_sources, event_io). `predict_t0` goes to `siren_rays.py` (particle-specific timing model).
4. Split `single_track_optimization.py` → pipeline.py + run.py; rename optimize.py → grid_search.py
5. Consolidate `lucid/losses.py` + `optimization/losses.py` → single `lucid/losses.py`. Move unused loss functions to `lucid/losses_archived.py`.
6. Unify `_common_propagation` and `_common_propagation_likelihood` → 1 function with mode parameters (position stop_gradient iters + make_hits function)
7. Deduplicate: spherical_to_cartesian (3→1), normalize (2→1, use `norm + epsilon` approach, test callers for epsilon sensitivity), jax_rotate_vector (2→1), get_particle_name (2→1)
8. Slim `utils.py` — move ROOT I/O to `sources/event_io.py`
9. Extract IsotropicSource/LaserSource from detector_params.py → sources/calibration_sources.py
10. Archive `bo_leap/` (unused — decide whether to remove at end of refactor)
11. Keep `make_hits_simulation` for calibration mode. Track mode uses `make_hits_likelihood` exclusively.

### 4.6 Packaging

- `pyproject.toml` declaring `lucid` package (distribution name: `lucid-sim`) with core deps
  (JAX, Flax, optax, numpy, scipy, plotly, h5py, uproot, matplotlib, seaborn, tqdm)
- `[training]` extra: PyTorch
- `[dev]` extra: pytest, nbmake
- Lazy `import torch` in `siren/core.py`
- `setuptools-scm` for automatic version from git tags
- `lucid/__init__.py` exports public API for clean imports (final state, after Phase 6+):
  ```python
  from lucid.simulation.simulator import setup_event_simulator
  from lucid.detector_params import DetectorParams, ParticleParams
  from lucid.geometry import DetectorGeometry  # added in Phase 6, defined in lucid/geometry/
  # etc.
  ```
  Phase 1 `__init__.py` exports `setup_event_simulator`, `DetectorParams`,
  `ParticleParams`, `generate_detector` only.
  
  Container locations: `DetectorGeometry` in `lucid/geometry/`,
  `SimConfig` in `lucid/simulation/`, `ParticleModel` in `lucid/sources/`.

### 4.7 Cleanup

- **Archive to `archived/`:** `time_problem/`, `time_diagnosis/`, `tests/` (old scripts), `notebooks/` (stale), `figures/`, `plots/`, `output/`, `bo_leap/`, `siren_training_OLD/`
- **Move unused loss functions** to `lucid/losses_archived.py` (recoverable without git history)
- **Archive dead code:** commented-out `first_arrival_nll` versions → `losses_archived.py`,
  dead `create_combined_loss_function` in optimization/losses.py → delete (already superseded),
  wildcard imports → replaced with explicit, duplicate imports → removed
- **Keep but note:** `propagation/geometry.py` (unused by production code, may be useful reference),
  `propagation/base.py:calculate_linear_index_base` (unused, but related to Phase 8 work)
- **Migrate notebooks:** 11 remaining to NamedTuple API
- **Migrate paper_notebooks** from diffCherenkov → `notebooks/reference/`
- **Final removal decision** at end of refactor for all archived items

### 4.8 Downstream Impact

All consumers of `tools.*` need import updates. Key locations:

| Consumer | Files | Changes needed |
|---|---|---|
| `s3df_jobs/` | 5 scripts | `from tools.*` → `from lucid.*` + new factory API (containers) |
| `production/` | 5 files | `from tools.*` → `from lucid.*` (minimal logic changes) |
| notebooks | 24 | Import paths + 11 need tuple→NamedTuple migration |
| `conftest.py` / tests | all | New pytest suite replaces old scripts |

The `lucid/__init__.py` public API re-exports minimize breakage — most code
only needs the top-level import path change, not deep submodule paths.

### 4.9 Testing

- New pytest suite: unit tests (params, losses, optics, photon step, sensor response, coordinates) + integration tests (full pipeline, gradient flow)
- Force CPU in tests via `JAX_PLATFORM_NAME=cpu` (fast, deterministic, no GPU needed)
- GPU tests marked with `@pytest.mark.gpu` (run on S3DF or manually)
- Small synthetic test fixtures (e.g., tiny SIREN model with hidden_features=8) — don't download full models in CI
- Golden output `.npz` files in `tests/data/` for numerical regression
- `nbmake` in CI to execute notebooks and catch API drift
- K-selection script: `python -m lucid.determine_K --config ... --threshold 0.01`
- CI/CD: GitHub Actions CPU tier on every PR; GPU tier nightly or manual trigger

### 4.10 JAX-Specific Patterns (from expert review)

Verified experimentally:

- **All NamedTuples (build-time and call-time) stay as NamedTuples.** Build-time containers
  (`DetectorGeometry`, `SimConfig`, `ParticleModel`) are never passed to JIT functions, so
  JAX pytree auto-registration is irrelevant. Call-time containers (`DetectorParams`,
  `ParticleParams`, `PropagationResult`, etc.) work naturally as JAX pytrees.

- **`PhotonRays.wavelengths` is Optional** — None for monochromatic, `(n_rays,)` when
  wavelength-active. The structure is fixed at setup time (not toggled mid-simulation),
  so the one-time rejit cost is acceptable. Build-time containers (DetectorGeometry,
  SimConfig, ParticleModel, MediumProperties) freely use Optional since they never
  enter JIT.

- **SimConfig must stay factory-level only** — never passed as a JIT argument. Its fields
  (mode string, integer shape parameters) are captured in closures at build time, matching
  the current code's pattern.

- **Custom VJP simplifies with NamedTuples:** The 6-line manual NaN sanitization becomes
  `g = jax.tree.map(lambda x: jnp.nan_to_num(x, ...), g)` — one line, auto-adapts to
  field changes.

- **remat placement:** Keep on photon_step (inner) and propagation_step (outer). Don't add
  remat inside optics.py functions — they're already inside the remat boundary.

- **Detector ABC methods can be called directly inside JIT** — geometry dimensions are
  Python constants captured at trace time. No need for the extra indirection of returning
  closures. The detector object is captured in the factory closure, its attributes become
  traced constants.

### 4.11 Packaging Details (from expert review)

- **Package name:** Use `lucid-sim` as PyPI distribution name (`lucid` is taken on PyPI
  by a Google neural network interpretability library). Import name stays `lucid`.

- **Version management:** `setuptools-scm` derives version from git tags automatically.
  Tag `v0.1.0` → version is `0.1.0`. No manual `__version__` to maintain.

- **Dependency groups:**
  ```toml
  [project]
  name = "lucid-sim"
  dependencies = [
      "jax>=0.4.20", "jaxlib>=0.4.20", "flax>=0.8.0", "optax>=0.1.7",
      "numpy>=1.24", "scipy>=1.10", "h5py>=3.8", "tqdm>=4.60",
      "plotly>=5.0", "matplotlib>=3.7", "seaborn>=0.12",
      "uproot>=5.0",
  ]

  [project.optional-dependencies]
  training = ["torch>=2.0"]
  dev = ["pytest>=7.0", "nbmake"]
  ```
  Lower bounds only (no upper bounds) — JAX evolves rapidly, upper bounds cause
  dependency hell. `gpjax` and `evosax` dropped (unused). `uproot` kept as core
  (11 MB, lightweight, needed for data mode).

- **Data strategy:**

  **Ship with the package** (total <1 MB, as `package_data` in `lucid/data/`):
  - `water/muon/siren_training/trained_model/photonsim_siren_weights.npz` (780 KB)
  - `water/muon/siren_training/trained_model/photonsim_siren.json` (4 KB)
  - `water/muon/photonsim_params.json` (4 KB)
  - `water/muon/t0.json` (8 KB)
  - `water/muon/range_params.json` (4 KB)
  - `water/electron/photonsim_params.json` (4 KB)
  - `spatial_overlap_integrals/*.json` (4 files, <1 KB each) — precomputed Gaussian overlap integrals used by overlap.py

  **Separate download** (1.7 GB, via `scripts/download_data.sh`):
  - `water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root`
  - Only needed for data mode and HPC evaluation scripts
  - Keep existing Google Drive + gdown script for now

  **Archive** (decide whether to remove at end of refactor):
  - `water/muon/siren_training_OLD/` — 51 MB of stale checkpoints (84 files), unused

- **Config files:** Small geometry configs (<10KB each) shipped as `package_data` inside
  `lucid/config/`. Loadable via `importlib.resources`.

- **Physics configs:** `SK_physics_config.json` (10,773 lines) and
  `JUNO_physics_config.json` (10,008 lines) have large uniform `qe_corrections` arrays.
  Keep as-is for now. Can be split into compact JSON + `.npy` later if needed.

- **CLI entry points:**
  ```toml
  [project.scripts]
  lucid-optimize = "lucid.optimization.run:main"
  lucid-train-siren = "lucid.siren.train:main"
  ```

- **Remove all `sys.path.insert` calls** — once installed via `pip install -e .`,
  all imports work directly.

---

## 5. Future Extensibility

Design constraints to respect during refactor — no code changes needed now,
but the architecture must not block these extensions.

### 5.1 Wavelength Dependence

**Implemented via `lucid/wavelength/` module** (ported from wavelength_dependency branch).

**Physical property split — medium vs detector:**
- **Medium properties** (water/ice physics): refractive index, scattering, absorption, dispersion
- **Detector properties** (PMT hardware): QE spectral response curve
- **Calibration parameters** (optimizable): scalar qe, qe_corrections, scatter_length, absorption_length

```python
class MediumProperties(NamedTuple):
    """Physical properties of the detection medium. Built from material data files."""
    material: str                              # 'water', 'ice'
    refractive_index: float                    # n (scalar, reference wavelength)
    speed_of_light: float                      # c/n in m/ns
    # Wavelength-dependent (from material data files):
    wavelength_grid: Optional[jnp.ndarray]     # (n_λ,) nm
    scatter_coeff: Optional[jnp.ndarray]       # (n_λ,) 1/m — Rayleigh
    absorption_coeff: Optional[jnp.ndarray]    # (n_λ,) 1/m
    refractive_index_curve: Optional[jnp.ndarray]  # (n_λ,) n(λ) dispersion
    # Mie scattering (ported, not used in sim loop):
    mie_scatter_coeff: Optional[jnp.ndarray]   # (n_λ,) 1/m
    mie_asymmetry: Optional[float]             # Henyey-Greenstein g parameter

    @classmethod
    def from_material(cls, material):
        """Load material properties from data files (water.json, etc.).
        Scalar properties always loaded. Wavelength arrays loaded if data available."""
```

MediumProperties is build-time (on DetectorGeometry), never enters JIT. Optional
fields are safe — structure is fixed at construction time.

QE spectral response curve lives on DetectorGeometry (detector hardware, not medium):
`DetectorGeometry.qe_curve: Optional[jnp.ndarray]`

**Architecture — always-on effective properties:**
- Propagation loop consumes per-photon effective property arrays, not raw DetectorParams scalars
- `effective_value(λ) = DetectorParams.scalar × medium.correction(λ)` for scatter, absorption
- `effective_qe(λ) = DetectorParams.qe × qe_curve(λ)` (QE uses detector curve, not medium)
- Monochromatic (wavelengths=None): effective arrays = broadcast scalars (corrections = 1.0)
- Wavelength-active: effective arrays vary per photon

**Wavelength lifecycle through the simulation:**

| Stage | Data | Location |
|---|---|---|
| **Generation** | `wavelengths (n_rays,)` sampled from Cherenkov spectrum | Simulator, after ray generation |
| **Derivation** | `eff_scatter, eff_absorption, eff_qe` each `(n_rays,)` | Simulator, before propagation loop |
| **Propagation** | Per-photon scatter/absorption via vmap | `photon_update_fn` arguments |
| **Sensor response** | Per-photon QE via vmap | `make_hits_*` arguments |
| **Output** | Not carried | Wavelengths consumed during derivation |

**Cherenkov spectrum sampling** (in `wavelength/spectrum.py`):
- `sample_cherenkov_wavelengths(medium, qe_curve, beta, n_photons, key)`
- Uses medium n(λ) dispersion + detector QE(λ) for importance sampling
- Called in the simulator (not inside ray generators — SIREN wasn't trained with wavelength)
- For calibration: laser = fixed wavelength, isotropic = source-dependent

**PhotonRays.wavelengths** is Optional — None for monochromatic, `(n_rays,)` when
wavelength-active. Structure fixed at setup time.

**Mie scattering** functions (Henyey-Greenstein phase function) ported to
`wavelength/scattering.py`, added to MediumProperties fields. Available for
external use. Not called inside the simulation loop for now.

### 5.2 Scintillation Light

- `ParticleModel` is light-type agnostic: interface is "produce `PhotonRays`"
- Cherenkov: SIREN model with cone emission
- Scintillation: isotropic emission with exponential time components
- The propagation pipeline doesn't care — it receives `PhotonRays` and propagates them

### 5.3 String-Based Detectors (IceCube/KM3NeT)

- `bounds_check` generalizes from "inside enclosed volume" to "in instrumented region"
- Propagation termination: "left region of interest" rather than "hit wall"
- Grid cells become 3D volumetric instead of surface-based
- New geometry class, no inheritance from existing shapes

### 5.4 Real Detector Geometries

SuperK pattern generalizes:
- `SuperK(Cylinder)` — inherits cylinder propagation, overrides sensor placement from ROOT
- `RealJUNO(Sphere)` — inherits sphere propagation, real PMT positions
- HK — just Cylinder with different dimensions
- All fit the registry pattern with no additional code beyond the detector class

### 5.5 Design Rules

1. Never hardcode `1.33`, `c/1.33`, `41.2°` — use `geometry.speed_of_light`, `geometry.medium`
2. Never assume Cherenkov — `ParticleModel` produces `PhotonRays`, pipeline is agnostic
3. Never assume enclosed volume — `bounds_check` = "is photon still relevant?"
4. `DetectorParams` = scalar calibration scale factors; medium/detector = fixed physics
5. QE spectral curve = detector hardware (`DetectorGeometry.qe_curve`), not medium
6. Optional fields on NamedTuples are fine for build-time containers. For call-time
   NamedTuples (PhotonRays), set structure once at setup, don't toggle mid-simulation

---

## 6. Design Decisions

| # | Decision | Choice | Reason |
|---|---|---|---|
| 1 | Likelihood vs soft-min | Likelihood is the path forward | Confirmed |
| 2 | Factory pattern | Single `setup_event_simulator` with containers | More robust than separate factories |
| 3 | Parameter representation | (theta, phi) canonical, .direction property | Unconstrained optimization |
| 4 | PyTorch dependency | Training-only optional extra | Lazy import in core.py |
| 5 | SIREN model format | .npz + .json (already done) | PyTorch-free inference |
| 6 | n_grad_iters | Settable on SimConfig, default derived from mode | Controls direction stop_gradient. Track=0 (always stop — optimization variable, gradients come from loss), calibration=2 (known source, direction gradient refines propagation). Position stop_gradient is separate (K vs 0, unified in Phase 4). |
| 7 | Dual reflection | Keep (wall=diffuse, sensor=specular) | Modeling choice |
| 8 | Geometry dispatch | Registry pattern, lowercase everywhere | Eliminates 8+ if/elif chains |
| 9 | Build-time containers | DetectorGeometry + SimConfig + ParticleModel | Clear separation of concerns |
| 10 | Pipeline data | PropagationResult, PhotonState, PhotonStepResult, PhotonRays | Replace fragile dicts/tuples |
| 11 | Propagation framework | Full ABC (Option A): 6 abstract methods + shared create_propagator | ~20-25% reduction |
| 12 | Propagator location | Lives on DetectorGeometry, built once | Reuse across SimConfigs |
| 13 | Backward compatibility | Clean break, no shims | No external users yet |
| 14 | Physics logic | No changes | Refactor is structural only |
| 15 | make_hits_simulation | Keep for calibration, track uses likelihood only | Calibration doesn't use time |
| 16 | normalize deduplication | Use `norm + epsilon` everywhere, test callers | `jnp.maximum` can zero gradients |
| 17 | Dead code policy | Archive (losses_archived.py, archived/ dir), don't delete | Decide what to remove at end of refactor |
| 18 | predict_t0 location | sources/siren_rays.py | Particle-specific timing model |
| 19 | gradient_analysis | Keep as lucid/gradient_analysis/, standardize notebook usage | Verify interface with toy tests first |
| 20 | visualization.py | Keep cylinder-only for now | 2D display doesn't generalize easily |
| 21 | Physics config format | Keep large JSON as-is for now | Can split later if needed |
| 22 | Combined loss formula | Don't touch | Working as designed |
| 23 | Wavelength module | `lucid/wavelength/` (medium.py + spectrum.py + scattering.py + data/) | Standalone, usable outside sim |
| 24 | Wavelength in sim | Always-on effective properties; monochromatic = scalar passthrough | Per-photon arrays via vmap |
| 25 | PhotonRays.wavelengths | Optional — None for monochromatic, (n_rays,) when active | Structure fixed at setup, one-time rejit acceptable |
| 26 | Mie in simulation | Ported to scattering.py + MediumProperties fields; not called in sim loop | Available for external use, add to sim later |
| 27 | wavelength_dependency branch | Port everything except Mie sim integration | Wavelength features on top of refactored structure |
| 28 | MediumProperties vs QE | Medium = scattering/absorption/dispersion; QE curve = detector hardware | Physical distinction: water physics vs PMT physics |
| 29 | Cherenkov spectrum sampling | `wavelength/spectrum.py`, called from simulator after ray generation | SIREN not wavelength-aware; wavelength assignment is separate step |
| 30 | ParticleModel.material | Not stored; validated at load time, raises on mismatch | Authoritative material comes from DetectorGeometry.medium |
| 31 | Effective property flow | Computed in simulator, captured in propagation closure | wavelengths → eff_scatter/absorption/qe → propagation loop + sensor response |

---

## 7. Key Metrics

| Metric | Current | After refactor |
|---|---|---|
| simulation.py | 900 lines, 1 file | 4 files in simulation/ |
| generate.py | 1500 lines, 1 file | 3 files in sources/ |
| utils.py | 1970 lines | ~800 (ROOT I/O moved) |
| Propagation code | ~2100 (3 near-duplicates) | ~1600 (ABC + shared framework, 20-25% reduction) |
| Losses | 2 files (780+420) | 1 consolidated file |
| Optimization | 1 monolith (1000 lines) | 3 focused files |
| Duplicate functions | 8 groups | 0 |
| if/elif detector chains | 8+ across 6 files | 0 (registry) |
| Automated tests | 0 | Full pytest suite |
| Notebooks | 24 unorganized | 4 categories |
