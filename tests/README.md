# LUCiD Test Suite

## Running tests

```bash
# Fast only — pure math, no JIT (~10s)
pytest tests/ -m "not slow"

# All tests (~15-20 min)
pytest tests/

# Slow tests only
pytest tests/ -m slow
```

## Test categories

### Fast (~150 tests, <10s) — pure math, no JIT

These run without building detectors or compiling propagators.

| File | Tests | What it covers |
|------|-------|----------------|
| test_imports.py | 1 (52 parametrized) | All lucid submodules import cleanly |
| test_params.py | 8 | DetectorParams, ParticleParams, normalize/denormalize, bounds |
| test_utils.py | 10 | Utility functions (spherical coords, smearing) |
| test_optics.py | 26 | Optics: normalize, reflection, scatter direction |
| test_optics_physics.py | 21 | Physics: law of reflection, orthonormal frames, Fresnel |
| test_losses.py | 7 | Poisson NLL, energy loss, segment logsumexp |
| test_losses_physics.py | 11 | Loss gradients and numerical stability |
| test_photon_step.py | 6 | Photon iteration functions (sample, update_factors) |
| test_photon_step_physics.py | 15 | Absorption, detection probability, TOF, step gradients |
| test_sensor_response.py | 6 | make_hits_simulation/data/likelihood aggregation |
| test_sensor_response_physics.py | 13 | Charge conservation, soft-min timing, differentiability |
| test_pipeline_types.py | 13 | JAX pytree compat, lax.scan, vmap, jit |
| test_wavelength_physics.py | 10 | Rayleigh scattering, Henyey-Greenstein, Mie phase functions |
| test_integration.py | 18 | Optics -> photon_step -> sensor_response chain (unit-level) |
| test_wavelength.py | 18 | Medium properties, QE curve loading, Cherenkov spectrum |

### Slow (~150 tests, marked `@pytest.mark.slow`) — JIT compilation required

These build detectors, compile propagators, or run full simulations.

**Geometry & propagator (3-30s each):**

| File | Tests | What it covers |
|------|-------|----------------|
| test_containers.py | 11 | SimConfig, DetectorGeometry.from_config |
| test_geometry.py | 13 | Detector construction (cylinder, sphere, box), sensor placement |
| test_registry.py | 13 | Detector registry, factory lookup, JIT compat |
| test_ray_intersection.py | 27 | Ray-geometry intersection for all detector types |
| test_sensor_map_validation.py | 13 | Grid auto-derivation, overcrowding validation |
| test_propagator_output.py | 9 | Propagator output structure (weights, positions, normals) |
| test_shared_propagator.py | 5 | Shared propagator equivalence across geometries |
| test_shared_propagator_differentiability.py | 8 | Gradient flow through shared propagator |
| test_propagation_differentiability.py | 15 | Gradient flow through propagation + photon step |

**Full simulation (30s+ each):**

| File | Tests | What it covers |
|------|-------|----------------|
| test_sk_like_integration.py | 11 | SK_like simulator: laser, isotropic, SIREN track; K convergence; gradients |
| test_wavelength_integration.py | 13 | Wavelength-mode simulation: scalar vs wavelength; QE weighting; physics consistency |

### Broken / disabled

Old debugging scripts with no `test_` functions:
test_all_fixes.py, test_combined_real_sensors.py, test_multibounce_jacobian.py,
test_normalize_fix.py, test_s1_bounce.py, test_s1b_hk.py, test_s2_tangent.py,
test_s3_combined.py, test_tangent_gradients.py

## Fixtures (conftest.py)

| Fixture | Scope | What it provides |
|---------|-------|-----------------|
| `key` | session | `jax.random.PRNGKey(42)` |
| `small_cylinder_config` | session | Path to WCTE_geom_config.json |
| `cylinder_detector` | session | Pre-built WCTE detector (cached) |
| `fixed_flat_hits` | session | Synthetic sensor hit data for unit tests |

## Config dependencies

Tests reference configs in `config/`:
- Geometry: `WCTE_geom_config.json` (small, fast), `SK_like_geom_config.json` (large, slow)
- Physics: `SK_physics_config.json` (composable format with medium_model + qe_curve references)
- Materials: `materials/water.json`
- PMT curves: `pmt/SK_QE.json`, `pmt/HK_QE.json`
