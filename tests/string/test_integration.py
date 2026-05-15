"""
Integration test: string propagator inside the K-iteration loop.

Tests:
  1. DetectorGeometry.from_config dispatches to StringTelescope + string propagator
  2. Propagator output feeds into the photon_step iteration correctly
  3. K-iteration loop with scatter/absorb produces finite outputs
  4. Cherenkov-like photons near a string produce detectable hits
  5. Visualization produces a plotly figure

Run: python string/test_integration.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.propagator import create_string_propagator
from lucid.simulation.photon_step import photon_iteration_update_factors_safe
from lucid.simulation.optics import normalize

passed = 0
failed = 0
errors = []

def run_test(fn, name):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except Exception as e:
        failed += 1
        errors.append((name, e))
        import traceback
        print(f"  FAIL  {name}:")
        traceback.print_exc()


CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
SIMPLE_NPZ = os.path.join(CONFIG_DIR, "icecube86_simple.npz")
FULL_NPZ = os.path.join(CONFIG_DIR, "icecube86_full.npz")
GEOM_JSON = os.path.join(CONFIG_DIR, "IceCube86_simple_geom_config.json")


# ──────────────────────────────────────────────────────────────────────
# Test: DetectorGeometry.from_config dispatch
# ──────────────────────────────────────────────────────────────────────

def test_detector_geometry_dispatch():
    """DetectorGeometry.from_config should create a StringTelescope and string propagator."""
    from lucid.geometry.detector_geometry import DetectorGeometry

    det_geom = DetectorGeometry.from_config(
        GEOM_JSON,
        temperature=0.2,
        detector_type='string',
        lambda_abs=100.0,
        lambda_scat=30.0,
    )

    assert isinstance(det_geom.detector, StringTelescope), \
        f"expected StringTelescope, got {type(det_geom.detector)}"
    assert det_geom.num_sensors == 4680
    assert det_geom.propagator is not None
    assert callable(det_geom.propagator)
    print(f"    detector: {det_geom.detector_type}, "
          f"{det_geom.num_sensors} sensors, "
          f"speed_of_light={det_geom.speed_of_light:.4f} m/ns")


# ──────────────────────────────────────────────────────────────────────
# Test: K-iteration propagation loop
# ──────────────────────────────────────────────────────────────────────

def test_k_iteration_loop():
    """Run K iterations of propagate + photon_step. Verify finite outputs."""
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=0.2)

    n_rays = 50
    K = 5
    SPEED = 0.2198  # m/ns in water

    # Place photons near string 0, aimed at the DOMs
    str_xy = det.string_anchors[0, :2]
    dom_z = det.string_anchors[0, 2] + det.dom_s_offsets[0, 30]

    rng = np.random.RandomState(42)
    origins_np = np.zeros((n_rays, 3))
    origins_np[:, 0] = str_xy[0] + rng.uniform(-0.1, 0.1, n_rays)
    origins_np[:, 1] = str_xy[1] + rng.uniform(-0.1, 0.1, n_rays)
    origins_np[:, 2] = dom_z + rng.uniform(-5, 5, n_rays)

    dirs_np = rng.randn(n_rays, 3)
    dirs_np /= np.linalg.norm(dirs_np, axis=1, keepdims=True)

    positions = jnp.array(origins_np)
    directions = jnp.array(dirs_np)
    times = jnp.zeros(n_rays)
    survival = jnp.ones(n_rays)

    scatter_lengths = jnp.full(n_rays, 30.0)
    absorption_lengths = jnp.full(n_rays, 100.0)
    wall_reflection_rate = jnp.float32(0.0)
    sensor_reflection_rate = jnp.float32(0.0)

    key = jax.random.PRNGKey(0)
    total_hits = 0

    for k in range(K):
        key, subkey = jax.random.split(key)
        rng_keys = jax.random.split(subkey, n_rays)

        result = prop(positions, directions)
        inside_sensor = result['inside_sensor']
        hit_sensor = jnp.max(inside_sensor, axis=0)
        hit_positions = result['positions']
        normals = result['normals']
        surface_distances = jnp.linalg.norm(hit_positions - positions, axis=1) - 1e-6

        (new_positions, new_directions, new_times,
         detect_probs, reflection_attenuations,
         continuing_factors) = jax.vmap(
            photon_iteration_update_factors_safe,
            in_axes=(0, 0, 0, 0, 0, 0, None, None, 0, 0, 0, None)
        )(positions, directions, times,
          surface_distances, normals,
          scatter_lengths, wall_reflection_rate, sensor_reflection_rate,
          absorption_lengths,
          hit_sensor, rng_keys, SPEED)

        inside_det = det.bounds_check(np.array(new_positions))
        continuing_factors = jnp.where(jnp.array(inside_det), continuing_factors, 0.0)

        n_hits_this_k = int(jnp.sum(hit_sensor))
        total_hits += n_hits_this_k

        survival = survival * continuing_factors
        positions = new_positions
        directions = new_directions
        times = new_times

    # Verify finite
    assert jnp.all(jnp.isfinite(positions)), "positions not finite"
    assert jnp.all(jnp.isfinite(directions)), "directions not finite"
    assert jnp.all(jnp.isfinite(survival)), "survival not finite"

    print(f"    K={K}, n_rays={n_rays}, total_hits_across_K={total_hits}")
    print(f"    survival: mean={float(survival.mean()):.4f}, "
          f"min={float(survival.min()):.4f}, max={float(survival.max()):.4f}")


# ──────────────────────────────────────────────────────────────────────
# Test: Cherenkov-like photons produce hits
# ──────────────────────────────────────────────────────────────────────

def test_cherenkov_hits():
    """Photons aimed directly at a DOM should produce hits.

    Place origin 1m from a DOM, direction pointing AT the DOM.
    Propagate 1 step. The inside_sensor flag should fire.
    """
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=0.2)

    dom_idx = 30  # string 0, DOM 30
    dom_pos = sp[dom_idx]

    # Aim at DOM from 1m away along x
    origin = dom_pos + jnp.array([1.0, 0.0, 0.0])
    direction = dom_pos - origin
    direction = direction / jnp.linalg.norm(direction)

    result = prop(origin[None, :], direction[None, :])
    inside = result['inside_sensor']
    hit = bool(jnp.any(inside))
    max_weight = float(jnp.max(result['sensor_weights']))

    assert hit, f"photon aimed at DOM should produce a hit; weights={max_weight}"
    print(f"    aimed at DOM {dom_idx}: hit={hit}, max_weight={max_weight:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Test: Full geometry with DeepCore (non-uniform z)
# ──────────────────────────────────────────────────────────────────────

def test_full_geometry_loads():
    """86-string geometry with DeepCore non-uniform z should load and propagate."""
    det = StringTelescope.from_npz(FULL_NPZ)
    assert det.n_str == 86
    assert det.n_sensors == 5160

    sp = jnp.array(det.all_points)
    prop = create_string_propagator(det, sp, det.S_radius, temperature=0.2)

    # Quick smoke: single ray
    origin = jnp.array([[0.0, 0.1, -2000.0]])
    direction = jnp.array([[1.0, 0.0, 0.0]])
    result = prop(origin, direction)
    assert result['sensor_weights'].shape[1] == 1
    print(f"    86 strings, {det.n_sensors} DOMs, "
          f"DeepCore curv_max={det.string_curv.max():.2f}m")


# ──────────────────────────────────────────────────────────────────────
# Test: Visualization
# ──────────────────────────────────────────────────────────────────────

def test_visualization():
    """visualize_geometry_wireframe should return a plotly Figure."""
    det = StringTelescope.from_npz(SIMPLE_NPZ)
    fig = det.visualize_geometry_wireframe(show_sensors=True)
    import plotly.graph_objects as go
    assert isinstance(fig, go.Figure), f"expected plotly Figure, got {type(fig)}"
    assert len(fig.data) > 0, "figure should have traces"
    print(f"    figure has {len(fig.data)} traces")


if __name__ == "__main__":
    print("=== DetectorGeometry dispatch ===")
    run_test(test_detector_geometry_dispatch, "detector_geometry_dispatch")

    print("\n=== K-iteration loop ===")
    run_test(test_k_iteration_loop, "k_iteration_loop")

    print("\n=== Cherenkov hits ===")
    run_test(test_cherenkov_hits, "cherenkov_hits")

    print("\n=== Full geometry (DeepCore) ===")
    run_test(test_full_geometry_loads, "full_geometry_loads")

    print("\n=== Visualization ===")
    run_test(test_visualization, "visualization")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
