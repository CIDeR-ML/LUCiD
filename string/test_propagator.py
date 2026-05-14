"""
Test suite for string propagator — from config load through ray propagation.

Tests:
  1. StringTelescope loads from NPZ correctly
  2. String propagator builds without error
  3. Propagator output dict has correct shapes
  4. Ray near a string produces non-zero weights
  5. Ray far from all strings produces zero weights
  6. Batch of rays produces correct output shape
  7. Gradients flow through the propagator

Run: python string/test_propagator.py
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


def get_detector():
    return StringTelescope.from_npz(SIMPLE_NPZ)


def get_propagator(det=None):
    if det is None:
        det = get_detector()
    sp = jnp.array(det.all_points)
    return create_string_propagator(
        det, sp, det.S_radius,
        temperature=0.2,
        lambda_abs=100.0,
        lambda_scat=30.0,
    ), det


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

def test_load_detector():
    det = get_detector()
    assert det.n_str == 78, f"expected 78 strings, got {det.n_str}"
    assert det.n_sensors == 4680, f"expected 4680 DOMs, got {det.n_sensors}"
    assert det.all_points.shape == (4680, 3)
    assert det.S_radius == 0.165
    print(f"    envelope: R={det.envelope_radius:.0f}m, "
          f"z=[{det.envelope_z_min:.0f}, {det.envelope_z_max:.0f}]")


def test_build_propagator():
    prop, det = get_propagator()
    assert callable(prop)
    print(f"    K_min={prop.sizing.K_min}, max_dom/seg={prop.sizing.max_dom_per_segment}")


def test_output_shape():
    prop, det = get_propagator()
    n_rays = 4
    origins = jnp.tile(jnp.array([0.0, 0.1, -1950.0]), (n_rays, 1))
    directions = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n_rays, 1))

    result = prop(origins, directions)
    max_dom = prop.sizing.max_dom_per_segment

    assert result['sensor_weights'].shape[1] == n_rays, \
        f"weights axis 1: {result['sensor_weights'].shape}"
    assert result['positions'].shape == (n_rays, 3), \
        f"positions: {result['positions'].shape}"
    assert result['normals'].shape == (n_rays, 3)
    assert result['inside_sensor'].shape[1] == n_rays
    print(f"    max_dom_per_segment={max_dom}, "
          f"sensor_weights shape={result['sensor_weights'].shape}")


def test_ray_near_string_hits():
    """Ray 0.1m from string 0, at the z of a specific DOM → non-zero weight.

    Inter-DOM spacing is 17m, sensor_radius is 0.165m. The ray must be
    placed AT the z of a DOM, not between DOMs, to be within sensor_radius.
    """
    prop, det = get_propagator()
    dom_k = 30
    z_dom = det.string_anchors[0, 2] + det.dom_s_offsets[0, dom_k]
    xy = det.string_anchors[0, :2]

    origin = jnp.array([xy[0], xy[1] + 0.1, z_dom])
    direction = jnp.array([1.0, 0.0, 0.0])

    result = prop(origin[None, :], direction[None, :])
    weights = result['sensor_weights']
    max_weight = float(jnp.max(weights))
    assert max_weight > 0.01, f"expected non-zero weight near DOM, got max={max_weight}"
    print(f"    max weight near string 0 DOM {dom_k}: {max_weight:.4f}")


def test_ray_far_from_strings():
    """Ray far outside the string array should produce zero weights."""
    prop, det = get_propagator()
    origin = jnp.array([[5000.0, 5000.0, -1950.0]])
    direction = jnp.array([[1.0, 0.0, 0.0]])

    result = prop(origin, direction)
    max_weight = float(jnp.max(result['sensor_weights']))
    assert max_weight < 1e-6, f"expected ~zero weight far from strings, got {max_weight}"


def test_gradient_flows():
    """Gradient of total weight sum wrt ray origin should be finite and non-zero."""
    prop, det = get_propagator()
    dom_k = 30
    z_dom = det.string_anchors[0, 2] + det.dom_s_offsets[0, dom_k]
    xy = det.string_anchors[0, :2]

    direction = jnp.array([[1.0, 0.0, 0.0]])

    def total_weight(origin_y):
        origin = jnp.array([[xy[0], origin_y, z_dom]])
        result = prop(origin, direction)
        return jnp.sum(result['sensor_weights'])

    y0 = xy[1] + 0.1
    grad = jax.grad(total_weight)(y0)
    assert jnp.isfinite(grad), f"gradient is not finite: {grad}"
    assert jnp.abs(grad) > 1e-6, f"gradient is ~zero: {grad}"
    print(f"    d(total_weight)/d(y) at y={y0:.2f}: {float(grad):.6f}")


if __name__ == "__main__":
    if not os.path.exists(SIMPLE_NPZ):
        print(f"Run 'python string/gen_icecube86.py' first to generate {SIMPLE_NPZ}")
        sys.exit(1)

    print("=== StringTelescope ===")
    run_test(test_load_detector, "load_detector")

    print("\n=== String Propagator ===")
    run_test(test_build_propagator, "build_propagator")
    run_test(test_output_shape, "output_shape")
    run_test(test_ray_near_string_hits, "ray_near_string_hits")
    run_test(test_ray_far_from_strings, "ray_far_from_strings")
    run_test(test_gradient_flows, "gradient_flows")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
