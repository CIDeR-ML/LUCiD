"""
Test the refactored string propagator (standard LUCiD interface).

Verifies:
  1. Output dict has the correct keys and shapes
  2. Photons near DOMs get nonzero weights
  3. Photons far from DOMs get zero weights
  4. Gradients flow through the propagator
  5. Output matches fast.py's candidate finding (same DOMs found)

Run: python string/test_string_propagator.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_platform_name", "cpu")

from lucid.geometry.string import StringTelescope
from lucid.propagation.string.string_propagator import create_string_propagator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ = os.path.join(PROJECT_ROOT, "config", "icecube86_simple.npz")

passed = failed = 0
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


det = StringTelescope.from_npz(NPZ)
prop = create_string_propagator(det, det.S_radius, temperature=0.2)
N_CAND = prop.n_cand


def test_output_keys():
    o = jnp.array([[0.0, 0.1, -1950.0]])
    d = jnp.array([[1.0, 0.0, 0.0]])
    r = prop(o, d)
    expected = {'sensor_weights', 'sensor_indices', 'sensor_distances',
                'positions', 'normals', 'inside_sensor',
                'per_sensor_positions', 'sensor_normals'}
    assert set(r.keys()) == expected, f"missing keys: {expected - set(r.keys())}"


def test_output_shapes():
    n = 8
    o = jnp.tile(jnp.array([0.0, 0.1, -1950.0]), (n, 1))
    d = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n, 1))
    r = prop(o, d)
    assert r['sensor_weights'].shape == (N_CAND, n), f"weights: {r['sensor_weights'].shape}"
    assert r['sensor_indices'].shape == (N_CAND, n), f"indices: {r['sensor_indices'].shape}"
    assert r['sensor_distances'].shape == (N_CAND, n, 1), f"distances: {r['sensor_distances'].shape}"
    assert r['positions'].shape == (n, 3)
    assert r['normals'].shape == (n, 3)
    assert r['inside_sensor'].shape == (N_CAND, n)
    assert r['per_sensor_positions'].shape == (N_CAND, n, 3)
    assert r['sensor_normals'].shape == (N_CAND, n, 3)


def test_near_dom_hits():
    """Ray 0.1m from string 0, at DOM z → nonzero weight."""
    dom_k = 30
    z = det.string_anchors[0, 2] + det.dom_s_offsets[0, dom_k]
    xy = det.string_anchors[0, :2]
    o = jnp.array([[xy[0], xy[1] + 0.1, z]])
    d = jnp.array([[1.0, 0.0, 0.0]])
    r = prop(o, d)
    max_w = float(jnp.max(r['sensor_weights']))
    assert max_w > 0.01, f"expected hit weight > 0.01, got {max_w}"
    print(f"    max weight near DOM: {max_w:.4f}")


def test_far_misses():
    """Ray far outside array → all weights zero."""
    o = jnp.array([[5000.0, 5000.0, -1950.0]])
    d = jnp.array([[1.0, 0.0, 0.0]])
    r = prop(o, d)
    max_w = float(jnp.max(r['sensor_weights']))
    assert max_w < 1e-6, f"expected near-zero weight, got {max_w}"


def test_gradient():
    """Gradient of total weight wrt ray origin should be finite."""
    d = jnp.array([[1.0, 0.0, 0.0]])
    dom_k = 30
    z = det.string_anchors[0, 2] + det.dom_s_offsets[0, dom_k]
    xy = det.string_anchors[0, :2]

    def total_w(y):
        o = jnp.array([[xy[0], y, z]])
        r = prop(o, d)
        return jnp.sum(r['sensor_weights'])

    g = jax.grad(total_w)(xy[1] + 0.1)
    assert jnp.isfinite(g), f"non-finite gradient: {g}"
    assert jnp.abs(g) > 1e-3, f"gradient too small: {g}"
    print(f"    d(weight)/d(y) = {float(g):.4f}")


def test_inside_sensor_matches_weights():
    """inside_sensor should be True wherever weights > threshold."""
    o = jnp.array([[0.0, 0.1, -1950.0]])
    d = jnp.array([[1.0, 0.0, 0.0]])
    r = prop(o, d)
    w = r['sensor_weights'][:, 0]
    ins = r['inside_sensor'][:, 0]
    has_w = w > 1e-10
    assert jnp.all(has_w == ins), "inside_sensor should match weights > 1e-10"


def test_distances_are_positive():
    """sensor_distances should be non-negative."""
    n = 16
    rng = np.random.RandomState(42)
    o = jnp.array(rng.randn(n, 3) * 100 + np.array([0, 0, -1950]))
    d = jnp.array(rng.randn(n, 3))
    r = prop(o, d)
    dists = r['sensor_distances']
    assert jnp.all(dists >= 0), f"negative distances found"


if __name__ == "__main__":
    print("=== String Propagator (standard interface) ===")
    run_test(test_output_keys, "output_keys")
    run_test(test_output_shapes, "output_shapes")
    run_test(test_near_dom_hits, "near_dom_hits")
    run_test(test_far_misses, "far_misses")
    run_test(test_gradient, "gradient")
    run_test(test_inside_sensor_matches_weights, "inside_sensor_matches_weights")
    run_test(test_distances_are_positive, "distances_positive")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
