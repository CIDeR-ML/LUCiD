"""
Test suite for string-telescope distance and snap kernels.

Tests skew_line_distance, snap_to_doms, string_z_range_check,
and candidate_doms_for_ray_string against numpy reference
implementations and known analytic results.

Run: python string/test_kernel.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_platform_name", "cpu")

from lucid.propagation.string.kernel import (
    skew_line_distance,
    snap_to_doms,
    string_z_range_check,
    candidate_doms_for_ray_string,
)


# ──────────────────────────────────────────────────────────────────────
# Reference implementations (numpy, no tricks)
# ──────────────────────────────────────────────────────────────────────

def ref_skew_distance(O, D, P, a):
    """Brute-force closest distance between ray and line using scipy."""
    w = O - P
    dd = np.dot(D, D)
    da = np.dot(D, a)
    wd = np.dot(w, D)
    wa = np.dot(w, a)
    denom = dd - da**2
    if abs(denom) < 1e-12:
        # parallel
        dist = np.sqrt(max(np.dot(w,w) - wa**2, 0.0))
        s = wa
        res = w - s * a
        t = -np.dot(res, D) / (dd + 1e-30)
        return dist, t, s
    t = (da * wa - wd) / denom
    s = (wa * dd - wd * da) / denom
    closest_ray = O + t * D
    closest_line = P + s * a
    dist = np.linalg.norm(closest_ray - closest_line)
    return dist, t, s


def ref_brute_nearest_dom(O, D, dom_positions):
    """Find nearest DOM to a ray by brute-force minimum distance."""
    best_dist = np.inf
    best_idx = -1
    best_t = 0.0
    for i, pos in enumerate(dom_positions):
        oc = O - pos
        d_norm = D / (np.linalg.norm(D) + 1e-30)
        t = -np.dot(oc, d_norm)
        closest = O + t * d_norm
        dist = np.linalg.norm(closest - pos)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
            best_t = t
    return best_idx, best_dist, best_t


# ──────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────

def assert_close(a, b, name, atol=1e-5, rtol=1e-5):
    a, b = float(a), float(b)
    if abs(a - b) > atol + rtol * abs(b):
        raise AssertionError(f"{name}: {a} != {b} (atol={atol}, rtol={rtol})")

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
        print(f"  FAIL  {name}: {e}")


# ──────────────────────────────────────────────────────────────────────
# skew_line_distance tests
# ──────────────────────────────────────────────────────────────────────

def test_horizontal_ray_vertical_string():
    """Horizontal ray at y=5, vertical string at origin → distance = 5."""
    O = jnp.array([0.0, 5.0, 0.0])
    D = jnp.array([1.0, 0.0, 0.0])
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    dist, t, s = skew_line_distance(O, D, P, a)
    assert_close(dist, 5.0, "distance")
    assert_close(t, 0.0, "t_ray")
    assert_close(s, 0.0, "s_string")


def test_offset_ray_vertical_string():
    """Ray at (10, 3, 7) going (0, 1, 0), string at (10, 0, 0) going z."""
    O = jnp.array([10.0, 3.0, 7.0])
    D = jnp.array([0.0, 1.0, 0.0])
    P = jnp.array([10.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    dist, t, s = skew_line_distance(O, D, P, a)
    ref_d, ref_t, ref_s = ref_skew_distance(
        np.array(O), np.array(D), np.array(P), np.array(a))
    assert_close(dist, ref_d, "distance")
    assert_close(t, ref_t, "t_ray")
    assert_close(s, ref_s, "s_string")


def test_diagonal_ray():
    """Diagonal ray vs vertical string — compare to numpy reference."""
    O = jnp.array([3.0, 4.0, 10.0])
    D = jnp.array([1.0, -1.0, -2.0])
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    dist, t, s = skew_line_distance(O, D, P, a)
    ref_d, ref_t, ref_s = ref_skew_distance(
        np.array(O), np.array(D), np.array(P), np.array(a))
    assert_close(dist, ref_d, "distance")
    assert_close(t, ref_t, "t_ray", atol=1e-4)
    assert_close(s, ref_s, "s_string", atol=1e-4)


def test_ray_through_string_center():
    """Ray passes directly through the string axis → distance = 0."""
    O = jnp.array([5.0, 0.0, 0.0])
    D = jnp.array([-1.0, 0.0, 0.0])
    P = jnp.array([0.0, 0.0, -10.0])
    a = jnp.array([0.0, 0.0, 1.0])
    dist, t, s = skew_line_distance(O, D, P, a)
    assert_close(dist, 0.0, "distance", atol=1e-4)


def test_parallel_ray_vertical_string():
    """Ray parallel to string (both along z), offset by 3m in y."""
    O = jnp.array([0.0, 3.0, 0.0])
    D = jnp.array([0.0, 0.0, 1.0])
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    dist, t, s = skew_line_distance(O, D, P, a)
    assert_close(dist, 3.0, "distance", atol=1e-3)
    assert_close(s, 0.0, "s_string", atol=1e-3)


def test_parallel_ray_offset_anchor():
    """Parallel ray at (2, 1, 5), string at (2, 1, 0) going z."""
    O = jnp.array([2.0, 1.0, 5.0])
    D = jnp.array([0.0, 0.0, -1.0])
    P = jnp.array([2.0, 1.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    dist, t, s = skew_line_distance(O, D, P, a)
    assert_close(dist, 0.0, "distance", atol=1e-3)
    assert_close(s, 5.0, "s_string", atol=1e-3)


def test_tilted_string():
    """Non-vertical string axis — compare to numpy reference."""
    O = jnp.array([5.0, 0.0, 0.0])
    D = jnp.array([0.0, 1.0, 0.0])
    tilt = np.array([0.0, 0.1, 1.0])
    a_np = tilt / np.linalg.norm(tilt)
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array(a_np)
    dist, t, s = skew_line_distance(O, D, P, a)
    ref_d, ref_t, ref_s = ref_skew_distance(
        np.array(O), np.array(D), np.array(P), a_np)
    assert_close(dist, ref_d, "distance", atol=1e-4)
    assert_close(t, ref_t, "t_ray", atol=1e-4)
    assert_close(s, ref_s, "s_string", atol=1e-4)


def test_unnormalized_ray_direction():
    """Ray direction not unit-length — should still give correct distance."""
    O = jnp.array([0.0, 5.0, 0.0])
    D = jnp.array([100.0, 0.0, 0.0])  # large magnitude
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    dist, t, s = skew_line_distance(O, D, P, a)
    assert_close(dist, 5.0, "distance", atol=1e-4)


def test_vmap_over_rays():
    """Vectorize over a batch of rays against one string."""
    origins = jnp.array([
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
    ])
    direction = jnp.array([1.0, 0.0, 0.0])
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])

    batched = jax.vmap(skew_line_distance, in_axes=(0, None, None, None))
    dists, ts, ss = batched(origins, direction, P, a)
    np.testing.assert_allclose(np.array(dists), [1.0, 2.0, 3.0], atol=1e-5)


def test_vmap_over_strings():
    """Vectorize over multiple strings for one ray."""
    O = jnp.array([0.0, 0.0, 0.0])
    D = jnp.array([1.0, 0.0, 0.0])
    anchors = jnp.array([
        [0.0, 1.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, -3.0, 0.0],
    ])
    a = jnp.array([0.0, 0.0, 1.0])

    batched = jax.vmap(skew_line_distance, in_axes=(None, None, 0, None))
    dists, ts, ss = batched(O, D, anchors, a)
    np.testing.assert_allclose(np.array(dists), [1.0, 5.0, 3.0], atol=1e-5)


def test_gradient_wrt_origin():
    """Distance should have smooth gradients wrt ray origin."""
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    D = jnp.array([1.0, 0.0, 0.0])

    def dist_fn(O):
        d, _, _ = skew_line_distance(O, D, P, a)
        return d

    O = jnp.array([0.0, 5.0, 0.0])
    grad = jax.grad(dist_fn)(O)
    # distance = |O.y| = 5, grad wrt O.y should be +1 (since O.y > 0)
    assert_close(grad[1], 1.0, "grad_y", atol=1e-4)
    assert_close(grad[0], 0.0, "grad_x", atol=1e-4)
    assert_close(grad[2], 0.0, "grad_z", atol=1e-4)


def test_gradient_near_parallel():
    """Gradients should not explode near-parallel case."""
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    D = jnp.array([0.0, 1e-6, 1.0])  # nearly parallel

    def dist_fn(O):
        d, _, _ = skew_line_distance(O, D, P, a)
        return d

    O = jnp.array([3.0, 0.0, 0.0])
    grad = jax.grad(dist_fn)(O)
    assert np.all(np.isfinite(np.array(grad))), f"non-finite gradient: {grad}"


# ──────────────────────────────────────────────────────────────────────
# snap_to_doms tests
# ──────────────────────────────────────────────────────────────────────

def test_snap_basic():
    """Snap to bracket pair for s* between two DOMs."""
    offsets = jnp.array([0.0, 17.0, 34.0, 51.0, 68.0, jnp.inf, jnp.inf])
    indices = snap_to_doms(25.0, offsets, 2)
    assert list(np.array(indices)) == [1, 2], f"expected [1,2], got {indices}"


def test_snap_at_bottom():
    """s* below bottom DOM — should clamp to [0, 1]."""
    offsets = jnp.array([0.0, 17.0, 34.0, 51.0, jnp.inf])
    indices = snap_to_doms(-5.0, offsets, 2)
    assert list(np.array(indices)) == [0, 1], f"expected [0,1], got {indices}"


def test_snap_at_top():
    """s* above top DOM — indices may include padding slots.

    With offsets=[0, 17, 34, 51, inf], s*=100 searchsorted gives k_right=4.
    Snap returns [3, 4] where index 4 is a padding slot (+inf). This is
    correct: downstream masking via dom_global_ids (which has -1 at padding
    positions) filters it out. The kernel does NOT know n_doms.
    """
    offsets = jnp.array([0.0, 17.0, 34.0, 51.0, jnp.inf])
    indices = snap_to_doms(100.0, offsets, 2)
    idx_list = list(np.array(indices))
    assert 3 in idx_list, f"expected top real DOM (idx 3) in {idx_list}"


def test_snap_exact_dom_position():
    """s* exactly on a DOM — bracket should include it."""
    offsets = jnp.array([0.0, 17.0, 34.0, 51.0, jnp.inf])
    indices = snap_to_doms(17.0, offsets, 2)
    idx_list = list(np.array(indices))
    assert 1 in idx_list, f"expected index 1 in {idx_list}"


def test_snap_4_doms():
    """4-DOM snap window for curved-string mode."""
    offsets = jnp.array([0.0, 9.0, 18.0, 27.0, 36.0, 45.0, jnp.inf, jnp.inf])
    indices = snap_to_doms(20.0, offsets, 4)
    idx_list = list(np.array(indices))
    assert len(idx_list) == 4
    assert 2 in idx_list, f"expected index 2 (18.0) in {idx_list}"
    assert 3 in idx_list, f"expected index 3 (27.0) in {idx_list}"


def test_snap_single_dom():
    """String with only 1 real DOM (rest are inf)."""
    offsets = jnp.array([0.0, jnp.inf, jnp.inf, jnp.inf])
    indices = snap_to_doms(0.0, offsets, 2)
    idx_list = list(np.array(indices))
    assert 0 in idx_list


# ──────────────────────────────────────────────────────────────────────
# string_z_range_check tests
# ──────────────────────────────────────────────────────────────────────

def test_z_range_inside():
    assert bool(string_z_range_check(50.0, 0.0, 100.0, 1.0))

def test_z_range_outside_below():
    assert not bool(string_z_range_check(-10.0, 0.0, 100.0, 1.0))

def test_z_range_within_padding():
    assert bool(string_z_range_check(-0.5, 0.0, 100.0, 1.0))

def test_z_range_outside_above():
    assert not bool(string_z_range_check(110.0, 0.0, 100.0, 1.0))


# ──────────────────────────────────────────────────────────────────────
# candidate_doms_for_ray_string integration test
# ──────────────────────────────────────────────────────────────────────

def test_candidate_doms_hit():
    """Ray passes within sensor_radius of a vertical string."""
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    n_doms = 5
    dz = 17.0
    s_offsets = jnp.array([k * dz for k in range(n_doms)] + [jnp.inf] * 3)
    global_ids = jnp.array([100 + k for k in range(n_doms)] + [-1] * 3)

    O = jnp.array([0.0, 0.1, 30.0])   # 0.1m from string, z=30 (between DOM 1 and 2)
    D = jnp.array([1.0, 0.0, 0.0])
    r_filter = 0.5

    ids, dist, t = candidate_doms_for_ray_string(
        O, D, P, a, s_offsets, global_ids,
        s_min=0.0, s_max=(n_doms - 1) * dz,
        r_filter=r_filter, n_dom_snap=2,
    )
    ids_list = list(np.array(ids))
    assert ids_list[0] != -1, f"expected hit, got {ids_list}"
    assert 101 in ids_list or 102 in ids_list, f"expected DOM 1 or 2, got {ids_list}"


def test_candidate_doms_miss_distance():
    """Ray too far from string — should return all -1."""
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    s_offsets = jnp.array([0.0, 17.0, 34.0, jnp.inf])
    global_ids = jnp.array([100, 101, 102, -1])

    O = jnp.array([0.0, 10.0, 20.0])   # 10m from string
    D = jnp.array([1.0, 0.0, 0.0])
    r_filter = 0.5

    ids, dist, t = candidate_doms_for_ray_string(
        O, D, P, a, s_offsets, global_ids,
        s_min=0.0, s_max=34.0,
        r_filter=r_filter, n_dom_snap=2,
    )
    ids_list = list(np.array(ids))
    assert all(i == -1 for i in ids_list), f"expected all -1, got {ids_list}"


def test_candidate_doms_miss_z_range():
    """Ray close to string axis but z is out of DOM range."""
    P = jnp.array([0.0, 0.0, 0.0])
    a = jnp.array([0.0, 0.0, 1.0])
    s_offsets = jnp.array([0.0, 17.0, 34.0, jnp.inf])
    global_ids = jnp.array([100, 101, 102, -1])

    O = jnp.array([0.0, 0.1, 500.0])   # 0.1m away, but z=500 (DOMs end at z=34)
    D = jnp.array([1.0, 0.0, 0.0])
    r_filter = 0.5

    ids, dist, t = candidate_doms_for_ray_string(
        O, D, P, a, s_offsets, global_ids,
        s_min=0.0, s_max=34.0,
        r_filter=r_filter, n_dom_snap=2,
    )
    ids_list = list(np.array(ids))
    assert all(i == -1 for i in ids_list), f"expected all -1 (z out of range), got {ids_list}"


def test_candidate_doms_matches_brute_force():
    """Verify snap finds the same DOM as brute-force nearest on a toy geometry."""
    P = jnp.array([10.0, 20.0, -100.0])
    a = jnp.array([0.0, 0.0, 1.0])
    n_doms = 10
    dz = 17.0
    dom_positions = np.array([[10.0, 20.0, -100.0 + k * dz] for k in range(n_doms)])
    s_offsets = jnp.array([k * dz for k in range(n_doms)] + [jnp.inf] * 2)
    global_ids = jnp.array(list(range(n_doms)) + [-1] * 2)

    rng = np.random.RandomState(42)
    for trial in range(50):
        O_np = rng.randn(3) * 50 + np.array([10.0, 20.0, 0.0])
        D_np = rng.randn(3)
        D_np = D_np / (np.linalg.norm(D_np) + 1e-10)

        bf_idx, bf_dist, bf_t = ref_brute_nearest_dom(O_np, D_np, dom_positions)

        ids, dist, t = candidate_doms_for_ray_string(
            jnp.array(O_np), jnp.array(D_np), P, a,
            s_offsets, global_ids,
            s_min=0.0, s_max=(n_doms - 1) * dz,
            r_filter=1000.0,  # wide filter so everything passes
            n_dom_snap=4,
        )
        ids_list = list(np.array(ids))
        assert bf_idx in ids_list, (
            f"trial {trial}: brute-force nearest DOM {bf_idx} not in snap candidates {ids_list}"
        )


# ──────────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== skew_line_distance ===")
    run_test(test_horizontal_ray_vertical_string, "horizontal_ray_vertical_string")
    run_test(test_offset_ray_vertical_string, "offset_ray_vertical_string")
    run_test(test_diagonal_ray, "diagonal_ray")
    run_test(test_ray_through_string_center, "ray_through_string_center")
    run_test(test_parallel_ray_vertical_string, "parallel_ray_vertical_string")
    run_test(test_parallel_ray_offset_anchor, "parallel_ray_offset_anchor")
    run_test(test_tilted_string, "tilted_string")
    run_test(test_unnormalized_ray_direction, "unnormalized_ray_direction")
    run_test(test_vmap_over_rays, "vmap_over_rays")
    run_test(test_vmap_over_strings, "vmap_over_strings")
    run_test(test_gradient_wrt_origin, "gradient_wrt_origin")
    run_test(test_gradient_near_parallel, "gradient_near_parallel")

    print("\n=== snap_to_doms ===")
    run_test(test_snap_basic, "snap_basic")
    run_test(test_snap_at_bottom, "snap_at_bottom")
    run_test(test_snap_at_top, "snap_at_top")
    run_test(test_snap_exact_dom_position, "snap_exact_dom_position")
    run_test(test_snap_4_doms, "snap_4_doms")
    run_test(test_snap_single_dom, "snap_single_dom")

    print("\n=== string_z_range_check ===")
    run_test(test_z_range_inside, "z_range_inside")
    run_test(test_z_range_outside_below, "z_range_outside_below")
    run_test(test_z_range_within_padding, "z_range_within_padding")
    run_test(test_z_range_outside_above, "z_range_outside_above")

    print("\n=== candidate_doms_for_ray_string ===")
    run_test(test_candidate_doms_hit, "candidate_doms_hit")
    run_test(test_candidate_doms_miss_distance, "candidate_doms_miss_distance")
    run_test(test_candidate_doms_miss_z_range, "candidate_doms_miss_z_range")
    run_test(test_candidate_doms_matches_brute_force, "candidate_doms_matches_brute_force")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
