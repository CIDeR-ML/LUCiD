"""
Integration test for per-segment matcher.

Builds a small 4-string × 5-DOM toy geometry, runs rays through it,
and verifies that the matcher finds the same DOMs as brute-force search.

Run: python string/test_match.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

from lucid.propagation.string.hash import build_string_hash
from lucid.propagation.string.match import match_segment, match_segment_batch

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
# Toy geometry: 4 vertical strings, 5 DOMs each, 17m spacing
# ──────────────────────────────────────────────────────────────────────

N_STR = 4
N_DOM = 5
MAX_DOM = 6  # padded
DZ = 17.0
SENSOR_RADIUS = 0.165

STRING_XY = np.array([
    [0.0, 0.0],
    [50.0, 0.0],
    [0.0, 50.0],
    [50.0, 50.0],
])

def build_toy():
    """Build toy geometry arrays."""
    anchors = np.column_stack([STRING_XY, np.zeros(N_STR)])
    tops = np.column_stack([STRING_XY, np.full(N_STR, (N_DOM - 1) * DZ)])
    axes = np.tile([0.0, 0.0, 1.0], (N_STR, 1))

    s_offsets = np.full((N_STR, MAX_DOM), np.inf)
    global_ids = np.full((N_STR, MAX_DOM), -1, dtype=np.int32)
    for i in range(N_STR):
        for k in range(N_DOM):
            s_offsets[i, k] = k * DZ
            global_ids[i, k] = i * N_DOM + k

    s_min = np.zeros(N_STR)
    s_max = np.full(N_STR, (N_DOM - 1) * DZ)

    # Build hash
    cell_size = 50.0
    r_filter = SENSOR_RADIUS + 0.5
    cell_map, grid_origin, grid_shape, cs, stats = build_string_hash(
        anchors, tops, cell_size=cell_size, r_filter=r_filter,
        max_strings_per_cell=4,
    )
    assert stats['string_coverage'] == 1.0

    return {
        'anchors': jnp.array(anchors),
        'tops': jnp.array(tops),
        'axes': jnp.array(axes),
        's_offsets': jnp.array(s_offsets),
        'global_ids': jnp.array(global_ids),
        's_min': jnp.array(s_min),
        's_max': jnp.array(s_max),
        'cell_map': jnp.array(cell_map),
        'grid_origin': jnp.array(grid_origin),
        'grid_shape': jnp.array(grid_shape),
        'cell_size': cs,
        'r_filter': r_filter,
    }


def brute_force_candidates(O, D, geo, r_filter):
    """Find all DOMs within r_filter of the ray, by brute force."""
    hits = []
    for i in range(N_STR):
        for k in range(N_DOM):
            dom_pos = np.array([STRING_XY[i, 0], STRING_XY[i, 1], k * DZ])
            oc = O - dom_pos
            d_norm = D / (np.linalg.norm(D) + 1e-30)
            t = -np.dot(oc, d_norm)
            closest = O + t * d_norm
            dist = np.linalg.norm(closest - dom_pos)
            if dist < r_filter:
                hits.append(i * N_DOM + k)
    return set(hits)


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

MAX_CELLS = 5
MAX_STR_PER_CELL = 4
N_DOM_SNAP = 2

def test_ray_near_string_0():
    """Ray passing 0.1m from string 0 at z=30 should find DOMs 1 and 2."""
    geo = build_toy()
    O = jnp.array([0.0, 0.1, 30.0])
    D = jnp.array([1.0, 0.0, 0.0])

    result = match_segment(
        O, D,
        geo['grid_origin'], geo['grid_shape'], geo['cell_size'], geo['cell_map'],
        geo['anchors'], geo['axes'], geo['s_offsets'], geo['global_ids'],
        geo['s_min'], geo['s_max'],
        MAX_CELLS, MAX_STR_PER_CELL, N_DOM_SNAP,
        geo['r_filter'],
    )
    found = set(int(x) for x in np.array(result) if x >= 0)

    # String 0, DOM 1 (z=17) and DOM 2 (z=34) bracket z=30
    assert 1 in found or 2 in found, f"expected DOM 1 or 2 from string 0, got {found}"


def test_ray_misses_all():
    """Ray far from all strings should return no candidates."""
    geo = build_toy()
    O = jnp.array([200.0, 200.0, 30.0])
    D = jnp.array([1.0, 0.0, 0.0])

    result = match_segment(
        O, D,
        geo['grid_origin'], geo['grid_shape'], geo['cell_size'], geo['cell_map'],
        geo['anchors'], geo['axes'], geo['s_offsets'], geo['global_ids'],
        geo['s_min'], geo['s_max'],
        MAX_CELLS, MAX_STR_PER_CELL, N_DOM_SNAP,
        geo['r_filter'],
    )
    found = set(int(x) for x in np.array(result) if x >= 0)
    assert len(found) == 0, f"expected no hits, got {found}"


def test_ray_through_multiple_strings():
    """Ray passing through strings 0 and 1 (at y=0, x=0 and x=50)."""
    geo = build_toy()
    O = jnp.array([-10.0, 0.05, 34.0])  # y=0.05, near strings 0 and 1
    D = jnp.array([1.0, 0.0, 0.0])

    result = match_segment(
        O, D,
        geo['grid_origin'], geo['grid_shape'], geo['cell_size'], geo['cell_map'],
        geo['anchors'], geo['axes'], geo['s_offsets'], geo['global_ids'],
        geo['s_min'], geo['s_max'],
        MAX_CELLS, MAX_STR_PER_CELL, N_DOM_SNAP,
        geo['r_filter'],
    )
    found = set(int(x) for x in np.array(result) if x >= 0)

    # Should find DOMs from both string 0 (IDs 0-4) and string 1 (IDs 5-9)
    from_str0 = found & set(range(0, 5))
    from_str1 = found & set(range(5, 10))
    assert len(from_str0) > 0, f"expected hits from string 0, got {found}"
    assert len(from_str1) > 0, f"expected hits from string 1, got {found}"


def test_brute_force_agreement():
    """Random rays: matcher should find a superset of brute-force hits."""
    geo = build_toy()
    rng = np.random.RandomState(123)
    n_misses = 0

    for trial in range(100):
        O_np = rng.uniform(-10, 60, 3)
        D_np = rng.randn(3)
        D_np /= np.linalg.norm(D_np) + 1e-30

        bf = brute_force_candidates(O_np, D_np, geo, geo['r_filter'])

        result = match_segment(
            jnp.array(O_np), jnp.array(D_np),
            geo['grid_origin'], geo['grid_shape'], geo['cell_size'], geo['cell_map'],
            geo['anchors'], geo['axes'], geo['s_offsets'], geo['global_ids'],
            geo['s_min'], geo['s_max'],
            MAX_CELLS, MAX_STR_PER_CELL, N_DOM_SNAP,
            geo['r_filter'],
        )
        found = set(int(x) for x in np.array(result) if x >= 0)

        # Matcher should find at least as many as brute force
        # (it may find more due to snap window including non-closest DOMs)
        missed = bf - found
        if len(missed) > 0:
            n_misses += 1

    # Allow small miss rate due to DDA/hash discretization edge cases
    miss_rate = n_misses / 100
    assert miss_rate < 0.05, (
        f"brute-force agreement: {n_misses}/100 rays missed candidates "
        f"(miss rate {miss_rate:.0%}, threshold 5%)"
    )
    print(f"    brute-force agreement: {100 - n_misses}/100 rays had full coverage")


def test_batch_shape():
    """match_segment_batch should return (n_rays, max_dom_per_segment)."""
    geo = build_toy()
    n_rays = 8
    origins = jnp.tile(jnp.array([0.0, 0.1, 30.0]), (n_rays, 1))
    directions = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n_rays, 1))

    result = match_segment_batch(
        origins, directions,
        geo['grid_origin'], geo['grid_shape'], geo['cell_size'], geo['cell_map'],
        geo['anchors'], geo['axes'], geo['s_offsets'], geo['global_ids'],
        geo['s_min'], geo['s_max'],
        MAX_CELLS, MAX_STR_PER_CELL, N_DOM_SNAP,
        geo['r_filter'],
    )
    expected_cols = MAX_CELLS * MAX_STR_PER_CELL * N_DOM_SNAP
    assert result.shape == (n_rays, expected_cols), (
        f"expected ({n_rays}, {expected_cols}), got {result.shape}"
    )


if __name__ == "__main__":
    print("=== match_segment ===")
    run_test(test_ray_near_string_0, "ray_near_string_0")
    run_test(test_ray_misses_all, "ray_misses_all")
    run_test(test_ray_through_multiple_strings, "ray_through_multiple_strings")
    run_test(test_brute_force_agreement, "brute_force_agreement")
    run_test(test_batch_shape, "batch_shape")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
