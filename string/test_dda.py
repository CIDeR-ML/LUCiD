"""
Test suite for 2D DDA traversal kernel.

Run: python string/test_dda.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

from lucid.propagation.string.dda import dda_traverse_2d

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


def valid_cells(cell_indices, valid_mask):
    """Extract valid cell indices as a python list."""
    ci = np.array(cell_indices)
    vm = np.array(valid_mask)
    return [int(ci[i]) for i in range(len(ci)) if vm[i]]


def test_horizontal_ray():
    """Horizontal ray should traverse cells along x axis."""
    origin = jnp.array([0.5, 0.5])
    direction = jnp.array([1.0, 0.0])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([5, 5])

    cells, mask = dda_traverse_2d(origin, direction, grid_origin, grid_shape, 1.0, 8)
    vc = valid_cells(cells, mask)
    # Should visit cells (0,0), (1,0), (2,0), (3,0), (4,0) → linear 0,5,10,15,20
    # With n_cy=5: linear_idx = ix * 5 + iy
    expected = [0*5+0, 1*5+0, 2*5+0, 3*5+0, 4*5+0]
    assert vc == expected, f"expected {expected}, got {vc}"


def test_vertical_ray():
    """Vertical ray should traverse cells along y axis."""
    origin = jnp.array([0.5, 0.5])
    direction = jnp.array([0.0, 1.0])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([5, 5])

    cells, mask = dda_traverse_2d(origin, direction, grid_origin, grid_shape, 1.0, 8)
    vc = valid_cells(cells, mask)
    expected = [0*5+0, 0*5+1, 0*5+2, 0*5+3, 0*5+4]
    assert vc == expected, f"expected {expected}, got {vc}"


def test_diagonal_ray():
    """45-degree ray should traverse diagonal cells."""
    origin = jnp.array([0.5, 0.5])
    direction = jnp.array([1.0, 1.0])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([4, 4])

    cells, mask = dda_traverse_2d(origin, direction, grid_origin, grid_shape, 1.0, 10)
    vc = valid_cells(cells, mask)
    # Diagonal ray hits cells along the diagonal
    # At tie-breaks (tmx == tmy), implementation picks x first
    assert len(vc) >= 4, f"expected >= 4 cells, got {len(vc)}: {vc}"
    # First cell must be (0,0) = linear 0
    assert vc[0] == 0, f"first cell should be 0, got {vc[0]}"


def test_negative_direction():
    """Ray going in -x direction."""
    origin = jnp.array([3.5, 0.5])
    direction = jnp.array([-1.0, 0.0])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([5, 5])

    cells, mask = dda_traverse_2d(origin, direction, grid_origin, grid_shape, 1.0, 8)
    vc = valid_cells(cells, mask)
    expected = [3*5+0, 2*5+0, 1*5+0, 0*5+0]
    assert vc == expected, f"expected {expected}, got {vc}"


def test_ray_exits_grid():
    """Ray starting near grid edge should exit and stop producing valid cells."""
    origin = jnp.array([3.5, 0.5])
    direction = jnp.array([1.0, 0.0])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([5, 5])

    cells, mask = dda_traverse_2d(origin, direction, grid_origin, grid_shape, 1.0, 10)
    vc = valid_cells(cells, mask)
    # Should visit (3,0) and (4,0) then exit
    assert len(vc) == 2, f"expected 2 cells, got {len(vc)}: {vc}"


def test_ray_outside_grid():
    """Ray starting completely outside grid bounds."""
    origin = jnp.array([100.0, 100.0])
    direction = jnp.array([1.0, 0.0])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([5, 5])

    cells, mask = dda_traverse_2d(origin, direction, grid_origin, grid_shape, 1.0, 8)
    vc = valid_cells(cells, mask)
    assert len(vc) == 0, f"expected 0 valid cells for ray outside grid, got {len(vc)}"


def test_zero_direction():
    """Near-zero direction should visit only starting cell."""
    origin = jnp.array([2.5, 2.5])
    direction = jnp.array([1e-25, 1e-25])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([5, 5])

    cells, mask = dda_traverse_2d(origin, direction, grid_origin, grid_shape, 1.0, 8)
    vc = valid_cells(cells, mask)
    assert len(vc) >= 1, "should visit at least starting cell"
    assert vc[0] == 2*5+2, f"starting cell should be (2,2)=12, got {vc[0]}"


def test_non_unit_cell_size():
    """Cell size != 1 — verify correct cell assignment."""
    origin = jnp.array([125.0, 50.0])
    direction = jnp.array([1.0, 0.0])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([10, 10])

    cells, mask = dda_traverse_2d(origin, direction, grid_origin, grid_shape,
                                  125.0, 8)
    vc = valid_cells(cells, mask)
    # origin at x=125 → cell ix=1. Should traverse ix=1,2,3,...
    assert vc[0] == 1*10+0, f"expected cell (1,0), got {vc[0]}"


def test_vmap_over_rays():
    """Batch of rays should produce correct independent traversals."""
    origins = jnp.array([
        [0.5, 0.5],
        [0.5, 2.5],
    ])
    directions = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([4, 4])

    batched = jax.vmap(dda_traverse_2d, in_axes=(0, 0, None, None, None, None))
    cells, masks = batched(origins, directions, grid_origin, grid_shape, 1.0, 6)

    vc0 = valid_cells(cells[0], masks[0])
    vc1 = valid_cells(cells[1], masks[1])

    # Ray 0: horizontal from (0,0) → [0,4,8,12]
    assert vc0[0] == 0*4+0, f"ray 0 first cell wrong: {vc0}"

    # Ray 1: vertical from (0,2) → [0*4+2, 0*4+3]
    assert vc1[0] == 0*4+2, f"ray 1 first cell wrong: {vc1}"


def test_large_cell_few_steps():
    """With cell_size >> L_max, only 1-2 cells should be visited.
    Mimics IceCube: L_max=110m, cell_size=125m."""
    origin = jnp.array([60.0, 60.0])
    direction = jnp.array([1.0, 0.3])
    grid_origin = jnp.array([0.0, 0.0])
    grid_shape = jnp.array([10, 10])

    cells, mask = dda_traverse_2d(origin, direction, grid_origin, grid_shape,
                                  125.0, 5)
    vc = valid_cells(cells, mask)
    # Large cells → ray crosses very few boundaries
    assert len(vc) <= 5, f"expected <= 5 cells, got {len(vc)}"
    assert len(vc) >= 1, f"expected >= 1 cell, got {len(vc)}"


if __name__ == "__main__":
    print("=== dda_traverse_2d ===")
    run_test(test_horizontal_ray, "horizontal_ray")
    run_test(test_vertical_ray, "vertical_ray")
    run_test(test_diagonal_ray, "diagonal_ray")
    run_test(test_negative_direction, "negative_direction")
    run_test(test_ray_exits_grid, "ray_exits_grid")
    run_test(test_ray_outside_grid, "ray_outside_grid")
    run_test(test_zero_direction, "zero_direction")
    run_test(test_non_unit_cell_size, "non_unit_cell_size")
    run_test(test_vmap_over_rays, "vmap_over_rays")
    run_test(test_large_cell_few_steps, "large_cell_few_steps")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
