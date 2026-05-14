"""
Test suite for 2D spatial hash builder.

Run: python string/test_hash.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from lucid.propagation.string.hash import build_string_hash


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


def test_single_vertical_string():
    """One vertical string at origin — should appear in exactly the cells
    covered by its halo."""
    anchors = np.array([[0.0, 0.0, 0.0]])
    tops = np.array([[0.0, 0.0, 100.0]])
    cell_map, origin, shape, cs, stats = build_string_hash(
        anchors, tops, cell_size=10.0, r_filter=1.0, max_strings_per_cell=2
    )
    assert stats['string_coverage'] == 1.0, f"coverage={stats['string_coverage']}"
    assert stats['overflow_count'] == 0
    n_cells_with_string = np.sum(cell_map[:, 0] == 0)
    assert n_cells_with_string >= 1, "string should appear in at least 1 cell"


def test_hex_grid_coverage():
    """86-string hex layout — all strings must be reachable."""
    rng = np.random.RandomState(0)
    n_str = 86
    spacing = 125.0
    # Approximate hex grid (rows of varying width)
    positions = []
    for ring in range(6):
        if ring == 0:
            positions.append([0.0, 0.0])
        else:
            for k in range(6 * ring):
                angle = 2 * np.pi * k / (6 * ring)
                positions.append([ring * spacing * np.cos(angle),
                                  ring * spacing * np.sin(angle)])
    positions = np.array(positions[:n_str])
    anchors = np.column_stack([positions, np.full(n_str, -2450.0)])
    tops = np.column_stack([positions, np.full(n_str, -1450.0)])

    cell_map, origin, shape, cs, stats = build_string_hash(
        anchors, tops, cell_size=125.0, r_filter=0.5,
        max_strings_per_cell=4
    )
    assert stats['string_coverage'] == 1.0, (
        f"coverage={stats['string_coverage']}, missing strings"
    )
    assert stats['overflow_count'] == 0, (
        f"overflow={stats['overflow_count']}, max_occ={stats['max_occupancy']}"
    )
    print(f"    cells={stats['total_cells']}, max_occ={stats['max_occupancy']}, "
          f"mean_occ={stats['mean_occupancy']:.2f}")


def test_overflow_detection():
    """Dense cluster where max_strings_per_cell is too small."""
    n_str = 10
    anchors = np.zeros((n_str, 3))  # all at same xy
    tops = np.zeros((n_str, 3))
    tops[:, 2] = 100.0

    cell_map, origin, shape, cs, stats = build_string_hash(
        anchors, tops, cell_size=100.0, r_filter=1.0,
        max_strings_per_cell=2
    )
    assert stats['overflow_count'] > 0, "expected overflow with 10 strings in 1 cell"


def test_tilted_string_footprint():
    """A string tilted so top is 50m from anchor in xy — should span
    multiple cells along its footprint."""
    anchors = np.array([[0.0, 0.0, 0.0]])
    tops = np.array([[50.0, 0.0, 200.0]])
    cell_map, origin, shape, cs, stats = build_string_hash(
        anchors, tops, cell_size=10.0, r_filter=1.0,
        max_strings_per_cell=2
    )
    n_cells_with_string = np.sum(cell_map[:, 0] == 0)
    # Footprint spans 50m in x → at least 5 cells plus halo
    assert n_cells_with_string >= 5, (
        f"tilted string should span >= 5 cells, got {n_cells_with_string}"
    )


def test_string_reachable_from_nearby_cell():
    """A point near the boundary of a string's cell should still find it."""
    anchors = np.array([[50.0, 50.0, 0.0]])
    tops = np.array([[50.0, 50.0, 100.0]])
    r_filter = 2.0
    cell_map, origin, shape, cs, stats = build_string_hash(
        anchors, tops, cell_size=10.0, r_filter=r_filter,
        max_strings_per_cell=2
    )
    # Check that the cell containing (50 + 1.5, 50) also has the string
    # (within r_filter=2.0)
    ci = int(np.floor((51.5 - origin[0]) / cs))
    cj = int(np.floor((50.0 - origin[1]) / cs))
    cell_idx = ci * shape[1] + cj
    found = 0 in cell_map[cell_idx]
    assert found, "string should be in adjacent cell within r_filter halo"


def test_empty_input():
    """Empty input should raise ValueError."""
    try:
        build_string_hash(
            np.zeros((0, 3)), np.zeros((0, 3)),
            cell_size=10.0, r_filter=1.0, max_strings_per_cell=2
        )
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_cell_index_consistency():
    """Cell indices returned should be valid and match grid dimensions."""
    n_str = 20
    rng = np.random.RandomState(42)
    xy = rng.randn(n_str, 2) * 100
    anchors = np.column_stack([xy, np.zeros(n_str)])
    tops = np.column_stack([xy, np.full(n_str, 100.0)])

    cell_map, origin, shape, cs, stats = build_string_hash(
        anchors, tops, cell_size=50.0, r_filter=1.0,
        max_strings_per_cell=4
    )
    total = shape[0] * shape[1]
    assert cell_map.shape[0] == total, f"shape mismatch: {cell_map.shape[0]} vs {total}"
    valid_entries = cell_map[cell_map >= 0]
    assert np.all(valid_entries < n_str), "string indices out of range"
    assert np.all(valid_entries >= 0), "negative string indices"


if __name__ == "__main__":
    print("=== build_string_hash ===")
    run_test(test_single_vertical_string, "single_vertical_string")
    run_test(test_hex_grid_coverage, "hex_grid_coverage")
    run_test(test_overflow_detection, "overflow_detection")
    run_test(test_tilted_string_footprint, "tilted_string_footprint")
    run_test(test_string_reachable_from_nearby_cell, "string_reachable_from_nearby_cell")
    run_test(test_empty_input, "empty_input")
    run_test(test_cell_index_consistency, "cell_index_consistency")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  {name}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed else 0)
