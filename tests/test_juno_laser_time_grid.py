"""Tests for the shared JUNO laser calibration time grid."""

import numpy as np
import pytest

from lucid.juno_laser.time_grid import TimeGrid


def test_time_grid_defines_edges_centers_and_bin_count_once():
    """An exactly tiled interval exposes one unambiguous grid convention."""
    grid = TimeGrid(0.0, 4.0, 2.0)
    assert grid.n_bins == 2
    np.testing.assert_allclose(grid.edges, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(grid.centers, [1.0, 3.0])


def test_time_grid_rejects_a_partially_filled_final_bin():
    """No caller may silently choose ceil, floor or round for partial bins."""
    with pytest.raises(ValueError, match="integer number of bins"):
        TimeGrid(0.0, 5.0, 2.0)


def test_time_grid_accepts_normal_floating_point_roundoff():
    """Binary roundoff must not reject a mathematically exact grid."""
    grid = TimeGrid(0.0, 0.3, 0.1)
    assert grid.n_bins == 3


def test_time_grid_validates_prediction_shape():
    """Likelihood inputs must have the grid's unique time dimension."""
    grid = TimeGrid(0.0, 4.0, 2.0)
    grid.validate_mass_shape(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="time dimension 2"):
        grid.validate_mass_shape(np.zeros((3, 3)))
