"""Unit tests for PMT-axis incidence diagnostics."""

from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest

from lucid.simulation.simulator import _sensor_outward_axes


def test_measured_pmt_directions_are_preferred_and_normalized():
    detector = SimpleNamespace(
        r=10.0,
        H=20.0,
        # Inward optical axes, deliberately not unit-normalized.
        pmt_directions=np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, -3.0]]),
    )
    # Positions deliberately disagree with the directions. This proves that
    # measured axes, rather than inferred surface normals, control the result.
    positions = np.array([[0.0, 10.0, 0.0], [10.0, 0.0, 0.0]])

    axes = _sensor_outward_axes(detector, positions)

    npt.assert_allclose(axes, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_algorithmic_cylinder_axes_follow_nearest_surface():
    detector = SimpleNamespace(r=1.0, H=2.0)
    positions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])

    axes = _sensor_outward_axes(detector, positions)

    npt.assert_allclose(
        axes,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    )


def test_zero_length_pmt_direction_is_rejected():
    detector = SimpleNamespace(
        r=1.0,
        H=2.0,
        pmt_directions=np.zeros((1, 3)),
    )

    with pytest.raises(ValueError, match="non-zero length"):
        _sensor_outward_axes(detector, np.zeros((1, 3)))
