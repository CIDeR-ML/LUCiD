"""Contract tests for the shared propagator.

There is no parity test against a second implementation: comparing two copies of the
same physics only proves they agree, not that either is correct, and it forces every
future change to be made twice. This checks the propagator against itself instead --
its output-key contract and its determinism.
"""
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from lucid.geometry import generate_detector
from lucid.propagation.shared import create_propagator

pytestmark = pytest.mark.slow

EXPECTED_KEYS = {'sensor_weights', 'sensor_indices', 'times', 'positions',
                 'normals', 'inside_sensor', 'per_sensor_positions', 'sensor_normals'}


class TestSharedPropagatorContract:
    def test_output_keys(self):
        det = generate_detector("config/WCTE_like_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        result = prop(jnp.zeros((1, 3)), jnp.array([[1., 0., 0.]]))
        assert set(result.keys()) == EXPECTED_KEYS

    def test_deterministic(self):
        det = generate_detector("config/WCTE_like_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        origins = jnp.zeros((2, 3))
        dirs = jnp.array([[1., 0., 0.], [0., 1., 0.]])
        r1 = prop(origins, dirs)
        r2 = prop(origins, dirs)
        for key in r1:
            npt.assert_array_equal(r1[key], r2[key])
