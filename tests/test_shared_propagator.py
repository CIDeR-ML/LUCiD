"""Contract tests for the shared propagator.

This file used to assert that `create_propagator` was bit-identical to the three
geometry-specific factories it replaced -- `create_photon_propagator`,
`create_sphere_photon_propagator`, `create_box_photon_propagator`. Those factories have been
deleted, so there is no longer a second implementation to compare against and the three
`test_matches_existing` cases went with them.

They passed right up to deletion, which is the point: a parity test between two copies of the
same code proves the copies agree, not that either is right, and it forces every future physics
change to be made twice or to fail here. `shared.py` has said since it was written that it
replaced those factories; nothing on main called them.

What remains checks the surviving propagator against itself: its output contract and its
determinism.
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
