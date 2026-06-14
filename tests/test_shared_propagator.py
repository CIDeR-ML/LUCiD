"""Tests for the shared propagator (Phase 9).

Verify that create_propagator produces bit-identical output to the
existing geometry-specific propagators for all 3 geometries.
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from lucid.propagation.shared import create_propagator
from lucid.geometry import generate_detector

pytestmark = pytest.mark.slow


def _compare_propagators(old_propagator, new_propagator, label):
    """Compare two propagators on multiple ray configurations."""
    test_cases = [
        # From center, axis-aligned
        (jnp.zeros((3, 3)),
         jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])),
        # Off-center, diagonal
        (jnp.array([[0.5, 0.3, 0.1], [-0.2, 0.4, -0.3]]),
         jnp.array([[0.7, 0.3, 0.5], [-0.5, 0.8, 0.2]])),
    ]

    for i, (origins, dirs) in enumerate(test_cases):
        dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)
        old = old_propagator(origins, dirs)
        new = new_propagator(origins, dirs)

        for key in old:
            npt.assert_allclose(
                old[key], new[key], atol=1e-5,
                err_msg=f"{label} case {i} key '{key}' mismatch")


class TestCylinderSharedPropagator:
    def test_matches_existing(self):
        """Shared propagator must match existing cylinder propagator."""
        from lucid.propagation.cylinder import create_photon_propagator

        det = generate_detector("config/WCTE_like_geom_config.json")
        sensor_points = jnp.array(det.all_points)
        sensor_radius = det.S_radius

        # Use explicit grid params matching the old factory defaults
        old = create_photon_propagator(
            sensor_points, sensor_radius,
            r=det.r, h=det.H, n_cap=150, n_angular=250, n_height=150,
            temperature=0.2, max_candidates_per_ray=4)
        new = create_propagator(
            det, sensor_points, sensor_radius,
            temperature=0.2, max_candidates_per_ray=4,
            n_cap=150, n_angular=250, n_height=150)

        _compare_propagators(old, new, "Cylinder")

    def test_output_keys(self):
        det = generate_detector("config/WCTE_like_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        result = prop(jnp.zeros((1, 3)), jnp.array([[1., 0., 0.]]))
        expected = {'sensor_weights', 'sensor_indices', 'times', 'positions',
                    'normals', 'inside_sensor', 'per_sensor_positions', 'sensor_normals'}
        assert set(result.keys()) == expected

    def test_deterministic(self):
        det = generate_detector("config/WCTE_like_geom_config.json")
        prop = create_propagator(det, jnp.array(det.all_points), det.S_radius)
        origins = jnp.zeros((2, 3))
        dirs = jnp.array([[1., 0., 0.], [0., 1., 0.]])
        r1 = prop(origins, dirs)
        r2 = prop(origins, dirs)
        for key in r1:
            npt.assert_array_equal(r1[key], r2[key])


class TestSphereSharedPropagator:
    def test_matches_existing(self):
        """Shared propagator must match existing sphere propagator."""
        from lucid.propagation.sphere import create_sphere_photon_propagator

        det = generate_detector("config/JUNO_geom_config.json")
        sensor_points = jnp.array(det.all_points)
        sensor_radius = det.S_radius

        # n_divisions=100 matches what old setup_event_simulator used for sphere
        # (n_divisions=50 was factory default but overcrowds JUNO's 10k sensors)
        old = create_sphere_photon_propagator(
            sensor_points, sensor_radius,
            sphere_radius=det.r, n_divisions=100,
            temperature=0.2, max_candidates_per_ray=4)

        new = create_propagator(
            det, sensor_points, sensor_radius,
            temperature=0.2, max_candidates_per_ray=4,
            n_divisions=100)

        _compare_propagators(old, new, "Sphere")


class TestBoxSharedPropagator:
    def test_matches_existing(self):
        """Shared propagator must match existing box propagator."""
        from lucid.propagation.box import create_box_photon_propagator

        det = generate_detector("config/nuSCOPE_geom_config.json")
        sensor_points = jnp.array(det.all_points)
        sensor_radius = det.S_radius

        old = create_box_photon_propagator(
            sensor_points, sensor_radius,
            length=det.L, width=det.W, height=det.H,
            n_x=125, n_y=125, n_z=125,
            temperature=0.2, max_candidates_per_ray=4)

        new = create_propagator(
            det, sensor_points, sensor_radius,
            temperature=0.2, max_candidates_per_ray=4,
            n_x=125, n_y=125, n_z=125)

        _compare_propagators(old, new, "Box")
