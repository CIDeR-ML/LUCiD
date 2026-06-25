"""Tests for utility functions (tools/utils.py)."""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.utils import spherical_to_cartesian, smear_times, smear_charges_SK_like


class TestSphericalToCartesian:
    def test_north_pole(self):
        result = spherical_to_cartesian(0.0, 0.0)
        npt.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-6)

    def test_equator_x(self):
        result = spherical_to_cartesian(jnp.pi / 2, 0.0)
        npt.assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)

    def test_equator_y(self):
        result = spherical_to_cartesian(jnp.pi / 2, jnp.pi / 2)
        npt.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)

    def test_south_pole(self):
        result = spherical_to_cartesian(jnp.pi, 0.0)
        npt.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-6)

    def test_output_is_unit(self):
        result = spherical_to_cartesian(0.7, 1.3)
        npt.assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-6)


class TestSmearTimes:
    def test_shape_preserved(self):
        key = jax.random.PRNGKey(0)
        times = jnp.ones(50) * 100.0
        result = smear_times(times, 0.4, key)
        assert result.shape == (50,)

    def test_resolution_scales(self):
        """Larger time_resolution → larger spread."""
        key = jax.random.PRNGKey(42)
        times = jnp.ones(1000) * 100.0
        small = smear_times(times, 0.1, key)
        large = smear_times(times, 2.0, key)
        assert jnp.std(small) < jnp.std(large)


class TestSmearCharges:
    def test_non_negative(self):
        key = jax.random.PRNGKey(42)
        counts = jnp.array([0.1, 0.5, 1.0, 10.0])
        result = smear_charges_SK_like(counts, key)
        assert jnp.all(result >= 0)
