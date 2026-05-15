"""Tests for photon iteration functions (photon_iteration_sample/update_factors)."""
import pytest
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.slow
import numpy.testing as npt

from lucid.simulation import (
    photon_iteration_sample,
    photon_iteration_update_factors,
    photon_iteration_update_factors_safe,
)


def _make_photon_args(key):
    """Standard photon iteration arguments for testing."""
    return dict(
        position=jnp.array([0.5, 0.5, 0.5]),
        direction=jnp.array([0.0, 0.0, 1.0]),
        time=0.0,
        surface_distance=1.5,
        normal=jnp.array([0.0, 0.0, -1.0]),
        scatter_length=50.0,
        wall_reflection_rate=0.5,
        sensor_reflection_rate=0.3,
        absorption_length=100.0,
        hit_sensor=True,
        rng_key=key,
        speed_of_light=0.2253,
    )


class TestPhotonIterationSample:
    def test_reference(self):
        key = jax.random.PRNGKey(42)
        args = _make_photon_args(key)
        new_pos, new_dir, new_time, detect_prob, refl_atten, cont_factor = \
            photon_iteration_sample(**args)
        # position: moved along direction by surface_distance, epsilon pushed via -normal (inward)
        npt.assert_allclose(new_pos, [0.5, 0.5, 2.0001], atol=1e-3)
        # direction: reflected (hit_sensor=True, sensor detected)
        npt.assert_allclose(new_dir, [0.0, 0.0, 1.0], atol=1e-5)
        # time advanced
        assert new_time > 0.0
        npt.assert_allclose(new_time, 6.657789707183838, atol=1e-4)

    def test_output_count(self):
        key = jax.random.PRNGKey(42)
        args = _make_photon_args(key)
        result = photon_iteration_sample(**args)
        assert len(result) == 6


class TestPhotonIterationUpdateFactors:
    def test_reference(self):
        key = jax.random.PRNGKey(42)
        args = _make_photon_args(key)
        new_pos, new_dir, new_time, detect_prob, refl_atten, cont_factor = \
            photon_iteration_update_factors(**args)
        npt.assert_allclose(new_pos, [0.5, 0.5, 2.0001], atol=1e-3)
        # update_factors reflects (expected value mode)
        npt.assert_allclose(new_dir, [0.0, 0.0, -1.0], atol=1e-5)
        npt.assert_allclose(new_time, 6.657789707183838, atol=1e-4)
        # detect_prob is a probability, not binary
        npt.assert_allclose(detect_prob, 0.6793118715286255, atol=1e-5)
        # refl_atten: exp(-1.5/100) ≈ 0.9851
        npt.assert_allclose(refl_atten, 0.9851119518280029, atol=1e-5)

    def test_detect_prob_in_01(self):
        key = jax.random.PRNGKey(42)
        args = _make_photon_args(key)
        _, _, _, detect_prob, _, _ = photon_iteration_update_factors(**args)
        assert 0.0 <= float(detect_prob) <= 1.0


class TestPhotonIterationSafe:
    def test_matches_update_factors(self):
        """Safe wrapper should produce identical forward pass output."""
        key = jax.random.PRNGKey(42)
        args = _make_photon_args(key)
        regular = photon_iteration_update_factors(**args)
        safe = photon_iteration_update_factors_safe(**args)
        for r, s in zip(regular, safe):
            npt.assert_allclose(r, s, atol=1e-7)

    def test_gradient_no_nan(self):
        """Custom VJP should sanitize NaN gradients."""
        key = jax.random.PRNGKey(42)

        def loss_fn(pos):
            args = _make_photon_args(key)
            args["position"] = pos
            out = photon_iteration_update_factors_safe(**args)
            return out[3]  # detect_prob

        pos = jnp.array([0.5, 0.5, 0.5])
        grad = jax.grad(loss_fn)(pos)
        assert jnp.all(jnp.isfinite(grad)), f"NaN/inf in gradient: {grad}"
