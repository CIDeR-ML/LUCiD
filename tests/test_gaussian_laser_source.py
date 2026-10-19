"""Tests for the empirical Gaussian calibration laser source."""

import jax
import jax.numpy as jnp
import numpy as np

from lucid.sources import (
    gaussian_beam_theta,
    generate_gaussian_laser_photons,
    gaussian_laser_source,
)


def test_gaussian_beam_theta_matches_formula():
    key = jax.random.PRNGKey(7)
    beam_width_deg = 37.0

    actual = gaussian_beam_theta(key, beam_width_deg, shape=(32,))

    uniforms = jax.random.uniform(key, shape=(32, 12))
    beam_width_rad = jnp.float32(beam_width_deg) * jnp.pi / 180.0
    rnd = (jnp.sum(uniforms, axis=-1) - 6.0) * beam_width_rad
    ang = jnp.sqrt((rnd * rnd) / (3610.0 ** 2))
    expected = jnp.arctan(jnp.sqrt(ang))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_gaussian_laser_obeys_calibration_source_contract():
    n_photons = 1024
    intensity = 2.5e6
    direction = jnp.array([1.0, 2.0, -3.0])
    position = jnp.array([0.1, -0.2, 0.3])

    vectors, origins, weights = generate_gaussian_laser_photons(
        position, direction, intensity, n_photons,
        jax.random.PRNGKey(11), beam_width_deg=37.0,
    )

    assert vectors.shape == (n_photons, 3)
    assert origins.shape == (n_photons, 3)
    assert weights.shape == (n_photons,)
    np.testing.assert_allclose(jnp.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(origins, jnp.tile(position, (n_photons, 1)))
    np.testing.assert_allclose(jnp.sum(weights), intensity, rtol=1e-6)


def test_gaussian_laser_factory_preserves_wavelength_metadata():
    source = gaussian_laser_source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, -1.0],
        beam_width_deg=37.0,
        wavelength=405.0,
    )

    assert float(source.beam_width_deg) == 37.0
    assert float(source.wavelength) == 405.0
    directions, origins, weights = source(64, jax.random.PRNGKey(3))
    assert directions.shape == (64, 3)
    assert origins.shape == (64, 3)
    assert weights.shape == (64,)
