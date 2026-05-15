"""Tests for ShotgunSource construction and Cherenkov sampling."""
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lucid.sources import shotgun_source, stack_shotgun_sources


def test_scalar_broadcast_stores_scalar_wavelength():
    s = shotgun_source([0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                       n_photons=16, wavelength=400.0, intensity=2.0)
    assert s.origins.shape == (16, 3)
    assert s.directions.shape == (16, 3)
    assert s.intensities.shape == (16,)
    assert np.allclose(s.origins, 0.0)
    assert np.allclose(s.directions, jnp.array([0.0, 0.0, 1.0]))
    assert np.allclose(s.intensities, 2.0)
    # Scalar wavelength stored as 0-d array (simulator broadcasts)
    assert s.wavelength.shape == ()
    assert float(s.wavelength) == 400.0


def test_per_photon_wavelengths():
    origins = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    dirs = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (4, 1))
    wl = jnp.array([300.0, 400.0, 500.0, 600.0])
    s = shotgun_source(origins, dirs, wavelength=wl, intensity=1.0)
    assert s.origins.shape == (4, 3)
    assert np.allclose(s.wavelength, wl)


def test_direction_normalization():
    s = shotgun_source([0., 0., 0.], [3.0, 4.0, 0.0], n_photons=1,
                       wavelength=400.0)
    assert np.allclose(jnp.linalg.norm(s.directions, axis=-1), 1.0, atol=1e-6)


def test_cherenkov_wavelength_is_none_until_simulator_samples():
    s = shotgun_source([0., 0., 0.], [0., 0., 1.], n_photons=2000,
                       wavelength='cherenkov')
    assert s.wavelength is None


def test_call_matches_calibration_contract():
    """__call__ returns 3-tuple (directions, origins, intensities), like
    IsotropicSource / LaserSource — wavelength is a separate attribute."""
    s = shotgun_source([0., 0., 0.], [0., 0., 1.], n_photons=8,
                       wavelength=400.0, intensity=1.0)
    d, o, i = s(n_photons=8, key=jax.random.PRNGKey(0))
    assert d.shape == (8, 3) and o.shape == (8, 3)
    assert i.shape == (8,)


def test_n_photons_mismatch_raises():
    with pytest.raises(ValueError, match="Inconsistent"):
        shotgun_source(jnp.zeros((4, 3)), jnp.zeros((5, 3)), wavelength=400.0)


def test_missing_n_photons_raises():
    with pytest.raises(ValueError, match="Cannot infer"):
        shotgun_source([0., 0., 0.], [0., 0., 1.], wavelength=400.0)


def test_stack_sources():
    sources = [
        shotgun_source([0., 0., 0.], [0., 0., 1.], n_photons=4, wavelength=400.0)
        for _ in range(3)
    ]
    batched = stack_shotgun_sources(sources)
    assert batched.origins.shape == (3, 4, 3)
    assert batched.directions.shape == (3, 4, 3)
    assert batched.intensities.shape == (3, 4)
    assert batched.wavelength.shape == (3,)  # 3 scalar wavelengths stacked


def test_stack_cherenkov_sources():
    sources = [
        shotgun_source([0., 0., 0.], [0., 0., 1.], n_photons=4, wavelength='cherenkov')
        for _ in range(3)
    ]
    batched = stack_shotgun_sources(sources)
    assert batched.wavelength is None


def test_stack_mixed_wavelength_raises():
    s1 = shotgun_source([0., 0., 0.], [0., 0., 1.], n_photons=4, wavelength=400.0)
    s2 = shotgun_source([0., 0., 0.], [0., 0., 1.], n_photons=4, wavelength='cherenkov')
    with pytest.raises(ValueError, match="wavelengths set or none set"):
        stack_shotgun_sources([s1, s2])
