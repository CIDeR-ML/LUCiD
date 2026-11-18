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


def test_call_returns_the_photons_it_was_given_in_the_calibration_order():
    """``__call__`` returns ``(directions, origins, intensities)``, the order IsotropicSource and
    LaserSource use, and they are the photons passed in: the origins and the per-photon
    intensities exactly, the directions normalised. The key and the medium index do not enter
    (a shotgun replays its photons; wavelength is a separate attribute).

    Every input is distinct per photon and per axis, so a swapped pair of outputs, a transposed
    array or a re-ordered photon fails, which equal-shape checks cannot see.
    """
    rng = np.random.default_rng(3)
    origins = rng.normal(size=(8, 3)).astype(np.float32)
    dirs = rng.normal(size=(8, 3)).astype(np.float32)
    inten = rng.uniform(0.5, 2.0, size=8).astype(np.float32)
    s = shotgun_source(origins, dirs, wavelength=400.0, intensity=inten)
    d, o, i = s(n_photons=8, key=jax.random.PRNGKey(0), n_medium=1.33)
    np.testing.assert_array_equal(np.asarray(o), origins)
    np.testing.assert_allclose(np.asarray(d), dirs / np.linalg.norm(dirs, axis=1, keepdims=True),
                               rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(i), inten)
    d2, o2, i2 = s(n_photons=8, key=jax.random.PRNGKey(9), n_medium=1.0)
    assert all(np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in ((d, d2), (o, o2), (i, i2)))


def test_n_photons_mismatch_raises():
    with pytest.raises(ValueError, match="Inconsistent"):
        shotgun_source(jnp.zeros((4, 3)), jnp.zeros((5, 3)), wavelength=400.0)


def test_missing_n_photons_raises():
    with pytest.raises(ValueError, match="Cannot infer"):
        shotgun_source([0., 0., 0.], [0., 0., 1.], wavelength=400.0)


def test_stack_sources_puts_case_c_at_index_c():
    """Stacking adds a leading case axis: ``batched.<field>[c]`` is source c's field, for
    origins, directions, intensities and the per-case scalar wavelength. The three cases differ
    in every field, so a stack in another order or along another axis fails."""
    rng = np.random.default_rng(4)
    sources = [shotgun_source(rng.normal(size=(4, 3)), rng.normal(size=(4, 3)),
                              wavelength=380.0 + 40.0 * c, intensity=rng.uniform(0.5, 2.0, 4))
               for c in range(3)]
    batched = stack_shotgun_sources(sources)
    assert batched.origins.shape == (3, 4, 3)
    for c, src in enumerate(sources):
        for field in ('origins', 'directions', 'intensities'):
            np.testing.assert_array_equal(np.asarray(getattr(batched, field)[c]),
                                          np.asarray(getattr(src, field)), err_msg=field)
    np.testing.assert_array_equal(np.asarray(batched.wavelength), [380.0, 420.0, 460.0])


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


@pytest.mark.parametrize("n_medium", [1.33, 1.309])
def test_laser_cone_opens_to_the_numerical_aperture_in_the_medium(n_medium):
    """A fibre of numerical aperture NA emits into a medium of index n within arcsin(NA / n) of its
    axis (Snell at the fibre tip): every photon inside, and the sample reaching the edge."""
    from lucid.sources.calibration_sources import generate_laser_photons
    na = 0.22
    d, _, _ = generate_laser_photons(jnp.zeros(3), jnp.array([0., 0., 1.]), 1.0, 20000,
                                     jax.random.PRNGKey(1), n_medium, fiber_NA=na)
    angle = np.arccos(np.clip(np.asarray(d)[:, 2], -1, 1))
    theta_max = np.arcsin(na / n_medium)
    assert angle.max() <= theta_max + 1e-5
    assert angle.max() > theta_max - 2e-3            # 20k photons fill the cone to its edge
