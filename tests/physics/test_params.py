"""Tests for DetectorParams, ParticleParams, and param utilities."""
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.detector_params import (
    ParticleParams,
    normalize_params, denormalize_params, default_bounds,
)

from tests.helpers.params import lab_dp


class TestDetectorParams:
    def test_construction_and_fields(self):
        dp = lab_dp()
        assert dp.scattering.scatter_length == 50.0
        assert dp.reflection.wall_reflection_rate == 0.5
        assert dp.reflection.sensor_reflection_rate == 0.3
        assert dp.absorption.absorption_length == 100.0
        assert dp.response.qe == 0.2
        assert dp.per_pmt.qe_corrections.shape == (10,)


class TestParticleParams:
    def test_direction_property(self):
        pp = ParticleParams(
            energy=500.0,
            position=jnp.array([0.0, 0.0, 0.0]),
            theta=0.5, phi=1.0, t0=0.0,
        )
        # spherical -> Cartesian with theta the polar angle from +z and phi the azimuth from +x
        d = pp.direction
        npt.assert_allclose(d, [np.sin(0.5) * np.cos(1.0), np.sin(0.5) * np.sin(1.0), np.cos(0.5)],
                            atol=1e-6)

    def test_direction_along_z(self):
        pp = ParticleParams(energy=500.0, position=jnp.zeros(3),
                            theta=0.0, phi=0.0, t0=0.0)
        npt.assert_allclose(pp.direction, [0.0, 0.0, 1.0], atol=1e-6)


class TestNormalizeDenormalize:
    def test_round_trip(self):
        dp = lab_dp()
        bounds_min, bounds_max = default_bounds(10)
        normed = normalize_params(dp, bounds_min, bounds_max)
        back = denormalize_params(normed, bounds_min, bounds_max)
        npt.assert_allclose(back.scattering.scatter_length, 50.0, atol=1e-4)
        npt.assert_allclose(back.response.qe, 0.2, atol=1e-5)
        npt.assert_allclose(back.reflection.wall_reflection_rate, 0.5, atol=1e-5)
        assert 0.0 <= float(normed.scattering.scatter_length) <= 1.0
        assert 0.0 <= float(normed.response.qe) <= 1.0
        lo, hi = default_bounds(50)
        assert lo.per_pmt.qe_corrections.shape == (50,) and hi.per_pmt.qe_corrections.shape == (50,)
