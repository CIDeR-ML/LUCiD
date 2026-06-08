"""Tests for DetectorParams, ParticleParams, and param utilities."""
import jax.numpy as jnp
import numpy.testing as npt

from lucid.detector_params import (
    DetectorParams, ParticleParams,
    normalize_params, denormalize_params, default_bounds,
)


class TestDetectorParams:
    def test_construction_and_fields(self):
        dp = DetectorParams(
            scatter_length=50.0,
            mie_scatter_length=1000.0,
            g=0.9,
            wall_reflection_rate=0.5,
            sensor_reflection_rate=0.3,
            absorption_length=100.0,
            qe=0.2,
            qe_corrections=jnp.ones(10),
        )
        assert dp.scatter_length == 50.0
        assert dp.wall_reflection_rate == 0.5
        assert dp.sensor_reflection_rate == 0.3
        assert dp.absorption_length == 100.0
        assert dp.qe == 0.2
        assert dp.qe_corrections.shape == (10,)


class TestParticleParams:
    def test_construction(self):
        pp = ParticleParams(
            energy=500.0,
            position=jnp.array([0.0, 0.0, 0.0]),
            theta=0.5, phi=1.0, t0=0.0,
        )
        assert pp.energy == 500.0
        assert pp.theta == 0.5

    def test_direction_property(self):
        pp = ParticleParams(
            energy=500.0,
            position=jnp.array([0.0, 0.0, 0.0]),
            theta=0.5, phi=1.0, t0=0.0,
        )
        d = pp.direction
        npt.assert_allclose(d, [0.2590347230434418, 0.40342268347740173, 0.8775825500488281], atol=1e-5)
        npt.assert_allclose(jnp.linalg.norm(d), 1.0, atol=1e-5)

    def test_direction_along_z(self):
        pp = ParticleParams(energy=500.0, position=jnp.zeros(3),
                            theta=0.0, phi=0.0, t0=0.0)
        npt.assert_allclose(pp.direction, [0.0, 0.0, 1.0], atol=1e-6)


class TestNormalizeDenormalize:
    def test_round_trip(self):
        dp = DetectorParams(
            scatter_length=50.0,
            mie_scatter_length=1000.0,
            g=0.9,
            wall_reflection_rate=0.5,
            sensor_reflection_rate=0.3,
            absorption_length=100.0,
            qe=0.2,
            qe_corrections=jnp.ones(10),
        )
        bounds_min, bounds_max = default_bounds(10)
        normed = normalize_params(dp, bounds_min, bounds_max)
        back = denormalize_params(normed, bounds_min, bounds_max)
        npt.assert_allclose(back.scatter_length, 50.0, atol=1e-4)
        npt.assert_allclose(back.qe, 0.2, atol=1e-5)
        npt.assert_allclose(back.wall_reflection_rate, 0.5, atol=1e-5)

    def test_normalized_in_01(self):
        dp = DetectorParams(
            scatter_length=50.0,
            mie_scatter_length=1000.0,
            g=0.9,
            wall_reflection_rate=0.5,
            sensor_reflection_rate=0.3,
            absorption_length=100.0,
            qe=0.2,
            qe_corrections=jnp.ones(10),
        )
        bounds_min, bounds_max = default_bounds(10)
        normed = normalize_params(dp, bounds_min, bounds_max)
        assert 0.0 <= float(normed.scatter_length) <= 1.0
        assert 0.0 <= float(normed.qe) <= 1.0


class TestDefaultBounds:
    def test_scatter_range(self):
        lo, hi = default_bounds(10)
        assert float(lo.scatter_length) == 0.0
        assert float(hi.scatter_length) == 100.0

    def test_qe_corrections_shape(self):
        lo, hi = default_bounds(50)
        assert lo.qe_corrections.shape == (50,)
        assert hi.qe_corrections.shape == (50,)
