"""Tests for the pluggable reflection models (lucid.simulation.reflection)."""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation.reflection import (
    ScalarReflection, scalar_reflection,
    AngularReflection, angular_reflection,
    pmt_reflectance, get_reflection_model,
)
from lucid.detector_params import create_default_detector_params


_DIR = jnp.array([0.3, 0.0, 0.95]) / jnp.linalg.norm(jnp.array([0.3, 0.0, 0.95]))
_NORMAL = jnp.array([0.0, 0.0, -1.0])
_KEY = jax.random.PRNGKey(0)


class TestScalarModel:
    def test_rates_select_by_surface(self):
        rp = ScalarReflection(wall_rate=jnp.asarray(0.5), sensor_rate=jnp.asarray(0.3))
        pw, _, lr_w = scalar_reflection(_DIR, _NORMAL, False, rp, jnp.asarray(0.0), _KEY)
        ps, _, lr_s = scalar_reflection(_DIR, _NORMAL, True, rp, jnp.asarray(0.0), _KEY)
        npt.assert_allclose(pw, 0.5)
        npt.assert_allclose(ps, 0.3)
        # scalar model carries no reflection DiCE score
        npt.assert_allclose(lr_w, 0.0)
        npt.assert_allclose(lr_s, 0.0)


class TestAngularModel:
    def _params(self):
        return AngularReflection(R0w=jnp.asarray(0.05), pw=jnp.asarray(1.5),
                                 fw=jnp.asarray(0.1), nr=jnp.asarray(2.8),
                                 nk=jnp.asarray(1.5), fs=jnp.asarray(0.2))

    def test_schlick_normal_incidence_is_R0(self):
        """At normal incidence (cosθ=1) the Schlick wall reflectance equals R0."""
        rp = self._params()
        # direction anti-parallel to normal → cth_inc = 1
        refl_prob, _, _ = angular_reflection(
            jnp.array([0.0, 0.0, 1.0]), _NORMAL, False, rp, jnp.asarray(400.0), _KEY)
        npt.assert_allclose(refl_prob, rp.R0w, atol=1e-5)

    def test_schlick_grazing_approaches_one(self):
        """At grazing incidence (cosθ→0) the Schlick wall reflectance → 1."""
        rp = self._params()
        grazing = jnp.array([0.99995, 0.0, 0.01])
        grazing = grazing / jnp.linalg.norm(grazing)
        refl_prob, _, _ = angular_reflection(
            grazing, _NORMAL, False, rp, jnp.asarray(400.0), _KEY)
        assert float(refl_prob) > 0.9

    def test_sensor_uses_fresnel(self):
        """Sensor reflectance matches the standalone pmt_reflectance at the angle."""
        rp = self._params()
        cth = jnp.abs(jnp.sum(_DIR * _NORMAL))
        expected = pmt_reflectance(jnp.clip(cth, 0., 1.), jnp.asarray(400.0), rp.nr, rp.nk)
        refl_prob, _, _ = angular_reflection(
            _DIR, _NORMAL, True, rp, jnp.asarray(400.0), _KEY)
        npt.assert_allclose(refl_prob, expected, atol=1e-5)

    def test_magnitude_gradients_are_pathwise(self):
        """R0w (wall) and nr (sensor) gradients flow and are finite (pathwise via sg(normal))."""
        rp = self._params()
        gw = jax.grad(lambda x: angular_reflection(
            _DIR, _NORMAL, False, rp._replace(R0w=x), jnp.asarray(400.0), _KEY)[0])(jnp.asarray(0.05))
        gs = jax.grad(lambda x: angular_reflection(
            _DIR, _NORMAL, True, rp._replace(nr=x), jnp.asarray(400.0), _KEY)[0])(jnp.asarray(2.8))
        assert jnp.isfinite(gw) and float(gw) > 0.0   # higher R0 → more reflection
        assert jnp.isfinite(gs)

    def test_spec_diff_score_finite(self):
        """The specular/diffuse-mix DiCE score is finite for reflected photons."""
        rp = self._params()
        _, _, lr = angular_reflection(_DIR, _NORMAL, False, rp, jnp.asarray(400.0), _KEY)
        assert jnp.isfinite(lr)


class TestRegistry:
    def test_scalar_and_angular_registered(self):
        fn_s, build_s = get_reflection_model('scalar')
        fn_a, build_a = get_reflection_model('angular')
        dp = create_default_detector_params(10)
        rp_s = build_s(dp)
        rp_a = build_a(dp)
        assert isinstance(rp_s, ScalarReflection)
        assert isinstance(rp_a, AngularReflection)
        npt.assert_allclose(rp_a.nr, dp.reflection.cathode_nr)

    def test_unknown_model_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown reflection model"):
            get_reflection_model('quantum')
