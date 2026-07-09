"""Tests for the pluggable reflection models (lucid.simulation.reflection)."""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation.reflection import (
    ScalarReflection, scalar_reflection,
    ScalarMixReflection, scalar_mix_reflection,
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


class TestScalarMixModel:
    def _params(self, fw=0.55, fs=0.90):
        return ScalarMixReflection(wall_rate=jnp.asarray(0.05), sensor_rate=jnp.asarray(0.25),
                                   wall_fspec=jnp.asarray(fw), sensor_fspec=jnp.asarray(fs))

    def test_registered(self):
        fn, builder = get_reflection_model('scalar_mix')
        assert fn is scalar_mix_reflection
        dp = create_default_detector_params(10)
        rp = builder(dp)
        assert isinstance(rp, ScalarMixReflection)

    def test_magnitude_matches_scalar_rates(self):
        """Reflection PROBABILITY is the plain wall/sensor rate (pathwise, fspec-independent)."""
        rp = self._params()
        pw, _, _ = scalar_mix_reflection(_DIR, _NORMAL, False, rp, jnp.asarray(0.0), _KEY)
        ps, _, _ = scalar_mix_reflection(_DIR, _NORMAL, True, rp, jnp.asarray(0.0), _KEY)
        npt.assert_allclose(pw, 0.05)
        npt.assert_allclose(ps, 0.25)

    def test_specular_fraction_monte_carlo(self):
        """The sampled specular fraction matches fspec (the discrete DiCE branch)."""
        rp = self._params(fw=0.3)
        from lucid.simulation.optics import compute_reflection_direction
        spec_dir = compute_reflection_direction(_DIR, _NORMAL)
        n_spec = 0
        n = 2000
        for i in range(n):
            _, d, _ = scalar_mix_reflection(_DIR, _NORMAL, False, rp, jnp.asarray(0.0),
                                            jax.random.PRNGKey(i))
            if bool(jnp.allclose(d, spec_dir, atol=1e-6)):
                n_spec += 1
        frac = n_spec / n
        assert abs(frac - 0.3) < 0.03, frac

    def test_dice_score_is_branch_logprob(self):
        """lr_score == log f on the specular branch, log(1-f) on the diffuse branch, and its
        gradient wrt fspec matches the analytic score derivative."""
        f = 0.3
        rp = self._params(fw=f)

        def lr_of_f(fval, key):
            rpf = self._params(fw=fval)
            _, _, lr = scalar_mix_reflection(_DIR, _NORMAL, False, rpf, jnp.asarray(0.0), key)
            return lr

        from lucid.simulation.optics import compute_reflection_direction
        spec_dir = compute_reflection_direction(_DIR, _NORMAL)
        seen = set()
        for i in range(200):
            key = jax.random.PRNGKey(i)
            _, d, lr = scalar_mix_reflection(_DIR, _NORMAL, False, rp, jnp.asarray(0.0), key)
            is_spec = bool(jnp.allclose(d, spec_dir, atol=1e-6))
            expected = jnp.log(f) if is_spec else jnp.log1p(-f)
            npt.assert_allclose(lr, expected, rtol=1e-5)
            g = jax.grad(lr_of_f)(jnp.asarray(f), key)
            expected_g = 1.0 / f if is_spec else -1.0 / (1.0 - f)
            npt.assert_allclose(g, expected_g, rtol=1e-4)
            seen.add(is_spec)
            if len(seen) == 2 and i > 50:
                break
        assert seen == {True, False}

    def test_default_fspec_is_fully_diffuse(self):
        """Configs that never set the fspec fields get 0.0 -> clipped to ~fully diffuse.
        Documents the behavior change vs 'scalar' (always-specular sensors)."""
        dp = create_default_detector_params(10)
        _, builder = get_reflection_model('scalar_mix')
        rp = builder(dp)
        npt.assert_allclose(rp.wall_fspec, 0.0)
        npt.assert_allclose(rp.sensor_fspec, 0.0)


class TestSamplePathHonorsModel:
    """The sampling (data/truth) path must use the SAME reflection model as the
    differentiable path — regression for the fix that unified photon_iteration_sample."""

    def _reflect_args(self, refl_params, key):
        # scatter_length huge -> always reaches the surface; sensor_rate=1 -> always reflects.
        return dict(
            position=jnp.zeros(3), direction=_DIR, time=jnp.asarray(0.0),
            surface_distance=jnp.asarray(1.0), normal=_NORMAL,
            scatter_length=jnp.asarray(1e9), mie_scatter_length=jnp.asarray(1e9),
            g=jnp.asarray(0.9), refl_params=refl_params, absorption_length=jnp.asarray(1e9),
            hit_sensor=jnp.asarray(True), lam=jnp.asarray(400.0), rng_key=key,
            speed_of_light=jnp.asarray(0.2254))

    def test_default_reflection_fn_is_scalar(self):
        from lucid.simulation.photon_step import photon_iteration_sample
        rp = ScalarReflection(wall_rate=jnp.asarray(1.0), sensor_rate=jnp.asarray(1.0))
        for i in range(20):
            k = jax.random.PRNGKey(i)
            a = photon_iteration_sample(**self._reflect_args(rp, k))
            b = photon_iteration_sample(**self._reflect_args(rp, k), reflection_fn=scalar_reflection)
            for x, y in zip(a, b):
                npt.assert_array_equal(x, y)   # default == explicit scalar, bit-for-bit

    def test_sample_path_honors_scalar_mix(self):
        from lucid.simulation.photon_step import photon_iteration_sample
        from lucid.simulation.optics import compute_reflection_direction
        specular = compute_reflection_direction(_DIR, _NORMAL)
        sca = ScalarReflection(wall_rate=jnp.asarray(1.0), sensor_rate=jnp.asarray(1.0))
        mix_spec = ScalarMixReflection(wall_rate=jnp.asarray(1.0), sensor_rate=jnp.asarray(1.0),
                                       wall_fspec=jnp.asarray(0.0), sensor_fspec=jnp.asarray(1.0))
        mix_diff = ScalarMixReflection(wall_rate=jnp.asarray(1.0), sensor_rate=jnp.asarray(1.0),
                                       wall_fspec=jnp.asarray(0.0), sensor_fspec=jnp.asarray(0.0))
        k = jax.random.PRNGKey(0)
        # scalar sensor reflection is specular
        d_sca = photon_iteration_sample(**self._reflect_args(sca, k), reflection_fn=scalar_reflection)[1]
        npt.assert_allclose(d_sca, specular, atol=1e-6)
        # scalar_mix sensor_fspec=1 -> also specular (honored)
        d_mix_spec = photon_iteration_sample(**self._reflect_args(mix_spec, k),
                                             reflection_fn=scalar_mix_reflection)[1]
        npt.assert_allclose(d_mix_spec, specular, atol=1e-6)
        # scalar_mix sensor_fspec=0 -> diffuse, NOT specular (proves the model is honored)
        d_mix_diff = photon_iteration_sample(**self._reflect_args(mix_diff, k),
                                             reflection_fn=scalar_mix_reflection)[1]
        assert not bool(jnp.allclose(d_mix_diff, specular, atol=1e-3))
