"""Tests for the pluggable reflection models (lucid.simulation.reflection)."""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation.reflection import (
    ScalarReflection, scalar_reflection,
    ScalarMixReflection, scalar_mix_reflection,
    AngularReflection, angular_reflection,
    pmt_reflectance, get_reflection_model, sensor_normal_reflectance,
    fresnel_rr, fresnel_rc, n_glass,
)
import numpy as np
import pytest
from lucid.detector_params import create_default_detector_params
from tests.helpers.paths import REPO


_DIR = jnp.array([0.3, 0.0, 0.95]) / jnp.linalg.norm(jnp.array([0.3, 0.0, 0.95]))
_NORMAL = jnp.array([0.0, 0.0, -1.0])
_KEY = jax.random.PRNGKey(0)
N_WATER = 1.33          # config/materials/water.json refractive_index


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
            jnp.array([0.0, 0.0, 1.0]), _NORMAL, False, rp, jnp.asarray(400.0), _KEY, n_medium=N_WATER)
        npt.assert_allclose(refl_prob, rp.R0w, atol=1e-5)

    def test_schlick_grazing_approaches_one(self):
        """At grazing incidence (cosθ→0) the Schlick wall reflectance → 1."""
        rp = self._params()
        grazing = jnp.array([0.99995, 0.0, 0.01])
        grazing = grazing / jnp.linalg.norm(grazing)
        refl_prob, _, _ = angular_reflection(
            grazing, _NORMAL, False, rp, jnp.asarray(400.0), _KEY, n_medium=N_WATER)
        assert float(refl_prob) > 0.9

    def test_sensor_uses_fresnel(self):
        """Sensor reflectance matches the standalone pmt_reflectance at the angle."""
        rp = self._params()
        cth = jnp.abs(jnp.sum(_DIR * _NORMAL))
        expected = pmt_reflectance(jnp.clip(cth, 0., 1.), jnp.asarray(400.0), rp.nr, rp.nk, N_WATER)
        refl_prob, _, _ = angular_reflection(
            _DIR, _NORMAL, True, rp, jnp.asarray(400.0), _KEY, n_medium=N_WATER)
        npt.assert_allclose(refl_prob, expected, atol=1e-5)

    def test_magnitude_gradients_are_pathwise(self):
        """R0w (wall) and nr (sensor) gradients flow and are finite (pathwise via sg(normal))."""
        rp = self._params()
        gw = jax.grad(lambda x: angular_reflection(
            _DIR, _NORMAL, False, rp._replace(R0w=x), jnp.asarray(400.0), _KEY, n_medium=N_WATER)[0])(jnp.asarray(0.05))
        gs = jax.grad(lambda x: angular_reflection(
            _DIR, _NORMAL, True, rp._replace(nr=x), jnp.asarray(400.0), _KEY, n_medium=N_WATER)[0])(jnp.asarray(2.8))
        assert jnp.isfinite(gw) and float(gw) > 0.0   # higher R0 → more reflection
        assert jnp.isfinite(gs)

    def test_spec_diff_score_is_branch_logprob(self):
        """The angular model's DiCE score is log f on the specular branch and log(1-f) on
        the diffuse one, with f = fw at walls; the specular share is f (binomial s.e. 0.002)."""
        rp = self._params()
        from lucid.simulation.optics import compute_reflection_direction
        spec_dir = compute_reflection_direction(_DIR, _NORMAL)
        keys = jax.random.split(jax.random.PRNGKey(1), 20000)
        _, d, lr = jax.vmap(lambda k: angular_reflection(
            _DIR, _NORMAL, False, rp, jnp.asarray(400.0), k, n_medium=N_WATER))(keys)
        is_spec = np.asarray(jnp.all(jnp.abs(d - spec_dir) < 1e-6, axis=1))
        f = float(rp.fw)
        npt.assert_allclose(np.asarray(lr), np.where(is_spec, np.log(f), np.log1p(-f)), rtol=1e-5)
        assert abs(is_spec.mean() - f) < 0.01



def _fresnel_sine_law(theta_i, n1, n2):
    """Unpolarised Fresnel reflectance from the sine/tangent forms (Born & Wolf, Sec. 1.5.2):
    r_s = -sin(ti - tt)/sin(ti + tt), r_p = tan(ti - tt)/tan(ti + tt), Snell n1 sin ti = n2 sin tt.
    An independent parametrisation of the cosine forms the library uses (valid for ti > 0,
    below the critical angle)."""
    tt = np.arcsin(n1 / n2 * np.sin(theta_i))
    rs = -np.sin(theta_i - tt) / np.sin(theta_i + tt)
    rp = np.tan(theta_i - tt) / np.tan(theta_i + tt)
    return 0.5 * (rs**2 + rp**2), np.cos(tt)


def _two_interface_bounce_sum(R1, R2, n_bounces=200):
    """Incoherent reflectance of a lossless layer (R1, T1 = 1 - R1) on a back face R2,
    summed bounce by bounce: R1 + T1 R2 T1 [1 + R1 R2 + (R1 R2)^2 + ...]."""
    T1 = 1.0 - R1
    return R1 + sum(T1 * R2 * (R1 * R2) ** k * T1 for k in range(n_bounces))


class TestFresnelAnalytic:
    """fresnel_rr / fresnel_rc / n_glass / pmt_reflectance against textbook optics."""

    @pytest.mark.parametrize("n1,n2", [(1.33, 1.49), (1.0, 1.5), (1.5, 1.33)])
    def test_normal_incidence(self, n1, n2):
        R, ct = fresnel_rr(jnp.asarray(1.0), n1, n2)
        npt.assert_allclose(R, ((n1 - n2) / (n1 + n2)) ** 2, rtol=1e-5)
        npt.assert_allclose(ct, 1.0, atol=1e-6)

    @pytest.mark.parametrize("n1,n2", [(1.33, 1.49), (1.0, 1.5), (1.5, 1.33), (1.49, 2.8)])
    def test_oblique_matches_sine_law(self, n1, n2):
        theta_c = np.arcsin(min(1.0, n2 / n1))
        for theta in np.linspace(0.05, min(1.5, theta_c - 0.02), 12):
            R, ct = fresnel_rr(jnp.asarray(np.cos(theta), jnp.float32), n1, n2)
            R_ref, ct_ref = _fresnel_sine_law(theta, n1, n2)
            npt.assert_allclose(R, R_ref, rtol=2e-4, atol=1e-6, err_msg=f"theta={theta}")
            npt.assert_allclose(ct, ct_ref, atol=1e-5)

    @pytest.mark.parametrize("n1,n2", [(1.33, 1.49), (1.0, 1.5), (1.5, 1.0)])
    def test_brewster_angle(self, n1, n2):
        """At tan(theta_B) = n2/n1, r_p = 0 and R = r_s^2/2 = ((n1^2-n2^2)/(n1^2+n2^2))^2 / 2."""
        theta_b = np.arctan(n2 / n1)
        R, _ = fresnel_rr(jnp.asarray(np.cos(theta_b), jnp.float32), n1, n2)
        npt.assert_allclose(R, 0.5 * ((n1**2 - n2**2) / (n1**2 + n2**2)) ** 2, rtol=1e-3)

    def test_total_internal_reflection(self):
        """Glass to water above the critical angle arcsin(1.33/1.5) = 62.5 deg reflects fully."""
        for theta_deg in (63.0, 70.0, 85.0):
            R, _ = fresnel_rr(jnp.asarray(np.cos(np.deg2rad(theta_deg)), jnp.float32), 1.5, 1.33)
            npt.assert_allclose(R, 1.0)

    @pytest.mark.parametrize("n_i,n,k", [(1.49, 2.8, 1.5), (1.33, 2.0, 0.3), (1.0, 1.5, 0.0)])
    def test_complex_index_normal_incidence(self, n_i, n, k):
        """R(1) = |(n_i - n~)/(n_i + n~)|^2 = ((n_i - n)^2 + k^2) / ((n_i + n)^2 + k^2)."""
        R = fresnel_rc(jnp.asarray(1.0), n_i, n + 1j * k)
        npt.assert_allclose(R, ((n_i - n) ** 2 + k**2) / ((n_i + n) ** 2 + k**2), rtol=1e-5)

    def test_complex_index_reduces_to_sine_law_when_lossless(self):
        for theta in np.linspace(0.05, 1.4, 8):
            R = fresnel_rc(jnp.asarray(np.cos(theta), jnp.float32), 1.49, 2.8 + 0j)
            npt.assert_allclose(R, _fresnel_sine_law(theta, 1.49, 2.8)[0], rtol=2e-4)

    def test_glass_index_is_borosilicate(self):
        """PMT window glass is borosilicate: n_d = 1.47 (Schott Borofloat 33) to 1.52
        (BK7), rising toward the UV (normal dispersion). A 1/lambda Cauchy term would
        give n ~ 10."""
        lam = jnp.linspace(300.0, 700.0, 9)
        n = np.asarray(n_glass(lam))
        assert np.all((n > 1.47) & (n < 1.52)), n
        assert np.all(np.diff(n) < 0)

    @pytest.mark.parametrize("nr,nk", [(2.8, 1.5), (2.0, 0.3)])
    def test_pmt_reflectance_normal_incidence(self, nr, nk):
        """At normal incidence both interfaces are closed form: water->glass
        ((1.33-n_g)/(1.33+n_g))^2, glass->cathode |(n_g-n~)/(n_g+n~)|^2, combined by
        the bounce series."""
        lam = 405.0
        ng = float(n_glass(jnp.asarray(lam)))
        R1 = ((N_WATER - ng) / (N_WATER + ng)) ** 2
        R2 = ((ng - nr) ** 2 + nk**2) / ((ng + nr) ** 2 + nk**2)
        npt.assert_allclose(pmt_reflectance(jnp.asarray(1.0), jnp.asarray(lam), nr, nk, N_WATER),
                            _two_interface_bounce_sum(R1, R2), rtol=1e-5)

    def test_pmt_reflectance_oblique_lossless_cathode(self):
        """Off normal, with a lossless cathode (k = 0) both interfaces follow the sine
        law: R1 at the water angle, R2 at the refracted glass angle. The water->glass
        reflectance R1 grows toward grazing, which is where a lost (1 - R1) shows."""
        lam, nr = 405.0, 2.8
        ng = float(n_glass(jnp.asarray(lam)))
        for theta in (0.3, 0.8, 1.2, 1.45):
            R1, ctg = _fresnel_sine_law(theta, N_WATER, ng)
            R2, _ = _fresnel_sine_law(np.arccos(ctg), ng, nr)
            npt.assert_allclose(
                pmt_reflectance(jnp.asarray(np.cos(theta), jnp.float32), jnp.asarray(lam), nr, 0.0, N_WATER),
                _two_interface_bounce_sum(R1, R2), rtol=5e-4, err_msg=f"theta={theta}")


def test_the_angular_model_reflects_off_the_cathode_from_the_detector_medium():
    """The first cathode interface is medium -> glass: ice (n = 1.309, config/materials/ice.json)
    gives its own closed-form ((n - n_g)/(n + n_g))^2, not water's 1.33."""
    import json
    n_ice = json.load(open(str(REPO / 'config/materials/ice.json')))['refractive_index']
    lam, nr, nk = 405.0, 2.8, 1.5
    ng = float(n_glass(jnp.asarray(lam)))
    R2 = ((ng - nr) ** 2 + nk**2) / ((ng + nr) ** 2 + nk**2)
    rp = AngularReflection(R0w=jnp.asarray(0.1), pw=jnp.asarray(5.0), fw=jnp.asarray(0.5),
                           nr=jnp.asarray(nr), nk=jnp.asarray(nk), fs=jnp.asarray(0.9))
    for n in (n_ice, N_WATER):
        fn, _ = get_reflection_model('angular', n_medium=n)
        p, _, _ = fn(jnp.array([0.0, 0.0, 1.0]), _NORMAL, True, rp, jnp.asarray(lam), _KEY)
        npt.assert_allclose(p, _two_interface_bounce_sum(((n - ng) / (n + ng)) ** 2, R2), rtol=1e-5)


class TestRegistry:
    def test_scalar_and_angular_registered(self):
        fn_s, build_s = get_reflection_model('scalar', n_medium=N_WATER)
        fn_a, build_a = get_reflection_model('angular', n_medium=N_WATER)
        dp = create_default_detector_params(10)
        rp_s = build_s(dp)
        rp_a = build_a(dp)
        assert isinstance(rp_s, ScalarReflection)
        assert isinstance(rp_a, AngularReflection)
        npt.assert_allclose(rp_a.nr, dp.reflection.cathode_nr)
        fn_m, build_m = get_reflection_model('scalar_mix', n_medium=N_WATER)
        assert fn_m is scalar_mix_reflection
        assert isinstance(build_m(dp), ScalarMixReflection)

    def test_unknown_model_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown reflection model"):
            get_reflection_model('quantum', n_medium=N_WATER)


class TestScalarMixModel:
    def _params(self, fw=0.55, fs=0.90):
        return ScalarMixReflection(wall_rate=jnp.asarray(0.05), sensor_rate=jnp.asarray(0.25),
                                   wall_fspec=jnp.asarray(fw), sensor_fspec=jnp.asarray(fs))


    def test_magnitude_matches_scalar_rates(self):
        """Reflection PROBABILITY is the plain wall/sensor rate (pathwise, fspec-independent)."""
        rp = self._params()
        pw, _, _ = scalar_mix_reflection(_DIR, _NORMAL, False, rp, jnp.asarray(0.0), _KEY)
        ps, _, _ = scalar_mix_reflection(_DIR, _NORMAL, True, rp, jnp.asarray(0.0), _KEY)
        npt.assert_allclose(pw, 0.05)
        npt.assert_allclose(ps, 0.25)

    def test_specular_fraction_monte_carlo(self):
        """The sampled specular fraction matches fspec (the discrete DiCE branch).
        Binomial standard error at n = 20000, f = 0.3 is 3.2e-3; atol 0.015 is ~5 sigma."""
        rp = self._params(fw=0.3)
        from lucid.simulation.optics import compute_reflection_direction
        spec_dir = compute_reflection_direction(_DIR, _NORMAL)
        keys = jax.random.split(jax.random.PRNGKey(0), 20000)
        d = jax.vmap(lambda k: scalar_mix_reflection(_DIR, _NORMAL, False, rp,
                                                     jnp.asarray(0.0), k)[1])(keys)
        frac = float(jnp.mean(jnp.all(jnp.abs(d - spec_dir) < 1e-6, axis=1)))
        assert abs(frac - 0.3) < 0.015, frac

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

    def test_default_fspec_is_sk_physical(self):
        """Configs that never set the fspec fields get the SK-lit specular fractions
        (wall 0.55, sensor 0.90) — the default 'scalar_mix' model's physical mixture."""
        dp = create_default_detector_params(10)
        _, builder = get_reflection_model('scalar_mix', n_medium=N_WATER)
        rp = builder(dp)
        npt.assert_allclose(rp.wall_fspec, 0.55)
        npt.assert_allclose(rp.sensor_fspec, 0.90)


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


class TestSensorNormalReflectance:
    """The reflection a quoted QE already contains: each model's sensor value at normal incidence."""

    def test_scalar_models_give_the_sensor_rate(self):
        lam = jnp.asarray(400.0)
        rp = ScalarReflection(wall_rate=jnp.asarray(0.5), sensor_rate=jnp.asarray(0.3))
        npt.assert_allclose(sensor_normal_reflectance(scalar_reflection, rp, lam), 0.3)
        rpm = ScalarMixReflection(wall_rate=jnp.asarray(0.5), sensor_rate=jnp.asarray(0.3),
                                  wall_fspec=jnp.asarray(0.5), sensor_fspec=jnp.asarray(0.9))
        npt.assert_allclose(sensor_normal_reflectance(scalar_mix_reflection, rpm, lam), 0.3)

    def test_angular_model_gives_its_fresnel_value(self):
        """The angular model's sensor R0 is the closed-form normal-incidence stack value."""
        lam = 405.0
        rp = AngularReflection(R0w=jnp.asarray(0.05), pw=jnp.asarray(1.5),
                               fw=jnp.asarray(0.1), nr=jnp.asarray(2.8),
                               nk=jnp.asarray(1.5), fs=jnp.asarray(0.2))
        ng = float(n_glass(jnp.asarray(lam)))
        R1 = ((N_WATER - ng) / (N_WATER + ng)) ** 2
        R2 = ((ng - 2.8) ** 2 + 1.5**2) / ((ng + 2.8) ** 2 + 1.5**2)
        npt.assert_allclose(sensor_normal_reflectance(get_reflection_model('angular', n_medium=N_WATER)[0], rp,
                                                      jnp.asarray(lam)),
                            _two_interface_bounce_sum(R1, R2), rtol=1e-5)
