"""Tests for the pluggable reflection models (lucid.simulation.reflection)."""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.simulation.reflection import (
    ScalarReflection, scalar_reflection,
    ScalarMixReflection, scalar_mix_reflection,
    AngularReflection, angular_reflection,
    pmt_reflectance, get_reflection_model, sensor_normal_reflectance,
    TabulatedTunedReflection, TabulatedScaledReflection,
    tabulated_tuned_reflection, tabulated_scaled_reflection,
    tabulated_pmt_only_reflection, tuned_wall_only_reflection,
    tuned_blacksheet_reflectance, tabulated_pmt_reflectance,
    tabulated_pmt_optical_outcome, load_pmt_reflectance_table,
    load_pmt_optical_table, _sample_phong_about_specular,
)
from lucid.detector_params import create_default_detector_params
from lucid.simulation.photon_step import photon_iteration_update_factors


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

    def test_tabulated_tuned_registered(self):
        fn, _ = get_reflection_model('tabulated_tuned')
        assert fn is tabulated_tuned_reflection

    def test_tabulated_scaled_registered_and_differentiable(self):
        fn, _ = get_reflection_model('tabulated_scaled')
        assert fn is tabulated_scaled_reflection
        table = jnp.full((108, 200), 0.25)

        def sensor_probability(scale):
            params = TabulatedScaledReflection(
                table, wall_scale=jnp.asarray(1.0), sensor_scale=scale)
            return fn(_DIR, _NORMAL, True, params, jnp.asarray(405.0), _KEY)[0]

        def wall_probability(scale):
            params = TabulatedScaledReflection(
                table, wall_scale=scale, sensor_scale=jnp.asarray(1.0))
            return fn(_DIR, _NORMAL, False, params, jnp.asarray(405.0), _KEY)[0]

        npt.assert_allclose(sensor_probability(0.8), 0.2, rtol=1e-6)
        npt.assert_allclose(jax.grad(sensor_probability)(jnp.asarray(0.8)), 0.25,
                            rtol=1e-6)
        assert float(jax.grad(wall_probability)(jnp.asarray(0.8))) > 0.0

    def test_tabulated_ablation_models_registered(self):
        assert get_reflection_model('tabulated_pmt_only')[0] is tabulated_pmt_only_reflection
        assert get_reflection_model('tuned_wall_only')[0] is tuned_wall_only_reflection

    def test_unknown_model_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown reflection model"):
            get_reflection_model('quantum')


class TestTabulatedTunedModel:
    def test_pmt_table_loader_averages_polarisations(self, tmp_path):
        raw = np.zeros((108 * 200, 8), dtype=np.float32)
        raw[:, 2] = 0.2
        raw[:, 3] = 0.4
        path = tmp_path / "synthetic_reflectance_table.txt"
        np.savetxt(path, raw)
        table = load_pmt_reflectance_table(path)
        npt.assert_allclose(table, 0.3, rtol=1e-6)

    def test_full_pmt_table_loader_preserves_all_outcomes(self, tmp_path):
        raw = np.zeros((108 * 200, 8), dtype=np.float32)
        raw[:, 2:4] = 0.2
        raw[:, 4:6] = 0.3
        raw[:, 6:8] = 0.5
        path = tmp_path / "synthetic_optical_table.txt"
        np.savetxt(path, raw)
        table = load_pmt_optical_table(path)
        npt.assert_allclose(table.reflectance, 0.2)
        npt.assert_allclose(table.transmittance, 0.3)
        npt.assert_allclose(table.absorptance, 0.5)

    def test_pmt_lookup_uses_fortran_bins(self):
        table = (jnp.arange(108 * 200, dtype=jnp.float32)
                 .reshape(108, 200) / (108 * 200))
        # λ=405 -> row 68. 45 degrees -> floor(45*200/90)=100.
        actual = tabulated_pmt_reflectance(
            jnp.cos(jnp.deg2rad(45.0)), jnp.asarray(405.0), table)
        npt.assert_allclose(actual, table[68, 100])

    def test_optical_outcomes_match_sgrefpmt_probability_budget(self):
        angle = jnp.arange(200, dtype=jnp.float32)[None, :]
        reflection = jnp.broadcast_to(0.10 + angle * 1e-4, (108, 200))
        absorption = jnp.broadcast_to(0.55 - angle * 1e-4, (108, 200))
        from lucid.simulation.reflection import PmtOpticalTable
        table = PmtOpticalTable(
            reflection, 1.0 - reflection - absorption, absorption)
        outcome = tabulated_pmt_optical_outcome(
            jnp.cos(jnp.deg2rad(37.0)), jnp.asarray(405.0), table)
        npt.assert_allclose(
            outcome.reflectance + outcome.absorptance + outcome.transmittance,
            1.0, atol=1e-7)

    def test_sk_exact_forward_has_forward_and_reverse_gradients(self):
        # A deliberately sloped table makes both derivatives observable. The
        # primal remains the exact hard SK bin despite the surrogate slope.
        wi = jnp.arange(108, dtype=jnp.float32)[:, None]
        ai = jnp.arange(200, dtype=jnp.float32)[None, :]
        table = 0.1 + 1e-4 * wi + 2e-4 * ai
        cth = jnp.cos(jnp.deg2rad(37.2))
        lam = jnp.asarray(405.2)
        fn = lambda c, w: tabulated_pmt_reflectance(c, w, table)
        primal, tangent = jax.jvp(fn, (cth, lam), (jnp.asarray(1e-3), jnp.asarray(1.0)))
        expected = table[int(np.floor(405.2 - 337.0)),
                         int(np.floor(37.2 / 0.45))]
        npt.assert_allclose(primal, expected, atol=1e-7)
        assert jnp.isfinite(tangent) and float(jnp.abs(tangent)) > 0.0
        reverse = jax.grad(lambda c: fn(c, lam))(cth)
        assert jnp.isfinite(reverse) and float(jnp.abs(reverse)) > 0.0

    def test_blacksheet_normal_incidence(self):
        # At 400 nm and normal incidence the correction is 2 and BSTREF=0.5,
        # so those factors cancel and leave the unpolarised Fresnel value.
        nbs = 2.01 / 1.34
        expected = ((nbs - 1.0) / (nbs + 1.0)) ** 2
        actual = tuned_blacksheet_reflectance(
            jnp.asarray(1.0), jnp.asarray(400.0))
        npt.assert_allclose(actual, expected, rtol=1e-5)

    def test_pmt_reflection_direction_is_specular(self):
        table = jnp.full((108, 200), 0.25)
        params = TabulatedTunedReflection(table)
        probability, direction, score = tabulated_tuned_reflection(
            _DIR, _NORMAL, True, params, jnp.asarray(405.0), _KEY)
        from lucid.simulation.optics import compute_reflection_direction
        npt.assert_allclose(probability, 0.25)
        npt.assert_allclose(direction, compute_reflection_direction(_DIR, _NORMAL),
                            atol=1e-6)
        npt.assert_allclose(score, 0.0)

    def test_pmt_reflectance_uses_sk4_polarization_columns(self):
        average = jnp.full((108, 200), 0.30)
        reflectance_s = jnp.full((108, 200), 0.20)
        reflectance_p = jnp.full((108, 200), 0.40)
        params = TabulatedTunedReflection(
            average, reflectance_s, reflectance_p)
        # The incidence plane is x-z. Physical s is y; physical p is the
        # in-plane vector perpendicular to the ray. SK-IV swaps the physical
        # labels when selecting the optical-table column.
        physical_s = jnp.array([0.0, 1.0, 0.0])
        physical_p = jnp.array([_DIR[2], 0.0, -_DIR[0]])
        r_s = tabulated_tuned_reflection(
            _DIR, _NORMAL, True, params, jnp.asarray(405.0), _KEY,
            polarization=physical_s)[0]
        r_p = tabulated_tuned_reflection(
            _DIR, _NORMAL, True, params, jnp.asarray(405.0), _KEY,
            polarization=physical_p)[0]
        npt.assert_allclose(r_s, 0.40, rtol=1e-6)
        npt.assert_allclose(r_p, 0.20, rtol=1e-6)

    def test_polarized_reflectance_has_matching_forward_reverse_derivative(self):
        ai = jnp.arange(200, dtype=jnp.float32)[None, :]
        average = jnp.broadcast_to(0.30 + 0.0 * ai, (108, 200))
        reflectance_s = jnp.broadcast_to(0.20 + 1e-4 * ai, (108, 200))
        reflectance_p = jnp.broadcast_to(0.40 - 1e-4 * ai, (108, 200))
        params = TabulatedTunedReflection(
            average, reflectance_s, reflectance_p)
        tangent = jnp.array([0.02, -0.01, 0.0])
        polarization = jnp.array([0.2, 0.9, -0.1])

        def observable(pol):
            return tabulated_tuned_reflection(
                _DIR, _NORMAL, True, params, jnp.asarray(405.0), _KEY,
                polarization=pol)[0]

        _, forward = jax.jvp(observable, (polarization,), (tangent,))
        reverse = jax.grad(observable)(polarization) @ tangent
        assert jnp.isfinite(forward) and float(jnp.abs(forward)) > 0.0
        npt.assert_allclose(forward, reverse, rtol=2e-5, atol=2e-6)

    def test_component_ablation_probabilities(self):
        params = TabulatedTunedReflection(jnp.full((108, 200), 0.25))
        pmt_at_sensor = tabulated_pmt_only_reflection(
            _DIR, _NORMAL, True, params, jnp.asarray(405.0), _KEY)[0]
        pmt_at_wall = tabulated_pmt_only_reflection(
            _DIR, _NORMAL, False, params, jnp.asarray(405.0), _KEY)[0]
        wall_at_sensor = tuned_wall_only_reflection(
            _DIR, _NORMAL, True, params, jnp.asarray(405.0), _KEY)[0]
        wall_at_wall = tuned_wall_only_reflection(
            _DIR, _NORMAL, False, params, jnp.asarray(405.0), _KEY)[0]
        npt.assert_allclose(pmt_at_sensor, 0.25)
        npt.assert_allclose(pmt_at_wall, 0.0)
        npt.assert_allclose(wall_at_sensor, 0.0)
        assert float(wall_at_wall) > 0.0

    def test_wall_reflection_directions_are_unit_and_inward(self):
        n = 2048
        keys = jax.random.split(jax.random.PRNGKey(91), n)
        cos_incidence = jnp.linspace(0.001, 1.0, n)
        directions = jnp.stack([
            jnp.sqrt(1.0 - cos_incidence**2),
            jnp.zeros(n),
            cos_incidence,
        ], axis=1)
        normals = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (n, 1))
        params = TabulatedTunedReflection(jnp.full((108, 200), 0.25))
        _, reflected, _ = jax.vmap(
            tabulated_tuned_reflection,
            in_axes=(0, 0, None, None, None, 0),
        )(directions, normals, False, params, jnp.asarray(405.0), keys)

        npt.assert_allclose(jnp.linalg.norm(reflected, axis=1), 1.0, atol=1e-6)
        assert bool(jnp.all(jnp.sum(reflected * normals, axis=1) < 0.0))

    def test_marginal_phong_sampler_preserves_power_ten_lobe(self):
        n = 8192
        keys = jax.random.split(jax.random.PRNGKey(918), n)
        inward = jnp.array([0.0, 0.0, 1.0])
        for specular in (inward, jnp.array([1.0, 0.0, 0.0])):
            samples = jax.vmap(
                _sample_phong_about_specular, in_axes=(None, None, 0)
            )(specular, inward, keys)
            cos_lobe = samples @ specular
            # A power-10 hemisphere has E[cos(theta)]=11/12. At exact
            # grazing, the eight-failure fallback shifts this by only 1/3072.
            assert abs(float(cos_lobe.mean()) - 11.0 / 12.0) < 0.006
            assert bool(jnp.all(samples @ inward >= -1e-6))
            npt.assert_allclose(
                jnp.linalg.norm(samples, axis=1), 1.0, atol=2e-6)

    def test_wall_reflection_direction_has_forward_and_reverse_gradients(self):
        params = TabulatedTunedReflection(jnp.full((108, 200), 0.25))
        tangent = jnp.array([0.0, 0.02, -0.01])

        def observable(direction):
            reflected = tabulated_tuned_reflection(
                direction, _NORMAL, False, params,
                jnp.asarray(405.0), _KEY)[1]
            return reflected @ jnp.array([0.2, -0.4, 0.7])

        _, forward = jax.jvp(observable, (_DIR,), (tangent,))
        reverse = jax.grad(observable)(_DIR) @ tangent
        assert jnp.isfinite(forward) and float(jnp.abs(forward)) > 0.0
        assert bool(jnp.all(jnp.isfinite(jax.grad(observable)(_DIR))))
        npt.assert_allclose(forward, reverse, rtol=2e-5, atol=2e-6)

    def test_expected_transport_separates_direct_pmt_response_from_reflection(self):
        """The downstream A(theta) response must not receive another 1-R factor."""
        params = TabulatedTunedReflection(jnp.full((108, 200), 0.25))
        common = dict(
            position=jnp.zeros(3), direction=_DIR, time=jnp.asarray(0.0),
            surface_distance=jnp.asarray(1.0), normal=_NORMAL,
            scatter_length=jnp.asarray(1e20),
            mie_scatter_length=jnp.asarray(1e20), g=jnp.asarray(0.0),
            refl_params=params, absorption_length=jnp.asarray(1e20),
            lam=jnp.asarray(405.0), rng_key=_KEY,
            speed_of_light=jnp.asarray(0.22),
            reflection_fn=tabulated_tuned_reflection,
            independent_sensor_detection=True,
        )
        sensor = photon_iteration_update_factors(hit_sensor=True, **common)
        wall = photon_iteration_update_factors(hit_sensor=False, **common)

        npt.assert_allclose(sensor[3], 1.0, rtol=1e-6)
        npt.assert_allclose(sensor[5], 0.25, rtol=1e-6)
        wall_r = tuned_blacksheet_reflectance(
            jnp.abs(jnp.sum(_DIR * _NORMAL)), jnp.asarray(405.0))
        npt.assert_allclose(wall[3], 1.0 - wall_r, rtol=1e-6)
        npt.assert_allclose(wall[5], wall_r, rtol=1e-6)


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

    def test_default_fspec_is_sk_physical(self):
        """Configs that never set the fspec fields get the SK-lit specular fractions
        (wall 0.55, sensor 0.90) — the default 'scalar_mix' model's physical mixture."""
        dp = create_default_detector_params(10)
        _, builder = get_reflection_model('scalar_mix')
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
        lam = jnp.asarray(405.0)
        rp = AngularReflection(R0w=jnp.asarray(0.05), pw=jnp.asarray(1.5),
                               fw=jnp.asarray(0.1), nr=jnp.asarray(2.8),
                               nk=jnp.asarray(1.5), fs=jnp.asarray(0.2))
        npt.assert_allclose(sensor_normal_reflectance(angular_reflection, rp, lam),
                            pmt_reflectance(jnp.asarray(1.0), lam, rp.nr, rp.nk), atol=1e-6)
