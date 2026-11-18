"""Capabilities restored after the 2026-11-01 deletion audit ("useful vs used"): each against an
independent reference."""
import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp
import pytest
from scipy.stats import logistic

from lucid.losses import first_arrival_nll, segment_logsumexp
from lucid.sources import isotropic_source, isotropic_source_random
from lucid.gradient_analysis import CALIB_SWEEPS, find_zero_crossing, get_param_value
from tests.helpers.params import lab_dp


class TestIsotropicSourceRandom:
    def test_keys_give_independent_realisations_where_the_lattice_does_not(self):
        rnd = isotropic_source_random([0.0, 0.0, 0.0])
        lat = isotropic_source([0.0, 0.0, 0.0])
        k0, k1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)
        assert not np.allclose(rnd(1000, k0, 1.33)[0], rnd(1000, k1, 1.33)[0])
        npt.assert_array_equal(lat(1000, k0, 1.33)[0], lat(1000, k1, 1.33)[0])

    def test_directions_are_isotropic_unit_vectors_with_the_source_intensity(self):
        n = 200_000
        d, o, w = isotropic_source_random([1.0, -2.0, 0.5], intensity=5000.0)(n, jax.random.PRNGKey(3), 1.33)
        d = np.asarray(d, np.float64)
        npt.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
        se = 1.0 / np.sqrt(n)
        assert np.all(np.abs(d.mean(0)) < 5 * se * np.sqrt(1 / 3))      # <x_i> = 0
        npt.assert_allclose((d ** 2).mean(0), 1 / 3, atol=5 * se)        # <x_i^2> = 1/3
        npt.assert_allclose(np.asarray(o), np.tile([1.0, -2.0, 0.5], (n, 1)))
        npt.assert_allclose(float(np.sum(w)), 5000.0, rtol=1e-5)


class TestSegmentLogsumexp:
    def test_is_logaddexp_per_segment(self):
        data = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
        idx = np.array([0, 0, 1, 1, 2])
        want = [np.logaddexp.reduce(data[idx == s]) for s in range(3)]
        npt.assert_allclose(segment_logsumexp(jnp.asarray(data), jnp.asarray(idx), 3), want, rtol=1e-6)

    def test_large_values_do_not_overflow(self):
        out = segment_logsumexp(jnp.array([500.0, 501.0, 499.0]), jnp.array([0, 0, 0]), 1)
        npt.assert_allclose(out[0], np.logaddexp.reduce([500.0, 501.0, 499.0]), rtol=1e-6)


class TestFirstArrivalNLL:
    """Against the order statistic of N iid logistic arrivals (scipy)."""

    @pytest.mark.parametrize('n', [1, 3, 7])
    def test_first_of_n_identical_photons_is_the_logistic_order_statistic(self, n):
        tau, t_pred, t_obs = 1.3, 10.0, 9.2
        log_w = jnp.zeros(n)                       # n photons of weight 1 -> N_s = n
        loss = first_arrival_nll(log_w, jnp.full(n, t_pred), jnp.zeros(n, jnp.int32),
                                 jnp.array([t_obs]), tau, 1)
        f = logistic.pdf(t_obs, loc=t_pred, scale=tau)
        sf = logistic.sf(t_obs, loc=t_pred, scale=tau)
        npt.assert_allclose(float(loss[0]), -np.log(n * f * sf ** (n - 1)), rtol=1e-5)

    def test_an_empty_sensor_contributes_zero_with_a_finite_gradient(self):
        log_w = jnp.array([0.0, 0.0])
        g = jax.grad(lambda t: first_arrival_nll(log_w, jnp.array([t, t]), jnp.array([0, 0]),
                                                 jnp.array([5.0, 5.0]), 1.0, 2).sum())(4.0)
        out = first_arrival_nll(log_w, jnp.array([4.0, 4.0]), jnp.array([0, 0]), jnp.array([5.0, 5.0]), 1.0, 2)
        assert float(out[1]) == 0.0 and np.isfinite(float(g))


class TestCalibSweeps:
    def test_every_axis_names_a_detector_parameter(self):
        dp = lab_dp()
        for sp in CALIB_SWEEPS:
            assert np.isfinite(get_param_value(dp, sp)), sp.field

    def test_zero_crossing_of_a_linear_gradient(self):
        v = np.linspace(0.0, 10.0, 11)
        npt.assert_allclose(find_zero_crossing(v, 2.0 * (v - 3.7)), 3.7, rtol=1e-12)
        assert np.isnan(find_zero_crossing(v, v + 1.0))
