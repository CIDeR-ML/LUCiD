"""Physics-level tests for the hits functions (lucid.simulation.hits).

Tests verify charge conservation, QE scaling, the mean first-arrival time,
and differentiability of the sensor aggregation pipeline.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.simulation.hits import (
    first_arrival_expected, first_arrival_sampled, photon_list_expected,
)
from tests.helpers.params import on_deposits


def _make_test_hits(n_det=10):
    """Create deterministic test data with known properties."""
    # 3 photons hitting sensor 0, 2 photons hitting sensor 3
    flat_weights = jnp.array([0.8, 0.6, 0.4, 0.9, 0.5])
    flat_indices = jnp.array([0, 0, 0, 3, 3])
    flat_times = jnp.array([10.0, 12.0, 15.0, 8.0, 9.0])
    qe_corrections = jnp.ones(n_det)
    return flat_weights, flat_indices, flat_times, n_det, qe_corrections


class TestChargeConservation:
    def test_total_charge_scales_with_qe(self):
        """Measured charge = sum(weights) * qe for each sensor."""
        w, idx, t, n, qe_corr = _make_test_hits()
        q1, _ = on_deposits(first_arrival_expected, w, idx, t, n, qe=0.2, qe_corrections=qe_corr)
        q2, _ = on_deposits(first_arrival_expected, w, idx, t, n, qe=0.4, qe_corrections=qe_corr)
        npt.assert_allclose(q1[0], 0.2 * (0.8 + 0.6 + 0.4), rtol=1e-6)
        npt.assert_allclose(q1[3], 0.2 * (0.9 + 0.5), rtol=1e-6)
        npt.assert_allclose(q2[0] / q1[0], 2.0, atol=1e-5)

    def test_per_sensor_qe_corrections(self):
        """qe_corrections scale charge per sensor independently."""
        w, idx, t, n, _ = _make_test_hits()
        uniform_corr = jnp.ones(n)
        double_corr = jnp.ones(n).at[0].set(2.0)  # sensor 0 gets 2x

        q_uniform, _ = on_deposits(first_arrival_expected, w, idx, t, n, qe=0.2,
                                   qe_corrections=uniform_corr)
        q_double, _ = on_deposits(first_arrival_expected, w, idx, t, n, qe=0.2,
                                  qe_corrections=double_corr)
        # Sensor 0 charge should double
        npt.assert_allclose(q_double[0] / q_uniform[0], 2.0, atol=1e-5)
        # Sensor 3 charge unchanged
        npt.assert_allclose(q_double[3], q_uniform[3], atol=1e-7)

    def test_empty_sensor_zero_charge(self):
        """Sensors with no photons should have zero charge."""
        w, idx, t, n, qe_corr = _make_test_hits()
        q, _ = on_deposits(first_arrival_expected, w, idx, t, n, qe=0.2, qe_corrections=qe_corr)
        # Sensors 1, 2, 4-9 have no photons
        for i in [1, 2, 4, 5, 6, 7, 8, 9]:
            assert q[i] == 0.0


class TestFirstArrivalTiming:
    """The time is E[first arrival | at least one photo-electron]; its occupancy limits."""

    def test_bright_sensor_reports_its_earliest_deposit(self):
        """50 photo-electrons per deposit: the first is from the earliest one but for e^-50."""
        w, idx, t, n, qe_corr = _make_test_hits()
        _, times = on_deposits(first_arrival_expected, 50.0 * w, idx, t, n, qe=1.0,
                               qe_corrections=qe_corr)
        npt.assert_allclose(times[0], 10.0, atol=1e-5)
        npt.assert_allclose(times[3], 8.0, atol=1e-5)

    def test_dim_sensor_reports_the_weighted_mean_arrival(self):
        """At mu -> 0 at most one photo-electron arrives, from deposit i with probability
        w_i / mu: sensor 0's [10, 12, 15] with weights [0.8, 0.6, 0.4] give 11.222 ns."""
        w, idx, t, n, qe_corr = _make_test_hits()
        _, times = on_deposits(first_arrival_expected, 1e-4 * w, idx, t, n, qe=1.0,
                               qe_corrections=qe_corr,
                               threshold=1e-12)
        npt.assert_allclose(times[0], (0.8 * 10 + 0.6 * 12 + 0.4 * 15) / 1.8, rtol=1e-4)

    def test_single_deposit_time_exact(self):
        """One deposit: its own time, whatever its weight, without the spread."""
        q, times = on_deposits(first_arrival_expected, jnp.array([3.0]), jnp.array([0]),
                               jnp.array([42.0]), 5, qe=1.0, qe_corrections=jnp.ones(5))
        npt.assert_allclose(times[0], 42.0, rtol=1e-7)

    def test_spread_moves_a_bright_deposit_earlier_and_a_dim_one_not(self):
        """With the spread the first of many smeared photo-electrons comes early; one alone is
        centred on the deposit (the Gaussian is symmetric)."""
        def first(weight):
            return float(on_deposits(first_arrival_expected, jnp.array([weight]), jnp.array([0]),
                                     jnp.array([42.0]), 2, qe=1.0,
                                     qe_corrections=jnp.ones(2), tts=2.0,
                                     threshold=1e-12)[1][0])
        npt.assert_allclose(first(1e-4), 42.0, atol=2e-3)
        assert first(20.0) < 42.0 - 2.0 * 1.5          # E[min of ~20 normals] = -1.87 sigma


class TestLikelihoodMode:
    def test_log_weights_correct(self):
        """log_w should be log(weight * qe * qe_correction) for valid photons."""
        w, idx, t, n, qe_corr = _make_test_hits()
        log_w, _, _, _ = on_deposits(photon_list_expected, w, idx, t, n, qe=0.2,
                                     qe_corrections=qe_corr)
        # Photon 0: weight=0.8, qe=0.2, corr=1.0 → qe_weight=0.16
        npt.assert_allclose(log_w[0], jnp.log(0.16), atol=1e-5)

    def test_total_charge_matches_simulation(self):
        """Likelihood total_charge should match simulation measured_charge."""
        w, idx, t, n, qe_corr = _make_test_hits()
        q_sim, _ = on_deposits(first_arrival_expected, w, idx, t, n, qe=0.2, qe_corrections=qe_corr)
        _, _, _, q_like = on_deposits(photon_list_expected, w, idx, t, n, qe=0.2,
                                      qe_corrections=qe_corr)
        npt.assert_allclose(q_sim, q_like, atol=1e-6)

    def test_invalid_photon_log_weight(self):
        """Photons with zero weight or non-positive time should get -1e10."""
        w = jnp.array([0.0, 0.5])   # first photon has zero weight
        idx = jnp.array([0, 1])
        t = jnp.array([10.0, -1.0])  # second has negative time
        qe_corr = jnp.ones(5)
        log_w, _, _, _ = on_deposits(photon_list_expected, w, idx, t, 5, qe=0.2,
                                     qe_corrections=qe_corr)
        assert float(log_w[0]) == -1e10  # zero weight
        assert float(log_w[1]) == -1e10  # negative time


class TestDataMode:
    def test_qe_one_passes_all(self):
        """With qe=1.0, all photons should pass QE sampling."""
        w, idx, t, n, _ = _make_test_hits()
        key = jax.random.PRNGKey(42)
        q, _ = on_deposits(first_arrival_sampled, w, idx, t, n, qe=1.0,
                           qe_corrections=jnp.ones(n), key=key)
        # Total charge should equal sum of weights at each sensor
        expected_sensor0 = 0.8 + 0.6 + 0.4
        npt.assert_allclose(q[0], expected_sensor0, atol=1e-5)

    def test_qe_zero_blocks_all(self):
        """With qe≈0, no photons should pass."""
        w, idx, t, n, _ = _make_test_hits()
        key = jax.random.PRNGKey(42)
        q, _ = on_deposits(first_arrival_sampled, w, idx, t, n, qe=1e-10,
                           qe_corrections=jnp.ones(n), key=key)
        npt.assert_allclose(q, jnp.zeros(n), atol=1e-6)

    N_KEYS = 40_000

    def _vmapped(self, qe, qe_corr):
        w, idx, t, n, _ = _make_test_hits()
        keys = jax.random.split(jax.random.PRNGKey(0), self.N_KEYS)
        q, tm = jax.jit(jax.vmap(lambda k: on_deposits(first_arrival_sampled,
            w, idx, t, n, qe=qe, qe_corrections=qe_corr, key=k)))(keys)
        return np.asarray(q, np.float64), np.asarray(tm, np.float64)

    def test_binomial_mean_with_per_sensor_corrections(self):
        """Each photon passes with p = qe * corr[sensor], so E[Q_s] = p_s * sum w. At
        qe = 0.3 with corrections 0.5 (sensor 0) and 2.0 (sensor 3): 0.27 and 0.84.
        The per-key std is < 0.5, so the 4e4-key mean is good to 2.5e-3."""
        corr = jnp.ones(10).at[0].set(0.5).at[3].set(2.0)
        q, _ = self._vmapped(0.3, corr)
        npt.assert_allclose(q[:, 0].mean(), 0.15 * (0.8 + 0.6 + 0.4), atol=0.012)
        npt.assert_allclose(q[:, 3].mean(), 0.60 * (0.9 + 0.5), atol=0.012)
        assert np.all(q[:, [1, 2, 4, 5]] == 0.0)

    def test_first_arrival_is_the_earliest_detected_photon(self):
        """With no TTS the time is the earliest photon that PASSED the QE draw: sensor 0's
        photons at 10, 12, 15 ns are first with probability p, (1-p)p, (1-p)^2 p (no hit:
        (1-p)^3, time 0). Binomial standard errors at 4e4 keys are <= 2.5e-3."""
        p = 0.4
        _, tm = self._vmapped(p, jnp.ones(10))
        t0 = tm[:, 0]
        for t_k, prob in [(10.0, p), (12.0, (1 - p) * p), (15.0, (1 - p) ** 2 * p),
                          (0.0, (1 - p) ** 3)]:
            npt.assert_allclose(np.mean(np.isclose(t0, t_k)), prob, atol=0.012,
                                err_msg=f"t = {t_k}")


class TestDifferentiability:
    def test_simulation_grad_wrt_weights(self):
        """first_arrival_expected should be differentiable w.r.t. flat_weights."""
        def loss(weights):
            qe_corr = jnp.ones(10)
            q, t = on_deposits(first_arrival_expected, weights, jnp.array([0, 0, 3]),
                               jnp.array([10.0, 12.0, 8.0]),
                               10, qe=0.2, qe_corrections=qe_corr)
            return jnp.sum(q)
        grad = jax.grad(loss)(jnp.array([0.8, 0.6, 0.9]))
        assert jnp.all(jnp.isfinite(grad))
        # Charge is linear in weights, so gradient should be constant (= qe)
        npt.assert_allclose(grad, jnp.array([0.2, 0.2, 0.2]), atol=1e-5)

    def test_likelihood_grad_wrt_weights(self):
        """photon_list_expected should be differentiable w.r.t. flat_weights."""
        def loss(weights):
            qe_corr = jnp.ones(10)
            log_w, _, _, total_q = on_deposits(photon_list_expected,
                weights, jnp.array([0, 0, 3]),
                jnp.array([10.0, 12.0, 8.0]),
                10, qe=0.2, qe_corrections=qe_corr)
            return jnp.sum(total_q)
        grad = jax.grad(loss)(jnp.array([0.8, 0.6, 0.9]))
        npt.assert_allclose(grad, jnp.array([0.2, 0.2, 0.2]), atol=1e-6)
