"""Physics-level tests for sensor response (make_hits_*) functions.

Tests verify charge conservation, QE scaling, soft-min behavior,
and differentiability of the sensor aggregation pipeline.
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_data, make_hits_likelihood,
)


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
        q1, _ = make_hits_simulation(w, idx, t, n, qe=0.2, qe_corrections=qe_corr)
        q2, _ = make_hits_simulation(w, idx, t, n, qe=0.4, qe_corrections=qe_corr)
        # Charge at sensor 0 should double when QE doubles
        npt.assert_allclose(q2[0] / q1[0], 2.0, atol=1e-5)

    def test_per_sensor_qe_corrections(self):
        """qe_corrections scale charge per sensor independently."""
        w, idx, t, n, _ = _make_test_hits()
        uniform_corr = jnp.ones(n)
        double_corr = jnp.ones(n).at[0].set(2.0)  # sensor 0 gets 2x

        q_uniform, _ = make_hits_simulation(w, idx, t, n, qe=0.2, qe_corrections=uniform_corr)
        q_double, _ = make_hits_simulation(w, idx, t, n, qe=0.2, qe_corrections=double_corr)
        # Sensor 0 charge should double
        npt.assert_allclose(q_double[0] / q_uniform[0], 2.0, atol=1e-5)
        # Sensor 3 charge unchanged
        npt.assert_allclose(q_double[3], q_uniform[3], atol=1e-7)

    def test_empty_sensor_zero_charge(self):
        """Sensors with no photons should have zero charge."""
        w, idx, t, n, qe_corr = _make_test_hits()
        q, _ = make_hits_simulation(w, idx, t, n, qe=0.2, qe_corrections=qe_corr)
        # Sensors 1, 2, 4-9 have no photons
        for i in [1, 2, 4, 5, 6, 7, 8, 9]:
            assert q[i] == 0.0


class TestSoftMinTiming:
    def test_approaches_true_min_at_low_temperature(self):
        """As temperature → 0, soft-min → true minimum arrival time."""
        w, idx, t, n, qe_corr = _make_test_hits()
        # Sensor 0 has times [10, 12, 15], true min = 10
        _, t_low = make_hits_simulation(w, idx, t, n, qe=0.2,
                                         qe_corrections=qe_corr, temperature=0.001)
        npt.assert_allclose(t_low[0], 10.0, atol=0.01)

    def test_higher_temperature_shifts_time(self):
        """Higher temperature → time shifts away from true minimum."""
        w, idx, t, n, qe_corr = _make_test_hits()
        _, t_low = make_hits_simulation(w, idx, t, n, qe=0.2,
                                         qe_corrections=qe_corr, temperature=0.001)
        _, t_high = make_hits_simulation(w, idx, t, n, qe=0.2,
                                          qe_corrections=qe_corr, temperature=1.0)
        # Higher temperature should give time ≤ true min (soft-min undershoots)
        assert float(t_high[0]) <= float(t_low[0]) + 0.01

    def test_single_photon_time_exact(self):
        """For a sensor with one photon, soft-min should return that photon's time."""
        w = jnp.array([1.0])
        idx = jnp.array([0])
        t = jnp.array([42.0])
        qe_corr = jnp.ones(5)
        q, times = make_hits_simulation(w, idx, t, 5, qe=1.0,
                                         qe_corrections=qe_corr, temperature=0.01)
        npt.assert_allclose(times[0], 42.0, atol=0.1)


class TestLikelihoodMode:
    def test_log_weights_correct(self):
        """log_w should be log(weight * qe * qe_correction) for valid photons."""
        w, idx, t, n, qe_corr = _make_test_hits()
        log_w, _, _, _ = make_hits_likelihood(w, idx, t, n, qe=0.2, qe_corrections=qe_corr)
        # Photon 0: weight=0.8, qe=0.2, corr=1.0 → qe_weight=0.16
        npt.assert_allclose(log_w[0], jnp.log(0.16), atol=1e-5)

    def test_total_charge_matches_simulation(self):
        """Likelihood total_charge should match simulation measured_charge."""
        w, idx, t, n, qe_corr = _make_test_hits()
        q_sim, _ = make_hits_simulation(w, idx, t, n, qe=0.2, qe_corrections=qe_corr)
        _, _, _, q_like = make_hits_likelihood(w, idx, t, n, qe=0.2, qe_corrections=qe_corr)
        npt.assert_allclose(q_sim, q_like, atol=1e-6)

    def test_invalid_photon_log_weight(self):
        """Photons with zero weight or non-positive time should get -1e10."""
        w = jnp.array([0.0, 0.5])   # first photon has zero weight
        idx = jnp.array([0, 1])
        t = jnp.array([10.0, -1.0])  # second has negative time
        qe_corr = jnp.ones(5)
        log_w, _, _, _ = make_hits_likelihood(w, idx, t, 5, qe=0.2, qe_corrections=qe_corr)
        assert float(log_w[0]) == -1e10  # zero weight
        assert float(log_w[1]) == -1e10  # negative time


class TestDataMode:
    def test_qe_one_passes_all(self):
        """With qe=1.0, all photons should pass QE sampling."""
        w, idx, t, n, _ = _make_test_hits()
        key = jax.random.PRNGKey(42)
        q, _ = make_hits_data(w, idx, t, n, qe=1.0, rng_key=key)
        # Total charge should equal sum of weights at each sensor
        expected_sensor0 = 0.8 + 0.6 + 0.4
        npt.assert_allclose(q[0], expected_sensor0, atol=1e-5)

    def test_qe_zero_blocks_all(self):
        """With qe≈0, no photons should pass."""
        w, idx, t, n, _ = _make_test_hits()
        key = jax.random.PRNGKey(42)
        q, _ = make_hits_data(w, idx, t, n, qe=1e-10, rng_key=key)
        npt.assert_allclose(q, jnp.zeros(n), atol=1e-6)


class TestDifferentiability:
    def test_simulation_grad_wrt_weights(self):
        """make_hits_simulation should be differentiable w.r.t. flat_weights."""
        def loss(weights):
            qe_corr = jnp.ones(10)
            q, t = make_hits_simulation(weights, jnp.array([0, 0, 3]),
                                         jnp.array([10.0, 12.0, 8.0]),
                                         10, qe=0.2, qe_corrections=qe_corr)
            return jnp.sum(q)
        grad = jax.grad(loss)(jnp.array([0.8, 0.6, 0.9]))
        assert jnp.all(jnp.isfinite(grad))
        # Charge is linear in weights, so gradient should be constant (= qe)
        npt.assert_allclose(grad, jnp.array([0.2, 0.2, 0.2]), atol=1e-5)

    def test_likelihood_grad_wrt_weights(self):
        """make_hits_likelihood should be differentiable w.r.t. flat_weights."""
        def loss(weights):
            qe_corr = jnp.ones(10)
            log_w, _, _, total_q = make_hits_likelihood(
                weights, jnp.array([0, 0, 3]),
                jnp.array([10.0, 12.0, 8.0]),
                10, qe=0.2, qe_corrections=qe_corr)
            return jnp.sum(total_q)
        grad = jax.grad(loss)(jnp.array([0.8, 0.6, 0.9]))
        assert jnp.all(jnp.isfinite(grad))
