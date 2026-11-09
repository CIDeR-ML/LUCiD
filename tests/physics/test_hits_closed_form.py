"""Exact values and gradients of ``first_arrival_expected``.

What survives of the old ``test_integration.py``, which built no detector and propagated no
photon: its smoke checks (finite outputs, unit vectors, a positive gradient) duplicated
test_optics / test_reflection / test_photon_step* / test_wavelength* and are gone. What is left
asserts closed-form values on hand-built hits:

- charge is linear in the photon weights and in QE, so ``Q_s = qe * c_s * sum_{i in s} w_i`` and
  its gradients are the summed weights exactly;
- the per-sensor time is the mean of the first-arrival distribution: deposit i is Poisson(w_i)
  photo-electrons at t_i, each smeared by N(0, tts^2), so P(first > t) = exp(-R(t)) with
  R(t) = sum_i w_i Phi((t - t_i) / tts), conditioned on at least one. Without the spread the first
  comes from deposit k with probability exp(-sum_{j<k} w_j) (1 - exp(-w_k)); with it the mean is
  integrated here by scipy quadrature, not by the module's node rule.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from lucid.simulation.hits import first_arrival_expected
from tests.helpers.params import on_deposits


def _mean_first(w, t, tts):
    """E[first arrival | at least one] in float64: the sorted-deposit sum, or quadrature."""
    w, t = np.asarray(w, np.float64), np.asarray(t, np.float64)
    mu = w.sum()
    if tts == 0:
        o = np.argsort(t)
        before = np.concatenate([[0.0], np.cumsum(w[o])[:-1]])
        return float(np.sum(np.exp(-before) * -np.expm1(-w[o]) * t[o]) / -np.expm1(-mu))
    lo, hi = t.min() - 10 * tts, t.max() + 10 * tts
    surv = lambda x: (np.exp(-np.sum(w * norm.cdf((x - t) / tts))) - np.exp(-mu)) / -np.expm1(-mu)
    return lo + quad(surv, lo, hi, points=list(t), limit=400, epsabs=1e-10)[0]


def test_charge_is_qe_times_summed_weight_and_time_is_the_mean_first_arrival():
    n_rays, n_sensors, qe = 10, 5, 0.2
    w = jnp.full(n_rays, 0.8)
    idx = jnp.zeros(n_rays, dtype=jnp.int32)
    t = jnp.linspace(10.0, 15.0, n_rays)
    q, t_meas = on_deposits(first_arrival_expected, w, idx, t, n_sensors, qe=qe,
                            qe_corrections=jnp.ones(n_sensors))
    npt.assert_allclose(np.asarray(q), [10 * 0.8 * qe, 0, 0, 0, 0], rtol=1e-6)
    # mu = 1.6 photo-electrons spread over 10..15 ns: the mean first arrival is 11.80 ns,
    # between the earliest deposit and the weight-averaged time 12.5
    npt.assert_allclose(float(t_meas[0]), _mean_first(np.full(n_rays, 0.16), t, 0.0), rtol=1e-6)
    # unlit sensors report time 0
    npt.assert_array_equal(np.asarray(t_meas[1:]), 0.0)


@pytest.mark.parametrize("tts", [0.5, 2.1])
def test_time_with_the_transit_time_spread_is_the_smeared_mean(tts):
    """Three deposits (one bright, one dim early, one late) on one sensor and one deposit on
    another: the node rule agrees with scipy quadrature to 2e-3 ns."""
    w = jnp.array([3.0, 0.2, 1.0, 0.5])
    idx = jnp.array([0, 0, 0, 1])
    t = jnp.array([20.0, 18.5, 31.0, 50.0])
    _, t_meas = on_deposits(first_arrival_expected, w, idx, t, 3, qe=1.0,
                            qe_corrections=jnp.ones(3), tts=tts)
    npt.assert_allclose(float(t_meas[0]), _mean_first(w[:3], t[:3], tts), atol=2e-3)
    npt.assert_allclose(float(t_meas[1]), _mean_first(w[3:], t[3:], tts), atol=2e-3)
    assert float(t_meas[2]) == 0.0


def test_total_charge_gradient_in_qe_is_the_summed_weight():
    """sum_s Q_s = qe * sum_i w_i, so d/dqe = sum_i w_i = 0.8 + 0.6 = 1.4 exactly."""
    def total(qe):
        q, _ = on_deposits(first_arrival_expected, jnp.array([0.8, 0.6]), jnp.array([0, 1]),
                           jnp.array([10.0, 12.0]), 5, qe=qe,
                           qe_corrections=jnp.ones(5))
        return jnp.sum(q)

    npt.assert_allclose(float(jax.grad(total)(0.2)), 1.4, rtol=1e-6)


def test_qe_correction_gradient_is_per_sensor_and_zero_when_unlit():
    """dQ_total/dc_s = qe * sum_{i in s} w_i: sensor 0 gets 0.2*(0.8+0.6) = 0.28, sensor 1
    0.2*0.5 = 0.10, unlit sensors exactly 0 (a leak through the segment sum would show here)."""
    def total(c):
        q, _ = on_deposits(first_arrival_expected, jnp.array([0.8, 0.6, 0.5]), jnp.array([0, 0, 1]),
                           jnp.array([10.0, 12.0, 8.0]), 5, qe=0.2,
                           qe_corrections=c)
        return jnp.sum(q)

    grad = jax.grad(total)(jnp.ones(5))
    npt.assert_allclose(np.asarray(grad), [0.28, 0.10, 0.0, 0.0, 0.0], rtol=1e-6, atol=0)
