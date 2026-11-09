"""Gradients of the photon step and the sensor response, against their closed forms.

The expected-value photon step deposits

    detect_prob = exp(-Dd / L_s) * exp(-Dd / L_M) * (1 - R) * exp(-Dd / L_a)

for a photon reaching a sensor a distance Dd away (L_s Rayleigh, L_M Mie, L_a absorption
length, R the sensor reflection rate): the probability of crossing Dd unscattered and
unabsorbed and then not being reflected. It does not depend on the random draw, so its
derivatives are exact numbers:

    d/dL_s = detect_prob * Dd / L_s**2      d/dL_a = detect_prob * Dd / L_a**2
    d/dR   = -detect_prob / (1 - R)         d/dDd  = -detect_prob * (1/L_s + 1/L_M + 1/L_a)

The propagator's own gradients (weights and hit positions against the geometry) are in
tests/propagation/test_shared_propagator_differentiability.py; the overlap kernel's slope in
tests/string/test_overlap_fix.py and tests/propagation/test_overlap_branches.py.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from lucid.simulation.photon_step import make_photon_iteration_update_factors_safe
from lucid.simulation.reflection import ScalarReflection
from lucid.simulation.hits import first_arrival_expected, photon_list_expected
from tests.helpers.params import on_deposits

# No detector or simulator is built here: these run by default.

DD, LS, LM, LA, R_SENSOR = 2.0, 50.0, 1.0e9, 100.0, 0.3
DETECT = np.exp(-DD / LS - DD / LM - DD / LA) * (1 - R_SENSOR)


def _detect_prob(scatter_length=LS, sensor_rate=R_SENSOR, absorption_length=LA,
                 surface_distance=DD):
    out = make_photon_iteration_update_factors_safe()(
        position=jnp.zeros(3), direction=jnp.array([0., 0., 1.]),
        time=0.0, surface_distance=surface_distance,
        normal=jnp.array([0., 0., -1.]),
        scatter_length=scatter_length, mie_scatter_length=LM, g=0.0,
        refl_params=ScalarReflection(wall_rate=jnp.asarray(0.5),
                                     sensor_rate=jnp.asarray(sensor_rate)),
        absorption_length=absorption_length, hit_sensor=True,
        lam=jnp.asarray(0.0), rng_key=jax.random.PRNGKey(42), speed_of_light=0.2253)
    return out[3]


class TestPhotonStepGradientFlow:
    """detect_prob and its derivatives against the closed forms in the module docstring.
    float32 throughout: rtol 1e-5 is ~100 ulp."""

    def test_detect_prob_value(self):
        npt.assert_allclose(float(_detect_prob()), DETECT, rtol=1e-6)

    def test_grad_detect_prob_wrt_scatter_length(self):
        g = jax.grad(lambda L: _detect_prob(scatter_length=L))(LS)
        npt.assert_allclose(float(g), DETECT * DD / LS ** 2, rtol=1e-5)

    def test_grad_detect_prob_wrt_reflection_rate(self):
        g = jax.grad(lambda R: _detect_prob(sensor_rate=R))(R_SENSOR)
        npt.assert_allclose(float(g), -DETECT / (1 - R_SENSOR), rtol=1e-5)

    def test_grad_attenuation_wrt_absorption_length(self):
        """Absorption is folded into detect_prob (reflection_attenuation is a constant 1)."""
        g = jax.grad(lambda L: _detect_prob(absorption_length=L))(LA)
        npt.assert_allclose(float(g), DETECT * DD / LA ** 2, rtol=1e-5)

    def test_grad_detect_prob_wrt_surface_distance(self):
        """The pathwise track gradient: Dd is live through reach and absorption."""
        g = jax.grad(lambda d: _detect_prob(surface_distance=d))(DD)
        npt.assert_allclose(float(g), -DETECT * (1 / LS + 1 / LM + 1 / LA), rtol=1e-5)

    def test_gradient_at_vanishing_surface_distance_is_the_closed_form(self):
        """A photon sitting on the surface it just left (Dd -> 0) was the historic NaN source;
        the step no longer scrubs cotangents, so its gradient must be both finite and right:
        at Dd = 1e-8, detect_prob = 1 - R and d/dDd = -(1 - R)(1/L_s + 1/L_M + 1/L_a)."""
        d0 = 1e-8
        g = jax.grad(lambda d: _detect_prob(surface_distance=d))(d0)
        assert np.isfinite(float(g))
        npt.assert_allclose(float(g), -(1 - R_SENSOR) * (1 / LS + 1 / LM + 1 / LA), rtol=1e-5)
        npt.assert_allclose(float(_detect_prob(surface_distance=d0)), 1 - R_SENSOR, rtol=1e-6)


class TestSensorResponseGradientChain:
    """Test that the full chain: weights → hits → loss is differentiable."""

    def test_charge_gradient_through_weights(self):
        """d(charge)/d(weight) should be qe (linear relationship)."""
        def loss(w0):
            weights = jnp.array([w0, 0.5])
            indices = jnp.array([0, 1])
            times = jnp.array([10., 12.])
            qe_corr = jnp.ones(5)
            q, _ = on_deposits(first_arrival_expected, weights, indices, times, 5,
                               qe=0.2, qe_corrections=qe_corr)
            return q[0]
        g = jax.grad(loss)(0.8)
        npt.assert_allclose(g, 0.2, atol=1e-5)  # d(w*qe)/dw = qe

    def test_time_gradient_through_weights_and_times(self):
        """The mean first arrival of deposits a = 0.2 w0 at 10 ns and b = 0.1 at 12 ns is
        m = [(1 - e^-a) 10 + e^-a (1 - e^-b) 12] / (1 - e^-(a+b)). More weight early pulls it
        earlier; dm/dt_k is the probability that the first photo-electron comes from deposit k.
        With the transit-time spread the same derivatives come out of the node rule, checked
        against its own central difference."""
        def m(w0, t0, tts=0.0):
            _, t = on_deposits(first_arrival_expected, jnp.array([w0, 0.5]), jnp.array([0, 0]),
                               jnp.array([t0, 12.]), 5, qe=0.2,
                               qe_corrections=jnp.ones(5), tts=tts)
            return t[0]

        def closed(w0):
            a, b = 0.2 * w0, 0.1
            return ((1 - np.exp(-a)) * 10 + np.exp(-a) * (1 - np.exp(-b)) * 12) / (1 - np.exp(-a - b))

        npt.assert_allclose(float(m(0.8, 10.)), closed(0.8), rtol=1e-6)
        dm_dw = (closed(0.8 + 1e-6) - closed(0.8 - 1e-6)) / 2e-6
        npt.assert_allclose(float(jax.grad(m, 0)(0.8, 10.)), dm_dw, rtol=1e-4)
        assert dm_dw < 0
        p_first = (1 - np.exp(-0.16)) / (1 - np.exp(-0.26))
        npt.assert_allclose(float(jax.grad(m, 1)(0.8, 10.)), p_first, rtol=1e-5)
        for arg, h in ((0, 0.05), (1, 0.05)):
            x = [0.8, 10.]
            hi = list(x); hi[arg] += h
            lo = list(x); lo[arg] -= h
            fd = (float(m(*hi, tts=2.0)) - float(m(*lo, tts=2.0))) / (2 * h)
            npt.assert_allclose(float(jax.grad(m, arg)(0.8, 10., 2.0)), fd, rtol=2e-2, atol=1e-4)

    def test_likelihood_gradient_through_log_weights(self):
        """sum(log(qe w)) + sum(qe w): d/dw0 = 1/w0 + qe."""
        def loss(w0):
            weights = jnp.array([w0, 0.5])
            indices = jnp.array([0, 1])
            times = jnp.array([10., 12.])
            log_w, _, _, total_q = on_deposits(photon_list_expected,
                weights, indices, times, 5, qe=0.2, qe_corrections=jnp.ones(5))
            return jnp.sum(log_w) + jnp.sum(total_q)
        npt.assert_allclose(float(jax.grad(loss)(0.8)), 1 / 0.8 + 0.2, rtol=1e-6)
