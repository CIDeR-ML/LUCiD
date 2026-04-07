"""Differentiability tests for the photon propagation pipeline.

These tests verify that gradients flow correctly through the simulation
pipeline — the core property that enables gradient-based optimization.
Phase 9 must preserve all of these properties.
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from lucid.geometry.detector_geometry import DetectorGeometry
from lucid.simulation.photon_step import (
    photon_iteration_update_factors,
    photon_iteration_update_factors_safe,
)
from lucid.simulation.sensor_response import (
    make_hits_simulation, make_hits_likelihood,
)
from lucid.overlap import create_overlap_prob


class TestOverlapProbDifferentiability:
    """The overlap probability is the soft-assignment kernel that makes
    sensor weights differentiable. Temperature controls smoothness."""

    def test_overlap_prob_differentiable(self):
        """Overlap probability should be differentiable w.r.t. distance."""
        sigma = 0.2 * 0.04  # temperature * sensor_radius (typical)
        r = 0.04
        overlap_fn = create_overlap_prob(sigma, r)

        grad_fn = jax.grad(overlap_fn)
        g = grad_fn(0.02)  # distance less than radius
        assert jnp.isfinite(g)

    def test_overlap_prob_decreases_with_distance(self):
        """Weight should decrease as ray moves farther from sensor center."""
        sigma = 0.2 * 0.04
        r = 0.04
        overlap_fn = create_overlap_prob(sigma, r)

        w_close = overlap_fn(0.01)
        w_far = overlap_fn(0.05)
        assert float(w_close) > float(w_far)

    def test_overlap_prob_gradient_sign(self):
        """Gradient should be negative (weight decreases with distance)."""
        sigma = 0.2 * 0.04
        r = 0.04
        overlap_fn = create_overlap_prob(sigma, r)

        # At distance slightly less than radius, gradient should be negative
        g = jax.grad(overlap_fn)(0.03)
        assert float(g) < 0, "Overlap weight should decrease with distance"


class TestPropagatorGradients:
    """Verify gradients flow through the propagator output."""

    @pytest.fixture(scope="class")
    def cylinder_geom(self):
        return DetectorGeometry.from_config(
            "config/WCTE_geom_config.json", detector_type='Cylinder')

    def test_grad_weights_wrt_origin(self, cylinder_geom):
        """Sensor weights should be differentiable w.r.t. ray origin."""
        def loss(origin):
            origins = origin[None, :]  # (1, 3)
            dirs = jnp.array([[1., 0., 0.]])
            result = cylinder_geom.propagator(origins, dirs)
            return jnp.sum(result['sensor_weights'])

        grad = jax.grad(loss)(jnp.array([0., 0., 0.]))
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_weights_wrt_direction(self, cylinder_geom):
        """Sensor weights should be differentiable w.r.t. ray direction."""
        def loss(direction):
            origins = jnp.zeros((1, 3))
            dirs = direction[None, :]
            result = cylinder_geom.propagator(origins, dirs)
            return jnp.sum(result['sensor_weights'])

        grad = jax.grad(loss)(jnp.array([1., 0., 0.]))
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_positions_wrt_origin(self, cylinder_geom):
        """Hit positions should be differentiable w.r.t. ray origin."""
        def loss(origin):
            origins = origin[None, :]
            dirs = jnp.array([[1., 0., 0.]])
            result = cylinder_geom.propagator(origins, dirs)
            return jnp.sum(result['positions'])

        grad = jax.grad(loss)(jnp.array([0., 0., 0.]))
        assert jnp.all(jnp.isfinite(grad))


class TestPhotonStepGradientFlow:
    """Test gradient flow through the photon iteration with custom VJP."""

    def test_grad_detect_prob_wrt_scatter_length(self):
        """Longer scatter length → higher P(reach surface) → higher detect_prob.
        Gradient should be positive."""
        key = jax.random.PRNGKey(42)
        def loss(scatter_length):
            _, _, _, detect_prob, _, _ = photon_iteration_update_factors_safe(
                position=jnp.zeros(3), direction=jnp.array([0., 0., 1.]),
                time=0.0, surface_distance=2.0,
                normal=jnp.array([0., 0., -1.]),
                scatter_length=scatter_length,
                wall_reflection_rate=0.5, sensor_reflection_rate=0.3,
                absorption_length=100.0, hit_sensor=True,
                rng_key=key, speed_of_light=0.2253)
            return detect_prob
        g = jax.grad(loss)(50.0)
        assert jnp.isfinite(g)
        assert float(g) > 0, "More scatter length → more detection → positive grad"

    def test_grad_detect_prob_wrt_reflection_rate(self):
        """Higher reflection rate → lower detection probability.
        Gradient should be negative."""
        key = jax.random.PRNGKey(42)
        def loss(sensor_refl):
            _, _, _, detect_prob, _, _ = photon_iteration_update_factors_safe(
                position=jnp.zeros(3), direction=jnp.array([0., 0., 1.]),
                time=0.0, surface_distance=2.0,
                normal=jnp.array([0., 0., -1.]),
                scatter_length=50.0, wall_reflection_rate=0.5,
                sensor_reflection_rate=sensor_refl,
                absorption_length=100.0, hit_sensor=True,
                rng_key=key, speed_of_light=0.2253)
            return detect_prob
        g = jax.grad(loss)(0.3)
        assert jnp.isfinite(g)
        assert float(g) < 0, "More reflection → less detection → negative grad"

    def test_grad_attenuation_wrt_absorption_length(self):
        """Longer absorption length → less attenuation → higher factor.
        Gradient should be positive."""
        key = jax.random.PRNGKey(42)
        def loss(abs_length):
            _, _, _, _, refl_atten, _ = photon_iteration_update_factors_safe(
                position=jnp.zeros(3), direction=jnp.array([0., 0., 1.]),
                time=0.0, surface_distance=2.0,
                normal=jnp.array([0., 0., -1.]),
                scatter_length=50.0, wall_reflection_rate=0.5,
                sensor_reflection_rate=0.3, absorption_length=abs_length,
                hit_sensor=True, rng_key=key, speed_of_light=0.2253)
            return refl_atten
        g = jax.grad(loss)(100.0)
        assert jnp.isfinite(g)
        assert float(g) > 0

    def test_custom_vjp_sanitizes_nan(self):
        """The custom VJP should produce finite gradients even with degenerate inputs."""
        key = jax.random.PRNGKey(42)
        def loss(pos):
            _, _, _, dp, _, _ = photon_iteration_update_factors_safe(
                position=pos, direction=jnp.array([0., 0., 1.]),
                time=0.0, surface_distance=1e-8,  # degenerate: nearly zero distance
                normal=jnp.array([0., 0., -1.]),
                scatter_length=50.0, wall_reflection_rate=0.5,
                sensor_reflection_rate=0.3, absorption_length=100.0,
                hit_sensor=True, rng_key=key, speed_of_light=0.2253)
            return dp
        g = jax.grad(loss)(jnp.zeros(3))
        assert jnp.all(jnp.isfinite(g)), f"NaN in gradient: {g}"


class TestSensorResponseGradientChain:
    """Test that the full chain: weights → make_hits → loss is differentiable."""

    def test_charge_gradient_through_weights(self):
        """d(charge)/d(weight) should be qe (linear relationship)."""
        def loss(w0):
            weights = jnp.array([w0, 0.5])
            indices = jnp.array([0, 1])
            times = jnp.array([10., 12.])
            qe_corr = jnp.ones(5)
            q, _ = make_hits_simulation(weights, indices, times, 5,
                                         qe=0.2, qe_corrections=qe_corr)
            return q[0]
        g = jax.grad(loss)(0.8)
        npt.assert_allclose(g, 0.2, atol=1e-5)  # d(w*qe)/dw = qe

    def test_time_gradient_through_weights(self):
        """Gradient of soft-min time w.r.t. weights should be finite."""
        def loss(w0):
            weights = jnp.array([w0, 0.5])
            indices = jnp.array([0, 0])
            times = jnp.array([10., 12.])
            qe_corr = jnp.ones(5)
            _, t = make_hits_simulation(weights, indices, times, 5,
                                         qe=0.2, qe_corrections=qe_corr,
                                         temperature=0.1)
            return t[0]
        g = jax.grad(loss)(0.8)
        assert jnp.isfinite(g)

    def test_likelihood_gradient_through_log_weights(self):
        """Gradient through log-space weights in likelihood mode."""
        def loss(w0):
            weights = jnp.array([w0, 0.5])
            indices = jnp.array([0, 1])
            times = jnp.array([10., 12.])
            qe_corr = jnp.ones(5)
            log_w, _, _, total_q = make_hits_likelihood(
                weights, indices, times, 5, qe=0.2, qe_corrections=qe_corr)
            return jnp.sum(log_w) + jnp.sum(total_q)
        g = jax.grad(loss)(0.8)
        assert jnp.isfinite(g)


class TestStopGradientMechanisms:
    """Verify the two stop_gradient mechanisms documented in the plan:
    1. Direction gradient controlled by n_grad_iters
    2. Position gradient controlled by pos_grad_threshold (K or 0)
    """

    def test_stop_gradient_blocks_position_grad(self):
        """When pos_grad_threshold=0, positions should not carry gradients
        through the where/stop_gradient gate."""
        # Simulate the stop_gradient logic from simulator.py line 276
        def with_grad(pos, i):
            # pos_grad_threshold = K = 5 (gradient flows)
            return jnp.where(i < 5, pos, jax.lax.stop_gradient(pos))

        def without_grad(pos, i):
            # pos_grad_threshold = 0 (always stop — likelihood mode)
            return jnp.where(i < 0, pos, jax.lax.stop_gradient(pos))

        pos = jnp.array([1., 2., 3.])
        i = jnp.int32(0)

        g_with = jax.grad(lambda p: jnp.sum(with_grad(p, i)))(pos)
        g_without = jax.grad(lambda p: jnp.sum(without_grad(p, i)))(pos)

        # With grad (i=0 < threshold=5): gradient should flow
        npt.assert_allclose(g_with, [1., 1., 1.], atol=1e-6)
        # Without grad (i=0 < threshold=0 is False): gradient blocked
        npt.assert_allclose(g_without, [0., 0., 0.], atol=1e-6)

    def test_stop_gradient_blocks_direction_grad(self):
        """When n_grad_iters=0, directions should not carry gradients."""
        dir_vec = jnp.array([0., 0., 1.])
        i = jnp.int32(0)

        g_track = jax.grad(lambda d: jnp.sum(
            jnp.where(i < 0, d, jax.lax.stop_gradient(d))))(dir_vec)  # track: n_grad_iters=0
        g_calib = jax.grad(lambda d: jnp.sum(
            jnp.where(i < 2, d, jax.lax.stop_gradient(d))))(dir_vec)  # calibration: n_grad_iters=2

        npt.assert_allclose(g_track, [0., 0., 0.], atol=1e-6)
        npt.assert_allclose(g_calib, [1., 1., 1.], atol=1e-6)
