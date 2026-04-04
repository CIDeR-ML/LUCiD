"""Physics-level tests for loss functions.

Tests verify gradient directions, numerical stability, and mathematical
properties of each loss function.
"""
import jax
import jax.numpy as jnp
import numpy.testing as npt

from lucid.losses import (
    counts_loss, energy_loss, segment_logsumexp,
    origin_time_loss, smooth_pinball, softplus,
    first_arrival_nll,
)


class TestCountsLossGradients:
    """Poisson NLL gradient should point toward the correct prediction."""

    def test_gradient_direction_overprediction(self):
        """When pred > true, gradient should push pred down (positive grad)."""
        true = jnp.array([10.0, 5.0])
        def loss(pred):
            return counts_loss(true, pred)
        grad = jax.grad(loss)(jnp.array([15.0, 8.0]))  # overprediction
        # d/d(pred) of Poisson NLL: 1 - true/pred > 0 when pred > true
        assert jnp.all(grad > 0)

    def test_gradient_direction_underprediction(self):
        """When pred < true, gradient should push pred up (negative grad)."""
        true = jnp.array([10.0, 5.0])
        def loss(pred):
            return counts_loss(true, pred)
        grad = jax.grad(loss)(jnp.array([5.0, 2.0]))  # underprediction
        assert jnp.all(grad < 0)

    def test_minimum_near_true(self):
        """Loss should be minimized near pred ≈ true."""
        true = jnp.array([10.0, 5.0, 8.0])
        losses = []
        for scale in [0.5, 0.8, 1.0, 1.2, 1.5]:
            losses.append(float(counts_loss(true, true * scale)))
        # Loss at scale=1.0 (perfect prediction) should be the minimum
        assert losses[2] <= min(losses[0], losses[1], losses[3], losses[4])


class TestEnergyLoss:
    def test_symmetric_in_ratio(self):
        """energy_loss = |log(sum_sim/sum_true)|, so doubling and halving give same loss."""
        true = jnp.array([10.0, 10.0])
        loss_double = energy_loss(true * 2, true)
        loss_half = energy_loss(true * 0.5, true)
        npt.assert_allclose(loss_double, loss_half, atol=1e-5)

    def test_monotonic_with_ratio(self):
        """Loss should increase as prediction deviates from truth."""
        true = jnp.array([10.0, 5.0])
        l1 = energy_loss(true * 1.1, true)
        l2 = energy_loss(true * 1.5, true)
        l3 = energy_loss(true * 2.0, true)
        assert l1 < l2 < l3


class TestSegmentLogsumexpStability:
    def test_large_values_no_overflow(self):
        """segment_logsumexp should handle large values without overflow."""
        data = jnp.array([500.0, 501.0, 499.0])
        indices = jnp.array([0, 0, 0])
        result = segment_logsumexp(data, indices, 1)
        assert jnp.all(jnp.isfinite(result))
        # logsumexp([500, 501, 499]) ≈ 501 + log(1 + exp(-1) + exp(-2))
        expected = 501.0 + jnp.log(1 + jnp.exp(-1.0) + jnp.exp(-2.0))
        npt.assert_allclose(result[0], expected, atol=0.01)

    def test_negative_values(self):
        """Should work correctly with negative values."""
        data = jnp.array([-10.0, -5.0, -20.0])
        indices = jnp.array([0, 0, 0])
        result = segment_logsumexp(data, indices, 1)
        assert jnp.all(jnp.isfinite(result))
        npt.assert_allclose(result[0], -5.0 + jnp.log(1 + jnp.exp(-5.0) + jnp.exp(-15.0)), atol=0.01)


class TestSmoothPinball:
    def test_asymmetric_penalty(self):
        """For tau < 0.5, negative residuals (early photons) penalized more."""
        tau = 0.1
        loss_early = smooth_pinball(-1.0, tau=tau)   # early photon (r < 0)
        loss_late = smooth_pinball(1.0, tau=tau)      # late photon (r > 0)
        assert float(loss_early) > float(loss_late)

    def test_minimum_is_finite(self):
        """Smooth pinball loss at a range of inputs should all be finite and non-negative."""
        tau = 0.2
        for r in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            val = smooth_pinball(r, tau=tau)
            assert float(val) >= 0.0
            assert jnp.isfinite(val)


class TestSoftplus:
    def test_approximates_relu(self):
        """softplus(x) ≈ max(x, 0) for large |x|."""
        npt.assert_allclose(softplus(10.0), 10.0, atol=1e-3)
        npt.assert_allclose(softplus(-10.0), 0.0, atol=1e-3)

    def test_smooth_at_zero(self):
        """softplus is smooth — gradient exists at x=0."""
        grad_pos = jax.grad(softplus)(0.1)
        grad_neg = jax.grad(softplus)(-0.1)
        assert jnp.isfinite(grad_pos)
        assert jnp.isfinite(grad_neg)
        # Should transition from ~0 (x<<0) to ~1 (x>>0)
        assert float(grad_neg) < float(grad_pos)
