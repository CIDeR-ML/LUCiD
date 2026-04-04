"""Tests for loss functions (tools/losses.py and tools/optimization/losses.py)."""
import jax.numpy as jnp
import numpy.testing as npt

from tools.losses import poisson_nll
from tools.optimization.losses import (
    energy_loss, counts_loss, segment_logsumexp,
)


class TestPoissonNLL:
    def test_reference(self):
        true_q = jnp.array([10.0, 5.0, 0.0, 3.0, 8.0])
        pred_q = jnp.array([9.0, 6.0, 0.1, 2.5, 7.5])
        result = poisson_nll(true_q, pred_q)
        npt.assert_allclose(result, 0.2918896973133087, atol=1e-6)

    def test_perfect_match_near_zero(self):
        q = jnp.array([10.0, 5.0, 3.0])
        result = poisson_nll(q, q)
        # perfect prediction should give small loss (not exactly 0 due to Poisson NLL form)
        assert result < 1.0

    def test_counts_loss_matches(self):
        """poisson_nll in losses.py and counts_loss in optimization/losses.py compute the same thing."""
        true_q = jnp.array([10.0, 5.0, 0.0, 3.0, 8.0])
        pred_q = jnp.array([9.0, 6.0, 0.1, 2.5, 7.5])
        npt.assert_allclose(poisson_nll(true_q, pred_q),
                            counts_loss(true_q, pred_q), atol=1e-7)


class TestEnergyLoss:
    def test_reference(self):
        sim = jnp.array([9.0, 6.0, 0.1, 2.5, 7.5])
        true = jnp.array([10.0, 5.0, 0.0, 3.0, 8.0])
        result = energy_loss(sim, true)
        npt.assert_allclose(result, 0.035228703171014786, atol=1e-6)

    def test_same_counts_zero_loss(self):
        q = jnp.array([10.0, 5.0, 3.0])
        result = energy_loss(q, q)
        npt.assert_allclose(result, 0.0, atol=1e-7)


class TestSegmentLogsumexp:
    def test_reference(self):
        data = jnp.array([1.0, 2.0, 3.0, 0.5, 1.5])
        indices = jnp.array([0, 0, 1, 1, 2])
        result = segment_logsumexp(data, indices, 3)
        npt.assert_allclose(result, [2.3132615089416504, 3.078889846801758, 1.5], atol=1e-5)

    def test_single_element_segments(self):
        data = jnp.array([2.0, 3.0, 4.0])
        indices = jnp.array([0, 1, 2])
        result = segment_logsumexp(data, indices, 3)
        # logsumexp of a single element is just that element
        npt.assert_allclose(result, [2.0, 3.0, 4.0], atol=1e-6)
