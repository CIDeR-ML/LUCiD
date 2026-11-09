"""The loss functions in lucid/losses.py against their closed forms (scipy / numpy)."""
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from scipy.stats import poisson

from lucid.losses import energy_loss, counts_loss

TRUE_Q = np.array([10.0, 5.0, 0.0, 3.0, 8.0])
PRED_Q = np.array([9.0, 6.0, 0.1, 2.5, 7.5])


class TestPoissonNLL:
    def test_is_the_poisson_negative_log_likelihood(self):
        """counts_loss = -sum log Poisson(true | pred), divided by sum(true) when normalised.

        (The previous pin 0.2918896973 is this value; float32 agrees with scipy to 1e-7.)
        """
        nll = -poisson.logpmf(TRUE_Q, PRED_Q).sum()
        npt.assert_allclose(counts_loss(jnp.asarray(TRUE_Q), jnp.asarray(PRED_Q)),
                            nll / TRUE_Q.sum(), rtol=1e-6)
        npt.assert_allclose(counts_loss(jnp.asarray(TRUE_Q), jnp.asarray(PRED_Q), normalize=False),
                            nll, rtol=1e-6)


class TestEnergyLoss:
    def test_is_the_absolute_log_ratio_of_totals(self):
        npt.assert_allclose(energy_loss(jnp.asarray(PRED_Q), jnp.asarray(TRUE_Q)),
                            abs(np.log(PRED_Q.sum() / TRUE_Q.sum())), rtol=1e-5)

    def test_same_counts_zero_loss(self):
        q = jnp.array([10.0, 5.0, 3.0])
        result = energy_loss(q, q)
        npt.assert_allclose(result, 0.0, atol=1e-7)

