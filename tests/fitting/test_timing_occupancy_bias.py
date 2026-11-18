"""The occupancy tables of the timing calibration: the Poisson-conditioned first-arrival bias."""
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy import integrate
from scipy.stats import norm

from lucid.fitting.timing import _occ_bias_mean


def _occ_bias_exact(mu):
    """E[min(z_1..z_N) | N ~ Poisson(mu), N >= 1], z_i ~ N(0, 1), from the survival function.

    P(min > x, N >= 1) = sum_n P(N=n) S(x)^n - P(N=0) = exp(-mu Phi(x)) - exp(-mu), so the
    conditioned survival is G(x) = (exp(-mu Phi(x)) - exp(-mu)) / (1 - exp(-mu)) and
    E[min] = int_0^inf G dx - int_-inf^0 (1 - G) dx. A different route from the library's
    (order statistics per n, then a Poisson mixture over a grid of mu).
    """
    G = lambda x: (np.exp(-mu * norm.cdf(x)) - np.exp(-mu)) / (-np.expm1(-mu))
    hi = integrate.quad(G, 0, np.inf, epsabs=1e-12)[0]
    lo = integrate.quad(lambda x: 1 - G(x), -np.inf, 0, epsabs=1e-12)[0]
    return hi - lo


class TestOccBiasMean:
    """The Poisson-conditioned occupancy early-bias E[min | N~Poisson(μ), N≥1]."""

    @pytest.mark.parametrize("mu", [1e-3, 0.1, 0.5, 1.0, 2.0, 4.0, 10.0, 30.0, 100.0, 300.0])
    def test_matches_the_survival_integral(self, mu):
        """Measured agreement <= 1.5e-5 relative over mu in [1e-3, 300]."""
        npt.assert_allclose(float(_occ_bias_mean(jnp.array(mu))), _occ_bias_exact(mu),
                            rtol=1e-4)

    def test_vanishes_as_mu_to_zero(self):
        npt.assert_allclose(float(_occ_bias_mean(jnp.array(1e-4))), 0.0, atol=1e-3)

    def test_nonzero_at_low_occupancy(self):
        """Unlike Blom-of-μ (which is 0 for μ≤1), the conditioned bias is small but
        NONZERO at low occupancy — the physically-correct behaviour that matters for
        the low-occupancy timing levers."""
        b = float(_occ_bias_mean(jnp.array(1.0)))
        assert b < -0.1   # hand value ≈ -0.28 (Poisson(1)-weighted Σ m_n)
        npt.assert_allclose(b, -0.28, atol=0.06)

    def test_negative_and_monotone(self):
        occ = jnp.array([0.5, 1.0, 2.0, 5.0, 20.0, 100.0])
        b = _occ_bias_mean(occ)
        assert jnp.all(b <= 0)
        assert jnp.all(jnp.diff(b) < 0)            # more photons → earlier
        # high occupancy → ≈ Blom E[min of ~100] ≈ -2.5
        npt.assert_allclose(float(_occ_bias_mean(jnp.array(100.0))), -2.5, atol=0.2)


def test_occupancy_tables_survive_a_jit_then_an_eager_call():
    """The occupancy bias/variance tables are built once; a first call inside jit must not
    leave tracers behind for the next trace or an eager call."""
    import jax
    from lucid.fitting import timing
    jitted = float(jax.jit(timing._occ_bias_mean)(1.0))
    assert float(timing._occ_bias_mean(1.0)) == jitted
    assert float(jax.jit(timing._occ_bias_var)(3.0)) == float(timing._occ_bias_var(3.0))
